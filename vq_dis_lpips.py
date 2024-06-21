import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from DynamicVectorQuantization.modules.discriminator.model import weights_init
from DynamicVectorQuantization.utils.utils import instantiate_from_config


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(
    weight: float, global_step: int, threshold: int = 0, value: float = 0.0
) -> float:
    return value if global_step < threshold else weight


def log(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(t + eps)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(logits_fake)


def vanilla_d_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )


def bce_discr_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor
) -> torch.Tensor:
    return (
        -log(1 - torch.sigmoid(logits_fake)) - log(torch.sigmoid(logits_real))
    ).mean()


def bce_gen_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -log(torch.sigmoid(logits_fake)).mean()


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        disc_config: dict,
        disc_init: bool,
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_conditional: bool = False,
        disc_adaptive_loss: bool = True,  # use adaptive weight or fixed
        disc_loss: str = "hinge",
        disc_weight_max: float = None,
        budget_loss_config: dict = None,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "bce"]

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_adaptive_loss = disc_adaptive_loss
        self.disc_weight_max = disc_weight_max
        self.discriminator_iter_start = disc_start
        self.budget_loss_config = budget_loss_config

        # Initialize the LPIPS model from the lpips library
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()

        # Initialize the discriminator
        self.discriminator = instantiate_from_config(disc_config)
        if disc_init:
            self.discriminator.apply(weights_init)

        # Set GAN loss functions
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "bce":
            self.disc_loss = bce_discr_loss
            self.gen_loss = bce_gen_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        # Initialize budget loss if config is provided
        if budget_loss_config is not None:
            self.budget_loss = instantiate_from_config(budget_loss_config)

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: nn.Module = None
    ) -> torch.Tensor:
        if last_layer is None:
            last_layer = self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach() * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss: torch.Tensor,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: nn.Module = None,
        cond: torch.Tensor = None,
        split: str = "train",
        gate: torch.Tensor = None,
    ):
        rec_loss = torch.abs(inputs - reconstructions)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss += self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss.mean()

        # Generator update
        if optimizer_idx == 0:
            logits_fake = self.discriminator(
                torch.cat((reconstructions, cond), dim=1)
                if cond is not None
                else reconstructions
            )
            g_loss = self.gen_loss(logits_fake)

            if self.disc_adaptive_loss:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                if self.disc_weight_max is not None:
                    d_weight.clamp_max_(self.disc_weight_max)
            else:
                d_weight = torch.tensor(self.disc_weight_max)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, self.discriminator_iter_start
            )
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            if gate is not None and self.budget_loss_config is not None:
                budget_loss = self.budget_loss(gate=gate)
                loss += budget_loss

            log = {
                f"{split}_total_loss": loss.clone().detach().mean(),
                f"{split}_quant_loss": codebook_loss.detach().mean(),
                f"{split}_nll_loss": nll_loss.detach().mean(),
                f"{split}_rec_loss": rec_loss.detach().mean(),
                f"{split}_p_loss": p_loss.detach().mean(),
                f"{split}_d_weight": d_weight.detach(),
                f"{split}_disc_factor": torch.tensor(disc_factor),
                f"{split}_g_loss": g_loss.detach().mean(),
            }
            if gate is not None and self.budget_loss_config is not None:
                log[f"{split}_budget_loss"] = budget_loss.detach().mean()

            return loss, log

        # Discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(
                torch.cat((inputs, cond), dim=1)
                if cond is not None
                else inputs.detach()
            )
            logits_fake = self.discriminator(
                torch.cat((reconstructions, cond), dim=1)
                if cond is not None
                else reconstructions.detach()
            )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}_disc_loss": d_loss.clone().detach().mean(),
                f"{split}_logits_real": logits_real.detach().mean(),
                f"{split}_logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log
