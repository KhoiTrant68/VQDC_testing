import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from DynamicVectorQuantization.modules.losses.vqperceptual_multidisc import VQLPIPSWithDiscriminator as OldVQLPIPSWithDiscriminator
from VQDC_testing.vq_dis_lpips import VQLPIPSWithDiscriminator as NewVQLPIPSWithDiscriminator

# Dummy configurations for testing
disc_start = 0
disc_config_orig = {
    "target": "DynamicVectorQuantization.modules.losses.vqperceptual_multidisc.VQLPIPSWithDiscriminator",
    "params": {
        "disc_start": 0,
        "disc_config": {
            "target": "DynamicVectorQuantization.modules.discriminator.model.NLayerDiscriminator",
            "params": {
                "input_nc": 3,
                "ndf": 64,
                "n_layers": 3,
                "use_actnorm": False
            }
        },
        "disc_init": True,
        "codebook_weight": 1.0,
        "pixelloss_weight": 1.0,
        "disc_factor": 1.0,
        "disc_weight": 1.0,
        "perceptual_weight": 1.0,
        "disc_conditional": False,
        "disc_loss": "hinge",
        "disc_weight_max": 0.75,
        "budget_loss_config": {
            "target": "DynamicVectorQuantization.modules.dynamic_modules.budget.BudgetConstraint_RatioMSE_DualGrain",
            "params": {
                "target_ratio": 0.5,
                "gamma": 10.0,
                "min_grain_size": 16,
                "max_grain_size": 32,
                "calculate_all": True
            }
        }
    }
}

disc_config_new = {
    "target": "VQDC_testing.vq_dis_lpips.VQLPIPSWithDiscriminator",
    "params": {
        "disc_start": 0,
        "disc_config": {
            "target": "VQDC_testing.discriminator.NLayerDiscriminator",
            "params": {
                "input_nc": 3,
                "ndf": 64,
                "n_layers": 3,
                "use_actnorm": False
            }
        },
        "disc_init": True,
        "codebook_weight": 1.0,
        "pixelloss_weight": 1.0,
        "disc_factor": 1.0,
        "disc_weight": 1.0,
        "perceptual_weight": 1.0,
        "disc_conditional": False,
        "disc_loss": "hinge",
        "disc_weight_max": 0.75,
        "budget_loss_config": {
            "target": "DynamicVectorQuantization.modules.dynamic_modules.budget.BudgetConstraint_RatioMSE_DualGrain",
            "params": {
                "target_ratio": 0.5,
                "gamma": 10.0,
                "min_grain_size": 16,
                "max_grain_size": 32,
                "calculate_all": True
            }
        }
    }
}

disc_init = True

# Initialize both old and new models
old_model = OldVQLPIPSWithDiscriminator(
    disc_start=disc_start,
    disc_config=disc_config_orig,
    disc_init=disc_init,
)

new_model = NewVQLPIPSWithDiscriminator(
    disc_start=disc_start,
    disc_config=disc_config_new,
    disc_init=disc_init,
)

# Set models to evaluation mode for consistency
old_model.eval()
new_model.eval()

# Create dummy data for testing
batch_size = 2
channels = 3
height = 64
width = 64

codebook_loss = torch.rand(batch_size)
inputs = torch.rand(batch_size, channels, height, width)
reconstructions = torch.rand(batch_size, channels, height, width)
optimizer_idx = 0
global_step = 1000
last_layer = None
cond = None
split = "test"
gate = None

# Copy the old model state to the new model to ensure same initialization
new_model.load_state_dict(deepcopy(old_model.state_dict()), strict=False)

# Run forward pass on both models

with torch.no_grad():
    old_loss, old_log = old_model(
        codebook_loss=codebook_loss,
        inputs=inputs,
        reconstructions=reconstructions,
        optimizer_idx=optimizer_idx,
        global_step=global_step,
        last_layer=last_layer,
        cond=cond,
        split=split,
        gate=gate,
    )

    new_loss, new_log = new_model(
        codebook_loss=codebook_loss,
        inputs=inputs,
        reconstructions=reconstructions,
        optimizer_idx=optimizer_idx,
        global_step=global_step,
        last_layer=last_layer,
        cond=cond,
        split=split,
        gate=gate,
    )

# Compare the outputs
def compare_tensors(t1, t2, tol=1e-6):
    return torch.allclose(t1, t2, atol=tol)

print("Loss comparison:", compare_tensors(old_loss, new_loss))

for key in old_log:
    print(f"{key} comparison:", compare_tensors(old_log[key], new_log[key]))
