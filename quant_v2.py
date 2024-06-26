import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class VQEmbedding(nn.Embedding):
    """VQ embedding module with EMA update and random restart of unused codes."""

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            # Initialize EMA buffers
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])
        distances = (
            inputs_flat.pow(2).sum(dim=1, keepdim=True)
            + codebook_t.pow(2).sum(dim=0, keepdim=True)
            - 2 * inputs_flat @ codebook_t
        )
        return distances.reshape(*inputs.shape[:-1], -1)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        return x + torch.rand_like(x) * std

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]
        vectors, idxs = vectors.reshape(-1, embed_dim), idxs.reshape(-1)
        one_hot_idxs = vectors.new_zeros(n_embed, vectors.shape[0])
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, vectors.shape[0])
        )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            if vectors.shape[0] < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            vectors_random = vectors[
                torch.randperm(vectors.shape[0], device=vectors.device)
            ][:n_embed]

            if dist.is_initialized():
                dist.broadcast(vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1)).add_((1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.view(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs)
            self._update_embedding()
        return self.embed(embed_idxs), embed_idxs

    def embed(self, idxs):
        return super().forward(idxs)


class VectorQuantize2(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim=None,
        accept_image_fmap=True,
        commitment_beta=0.25,
        decay=0.99,
        restart_unused_codes=True,
        channel_last=False,
    ):
        super().__init__()
        self.accept_image_fmap = accept_image_fmap
        self.beta = commitment_beta
        self.channel_last = channel_last
        self.codebook = VQEmbedding(
            codebook_size,
            codebook_dim,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
        )
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x, codebook_mask=None, *ignorewargs, **ignorekwargs):
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        if not self.channel_last and not self.accept_image_fmap:
            x = rearrange(x, "b d n -> b n d").contiguous()

        flatten = rearrange(x, "h ... d -> h (...) d").contiguous()
        x_q, x_code = self.codebook(flatten)

        if codebook_mask is not None:
            codebook_mask = (
                rearrange(codebook_mask, "b c h w -> b (h w) c").contiguous()
                if codebook_mask.dim() == 4
                else codebook_mask.unsqueeze(-1)
            )
            loss = self.beta * torch.mean(
                (x_q.detach() - x) ** 2 * codebook_mask
            ) + torch.mean((x_q - x.detach()) ** 2 * codebook_mask)
        else:
            loss = self.beta * torch.mean((x_q.detach() - x) ** 2) + torch.mean(
                (x_q - x.detach()) ** 2
            )

        x_q = x + (x_q - x).detach()

        if not self.channel_last and not self.accept_image_fmap:
            x_q = rearrange(x_q, "b n d -> b d n").contiguous()

        if self.accept_image_fmap:
            x_q = rearrange(x_q, "b (h w) c -> b c h w", h=height, w=width).contiguous()
            x_code = rearrange(
                x_code, "b (h w) ... -> b h w ...", h=height, w=width
            ).contiguous()

        return x_q, loss, (None, None, x_code)

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        distances = self.codebook.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)

        if stochastic:
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1).reshape(*soft_code.shape[:-1])
        else:
            code = distances.argmin(dim=-1)

        return soft_code, code

    def get_codebook_entry(self, indices, *kwargs):
        return self.codebook.embed(indices)
