import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

import pytorch_lightning as pl
from taming.diffusion_modules import AttnBlock, Downsample, Normalize, ResnetBlock
from dynamic_utils import instantiate_from_config


class TripleGrainEncoder(pl.LightningModule):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=128,
        router_config=None,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        self.down, self.down_resolutions = self._make_downsampling_layers(
            ch, ch_mult, num_res_blocks, attn_resolutions, dropout, resamp_with_conv
        )

        self.mid_coarse = self._make_middle_layers(ch * ch_mult[-1], dropout)
        self.norm_out_coarse = Normalize(ch * ch_mult[-1])
        self.conv_out_coarse = nn.Conv2d(
            ch * ch_mult[-1], z_channels, kernel_size=3, stride=1, padding=1
        )

        ch_median = ch * ch_mult[-2]
        self.mid_median = self._make_middle_layers(ch_median, dropout)
        self.norm_out_median = Normalize(ch_median)
        self.conv_out_median = nn.Conv2d(
            ch_median, z_channels, kernel_size=3, stride=1, padding=1
        )

        ch_fine = ch * ch_mult[-3]
        self.mid_fine = self._make_middle_layers(ch_fine, dropout)
        self.norm_out_fine = Normalize(ch_fine)
        self.conv_out_fine = nn.Conv2d(
            ch_fine, z_channels, kernel_size=3, stride=1, padding=1
        )

        self.router = instantiate_from_config(router_config)

    def _make_downsampling_layers(
        self, ch, ch_mult, num_res_blocks, attn_resolutions, dropout, resamp_with_conv
    ):
        down = nn.ModuleList()
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        down_resolutions = []
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down.append(nn.ModuleDict({"block": block, "attn": attn}))
            down_resolutions.append(curr_res)

            if i_level != self.num_resolutions - 1:
                down[-1]["downsample"] = Downsample(block_in, resamp_with_conv)
                curr_res //= 2

        return down, down_resolutions

    def _make_middle_layers(self, ch, dropout):
        mid = nn.ModuleDict(
            {
                "block_1": ResnetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ),
                "attn_1": AttnBlock(ch),
                "block_2": ResnetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ),
            }
        )
        return mid

    def _apply_middle_layers(self, x, mid, temb):
        x = mid["block_1"](x, temb)
        x = mid["attn_1"](x)
        x = mid["block_2"](x, temb)
        return x

    def forward(self, x, x_entropy=None):
        print('new', x.shape)

        assert (
            x.shape[2] == x.shape[3] == self.resolution
        ), f"{x.shape[2]}, {x.shape[3]}, {self.resolution}"

        temb = None

        hs = [self.conv_in(x)]
        print('new_hs', hs.shape)
        for i_level, down in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                h = down["block"][i_block](hs[-1], temb)
                if down["attn"]:
                    h = down["attn"][i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(down["downsample"](hs[-1]))

            if i_level == self.num_resolutions - 2:
                h_median = hs[-1]
            elif i_level == self.num_resolutions - 3:
                h_fine = hs[-1]

        h_coarse = hs[-1]

        h_coarse = self._apply_middle_layers(h_coarse, self.mid_coarse, temb)
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = F.silu(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        h_median = self._apply_middle_layers(h_median, self.mid_median, temb)
        h_median = self.norm_out_median(h_median)
        h_median = F.silu(h_median)
        h_median = self.conv_out_median(h_median)

        h_fine = self._apply_middle_layers(h_fine, self.mid_fine, temb)
        h_fine = self.norm_out_fine(h_fine)
        h_fine = F.silu(h_fine)
        h_fine = self.conv_out_fine(h_fine)

        gate = self.router(
            h_fine=h_fine, h_median=h_median, h_coarse=h_coarse, entropy=x_entropy
        )
        if self.training:
            gate = F.gumbel_softmax(gate, tau=1, dim=-1, hard=True)
        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)

        h_coarse = h_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        h_median = h_median.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

        indices_repeat = (
            indices.repeat_interleave(4, dim=-1)
            .repeat_interleave(4, dim=-2)
            .unsqueeze(1)
        )

        h_triple = torch.where(indices_repeat == 0, h_coarse, h_median)
        h_triple = torch.where(indices_repeat == 1, h_median, h_triple)
        h_triple = torch.where(indices_repeat == 2, h_fine, h_triple)

        if self.training:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(4, dim=-1).repeat_interleave(
                4, dim=-2
            )
            h_triple = h_triple * gate_grad

        coarse_mask = 0.0625 * torch.ones_like(indices_repeat).to(x.device)
        median_mask = 0.25 * torch.ones_like(indices_repeat).to(x.device)
        fine_mask = 1.0 * torch.ones_like(indices_repeat).to(x.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, median_mask)
        codebook_mask = torch.where(indices_repeat == 1, median_mask, codebook_mask)
        codebook_mask = torch.where(indices_repeat == 2, fine_mask, codebook_mask)

        return {
            "h_triple": h_triple,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }
