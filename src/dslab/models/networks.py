"""Neural network building blocks (PyTorch)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DeepESDNet(nn.Module):
    """DeepESD (Baño-Medina et al. 2020; CORDEX-ML-Bench example): three 3x3 convolutions on the coarse
    predictor grid, then one dense layer to every target grid point. Here one network predicts all
    variables jointly from a shared convolutional trunk."""

    def __init__(self, n_in: int, in_hw: tuple[int, int], n_out: int, out_hw: tuple[int, int],
                 filters=(50, 25, 1)):
        super().__init__()
        chans = [n_in, *filters]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans[:-1], chans[1:]))
        self.n_out, self.out_hw = n_out, tuple(out_hw)
        self.dense = nn.Linear(filters[-1] * in_hw[0] * in_hw[1], n_out * out_hw[0] * out_hw[1])

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return self.dense(x.flatten(1)).view(len(x), self.n_out, *self.out_hw)


def _groups(c: int) -> int:
    return min(32, c // 4) if c >= 8 else 1


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(_groups(cin), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = nn.GroupNorm(_groups(cout), cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.emb = nn.Linear(emb_dim, 2 * cout) if emb_dim else None  # FiLM scale/shift

    def forward(self, x, emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.emb is not None:
            scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] (flow-matching time)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class UNet(nn.Module):
    """Plain residual U-Net. Resolution halves at every level after the first, so mults=(1, 2, 4, 8)
    on a 128x128 grid reaches 16x16 (the predictor resolution) at the bottom. With time_dim set,
    every block is FiLM-conditioned on a time embedding (for flow matching / diffusion)."""

    def __init__(self, cin: int, cout: int, base: int = 32, mults=(1, 2, 4, 8), blocks: int = 2,
                 dropout: float = 0.0, time_dim: int | None = None):
        super().__init__()
        self.time_dim = time_dim
        emb = 4 * time_dim if time_dim else None
        if time_dim:
            self.time_mlp = nn.Sequential(nn.Linear(time_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        chans = [base * m for m in mults]
        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        c = chans[0]
        for i, ch in enumerate(chans):
            self.down.append(nn.ModuleList(ResBlock(c if j == 0 else ch, ch, emb, dropout) for j in range(blocks)))
            c = ch
            if i < len(chans) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.mid = nn.ModuleList([ResBlock(c, c, emb, dropout), ResBlock(c, c, emb, dropout)])
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.up.append(nn.ModuleList(ResBlock((c + ch) if j == 0 else ch, ch, emb, dropout)
                                         for j in range(blocks)))
            c = ch
            if i > 0:
                self.upsample.append(nn.Conv2d(ch, chans[i - 1], 3, padding=1))
                c = chans[i - 1]
        self.out = nn.Sequential(nn.GroupNorm(_groups(c), c), nn.SiLU(), nn.Conv2d(c, cout, 3, padding=1))

    def forward(self, x, t=None):
        emb = self.time_mlp(timestep_embedding(t, self.time_dim)) if self.time_dim else None
        h = self.stem(x)
        skips = []
        for i, blocks in enumerate(self.down):
            for b in blocks:
                h = b(h, emb)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        for b in self.mid:
            h = b(h, emb)
        for i, blocks in enumerate(self.up):
            h = torch.cat([h, skips.pop()], dim=1)
            for b in blocks:
                h = b(h, emb)
            if i < len(self.upsample):
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


def sample_grid(lat: np.ndarray, lon: np.ndarray, pred_lat: np.ndarray, pred_lon: np.ndarray) -> torch.Tensor:
    """grid_sample coordinates (1, H, W, 2) that bilinearly interpolate the regular, ascending
    predictor grid to the target grid points (align_corners=True: -1/+1 = first/last grid centre)."""
    gx = 2 * (lon - pred_lon[0]) / (pred_lon[-1] - pred_lon[0]) - 1
    gy = 2 * (lat - pred_lat[0]) / (pred_lat[-1] - pred_lat[0]) - 1
    if np.abs(gx).max() > 1 or np.abs(gy).max() > 1:
        raise ValueError("target grid extends beyond the predictor grid")
    return torch.from_numpy(np.stack([gx, gy], axis=-1)[None].astype(np.float32))


class RegridUNet(nn.Module):
    """Interpolates the coarse predictors bilinearly onto the target grid (true lat/lon positions),
    appends static fields (standardised orography) and applies a U-Net."""

    def __init__(self, n_pred: int, n_out: int, out_hw: tuple[int, int], n_static: int = 1, **unet_kw):
        super().__init__()
        H, W = out_hw
        self.register_buffer("grid", torch.zeros(1, H, W, 2))
        self.register_buffer("static", torch.zeros(1, n_static, H, W))
        self.unet = UNet(n_pred + n_static, n_out, **unet_kw)

    def set_geometry(self, grid: torch.Tensor, static: torch.Tensor) -> None:
        self.grid.copy_(grid)
        self.static.copy_(static)

    def upsample(self, x):
        g = self.grid.expand(len(x), -1, -1, -1)
        up = F.grid_sample(x, g, mode="bilinear", align_corners=True)
        return torch.cat([up, self.static.expand(len(x), -1, -1, -1)], dim=1)

    def forward(self, x, t=None):
        return self.unet(self.upsample(x), t)
