"""U-Net regression on the target grid: predictors bilinearly interpolated to the target grid points
(true lat/lon positions) plus standardised orography as input channels."""

from __future__ import annotations

import numpy as np
import torch

from ..data import Grid
from .base import register
from .networks import RegridUNet, sample_grid
from .torch_base import TorchRegressor


@register("unet")
class UNetRegressor(TorchRegressor):
    summary = "residual U-Net on the target grid; predictors interpolated to the grid points + orography"
    arch_defaults = dict(base=32, mults=[1, 2, 4, 8], blocks=2, dropout=0.0, lr=3e-4, weight_decay=1e-4,
                         batch_size=16, max_epochs=60, patience=8)

    def build_network(self, n_in, in_hw, n_out, out_hw):
        p = self.params
        return RegridUNet(n_in, n_out, out_hw, n_static=1, base=p["base"], mults=tuple(p["mults"]),
                          blocks=p["blocks"], dropout=p["dropout"])

    def set_geometry(self, net, grid: Grid) -> None:
        orog = (grid.orog - grid.orog.mean()) / grid.orog.std()
        net.set_geometry(sample_grid(grid.lat, grid.lon, grid.pred_lat, grid.pred_lon),
                         torch.from_numpy(orog.astype(np.float32))[None, None])
