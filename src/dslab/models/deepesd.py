"""DeepESD: the CORDEX-ML-Bench reference CNN (3 conv layers on the coarse grid + dense output layer)."""

from __future__ import annotations

from .base import register
from .networks import DeepESDNet
from .torch_base import TorchRegressor


@register("deepesd")
class DeepESD(TorchRegressor):
    summary = "benchmark reference CNN: 3 conv layers on the coarse grid + dense layer to every grid point"
    arch_defaults = dict(filters=[50, 25, 1], lr=1e-4, batch_size=100, max_epochs=2000, patience=30,
                         lr_patience=1000, amp=False)  # original recipe: Adam 1e-4, early stopping only

    def build_network(self, n_in, in_hw, n_out, out_hw):
        return DeepESDNet(n_in, in_hw, n_out, out_hw, filters=tuple(self.params["filters"]))
