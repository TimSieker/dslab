"""Normalisation shared by all models. Statistics always come from the training split only."""

from __future__ import annotations

import numpy as np


class PredictorScaler:
    """Standardise predictors per channel and grid point; NaNs (missing 850 hPa values in some
    domains) become 0 = the training mean after scaling."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> PredictorScaler:
        x64 = x.astype(np.float64)
        self.mean = np.nanmean(x64, axis=0).astype(np.float32)
        self.std = np.maximum(np.nanstd(x64, axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> PredictorScaler:
        return cls(state["mean"], state["std"])


class TargetTransform:
    """Per-variable invertible transform of the targets, e.g. {"pr": "sqrt"}.

    Supported: "none", "sqrt", "log1p" (pr in mm/day). Inverse transforms clip at 0 for pr-like
    transforms, so model outputs never produce negative precipitation.
    """

    KINDS = ("none", "sqrt", "log1p")

    def __init__(self, variables: tuple[str, ...], kinds: dict[str, str] | None = None):
        kinds = kinds or {}
        unknown = set(kinds) - set(variables)
        if unknown:
            raise KeyError(f"transform given for unknown variables {unknown}")
        self.variables = tuple(variables)
        self.kinds = {v: kinds.get(v, "none") for v in variables}
        for k in self.kinds.values():
            if k not in self.KINDS:
                raise ValueError(f"unknown transform {k!r}; choose from {self.KINDS}")

    def forward(self, y: np.ndarray) -> np.ndarray:
        """y: (..., V, H, W) in physical units."""
        out = np.empty_like(y, dtype=np.float32)
        for i, v in enumerate(self.variables):
            k, yi = self.kinds[v], y[..., i, :, :]
            out[..., i, :, :] = (np.sqrt(np.maximum(yi, 0)) if k == "sqrt"
                                 else np.log1p(np.maximum(yi, 0)) if k == "log1p" else yi)
        return out

    def inverse(self, z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float32)
        for i, v in enumerate(self.variables):
            k, zi = self.kinds[v], z[..., i, :, :]
            out[..., i, :, :] = (np.maximum(zi, 0) ** 2 if k == "sqrt"
                                 else np.expm1(np.maximum(zi, 0)) if k == "log1p" else zi)
        return out
