"""Day-of-year climatology: the floor every model has to beat. Ignores the predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from ..data import Grid, Split
from .base import Downscaler, register


def doy365(time: np.ndarray) -> np.ndarray:
    """Day of a 365-day year (0..364); 29 February maps onto 28 February."""
    t = pd.DatetimeIndex(time)
    doy = t.dayofyear.values - 1
    return doy - ((t.is_leap_year & (doy >= 59)).astype(int))


@register("climatology")
class Climatology(Downscaler):
    """Training-period mean for each calendar day, smoothed with a circular `window`-day running mean."""

    summary = "mean of each calendar day over the training years (31-day smoothing); ignores the predictors"

    def __init__(self, variables, predictors, window: int = 31):
        super().__init__(variables, predictors, window=window)
        self.clim = None  # (365, V, H, W)

    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        doy = doy365(train.time)
        clim = np.stack([train.y[doy == d].mean(axis=0, dtype=np.float64) for d in range(365)])
        self.clim = uniform_filter1d(clim, size=self.params["window"], axis=0, mode="wrap").astype(np.float32)
        return {"n_train_days": len(train)}

    def predict(self, split: Split, grid: Grid, n_members: int = 1, seed: int = 0) -> np.ndarray:
        return self.clim[doy365(split.time)][None]

    def state_dict(self):
        return {"clim": self.clim}

    def load_state_dict(self, state):
        self.clim = state["clim"]
