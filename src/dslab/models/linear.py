"""Linear baseline: ridge regression on the leading principal components of the predictors.

Classic perfect-prognosis regression. Predictors are standardised per channel and grid point and
flattened; the targets (pr optionally sqrt-transformed) are regressed at every target grid point
on the k leading PCs with ridge penalty alpha. k and alpha are chosen per variable by validation
RMSE in physical units.
"""

from __future__ import annotations

import time

import numpy as np

from ..data import Grid, Split
from ..preprocessing import PredictorScaler, TargetTransform
from .base import Downscaler, register


@register("linear")
class LinearRidge(Downscaler):
    def __init__(self, variables, predictors, transform=None,
                 n_pcs=(50, 100, 200, 400, 800, 1600, 3840), alphas=(0.0, 1e1, 1e2, 1e3, 1e4, 1e5)):
        super().__init__(variables, predictors, transform=dict(transform or {}),
                         n_pcs=list(n_pcs), alphas=[float(a) for a in alphas])
        self.tf = TargetTransform(self.variables, transform)
        self.scaler = None
        self.components = None  # (V, k_max, P) right singular vectors; rows beyond k_v are zero
        self.coef = None  # (V, k_max, H*W)
        self.y_mean = None  # (V, H*W) in transformed space

    def _project(self, x: np.ndarray, v: int) -> np.ndarray:
        z = self.scaler.transform(x).reshape(len(x), -1)
        return z @ self.components[v].T

    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        t0 = time.time()
        self.scaler = PredictorScaler().fit(train.x)
        X = self.scaler.transform(train.x).reshape(len(train), -1).astype(np.float64)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        n_pcs = [k for k in self.params["n_pcs"] if k <= len(s)]
        k_max = max(n_pcs)
        Pval = self.scaler.transform(val.x).reshape(len(val), -1) @ Vt[:k_max].T  # (Tv, k_max)

        ztr = self.tf.forward(train.y).reshape(len(train), len(self.variables), -1)
        V, HW = ztr.shape[1:]
        self.components = np.zeros((V, k_max, X.shape[1]), np.float32)
        self.coef = np.zeros((V, k_max, HW), np.float32)
        self.y_mean = ztr.mean(axis=0)
        info = {"explained_variance": (np.cumsum(s**2) / np.sum(s**2))[[k - 1 for k in n_pcs]].tolist(),
                "n_pcs": n_pcs, "alphas": self.params["alphas"], "val_rmse": {}, "selected": {}}

        for v, var in enumerate(self.variables):
            B = U[:, :k_max].T @ (ztr[:, v] - self.y_mean[v])  # (k_max, HW)
            yval = val.y[:, v].reshape(len(val), -1)
            scores = np.full((len(n_pcs), len(self.params["alphas"])), np.nan)
            for i, k in enumerate(n_pcs):
                for j, a in enumerate(self.params["alphas"]):
                    shrink = s[:k] / (s[:k] ** 2 + a)
                    zhat = (Pval[:, :k] * shrink) @ B[:k] + self.y_mean[v]
                    yhat = self._inverse_one(zhat, v)
                    scores[i, j] = np.sqrt(np.mean((yhat - yval) ** 2))
            i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
            k, a = n_pcs[i], self.params["alphas"][j]
            self.components[v, :k] = Vt[:k]
            self.coef[v, :k] = (s[:k] / (s[:k] ** 2 + a))[:, None] * B[:k]
            info["val_rmse"][var] = scores.tolist()
            info["selected"][var] = {"n_pcs": int(k), "alpha": float(a), "val_rmse": float(scores[i, j])}
        info["fit_seconds"] = round(time.time() - t0, 1)
        return info

    def _inverse_one(self, z: np.ndarray, v: int) -> np.ndarray:
        """Inverse transform for variable v of a (T, HW) array."""
        kind = self.tf.kinds[self.variables[v]]
        if kind == "sqrt":
            return np.maximum(z, 0) ** 2
        if kind == "log1p":
            return np.expm1(np.maximum(z, 0))
        return z

    def predict(self, split: Split, grid: Grid, n_members: int = 1, seed: int = 0) -> np.ndarray:
        out = np.empty((1, len(split), len(self.variables), *grid.shape), np.float32)
        for v in range(len(self.variables)):
            z = self._project(split.x, v) @ self.coef[v] + self.y_mean[v]
            out[0, :, v] = self._inverse_one(z, v).reshape(len(split), *grid.shape)
        return out

    def state_dict(self):
        return {"x_mean": self.scaler.mean, "x_std": self.scaler.std, "components": self.components,
                "coef": self.coef, "y_mean": self.y_mean}

    def load_state_dict(self, state):
        self.scaler = PredictorScaler(state["x_mean"], state["x_std"])
        self.components, self.coef, self.y_mean = state["components"], state["coef"], state["y_mean"]
