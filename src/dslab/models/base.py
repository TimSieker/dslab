"""Common interface of all downscaling models.

A model is fitted on the train split (val split for model selection / early stopping) and
predicts physical-unit targets of shape (member, time, variable, y, x). Deterministic models
return a single member. To add a model: subclass Downscaler, decorate with @register("<name>"),
import it in models/__init__.py, and add configs/models/<name>.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..data import Grid, Split

REGISTRY: dict[str, type[Downscaler]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def build_model(name: str, variables, predictors, **params) -> Downscaler:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](variables=variables, predictors=predictors, **params)


class Downscaler:
    name = ""
    stochastic = False  # True: predict() draws n_members samples

    def __init__(self, variables, predictors, **params):
        self.variables = tuple(variables)
        self.predictors = tuple(predictors)
        self.params = params

    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        """Fit the model; return a JSON-serialisable dict with fit information."""
        raise NotImplementedError

    def predict(self, split: Split, grid: Grid, n_members: int = 1, seed: int = 0) -> np.ndarray:
        """Return (M, T, V, H, W) float32 in physical units."""
        raise NotImplementedError

    # Persistence: model.yaml (what to build) + state (what was learned).
    # Array-based models use state_dict(); torch models override save_state/load_state.
    def state_dict(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def save_state(self, run_dir: Path) -> None:
        np.savez(run_dir / "state.npz", **self.state_dict())

    def load_state(self, run_dir: Path) -> None:
        with np.load(run_dir / "state.npz") as f:
            self.load_state_dict({k: f[k] for k in f.files})

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = {"model": self.name, "variables": list(self.variables),
                "predictors": list(self.predictors), "params": self.params}
        (run_dir / "model.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        self.save_state(run_dir)

    @staticmethod
    def load(run_dir: Path) -> Downscaler:
        spec = yaml.safe_load((run_dir / "model.yaml").read_text())
        model = build_model(spec["model"], spec["variables"], spec["predictors"], **spec["params"])
        model.load_state(run_dir)
        return model
