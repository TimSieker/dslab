"""Experiment definitions: which files form the train, validation and test splits.

An experiment is a YAML file in configs/experiments/ (see alps_esd.yaml). Adding another
CORDEX-ML-Bench experiment or domain only needs a new YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
WORK = REPO_ROOT / "work"  # symlink to /work/bb1596/b384340/downscaling-lab


@dataclass(frozen=True)
class Source:
    """A predictor file and the matching target file (None if no ground truth is available)."""

    predictors: Path
    target: Path | None


@dataclass(frozen=True, eq=False)  # identity hash: usable as an lru_cache key
class Experiment:
    name: str
    domain: str
    description: str
    data_root: Path
    variables: tuple[str, ...]
    predictors: tuple[str, ...]
    static: Path
    train: Source
    val_years: tuple[int, ...]
    tests: dict[str, Source]
    default_splits: tuple[str, ...]
    combined_splits: dict[str, tuple[str, ...]]  # evaluation-only: test splits concatenated in time
    default_eval_splits: tuple[str, ...]
    reference: dict  # published scorecard to compare against: experiment, domain, split -> triangle
    grid_crs: dict | None  # projection of the target grid (for maps); None = regular lat/lon
    truth: str  # description of the target data, for figure captions
    benchmark_tests: tuple[dict, ...]  # official tests: {test, name, split}

    @classmethod
    def load(cls, name_or_path: str | Path) -> Experiment:
        """Load by name (configs/experiments/<name>.yaml) or by path."""
        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIG_DIR / "experiments" / f"{name_or_path}.yaml"
        cfg = yaml.safe_load(path.read_text())
        root = Path(cfg["data_root"])

        def source(d: dict) -> Source:
            return Source(root / d["predictors"], root / d["target"] if d.get("target") else None)

        return cls(
            name=cfg["name"],
            domain=cfg["domain"],
            description=cfg.get("description", ""),
            data_root=root,
            variables=tuple(cfg["variables"]),
            predictors=tuple(cfg["predictors"]),
            static=root / cfg["static"],
            train=source(cfg["train"]),
            val_years=tuple(cfg["val_years"]),
            tests={k: source(v) for k, v in cfg["tests"].items()},
            default_splits=tuple(cfg["default_splits"]),
            combined_splits={k: tuple(v) for k, v in (cfg.get("combined_splits") or {}).items()},
            default_eval_splits=tuple(cfg.get("default_eval_splits", cfg["default_splits"])),
            reference=cfg.get("reference") or {},
            grid_crs=cfg.get("grid_crs"),
            truth=cfg.get("truth", "target data"),
            benchmark_tests=tuple(cfg.get("benchmark_tests") or ()),
        )

    @property
    def splits(self) -> list[str]:
        return ["train", "val", *self.tests]

    def source(self, split: str) -> Source:
        if split in ("train", "val"):
            return self.train
        if split not in self.tests:
            raise KeyError(f"unknown split {split!r} for {self.name}; available: {self.splits}")
        return self.tests[split]

    # Output locations on WORK, one directory per experiment
    @property
    def runs_dir(self) -> Path:
        return WORK / "runs" / self.name

    @property
    def predictions_dir(self) -> Path:
        return WORK / "predictions" / self.name

    @property
    def eval_dir(self) -> Path:
        return WORK / "eval" / self.name
