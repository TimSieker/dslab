"""Load evaluation results for analysis (used by notebooks/evaluation.ipynb)."""

from __future__ import annotations

import json

import pandas as pd
import xarray as xr
import yaml

from .experiment import REPO_ROOT, Experiment

REFERENCE_CSV = REPO_ROOT / "docs" / "reference" / "supplement_S12_S13_scorecards.csv"

# Direction of "better" per metric: -1 lower is better, +1 higher is better
BETTER = {"PSS": 1, "Corr (temporal)": 1}


def list_runs(exp: Experiment) -> list[str]:
    if not exp.eval_dir.exists():
        return []
    return sorted(p.name for p in exp.eval_dir.iterdir() if p.is_dir() and any(p.glob("*/scores.json")))


def load_scores(exp: Experiment) -> pd.DataFrame:
    """Long table: run, split, variable, metric, value, n_members."""
    rows = []
    for f in sorted(exp.eval_dir.glob("*/*/scores.json")):
        r = json.loads(f.read_text())
        for var, scores in r["scores"].items():
            for metric, value in scores.items():
                rows.append({"run": r["run"], "split": r["split"], "variable": var, "metric": metric,
                             "value": value, "n_members": r["n_members"]})
    return pd.DataFrame(rows, columns=["run", "split", "variable", "metric", "value", "n_members"])


def load_reference(exp: Experiment) -> pd.DataFrame:
    """Published scorecard values (supplement S12/S13) mapped onto this experiment's split names."""
    ref = exp.reference
    if not ref:
        return pd.DataFrame(columns=["model", "split", "variable", "metric", "value"])
    df = pd.read_csv(REFERENCE_CSV)
    df = df[(df.domain == ref["domain"]) & (df.experiment == ref["experiment"])]
    triangle_to_split = {tri: split for split, tri in ref["splits"].items()}
    df = df.assign(split=df.triangle.map(triangle_to_split)).dropna(subset=["split"])
    return df[["model", "split", "variable", "metric", "value"]].reset_index(drop=True)


def scorecard(exp: Experiment, split: str, runs: list[str] | None = None,
              ref_models: list[str] | None = None, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wide table (rows: our runs, then published models; columns: (variable, metric)) for one split,
    restricted to the published scorecard metrics."""
    from .metrics import SCORECARD

    scores = load_scores(exp) if scores is None else scores
    ours = scores[scores.split == split]
    if runs is not None:
        ours = ours[ours.run.isin(runs)]
    ours = ours.rename(columns={"run": "model"}).assign(source="this study")
    ref = load_reference(exp)
    ref = ref[ref.split == split]
    if ref_models is not None:
        ref = ref[ref.model.isin(ref_models)]
    ref = ref.assign(source="published")
    both = pd.concat([ours, ref], ignore_index=True)
    cols = [(v, m) for v, ms in SCORECARD.items() for m in ms]
    both = both[[(v, m) in cols for v, m in zip(both.variable, both.metric)]]
    wide = both.pivot_table(index=["source", "model"], columns=["variable", "metric"], values="value", sort=False)
    order = [c for c in cols if c in wide.columns]
    return wide[order]


def load_diagnostics(exp: Experiment, run: str, split: str) -> xr.Dataset:
    return xr.open_dataset(exp.eval_dir / run / split / "diagnostics.nc")


def load_fields(exp: Experiment, split: str, dates, var: str, run: str | None = None, member: int = 0):
    """Fields of single days, (D, H, W): ground truth (run=None, read from the cache) or one member
    of a run's predictions. Only plain (not combined) splits."""
    import numpy as np

    from .data import _load_source

    dates = np.asarray(dates, dtype="datetime64[D]")
    if run is None:
        time, _, y = _load_source(exp, exp.source(split))
        idx = [int(np.flatnonzero(time.astype("datetime64[D]") == d)[0]) for d in dates]
        return np.asarray(y[idx, exp.variables.index(var)])
    with xr.open_dataset(exp.predictions_dir / run / f"{split}.nc") as ds:
        days = ds["time"].values.astype("datetime64[D]")
        idx = [int(np.flatnonzero(days == d)[0]) for d in dates]
        return ds[var].isel(member=member, time=idx).values


def run_info(exp: Experiment, runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        d = exp.runs_dir / run
        spec = yaml.safe_load((d / "model.yaml").read_text()) if (d / "model.yaml").exists() else {}
        info = json.loads((d / "fit_info.json").read_text()) if (d / "fit_info.json").exists() else {}
        meta = yaml.safe_load((d / "run.yaml").read_text()) if (d / "run.yaml").exists() else {}
        rows.append({"run": run, "model": spec.get("model"), "params": spec.get("params"),
                     "selected": info.get("selected"), "fit_seconds": info.get("wall_seconds"),
                     "fitted": meta.get("time"), "git_commit": (meta.get("git_commit") or "")[:8]})
    return pd.DataFrame(rows).set_index("run")
