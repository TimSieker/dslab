"""Load evaluation results for analysis (used by notebooks/evaluation.ipynb)."""

from __future__ import annotations

import json
from pathlib import Path

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


def rank_vs_published(card: pd.DataFrame) -> pd.DataFrame:
    """For each of our runs and metric: 'k / N' = number of published entries that score better, out of N.
    card: output of scorecard() without ref_models (all published entries)."""
    pub = card.xs("published", level="source")
    ours = card.xs("this study", level="source")
    out = pd.DataFrame(index=ours.index, columns=card.columns, dtype=object)
    for col in card.columns:
        p = pub[col].dropna()
        hi = BETTER.get(col[1], -1) > 0
        for run, v in ours[col].items():
            if pd.notna(v):
                n_better = int((p > v).sum() if hi else (p < v).sum())
                out.loc[run, col] = f"{n_better} / {len(p)}"
    return out


def load_diagnostics(exp: Experiment, run: str, split: str) -> xr.Dataset:
    return xr.open_dataset(exp.eval_dir / run / split / "diagnostics.nc")


def load_fields(exp: Experiment, split: str, dates, var: str, run: str | None = None, member=0):
    """Fields of single days, (D, H, W): ground truth (run=None, read from the cache) or one member of a
    run's predictions (member="mean": ensemble mean). Only plain (not combined) splits."""
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
        da = ds[var].isel(time=idx)
        return (da.mean("member") if member == "mean" else da.isel(member=member)).values


def load_history(exp: Experiment, run: str) -> pd.DataFrame | None:
    """Training history of a neural run (None for models without one). Column `epochs_done` = passes over
    the training data completed at that record; `attrs["loss_name"]` names the training loss."""
    f = exp.runs_dir / run / "fit_info.json"
    info = json.loads(f.read_text()) if f.exists() else {}
    if not info.get("history"):
        return None
    h = pd.DataFrame(info["history"])
    h["epochs_done"] = h["epoch"] if "samples" in h else h["epoch"] + 1
    h.attrs["loss_name"] = info.get("loss_name", "MSE of standardised targets")
    return h


def _split_time(exp: Experiment, split: str):
    """Time axis of a split (from the data cache; combined splits concatenated)."""
    import numpy as np

    from .data import _cache_dir, _load_source

    parts = exp.combined_splits.get(split, (split,))
    times = []
    for part in parts:
        src = exp.source(part)
        d = _cache_dir(exp, src)
        if not (d / "time.npy").exists():
            _load_source(exp, src)
        t = np.load(d / "time.npy")
        if part in ("train", "val"):
            is_val = np.isin(pd.DatetimeIndex(t).year, exp.val_years)
            t = t[is_val] if part == "val" else t[~is_val]
        times.append(t)
    return times


def describe_split(exp: Experiment, split: str) -> str:
    """One line: period(s), number of days and predictor type of a split."""
    times = _split_time(exp, split)
    parts = exp.combined_splits.get(split, (split,))
    n = sum(len(t) for t in times)
    years = ", ".join(map(str, exp.val_years))
    if split == "val":
        period = years
    elif split == "train":
        period = f"{pd.Timestamp(times[0][0]).year}–{pd.Timestamp(times[0][-1]).year} without {years}"
    else:
        period = " + ".join(f"{pd.Timestamp(t[0]).year}–{pd.Timestamp(t[-1]).year}" for t in times)
        if len(parts) > 1:
            period += " pooled"
    path = str(exp.source(parts[0]).predictors)
    gcm = Path(path).stem.split("_")[0]
    kind = (f"imperfect predictors: {gcm} GCM fields" if "imperfect" in path
            else f"perfect predictors: upscaled {gcm}-driven RCM fields")
    return f"`{split}` ({period}; {n} days; {kind})"


def caption(exp: Experiment, shown: str, splits, runs, scores: pd.DataFrame | None = None):
    """Markdown block stating what a figure shows, what the models were trained on and what they are
    evaluated on (derived from the experiment config and the data cache, so it stays correct)."""
    from IPython.display import Markdown

    splits = [splits] if isinstance(splits, str) else list(splits)
    runs_txt = ", ".join(runs)
    if scores is not None and len(scores):
        m = scores[scores.run.isin(runs)].groupby("run").n_members.max()
        runs_txt = ", ".join(f"{r} ({int(m[r])} members)" if m.get(r, 1) > 1 else r for r in runs)
        if all(m.get(r, 1) == 1 for r in runs):
            runs_txt += " (all deterministic)"
    text = (f"**Shown:** {shown}  \n"
            f"**Trained on:** {describe_split(exp, 'train')}; model selection on {describe_split(exp, 'val')}.  \n"
            f"**Evaluated on:** " + "; ".join(describe_split(exp, s) for s in splits) +
            f". Truth: {exp.truth}.  \n"
            f"**Runs:** {runs_txt}")
    return Markdown(text)


def benchmark_tests(exp: Experiment, runs: list[str] | None = None) -> pd.DataFrame:
    """The benchmark's tests for this experiment: period, predictors, whether they enter the published
    scorecard, and which runs have been evaluated on them. Reads only the config and eval directories."""
    import re

    runs = list_runs(exp) if runs is None else runs
    rows = []
    for t in exp.benchmark_tests:
        split = t["split"]
        parts = exp.combined_splits.get(split, (split,))
        files = [exp.source(part).predictors for part in parts]
        periods = " + ".join(m for f in files for m in re.findall(r"\d{4}-\d{4}", f.stem))
        gcm = files[0].stem.split("_")[0]
        kind = "imperfect (GCM)" if "imperfect" in str(files[0]) else "perfect (upscaled RCM)"
        done = [r for r in runs if (exp.eval_dir / r / split / "scores.json").exists()]
        rows.append({"test": t["test"], "name": t["name"], "split": split, "period": periods.replace("-", "–"),
                     "predictors": kind, "driving GCM": gcm,
                     "in published scorecard": "yes" if split in exp.reference.get("splits", {}) else "",
                     "evaluated runs": ", ".join(done) or "—"})
    return pd.DataFrame(rows).set_index("test")


def _units(units: str, standard_name: str) -> str:
    import re

    if units == "1":
        return "kg/kg" if "humidity" in standard_name else "–"
    return re.sub(r"(\S+) (\S+)-1", r"\1/\2", units)


def _grid_text(ds: xr.Dataset, exp: Experiment) -> str:
    """'16 × 16, 2° lat/lon (32–62°N, 4°W–26°E)' or '128 × 128, 12.5 km, Lambert conformal'."""
    if ds["lat"].ndim == 1:
        lat, lon = ds["lat"].values, ds["lon"].values
        ew = lambda v: f"{abs(v):g}°{'W' if v < 0 else 'E'}"  # noqa: E731
        return (f"{len(lat)} × {len(lon)}, {abs(lat[1] - lat[0]):g}° lat/lon "
                f"({lat.min():g}–{lat.max():g}°N, {ew(lon.min())}–{ew(lon.max())})")
    ydim, xdim = ds["lat"].dims
    dx = abs(float(ds[xdim][1] - ds[xdim][0]))
    proj = {"lcc": "Lambert conformal"}.get((exp.grid_crs or {}).get("proj"), "curvilinear")
    return f"{ds.sizes[ydim]} × {ds.sizes[xdim]}, {dx:g} {ds[xdim].attrs.get('units', '')}, {proj}"


def variables_table(exp: Experiment) -> pd.DataFrame:
    """Predictors, static fields and target variables of the experiment, with long names, pressure levels,
    units and grids read from the training files (metadata only)."""
    rows = []
    with xr.open_dataset(exp.train.predictors) as ds:
        grid = _grid_text(ds, exp)
        groups: dict[str, list[str]] = {}
        for name in exp.predictors:  # "u_850" -> quantity u at 850 hPa
            qty, _, level = name.rpartition("_")
            groups.setdefault(qty if level.isdigit() else name, []).append(name)
        for qty, names in groups.items():
            a = ds[names[0]].attrs
            levels = [n.rpartition("_")[2] for n in names if n.rpartition("_")[2].isdigit()]
            rows.append({"role": "predictor", "variable": qty, "description": a.get("long_name", ""),
                         "levels": ", ".join(levels) + " hPa" if levels else "surface",
                         "channels": ", ".join(names),
                         "units": _units(a.get("units", ""), a.get("standard_name", "")), "grid": grid})
    with xr.open_dataset(exp.static) as ds:
        for name in ds.data_vars:
            a = ds[name].attrs
            rows.append({"role": "static", "variable": name, "description": a.get("long_name", ""),
                         "levels": "surface", "channels": name,
                         "units": _units(a.get("units", ""), a.get("standard_name", "")),
                         "grid": _grid_text(ds, exp)})
    with xr.open_dataset(exp.train.target) as ds:
        for name in exp.variables:
            a = ds[name].attrs
            rows.append({"role": "target (downscaled)", "variable": name, "description": a.get("long_name", ""),
                         "levels": "surface", "channels": name,
                         "units": _units(a.get("units", ""), a.get("standard_name", "")),
                         "grid": _grid_text(ds, exp)})
    return pd.DataFrame(rows).set_index(["role", "variable"])


def run_info(exp: Experiment, runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        d = exp.runs_dir / run
        spec = yaml.safe_load((d / "model.yaml").read_text()) if (d / "model.yaml").exists() else {}
        info = json.loads((d / "fit_info.json").read_text()) if (d / "fit_info.json").exists() else {}
        meta = yaml.safe_load((d / "run.yaml").read_text()) if (d / "run.yaml").exists() else {}
        from .models import REGISTRY

        cls = REGISTRY.get(spec.get("model"))
        rows.append({"run": run, "model": spec.get("model"), "summary": getattr(cls, "summary", ""),
                     "n_params": info.get("n_params"), "hardware": info.get("device", "CPU"),
                     "fit_seconds": info.get("wall_seconds"), "epochs": info.get("epochs"),
                     "selected": info.get("selected"), "params": spec.get("params"),
                     "fitted": meta.get("time"), "git_commit": (meta.get("git_commit") or "")[:8]})
    return pd.DataFrame(rows).set_index("run")
