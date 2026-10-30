"""Evaluate one run on one split.

Writes work/eval/<exp>/<run>/<split>/
    scores.json       scalar scores per variable (scorecard metrics + development metrics)
    diagnostics.nc    small arrays for plotting: maps, spectra, histograms, quantiles, seasonal cycle,
                      domain-mean series, rank histogram (ensembles)
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import metrics as M
from .data import load_grid, load_split
from .experiment import Experiment

QUANTILES = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999])
PLOT_BINS = {
    "pr": np.concatenate([[0.0], np.logspace(-1, np.log10(600), 60)]),  # mm/day, first bin = dry
    "tasmax": M.PSS_BINS,  # degC
}


def _parts(exp: Experiment, split: str) -> tuple[str, ...]:
    return exp.combined_splits.get(split, (split,))


def load_truth(exp: Experiment, split: str):
    """(time, y) with y (T, V, H, W); combined splits are concatenated in time."""
    parts = [load_split(exp, s) for s in _parts(exp, split)]
    if any(p.y is None for p in parts):
        raise ValueError(f"no ground truth for {split}")
    return np.concatenate([p.time for p in parts]), np.concatenate([p.y for p in parts])


def load_pred(exp: Experiment, run: str, split: str, var: str):
    """(time, pred) with pred (M, T, H, W)."""
    times, arrs = [], []
    for s in _parts(exp, split):
        with xr.open_dataset(exp.predictions_dir / run / f"{s}.nc") as ds:
            arrs.append(ds[var].transpose("member", "time", ...).values)
            times.append(ds["time"].values)
    return np.concatenate(times), np.concatenate(arrs, axis=1)


def variable_scores(var: str, pred: np.ndarray, truth: np.ndarray, time: np.ndarray,
                    spec: M.RadialSpectrum) -> tuple[dict, dict]:
    """Scores and plotting diagnostics for one variable. pred (M, T, H, W), truth (T, H, W)."""
    t_index = pd.DatetimeIndex(time)
    years, months = t_index.year.values, t_index.month.values
    n_mem = pred.shape[0]
    ens = pred.mean(axis=0) if n_mem > 1 else pred[0]
    idx_members = pred[:M.N_INDEX_MEMBERS]
    s, d = {}, {}

    # Ensemble-mean errors
    err = ens.astype(np.float64) - truth
    s["RMSE"] = float(np.sqrt(np.mean(err**2)))
    d["mean_true"] = M.time_mean(truth)
    d["bias_mean"] = M.time_mean(ens) - d["mean_true"]
    s["Mean Bias"] = M.rms_space(d["bias_mean"])
    d["rmse_map"] = np.sqrt(np.mean(err**2, axis=0))
    d["corr_map"] = M.temporal_corr(ens, truth)
    s["MAE"] = float(np.mean(np.abs(err)))
    s["Bias (domain mean)"] = float(np.mean(err))
    s["Corr (temporal)"] = float(np.nanmean(d["corr_map"]))
    del err

    # Extreme / climate indices: bias map per member (<= 5), averaged over members
    if var == "pr":
        indices = {"SDII": M.sdii, "Rx1day": lambda a: M.annual_max_mean(a, years)}
    else:
        indices = {"TXx": lambda a: M.annual_max_mean(a, years)}
    for name, fn in indices.items():
        true_map = fn(truth)
        with warnings.catch_warnings():  # grid points never wet in any member stay NaN (dropped later)
            warnings.simplefilter("ignore", RuntimeWarning)
            bias = np.nanmean([fn(m) - true_map for m in idx_members], axis=0)
        d[f"{name}_true"], d[f"{name}_bias"] = true_map, bias
        s[name] = M.rms_space(bias)
    if var == "tasmax":  # IAV on one member
        d["IAV_true"] = M.interannual_std(truth, years)
        d["IAV_bias"] = M.interannual_std(pred[0], years) - d["IAV_true"]
        s["IAV"] = M.rms_space(d["IAV_bias"])
    d["p99_true"] = M.quantile_map(truth, 0.99)
    d["p99_bias"] = M.quantile_map(pred[0], 0.99) - d["p99_true"]
    s["P99 bias"] = M.rms_space(d["p99_bias"])
    if var == "pr":
        d["wetfrac_true"] = M.wet_fraction(truth)
        d["wetfrac_bias"] = M.wet_fraction(pred[0]) - d["wetfrac_true"]
        s["Wet-day freq bias"] = M.rms_space(d["wetfrac_bias"])

    # Distributions: LHD (pr) / PSS (tasmax) per member, averaged over all members
    if var == "pr":
        h_true = M.histogram(truth, M.LHD_BINS)
        s["LHD"] = float(np.nanmean([M.lhd(M.histogram(m, M.LHD_BINS), h_true) for m in pred]))
        to_plot = lambda a: a  # noqa: E731
    else:
        h_true = M.histogram(truth - 273.15, M.PSS_BINS)
        s["PSS"] = float(np.nanmean([M.pss(M.histogram(m - 273.15, M.PSS_BINS), h_true) for m in pred]))
        to_plot = lambda a: a - 273.15  # noqa: E731
    d["hist_true"] = M.histogram(to_plot(truth), PLOT_BINS[var])
    d["hist_pred"] = np.mean([M.histogram(to_plot(m), PLOT_BINS[var]) for m in pred], axis=0)
    d["quant_true"] = np.quantile(truth, QUANTILES)
    d["quant_pred"] = np.quantile(pred[0], QUANTILES)

    # Spectra: RALSD on one member
    d["spec_true"] = spec.time_mean(truth)
    d["spec_pred"] = spec.time_mean(pred[0])
    s["RALSD"] = M.ralsd(d["spec_true"], d["spec_pred"])

    # Time structure
    d["season_true"] = np.array([truth[months == m].mean() for m in range(1, 13)])
    d["season_pred"] = np.array([ens[months == m].mean() for m in range(1, 13)])
    d["series_true"] = truth.mean(axis=(1, 2))
    d["series_pred"] = ens.mean(axis=(1, 2))

    # Probabilistic scores (CRPS = MAE for deterministic models)
    s["CRPS"] = M.crps_ensemble(pred, truth)
    if n_mem > 1:
        s["Spread/skill"] = M.spread_skill(pred, truth)
        d["rank_hist"] = M.rank_histogram(pred, truth)
        d["spec_ensmean"] = spec.time_mean(ens)
    return s, d


def _diag_dataset(var: str, d: dict, grid, time) -> xr.Dataset:
    ydim, xdim = grid.dims
    dims = {
        "map": (ydim, xdim),
        "spec": ("wavenumber",),
        "hist": (f"{var}_bin",),
        "quant": ("quantile",),
        "season": ("month",),
        "series": ("time",),
        "rank": ("rank",),
    }
    data = {}
    for name, arr in d.items():
        kind = ("spec" if name.startswith("spec") else "hist" if name.startswith("hist")
                else "quant" if name.startswith("quant") else "season" if name.startswith("season")
                else "series" if name.startswith("series") else "rank" if name == "rank_hist" else "map")
        data[f"{var}_{name}"] = (dims[kind], np.asarray(arr, dtype=np.float32))
    edges = PLOT_BINS[var]
    coords = {f"{var}_bin": 0.5 * (edges[1:] + edges[:-1]), f"{var}_bin_lower": (f"{var}_bin", edges[:-1]),
              f"{var}_bin_upper": (f"{var}_bin", edges[1:]), "quantile": QUANTILES,
              "month": np.arange(1, 13), "time": time}
    return xr.Dataset(data, coords=coords)


def evaluate_run(exp: Experiment, run: str, split: str, overwrite: bool = False) -> dict:
    out = exp.eval_dir / run / split
    if (out / "scores.json").exists() and not overwrite:
        print(f"[evaluate] {run} {split}: exists, skipping (use --overwrite)")
        return json.loads((out / "scores.json").read_text())
    t0 = time.time()
    grid = load_grid(exp)
    spec = M.RadialSpectrum(grid.shape)
    t_true, y_true = load_truth(exp, split)

    scores, dsets, n_mem = {}, [], None
    for v, var in enumerate(exp.variables):
        t_pred, pred = load_pred(exp, run, split, var)
        if not np.array_equal(t_pred.astype("datetime64[D]"), t_true.astype("datetime64[D]")):
            raise ValueError(f"{run}/{split}: prediction and truth time axes differ")
        n_mem = pred.shape[0]
        s, d = variable_scores(var, pred, y_true[:, v], t_true, spec)
        scores[var] = s
        dsets.append(_diag_dataset(var, d, grid, t_true))
        del pred

    diag = xr.merge(dsets)
    diag = diag.assign_coords(lat=(grid.dims, grid.lat), lon=(grid.dims, grid.lon),
                              **{grid.dims[0]: grid.y, grid.dims[1]: grid.x})
    diag["orog"] = (grid.dims, grid.orog)
    diag.attrs.update(run=run, split=split, n_members=n_mem)
    out.mkdir(parents=True, exist_ok=True)
    diag.to_netcdf(out / "diagnostics.nc")
    result = {"experiment": exp.name, "run": run, "split": split, "n_members": n_mem,
              "n_days": int(len(t_true)), "period": [str(t_true[0])[:10], str(t_true[-1])[:10]],
              "eval_seconds": round(time.time() - t0, 1), "scores": scores}
    (out / "scores.json").write_text(json.dumps(result, indent=1))
    print(f"[evaluate] {run} {split}: {result['eval_seconds']} s  " + "  ".join(
        f"{var}: RMSE {scores[var]['RMSE']:.3f} RALSD {scores[var]['RALSD']:.3g}" for var in scores), flush=True)
    return result
