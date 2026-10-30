"""Metric tests: synthetic identities, plus a cross-check against the organisers' own functions
(ml-benchmark/evaluation) on stored predictions when those are available.

    pytest tests/test_metrics.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dslab import metrics as M

ORG_EVAL = Path("/home/b/b384340/cordex-ml-bench/repos/ml-benchmark/evaluation")


def test_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    pred, truth = rng.normal(size=(6, 5, 4, 3)), rng.normal(size=(5, 4, 3))
    skill = np.abs(pred - truth).mean(0)
    spread = np.abs(pred[:, None] - pred[None]).sum((0, 1)) / (2 * 6 * 5)
    assert M.crps_ensemble(pred, truth, chunk=2) == pytest.approx((skill - spread).mean())
    assert M.crps_ensemble(pred[:1], truth) == pytest.approx(np.abs(pred[0] - truth).mean())


def test_rank_histogram_and_identities():
    rng = np.random.default_rng(1)
    pred, truth = rng.normal(size=(9, 50, 8, 8)), rng.normal(size=(50, 8, 8))
    h = M.rank_histogram(pred, truth, stride=1)
    assert len(h) == 10 and h.sum() == truth.size
    a = rng.gamma(0.5, 4, size=(30, 16, 16))
    spec = M.RadialSpectrum((16, 16))
    assert M.ralsd(spec.time_mean(a), spec.time_mean(a)) == 0.0
    hist = M.histogram(a, M.LHD_BINS)
    assert M.pss(hist, hist) == pytest.approx(1.0)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"org_{name}", ORG_EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("run", ["linear", "climatology"])
def test_against_organisers(run):
    """Recompute scorecard metrics with the organisers' diagnostics/indices on the val split."""
    from dslab.evaluate import load_pred, load_truth
    from dslab.experiment import Experiment
    from dslab.io import to_dataset
    from dslab.data import load_grid, load_split

    exp = Experiment.load("alps_esd")
    scores_file = exp.eval_dir / run / "val" / "scores.json"
    if not ORG_EVAL.exists() or not scores_file.exists():
        pytest.skip("organisers' code or evaluated val predictions not available")
    D, I = _load_module("diagnostics"), _load_module("indices")
    ours = json.loads(scores_file.read_text())["scores"]
    grid, split = load_grid(exp), load_split(exp, "val")
    _, y = load_truth(exp, "val")
    preds = np.stack([load_pred(exp, run, "val", v)[1][0] for v in exp.variables], axis=1)
    gt = to_dataset(y[None], split, grid).isel(member=0)
    pm = to_dataset(preds[None], split, grid).isel(member=0)
    rms = M.rms_space

    for v in exp.variables:
        ref = {
            "RMSE": rms(D.rmse(gt, pm, var=v, dim=["time"])[v].values),
            "Mean Bias": rms(D.bias_index(gt, pm, I.mean, var=v)[v].values),
            "RALSD": D.ralsd(*D.psd(gt, pm, var=v)),
        }
        if v == "pr":
            ref["SDII"] = rms(D.bias_index(gt, pm, I.sdii, var=v)[v].values)
            ref["Rx1day"] = rms(D.bias_index(gt, pm, I.rx1day, var=v)[v].values)
        else:
            ref["TXx"] = rms(D.bias_index(gt, pm, I.txx, var=v)[v].values)
            ref["IAV"] = rms(D.bias_index(gt, pm, I.interannual_var, var=v)[v].values)
        for name, value in ref.items():
            assert ours[v][name] == pytest.approx(float(value), rel=1e-4, abs=1e-6), (v, name)
