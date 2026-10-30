"""Adaptive flow matching: loss identity, a tiny training run, sampling and save/load (CPU, synthetic data).

    pytest tests/test_afm.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("physicsnemo")

from dslab.data import Grid, Split  # noqa: E402
from dslab.models import Downscaler, build_model  # noqa: E402
from dslab.models.afm import SIGMA_DATA  # noqa: E402

H = W = 32
TINY = dict(model_channels=8, channel_mult=[1, 2], num_blocks=1, attn_resolutions=[], dropout=0.0,
            batch_size=8, micro_batch=4, train_samples=64, eval_every=32, n_members=3, n_steps=4,
            predict_batch=4, amp=False)


def _data(n=24, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange("2000-01-01", n, dtype="datetime64[D]").astype("datetime64[ns]")
    x = rng.normal(size=(n, 3, 8, 8)).astype(np.float32)
    pr = rng.gamma(0.5, 4, size=(n, H, W)).astype(np.float32)
    tx = 280 + rng.normal(size=(n, H, W)).astype(np.float32)
    split = Split("train", time, x, np.stack([pr, tx], axis=1), ("pr", "tasmax"), ("a", "b", "c"))
    lat, lon = np.meshgrid(np.linspace(40, 50, H), np.linspace(5, 15, W), indexing="ij")
    grid = Grid(lat=lat, lon=lon, y=np.arange(H), x=np.arange(W), dims=("y", "x"),
                orog=rng.uniform(0, 2000, (H, W)).astype(np.float32),
                pred_lat=np.linspace(38, 52, 8), pred_lon=np.linspace(3, 17, 8))
    return split, grid


def test_loss_identity():
    """||F - target||^2 used in the code equals the paper's w(sigma) ||D - x1||^2."""
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(5, 2, 1, 1, generator=g) * 1.5 + 1e-3
    x1, n, F = (torch.randn(5, 2, 4, 4, generator=g, dtype=torch.float64) for _ in range(3))
    sigma = sigma.double()
    norm = torch.sqrt(sigma**2 + SIGMA_DATA**2)
    c_skip, c_out = SIGMA_DATA**2 / norm**2, sigma * SIGMA_DATA / norm
    D = c_skip * (x1 + sigma * n) + c_out * F
    w = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2
    target = sigma * x1 / (SIGMA_DATA * norm) - SIGMA_DATA * n / norm
    assert torch.allclose(w * (D - x1) ** 2, (F - target) ** 2, rtol=1e-10)


def test_fit_sample_save_load(tmp_path):
    torch.manual_seed(0)
    split, grid = _data()
    val = Split("val", split.time[:8], split.x[:8], split.y[:8], split.variables, split.predictors)
    model = build_model("afm", split.variables, split.predictors, **TINY)
    info = model.fit(split, val, grid)
    assert len(info["history"]) == 2 and np.isfinite(info["best_val_loss"])
    assert info["sigma_max"] != [1.0, 1.0]  # adapted towards the encoder error
    pred = model.predict(val, grid, n_members=3, seed=1)
    assert pred.shape == (3, 8, 2, H, W) and np.isfinite(pred).all()
    assert (pred[:, :, 0] >= 0).all()  # no negative precipitation
    assert not np.allclose(pred[0], pred[1])  # members differ
    model.save(tmp_path)
    again = Downscaler.load(tmp_path)
    np.testing.assert_allclose(again.predict(val, grid, n_members=3, seed=1), pred, rtol=1e-5, atol=1e-5)
