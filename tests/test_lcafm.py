"""Law-conditioned AFM: the exact (Woodbury) posterior against a dense solve, gradients to the encoder,
the laws' calibration on training truth, and a tiny training run (CPU, synthetic data).

    pytest tests/test_lcafm.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("physicsnemo")

from dslab.data import Grid, Split  # noqa: E402
from dslab.laws import CoarseOperator, LawSet  # noqa: E402
from dslab.models import Downscaler, build_model  # noqa: E402
from dslab.models.lcafm import LawConditioning  # noqa: E402

H = W = 32
PREDICTORS = tuple(f"{v}_{lev}" for v in "uvtzq" for lev in (850, 700, 500))


def _grid(seed=0, n=H):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(np.linspace(40, 50, n), np.linspace(5, 15, n), indexing="ij")
    return Grid(lat=lat, lon=lon, y=np.arange(n), x=np.arange(n), dims=("y", "x"),
                orog=rng.uniform(0, 2000, (n, n)).astype(np.float32),
                pred_lat=np.linspace(38, 52, 8), pred_lon=np.linspace(3, 17, 8))


def _split(grid, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange("2000-01-01", n_days * 9, 9, dtype="datetime64[D]").astype("datetime64[ns]")
    base = {"u": 8.0, "v": 3.0, "t": 270.0, "z": 0.0, "q": 0.004}
    x = np.zeros((n_days, len(PREDICTORS), 8, 8), np.float32)
    for i, name in enumerate(PREDICTORS):
        v, lev = name.split("_")
        lev = int(lev)
        if v == "z":
            x[:, i] = {850: 1500, 700: 3000, 500: 5600}[lev] + rng.normal(0, 30, (n_days, 8, 8))
        elif v == "t":
            x[:, i] = {850: 280, 700: 270, 500: 255}[lev] + rng.normal(0, 3, (n_days, 8, 8))
        else:
            x[:, i] = base[v] + rng.normal(0, 0.2 * abs(base[v]) + 1e-4, (n_days, 8, 8))
    op = CoarseOperator.from_grid(grid)
    from dslab.laws import free_temperature
    tx = free_temperature(x, PREDICTORS, op, grid.orog) + 8 + rng.normal(0, 1, (n_days, *grid.shape))
    pr = rng.gamma(0.6, 4, (n_days, *grid.shape))
    y = np.stack([pr, tx], 1).astype(np.float32)
    return Split("train", time, x, y, ("pr", "tasmax"), PREDICTORS)


def test_coarse_operator():
    op = CoarseOperator.from_grid(_grid())
    assert np.allclose(op.lowpass(np.ones((H, W))), 1, atol=1e-5)  # constants pass the low-pass
    x = np.random.default_rng(0).normal(size=(3, H, W))
    assert np.allclose(op.lowpass(op.lowpass(x)), op.lowpass(x), atol=1e-4)  # projector: P P = P
    assert np.allclose((op.lowpass(x) * x[::-1]).sum(), (x * op.lowpass(x[::-1])).sum(), rtol=1e-4)  # P = P^T
    assert op.Q.shape[1] == op.A.shape[0] < H * W


def _dense_posterior(op, tau, s, m0, b, eps_e, eta, highpass):
    N = m0.shape[1]
    J0 = np.eye(N) - op.Q.astype(np.float64) @ op.Q.T.astype(np.float64) if highpass else np.eye(N)
    out_m, out_n = [], []
    for i in range(len(m0)):
        J = J0 / tau[i][:, None]
        M = s**2 * J @ J.T + np.eye(N)
        K = s**2 * J.T @ np.linalg.inv(M)
        e = s * eps_e[i]
        out_m.append(m0[i] + K @ (b[i] / tau[i] - J @ m0[i]))
        out_n.append(e + K @ (eta[i] - J @ e))
    return np.array(out_m), np.array(out_n)


@pytest.mark.parametrize("kind", ["highpass", "full"])
def test_posterior_matches_dense(kind):
    n = 16
    grid = _grid(n=n)
    op = CoarseOperator.from_grid(grid)
    rng = np.random.default_rng(1)
    tol = rng.uniform(0.2, 1.5, (12, n, n)).astype(np.float32)
    cb = LawConditioning(op, [None, kind], [None, tol])
    B, s = 3, 0.7
    m0 = torch.from_numpy(rng.normal(size=(B, 2, n, n)).astype(np.float32))
    b = torch.from_numpy(rng.normal(size=(B, 2, n, n)).astype(np.float32))
    month = torch.tensor([0, 5, 11])
    gen = torch.Generator().manual_seed(3)
    m, eps = cb(m0, b, month, torch.tensor([1.0, s]), gen)
    gen = torch.Generator().manual_seed(3)  # the same draws, in the same order
    eps_e = torch.randn(m0.shape, generator=gen)[:, 1].flatten(1).double().numpy()
    eta = torch.randn(m0.shape, generator=gen)[:, 1].flatten(1).double().numpy()
    tau = tol.reshape(12, -1)[month.numpy()].astype(np.float64)
    dm, dn = _dense_posterior(op, tau, s, m0[:, 1].flatten(1).double().numpy(), b[:, 1].flatten(1).double().numpy(),
                              eps_e, eta, kind == "highpass")
    np.testing.assert_allclose(m[:, 1].flatten(1).numpy(), dm, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose((s * eps[:, 1]).flatten(1).numpy(), dn, rtol=1e-4, atol=1e-4)
    assert torch.equal(m[:, 0], m0[:, 0])  # unconstrained channel: white base


def test_posterior_many_threads():
    """The batched Cholesky solve stays exact with many CPU threads (MKL's batched LU did not, on Levante)."""
    n, threads = 16, torch.get_num_threads()
    op = CoarseOperator.from_grid(_grid(n=n))
    tol = np.random.default_rng(2).uniform(0.2, 1.5, (12, n, n)).astype(np.float32)
    cb = LawConditioning(op, ["highpass", None], [tol, None])
    args = (torch.randn(64, 2, n, n), torch.randn(64, 2, n, n), torch.randint(0, 12, (64,)), torch.tensor([0.7, 1.0]))
    try:
        torch.set_num_threads(1)
        ref = cb(*args, torch.Generator().manual_seed(0))[0]
        torch.set_num_threads(24)
        out = cb(*args, torch.Generator().manual_seed(0))[0]
    finally:
        torch.set_num_threads(threads)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gradient_reaches_encoder():
    n = 16
    op = CoarseOperator.from_grid(_grid(n=n))
    cb = LawConditioning(op, ["highpass", "full"], [np.ones((12, n, n), np.float32)] * 2)
    m0 = torch.randn(2, 2, n, n, requires_grad=True)
    m, _ = cb(m0, torch.zeros(2, 2, n, n), torch.tensor([1, 2]), torch.tensor([0.5, 0.5]))
    m.square().sum().backward()
    assert m0.grad is not None and torch.isfinite(m0.grad).all() and m0.grad.abs().sum() > 0


def test_laws_calibrated_on_training_truth():
    grid = _grid()
    train = _split(grid)
    law = LawSet.fit(train, grid, ("T", "Tfull", "O"))
    b = law.phi(train, grid)
    from dslab.laws import months
    mon = months(train.time)
    for name in law.names:
        v = list(train.variables).index(law.variable(name))
        assert law.residual(name, train.y[:, v], b[name], mon, grid) == pytest.approx(1.0, rel=1e-3)
    again = LawSet.from_state(law.state())
    assert again.sb_params["hw"] == law.sb_params["hw"]


TINY = dict(model_channels=8, channel_mult=[1, 2], num_blocks=1, attn_resolutions=[], dropout=0.0,
            batch_size=8, micro_batch=4, train_samples=64, eval_every=32, n_members=3, n_steps=4,
            predict_batch=4, amp=False, laws=["Tfull", "O"])


def test_lcafm_fit_sample_save_load(tmp_path, monkeypatch):
    import dslab.laws as laws_mod
    monkeypatch.setattr(laws_mod, "CACHE", tmp_path / "cache")
    torch.manual_seed(0)
    grid = _grid()
    train = _split(grid)
    val = Split("val", train.time[:8], train.x[:8], train.y[:8], train.variables, train.predictors)
    model = build_model("lcafm", train.variables, train.predictors, **TINY)
    info = model.fit(train, val, grid)
    assert np.isfinite(info["best_val_loss"]) and set(info["laws"]) == {"Tfull", "O"}
    pred = model.predict(val, grid, n_members=3, seed=1)
    assert pred.shape == (3, 8, 2, H, W) and np.isfinite(pred).all() and (pred[:, :, 0] >= 0).all()
    run = tmp_path / "run"
    model.save(run)
    again = Downscaler.load(run)
    np.testing.assert_allclose(again.predict(val, grid, n_members=3, seed=1), pred, rtol=1e-4, atol=1e-4)
    again.params["base_only"] = True
    base = again.predict(val, grid, n_members=2, seed=1)
    assert base.shape == (2, 8, 2, H, W) and not np.allclose(base[0], base[1])
