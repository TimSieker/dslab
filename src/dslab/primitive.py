"""The primitive equations of the atmosphere (Coti Zelati, Huang, Kukavica, Temam & Ziane 2015, Nonlinearity 28,
625; eqs. 2.11-2.16, pressure coordinates) evaluated on daily data with centred differences: which of them can be
checked with the fields we have, and how well do they hold? (`dslab laws --exp alps_esd --primitive`)

    momentum (2.11)    ∂v/∂t + (v·∇)v + ω ∂v/∂p + f k×v + ∇Φ + A_v v = S_v
    hydrostatic (2.12) ∂Φ/∂p + R T / p = 0
    continuity (2.13)  ∇·v + ∂ω/∂p = 0
    temperature (2.14) ∂T/∂t + v·∇T + ω ∂T/∂p - R ω T / (c_p p) + A_T T ∈ (L / (c_p p)) ω⁻ H(q - q_s) F(T) + S_T
    humidity (2.15)    ∂q/∂t + v·∇q + ω ∂q/∂p + A_q q ∈ -(1/p) ω⁻ H(q - q_s) F(T) + S_q

Only the daily 2° predictors (u, v, t, z, q at 850/700/500 hPa) hold a 3-D state, so the tests run on them:
    P-hyd   hydrostatic balance integrated over 850-700 and 700-500 hPa (thickness), with T and with T_v
    P-mom   horizontal momentum at 850/700/500 hPa: geostrophic part f k×v + g∇z alone, and with the daily
            tendency and the horizontal advection; ω ∂v/∂p (no ω) and friction / diffusion are not available
    P-hum   humidity equation integrated over 850-500 hPa with continuity (flux form): the layer's condensation
            ≈ -∂W/∂t - ∇·Q, compared with the RCM's 2° precipitation (all condensate falls out; the layer below
            850 hPa, evaporation and the vertical fluxes ω q through 850 and 500 hPa are missing)
At 12 km only the hydrostatic equation can be checked (law Hyd in dslab.candidate_laws). Derivatives: centred
differences on the 2° grid (sphere), and in time over the neighbouring days (np.gradient on the day axis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_grid, load_split
from .experiment import Experiment
from .laws import CoarseOperator, months

G0, RD, OMEGA, R_EARTH = 9.81, 287.04, 7.292e-5, 6.371e6
SPLITS = ("train", "val", "historical_perfect", "mid_century_perfect", "end_century_perfect",
          "historical_imperfect", "end_century_imperfect")


def _ddx(a, lat, dlon):
    return np.gradient(a, axis=-1) / (R_EARTH * np.cos(lat)[:, None] * dlon)


def _ddy(a, dlat):
    return np.gradient(a, axis=-2) / (R_EARTH * dlat)


def _rel(err, ref, mon):
    """RMS of err relative to the RMS of the daily anomalies of ref (about its monthly mean)."""
    clim = np.stack([ref[mon == k].mean(0) for k in range(12)])
    return float(np.sqrt(np.mean(err**2) / np.mean((ref - clim[mon]) ** 2)))


def run(exp: Experiment, splits=SPLITS) -> pd.DataFrame:
    from .candidate_laws import _Ctx, _monthly_mean  # full 2° cells and cell means of the RCM's precipitation

    grid = load_grid(exp)
    op = CoarseOperator.from_grid(grid)
    lat = np.radians(grid.pred_lat)
    dlat, dlon = lat[1] - lat[0], np.radians(grid.pred_lon[1] - grid.pred_lon[0])
    f = (2 * OMEGA * np.sin(lat))[:, None]
    inner = (slice(None), slice(1, -1), slice(1, -1))
    rows, gain = [], None
    for split in [s for s in splits if s in exp.splits]:
        s = load_split(exp, split)
        X = {p: s.x[:, i].astype(np.float64) for i, p in enumerate(s.predictors)}
        mon = months(s.time)
        tdays = s.time.astype("datetime64[D]").astype(np.float64)
        ddt = lambda a: np.gradient(a, tdays, axis=0) / 86400.0  # noqa: E731  per second

        # P-hyd: g (z_top - z_bottom) = R_d * mean(T_v) * ln(p_bottom / p_top)
        for bot, top in ((850, 700), (700, 500)):
            dz = X[f"z_{top}"] - X[f"z_{bot}"]
            for kind in ("T", "T_v"):
                tb, tt = X[f"t_{bot}"], X[f"t_{top}"]
                if kind == "T_v":
                    tb, tt = tb * (1 + 0.608 * X[f"q_{bot}"]), tt * (1 + 0.608 * X[f"q_{top}"])
                pred = RD / G0 * 0.5 * (tb + tt) * np.log(bot / top)
                e = (dz - pred)[inner]
                rows.append({"test": "P-hyd", "item": f"thickness {bot}-{top} hPa, {kind}", "split": split,
                             "bias": float(e.mean()), "RMSE": float(np.sqrt((e**2).mean())),
                             "relative RMSE": _rel(e, dz[inner], mon),
                             "corr": float(np.corrcoef(dz[inner].ravel(), pred[inner].ravel())[0, 1])})

        # P-mom: residuals of the horizontal momentum equation at each level (m s-2)
        for lev in (850, 700, 500):
            u, v, z = X[f"u_{lev}"], X[f"v_{lev}"], X[f"z_{lev}"]
            pgf = (G0 * _ddx(z, lat, dlon), G0 * _ddy(z, dlat))
            cor = (-f * v, f * u)
            adv = (u * _ddx(u, lat, dlon) + v * _ddy(u, dlat), u * _ddx(v, lat, dlon) + v * _ddy(v, dlat))
            ten = (ddt(u), ddt(v))
            rms = lambda c: float(np.sqrt(np.mean(c[0][inner] ** 2 + c[1][inner] ** 2)))  # noqa: E731
            geo = (cor[0] + pgf[0], cor[1] + pgf[1])
            full = (ten[0] + adv[0] + geo[0], ten[1] + adv[1] + geo[1])
            rows.append({"test": "P-mom", "item": f"{lev} hPa", "split": split, "RMS pressure gradient": rms(pgf),
                         "RMS Coriolis": rms(cor), "RMS tendency": rms(ten), "RMS advection": rms(adv),
                         "RMS residual geostrophic": rms(geo), "RMS residual with tendency + advection": rms(full),
                         "residual / pressure gradient (geostrophic)": rms(geo) / rms(pgf),
                         "residual / pressure gradient (with tendency + advection)": rms(full) / rms(pgf)})

        # P-hum: layer 850-500 hPa, W = (1/g) ∫ q dp, Q = (1/g) ∫ q v dp (trapezoid over the three levels)
        dp = {(850, 700): 15000.0, (700, 500): 20000.0}

        def layer(fn):
            return sum(0.5 * (fn(a) + fn(b)) * w for (a, b), w in dp.items()) / G0

        W = layer(lambda lv: X[f"q_{lv}"])
        Qx = layer(lambda lv: X[f"q_{lv}"] * X[f"u_{lv}"])
        Qy = layer(lambda lv: X[f"q_{lv}"] * X[f"v_{lv}"])
        divQ = _ddx(Qx, lat, dlon) + np.gradient(Qy * np.cos(lat)[:, None], axis=-2) / (R_EARTH * dlat * np.cos(lat)[:, None])
        budget = (-ddt(W) - divQ) * 86400.0  # kg m-2 day-1 = mm/day
        ctx = _Ctx(exp, s, grid, op, {"pr"})
        idx, ij = ctx.cells()
        pr = s.y[:, list(s.variables).index("pr")].reshape(len(s), -1).astype(np.float64)
        pc = np.stack([pr[:, idx == k].mean(1) for k in range(len(ij))], 1)  # RCM 2° precipitation
        bc = np.stack([budget[:, i, j] for i, j in ij], 1)
        pclim = np.stack([pc[mon == k].mean(0) for k in range(12)])
        bclim = np.stack([bc[mon == k].mean(0) for k in range(12)])
        pa, ba = pc - pclim[mon], bc - bclim[mon]
        if split == "train":  # one gain, fitted on the training years' anomalies
            gain = float((pa * ba).sum() / (ba * ba).sum())
            p_clim_train, b_clim_train = pclim, bclim
        pred = p_clim_train[mon] + gain * (bc - b_clim_train[mon])
        e = pc - pred
        rows.append({"test": "P-hum", "item": "850-500 hPa layer budget vs. 2° precipitation", "split": split,
                     "gain": gain, "mean budget (mm/day)": float(bc.mean()), "mean precipitation (mm/day)": float(pc.mean()),
                     "corr of daily anomalies": float(np.corrcoef(pa.ravel(), ba.ravel())[0, 1]),
                     "skill vs climatology": 1 - float(np.mean(e**2) / np.mean((pc - p_clim_train[mon]) ** 2))})
        print(f"[primitive] {split} done", flush=True)
    return pd.DataFrame(rows)


def save(exp: Experiment) -> pd.DataFrame:
    df = run(exp)
    out = exp.eval_dir / "laws"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "primitive.csv", index=False)
    print(df.round(4).to_string())
    return df
