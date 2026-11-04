"""How well the LC-AFM laws (dslab.laws) hold on the RCM truth, on the training years and on every evaluation
split; written by `dslab laws` to work/eval/<exp>/laws/diagnostics.nc and plotted in notebooks/laws.ipynb.

For law n with right-hand side b = J phi + c and tolerance tau (both from dslab.laws, fitted on the training
truth), per day and grid point:

    r = J x - b               residual (physical units)      z = r / tau     normalised residual
    J x - clim_m(J x)         residual of the climatology baseline (monthly mean of J x on the training years)
    J x - J phi               residual of the physics alone (without the fitted monthly offset c)

On the training years mean(z^2) = 1 at every grid point and month by construction, so the training numbers are
a consistency check; the evaluation splits are the test. Statistics are accumulated over chunks of days.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from .data import load_grid, load_split
from .experiment import Experiment
from .laws import LAWS, fit_cached, months, phi_cached

SPLITS = ("train", "val", "historical_perfect", "mid_century_perfect", "end_century_perfect",
          "historical_imperfect", "end_century_imperfect")
WHITE_SPLITS = ("val", "historical_perfect", "end_century_perfect")  # noise-shaping test (afm-base exists there)


def _afm_base(exp: Experiment, split: str, variables=("pr", "tasmax")) -> dict:
    """Member 0 of the trained AFM's white base E + sigma_max eps (`dslab predict --base-only`), if available."""
    path = exp.predictions_dir / "afm-base" / f"{split}.nc"
    if not path.exists():
        return {}
    with xr.open_dataset(path) as ds:
        return {v: ds[v].isel(member=0).transpose("time", ...).values.astype(np.float32) for v in variables}
Z_EDGES = np.linspace(-8, 8, 81)
CHUNK = 730


def _prior_noise(exp: Experiment, run: str) -> dict[str, float] | None:
    """Physical-unit prior noise s = sigma_max * y_std of a trained (LC-)AFM run, per variable."""
    from .models import Downscaler

    path = exp.runs_dir / run
    if not (path / "state.pt").exists():
        return None
    model = Downscaler.load(path)
    s = model.net.sigma_max.numpy() * np.asarray(model.y_std).ravel()
    return {v: float(s[i]) for i, v in enumerate(model.variables)}


class SplitStats:
    """Accumulates the diagnostics of one law on one split over chunks of days. Entries that are NaN (masked,
    e.g. dry days for a wet-day law, or missing inputs) are not counted."""

    def __init__(self, shape):
        H, W = shape
        self.tot = dict.fromkeys(("z2", "law", "clim", "phys", "Jx2", "c1", "c2", "r", "n"), 0.0)
        self.mon = {k: np.zeros(12) for k in ("z2", "r", "clim", "law", "n")}
        self.map = {k: np.zeros((H, W)) for k in ("z2", "r", "rj", "n", "nj")}
        self.rm, self.nm = np.zeros((12, H, W)), np.zeros((12, H, W))
        self.hist = np.zeros(len(Z_EDGES) - 1)

    def add(self, Jx, b, tol, off, clim, m, jja):
        """Jx: the law's left-hand side (t, H, W); b = J phi + offset; tol, off, clim: (12, H, W); m: months."""
        r = Jx - b
        z = r / tol[m]
        rc = Jx - clim[m]
        rp = r + off[m]  # J x - J phi, without the fitted offset
        v = np.isfinite(z) & np.isfinite(rc)
        vf = v.astype(np.float64)
        Jx, r, z, rc, rp = (np.where(v, a, 0.0) for a in (Jx, r, z, rc, rp))
        t = self.tot
        t["z2"] += (z**2).sum()
        t["law"] += (r**2).sum()
        t["clim"] += (rc**2).sum()
        t["phys"] += (rp**2).sum()
        t["Jx2"] += (Jx**2).sum()
        t["r"] += r.sum()
        t["c1"] += ((np.abs(z) <= 1) & v).sum()
        t["c2"] += ((np.abs(z) <= 2) & v).sum()
        t["n"] += vf.sum()
        for k, a in (("z2", z**2), ("r", r), ("clim", rc**2), ("law", r**2), ("n", vf)):
            np.add.at(self.mon[k], m, a.sum(axis=(1, 2)))
        self.map["z2"] += (z**2).sum(axis=0)
        self.map["r"] += r.sum(axis=0)
        self.map["n"] += vf.sum(axis=0)
        self.map["rj"] += r[jja].sum(axis=0)
        self.map["nj"] += vf[jja].sum(axis=0)
        np.add.at(self.rm, m, r)
        np.add.at(self.nm, m, vf)
        self.hist += np.histogram(z[v], bins=Z_EDGES)[0]

    def result(self) -> dict:
        t, mo, mp = self.tot, self.mon, self.map
        n = max(t["n"], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"ms_z": t["z2"] / n, "mse_law": t["law"] / n, "mse_clim": t["clim"] / n,
                    "mse_phys": t["phys"] / n, "ms_Jx": t["Jx2"] / n, "bias": t["r"] / n,
                    "cover1": t["c1"] / n, "cover2": t["c2"] / n, "valid_fraction": np.nan,
                    "ms_z_month": mo["z2"] / mo["n"], "bias_month": mo["r"] / mo["n"],
                    "mse_clim_month": mo["clim"] / mo["n"], "mse_law_month": mo["law"] / mo["n"],
                    "ms_z_map": mp["z2"] / mp["n"], "bias_map": mp["r"] / mp["n"], "bias_map_jja": mp["rj"] / mp["nj"],
                    "bias_map_month": self.rm / self.nm, "z_hist": self.hist / (n * np.diff(Z_EDGES)),
                    "n": t["n"]}


SCALARS = ("ms_z", "mse_law", "mse_clim", "mse_phys", "ms_Jx", "bias", "cover1", "cover2")
MONTHLY = ("ms_z_month", "bias_month", "mse_clim_month", "mse_law_month")
MAPS = ("ms_z_map", "bias_map", "bias_map_jja")


def assemble(names, splits, results: dict, grid, variables, n_days, tol, offset) -> xr.Dataset:
    """Dataset of the per-(law, split) results {(law, split): SplitStats.result()}; tol/offset: {law: (12, H, W)}."""
    ydim, xdim = grid.dims
    L, S = len(names), len(splits)
    ds = xr.Dataset(coords={"law": list(names), "split": list(splits), "month": np.arange(1, 13),
                            "z": 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1]), ydim: grid.y, xdim: grid.x})
    ds = ds.assign_coords(lat=((ydim, xdim), grid.lat), lon=((ydim, xdim), grid.lon),
                          law_variable=("law", [variables[n] for n in names]), n_days=("split", n_days))

    def stack(key, shape, dtype=np.float64):
        a = np.full((L, S, *shape), np.nan, dtype)
        for (n, s), r in results.items():
            a[list(names).index(n), list(splits).index(s)] = r[key]
        return a

    H, W = grid.shape
    for k in SCALARS:
        ds[k] = (("law", "split"), stack(k, ()))
    for k in MONTHLY:
        ds[k] = (("law", "split", "month"), stack(k, (12,)))
    for k in MAPS:
        ds[k] = (("law", "split", ydim, xdim), stack(k, (H, W), np.float32))
    ds["z_hist"] = (("law", "split", "z"), stack("z_hist", (len(Z_EDGES) - 1,)))
    ds["bias_map_month"] = (("law", "split", "month", ydim, xdim), stack("bias_map_month", (12, H, W), np.float32))
    # spatial RMS of the monthly mean residual: how far the law's training offsets are off (0 on training)
    ds["drift_rms_month"] = np.sqrt((ds["bias_map_month"].astype(np.float64) ** 2).mean((ydim, xdim)))
    ds["tol"] = (("law", "month", ydim, xdim), np.stack([tol[n] for n in names]).astype(np.float32))
    ds["offset"] = (("law", "month", ydim, xdim), np.stack([offset[n] for n in names]).astype(np.float32))
    ds["orog"] = ((ydim, xdim), grid.orog)
    return ds


def compute(exp: Experiment, splits=None, ref_run: str = "afm") -> xr.Dataset:
    grid = load_grid(exp)
    train = load_split(exp, "train")
    law = fit_cached(train, grid)
    names = list(law.names)
    splits = [s for s in (splits or SPLITS) if s in exp.splits]
    H, W = grid.shape

    # climatology baseline: monthly mean of J x on the training years
    mon_tr = months(train.time)
    clim = {}
    for n in names:
        v = list(train.variables).index(law.variable(n))
        acc, cnt = np.zeros((12, H, W)), np.zeros(12)
        for i in range(0, len(train), CHUNK):
            Jx = law.apply_J(n, train.y[i:i + CHUNK, v].astype(np.float64), grid)
            m = mon_tr[i:i + CHUNK]
            np.add.at(acc, m, Jx)
            cnt += np.bincount(m, minlength=12)
        clim[n] = acc / cnt[:, None, None]
    del train

    # conservative prior-noise proxy for the noise-shaping test (as dslab.candidate_laws.noise_levels): RMS of the
    # sub-2° daily anomaly of each target on the training years, from the high-pass laws' climatology
    train = load_split(exp, "train")
    s_noise = {}
    for n in names:
        if law.filtered(n):
            v = list(train.variables).index(law.variable(n))
            Jx = law.apply_J(n, train.y[::5, v].astype(np.float64), grid)
            s_noise[law.variable(n)] = float(np.sqrt(np.mean((Jx - clim[n][mon_tr[::5]]) ** 2)))
    del train
    results, n_days = {}, np.zeros(len(splits), int)
    white = np.full((len(names), len(splits)), np.nan)
    afmbase = np.full((len(names), len(splits)), np.nan)
    for j, split in enumerate(splits):
        s = load_split(exp, split)
        if s.y is None:
            continue
        b_all = phi_cached(law, s, grid)
        mon = months(s.time)
        jja = np.isin(mon, [5, 6, 7])
        n_days[j] = len(s)
        base = _afm_base(exp, split) if split in WHITE_SPLITS else {}
        for i_n, n in enumerate(names):
            v = list(s.variables).index(law.variable(n))
            tol, off = law.tol[n].astype(np.float64), law.offset[n].astype(np.float64)
            fields = {"truth": s.y[:, v]}
            if split in WHITE_SPLITS:  # noise shaping: truth + white noise, and the trained AFM's white base
                rng = np.random.default_rng(1)
                fields["white"] = s.y[:, v] + np.float32(s_noise[law.variable(n)]) * rng.standard_normal(
                    s.y[:, v].shape, np.float32)
                if base:
                    fields["afmbase"] = base[law.variable(n)]
            for kind, x in fields.items():
                st = SplitStats(grid.shape)
                for i in range(0, len(s), CHUNK):
                    Jx = law.apply_J(n, x[i:i + CHUNK].astype(np.float64), grid)
                    st.add(Jx, b_all[n][i:i + CHUNK], tol, off, clim[n], mon[i:i + CHUNK], jja[i:i + CHUNK])
                if kind == "truth":
                    results[(n, split)] = st.result()
                else:
                    (white if kind == "white" else afmbase)[i_n, j] = st.result()["ms_z"]
        print(f"[laws] {split}: {len(s)} days, mean z^2 " + ", ".join(
            f"{n} {results[(n, split)]['ms_z']:.3f}" for n in names), flush=True)

    ds = assemble(names, splits, results, grid, {n: law.variable(n) for n in names}, n_days, law.tol, law.offset)
    ds = ds.assign_coords(law_highpass=("law", [int(law.filtered(n)) for n in names]))
    ds["ms_z_white"] = (("law", "split"), white)
    ds["ms_z_afmbase"] = (("law", "split"), afmbase)
    ds.attrs["noise_levels"] = json.dumps(s_noise)
    noise = _prior_noise(exp, ref_run)
    if noise is not None:
        s_law = np.array([noise[law.variable(n)] for n in names])
        ds["prior_noise"] = ("law", s_law)
        ds["weight"] = (("law", "month", *grid.dims),
                        (s_law[:, None, None, None] ** 2 / (s_law[:, None, None, None] ** 2 + ds["tol"].values ** 2)))
        ds.attrs["prior_noise_run"] = ref_run
    ds.attrs.update(experiment=exp.name, law_info=json.dumps(law.info))
    return ds


def summary(ds: xr.Dataset) -> pd.DataFrame:
    """One row per law and split: skill beyond climatology, calibration, coverage, drift."""
    rows = []
    info = json.loads(ds.attrs.get("law_info", "{}"))
    for n in ds.law.values:
        alone = info.get(str(n), {}).get("physics_alone", True)  # False: the term is meaningless without offset
        for s in ds.split.values:
            d = ds.sel(law=n, split=s)
            rows.append({"law": n, "split": s,
                         "skill vs climatology": 1 - float(d.mse_law / d.mse_clim),
                         "skill of physics alone": 1 - float(d.mse_phys / d.mse_clim) if alone else np.nan,
                         "mean z²": float(d.ms_z), "inflation α": float(np.sqrt(d.ms_z)),
                         "|z| ≤ 1": float(d.cover1), "|z| ≤ 2": float(d.cover2),
                         "mean residual": float(d.bias)})
    return pd.DataFrame(rows).set_index(["law", "split"])


def run(exp: Experiment, splits=None, ref_run: str = "afm") -> xr.Dataset:
    ds = compute(exp, splits, ref_run)
    out = exp.eval_dir / "laws"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "diagnostics.nc")
    print(summary(ds).round(3).to_string())
    print(f"[laws] written {out / 'diagnostics.nc'}")
    return ds
