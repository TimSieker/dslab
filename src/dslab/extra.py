"""Extra fields of the same RCM run from the DKRZ CORDEX pool, cut to the benchmark window, and the grid
geometry needed to use them in laws (wind components on the grid axes, gradients, divergence).

    dslab extract --exp alps_esd            # -> work/cache/<domain>/extra/<period>/<field>.npy (+ time.npy)

Stored daily fields (T, 128, 128) float32, aligned with the benchmark's daily time axis:
    ua850 va850      daily-mean 850 hPa wind, eastward / northward (m/s); NaN where 850 hPa is below ground
    zg500            daily-mean 500 hPa geopotential height (m)
    huss ps tas tasmin   near-surface specific humidity (kg/kg), surface pressure (Pa), 2 m temperature and
                     daily minimum (K)
    evspsbl          evaporation (mm/day, positive upward; from kg m-2 s-1)
    uas vas          10 m wind, mean of the four instantaneous 6-hourly values 06, 12, 18, 24 UTC (m/s)
    quas qvas        mean of the four 6-hourly products huss * uas, huss * vas (kg/kg m/s): the near-surface
                     moisture flux per unit mass including its sub-daily covariance (huss from the 3-hourly file)
    upslope_pos      mean of the four 6-hourly positive parts of the terrain-forced ascent (V_10m . grad h)_+ (m/s)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import Grid
from .experiment import WORK, Experiment

R_EARTH = 6.371e6
DAILY = {"ua850": 1.0, "va850": 1.0, "zg500": 1.0, "huss": 1.0, "ps": 1.0, "tas": 1.0, "tasmin": 1.0,
         "evspsbl": 86400.0}
SUBDAILY = ("uas", "vas", "quas", "qvas", "upslope_pos")
FIELDS = tuple(DAILY) + SUBDAILY


# ---- geometry of the target grid
@lru_cache(maxsize=4)
def _geometry(lat_bytes: bytes, lon_bytes: bytes, shape: tuple) -> dict:
    lat = np.frombuffer(lat_bytes).reshape(shape)
    lon = np.frombuffer(lon_bytes).reshape(shape)
    la, lo = np.radians(lat), np.radians(lon)
    out = {}
    for name, axis in (("x", 1), ("y", 0)):  # x: increasing column index, y: increasing row index
        de = np.gradient(lo, axis=axis) * np.cos(la)  # eastward displacement per grid step (radians of arc)
        dn = np.gradient(la, axis=axis)
        step = np.hypot(de, dn)
        out[f"e{name}"] = (de / step, dn / step)  # unit vector of the grid axis in (east, north)
        out[f"d{name}"] = R_EARTH * step  # metres per grid step along that axis
    return out


def geometry(grid: Grid) -> dict:
    """Unit vectors of the grid axes in (east, north) and the true spacing (m) per grid step, at every point."""
    lat = np.ascontiguousarray(grid.lat, np.float64)
    lon = np.ascontiguousarray(grid.lon, np.float64)
    return _geometry(lat.tobytes(), lon.tobytes(), lat.shape)


def to_grid_axes(u_east, v_north, geo):
    """Wind components along the grid's x and y axes (the Lambert grid is conformal: the axes are orthogonal)."""
    (exe, exn), (eye, eyn) = geo["ex"], geo["ey"]
    return u_east * exe + v_north * exn, u_east * eye + v_north * eyn


def gradient(f, geo):
    """(df/dx, df/dy) along the grid axes, for f (..., H, W); centred differences, one-sided at the edges."""
    return np.gradient(f, axis=-1) / geo["dx"], np.gradient(f, axis=-2) / geo["dy"]


def divergence(fx, fy, geo):
    """d fx / dx + d fy / dy for grid-axis components (choice: local true spacing, without the map-factor
    derivative term, which is below 2% of the spacing change across this window)."""
    return np.gradient(fx, axis=-1) / geo["dx"] + np.gradient(fy, axis=-2) / geo["dy"]


# ---- extraction
def _years_of(path: Path) -> tuple[int, int]:
    a, b = re.findall(r"_(\d{8,12})-(\d{8,12})\.nc$", path.name)[0]
    return int(a[:4]), int(b[:4])


def _files(cfg: dict, freq: str, var: str, y0: int, y1: int) -> list[Path]:
    out = []
    for scen, (lo, hi) in (("historical", (1949, 2005)), ("rcp85", (2006, 2100))):
        if y1 < lo or y0 > hi:
            continue
        base = Path(cfg["root"]) / scen / cfg["run"] / freq / var
        # one version directory: "latest" (a link to the newest) if present, else the newest vYYYYMMDD
        versions = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
        if not versions:
            continue
        version = "latest" if "latest" in versions else versions[-1]
        for f in sorted((base / version).glob(f"{var}_*.nc")):
            a, b = _years_of(f)
            if freq != "day":  # sub-daily files end at 00 UTC of the next year
                b = a
            if b >= y0 and a <= y1:
                out.append(f)
    if not out:
        raise FileNotFoundError(f"no {freq}/{var} files for {y0}-{y1} under {cfg['root']}")
    return out


def _read(files, var, window, times=None) -> xr.DataArray:
    (x0, x1), (y0, y1) = window["x"], window["y"]
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            a = ds[var].isel(x=slice(x0, x1), y=slice(y0, y1))
            if times is not None:
                a = a.sel(time=a.time[np.isin(a.time.values, times)])
            parts.append(a.load())
    return xr.concat(parts, "time")


def _extract_daily(cfg, var, y0, y1, days):
    a = _read(_files(cfg, "day", var, y0, y1), var, cfg["window"])
    a = a.assign_coords(time=a.time.values.astype("datetime64[D]")).sel(time=days)
    return a.values.astype(np.float32) * np.float32(DAILY[var])


def _extract_subdaily(cfg, y0, y1, days, grid: Grid):
    """Daily means of the 6-hourly instantaneous 10 m wind, of huss * wind and of the positive terrain-forced
    ascent, from the samples at 06, 12, 18 UTC of each day and 00 UTC of the next."""
    geo = geometry(grid)
    hx, hy = (g.astype(np.float32) for g in gradient(grid.orog.astype(np.float64), geo))
    out = {k: [] for k in SUBDAILY}
    for year in range(y0, y1 + 1):
        u = _read(_files(cfg, "6hr", "uas", year, year), "uas", cfg["window"])
        v = _read(_files(cfg, "6hr", "vas", year, year), "vas", cfg["window"])
        q = _read(_files(cfg, "3hr", "huss", year, year), "huss", cfg["window"], times=u.time.values)
        if not (np.array_equal(u.time.values, v.time.values) and np.array_equal(u.time.values, q.time.values)):
            raise ValueError(f"{year}: 6-hourly uas/vas and 3-hourly huss times do not match")
        day = (u.time.values - np.timedelta64(1, "s")).astype("datetime64[D]")  # 00 UTC closes the day before
        uu, vv, qq = (a.values.astype(np.float32) for a in (u, v, q))
        ux, vy = to_grid_axes(uu, vv, geo)
        fields = {"uas": uu, "vas": vv, "quas": qq * uu, "qvas": qq * vv,
                  "upslope_pos": np.maximum(ux * hx + vy * hy, 0.0)}
        keep = np.isin(day, days)
        for k, f in fields.items():
            df = pd.DataFrame({"day": day[keep]})
            idx = df.groupby("day").indices
            if any(len(i) != 4 for i in idx.values()):
                raise ValueError(f"{year}: not 4 samples per day for {k}")
            fk = f[keep]
            out[k].append(np.stack([fk[i].mean(axis=0, dtype=np.float64) for _, i in sorted(idx.items())]
                                   ).astype(np.float32))
        del uu, vv, qq, ux, vy, fields
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def extra_dir(exp: Experiment, period: str) -> Path:
    return WORK / "cache" / exp.domain / "extra" / period


def _task(args) -> dict:
    """One extraction task (worker process): one daily field, or all sub-daily products, of one period."""
    exp_name, period, what = args
    exp = Experiment.load(exp_name)
    from .data import load_grid

    cfg = exp.extra
    y0, y1 = cfg["periods"][period]
    days = np.arange(f"{y0}-01-01", f"{y1 + 1}-01-01", dtype="datetime64[D]")
    out = extra_dir(exp, period)
    if isinstance(what, tuple):  # ("subdaily", first year, last year): one block, saved as a partial file
        _, b0, b1 = what
        block = days[(days >= np.datetime64(f"{b0}-01-01")) & (days < np.datetime64(f"{b1 + 1}-01-01"))]
        np.savez(out / f"_subdaily_{b0}.npz", **_extract_subdaily(cfg, b0, b1, block, load_grid(exp)))
        print(f"[extract] {period} sub-daily {b0}-{b1} done", flush=True)
        return {}
    if what == "pr_check":
        return {"pr_check": _check_pr(exp, cfg, y0, days)}
    else:
        arrays = {what: _extract_daily(cfg, what, y0, y1, days)}
        src = f"day/{what} x {DAILY[what]}"
    info = {}
    for k, a in arrays.items():
        np.save(out / f"{k}.npy", a)
        info[k] = {"source": src, "shape": list(a.shape), "nan_fraction": float(np.isnan(a).mean()),
                   "mean": float(np.nanmean(a))}
        print(f"[extract] {period} {k}: {a.shape}, NaN {info[k]['nan_fraction']:.3f}", flush=True)
    return {"fields": info}


def extract(exp: Experiment, periods=None, fields=FIELDS, workers: int = 8) -> None:
    """Extract `fields` for `periods` (default: all configured) in parallel worker processes."""
    from multiprocessing import get_context

    periods = list(periods or exp.extra["periods"])
    tasks = []
    for period in periods:
        y0, y1 = exp.extra["periods"][period]
        out = extra_dir(exp, period)
        out.mkdir(parents=True, exist_ok=True)
        days = np.arange(f"{y0}-01-01", f"{y1 + 1}-01-01", dtype="datetime64[D]")
        np.save(out / "time.npy", days.astype("datetime64[ns]") + np.timedelta64(12, "h"))
        if any(f in SUBDAILY for f in fields):  # slowest first: 5-year blocks of the sub-daily products
            tasks += [(exp.name, period, ("subdaily", b, min(b + 4, y1))) for b in range(y0, y1 + 1, 5)]
        tasks += [(exp.name, period, f) for f in fields if f in DAILY]
        tasks.append((exp.name, period, "pr_check"))
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_task, tasks, chunksize=1)
    for period in periods:
        out = extra_dir(exp, period)
        parts = sorted(out.glob("_subdaily_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
        if parts:  # join the blocks
            loaded = [np.load(f) for f in parts]
            for k in SUBDAILY:
                a = np.concatenate([d[k] for d in loaded])
                np.save(out / f"{k}.npy", a)
                results.append({"fields": {k: {"source": "6hr uas/vas, 3hr huss: mean of the values at 06, 12, 18, "
                                                         "24 UTC", "shape": list(a.shape), "mean": float(np.nanmean(a))}}})
                tasks.append((exp.name, period, k))
            for f in parts:
                f.unlink()
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
        meta.update(period=exp.extra["periods"][period], window=exp.extra["window"], root=exp.extra["root"],
                    run=exp.extra["run"])
        for (_, p, _), r in zip(tasks, results):
            if p == period:
                meta.setdefault("fields", {}).update(r.get("fields", {}))
                if "pr_check" in r:
                    meta["pr_check"] = r["pr_check"]
                    print(f"[extract] {period} pr check: {r['pr_check']}", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))


def _check_pr(exp: Experiment, cfg: dict, year: int, days) -> dict:
    from .data import load_split

    first = days[days.astype("datetime64[Y]").astype(int) + 1970 == year]
    pool = _read(_files(cfg, "day", "pr", year, year), "pr", cfg["window"])
    pool = pool.assign_coords(time=pool.time.values.astype("datetime64[D]")).sel(time=first).values * 86400.0
    for split in exp.splits:
        s = load_split(exp, split)
        if s.y is None:
            continue
        sel = np.isin(s.time.astype("datetime64[D]"), first)
        if sel.sum() == len(first):
            bench = s.y[sel, list(s.variables).index("pr")]
            return {"split": split, "days": int(len(first)), "max_abs_diff_mm_day": float(np.abs(pool - bench).max()),
                    "mean_pool": float(pool.mean()), "mean_benchmark": float(bench.mean())}
    return {"split": None, "note": f"no benchmark split covers {year}"}


def load_extra(exp: Experiment, time: np.ndarray, fields) -> dict[str, np.ndarray]:
    """Extra fields for the given daily times (any split): (T, H, W) each, from the extracted periods."""
    days = time.astype("datetime64[D]")
    out = {}
    for period, (y0, y1) in exp.extra["periods"].items():
        d = extra_dir(exp, period)
        if not (d / "time.npy").exists():
            continue
        pt = np.load(d / "time.npy").astype("datetime64[D]")
        sel = np.isin(days, pt)
        if not sel.any():
            continue
        idx = np.searchsorted(pt, days[sel])
        for f in fields:
            a = np.load(d / f"{f}.npy", mmap_mode="r")
            out.setdefault(f, np.full((len(days), *a.shape[1:]), np.nan, np.float32))[sel] = a[idx]
    missing = [f for f in fields if f not in out or np.isnan(out[f]).all(axis=(1, 2)).any()]
    if missing:
        raise FileNotFoundError(f"extra fields {missing} not extracted for all days; run `dslab extract`")
    return out


# ---- the driving GCM's daily fields on the 2° predictor grid
def _overlap(lo_src, hi_src, lo_dst, hi_dst):
    """(n_dst, n_src) overlap lengths of 1-D intervals."""
    return np.clip(np.minimum(hi_dst[:, None], hi_src[None]) - np.maximum(lo_dst[:, None], lo_src[None]), 0, None)


def _conservative_weights(ds, pred_lat, pred_lon):
    """Separable first-order conservative weights from the GCM's cells (bounds in the file) to 2° cells
    centred on the predictor grid: area overlap in sin(latitude) x longitude, rows normalised."""
    lb = ds["lat_bnds"].values
    wl = _overlap(np.sin(np.radians(lb[:, 0])), np.sin(np.radians(lb[:, 1])),
                  np.sin(np.radians(pred_lat - 1.0)), np.sin(np.radians(pred_lat + 1.0)))
    ob = (ds["lon_bnds"].values + 180.0) % 360.0 - 180.0  # to [-180, 180)
    ob[:, 1] = np.where(ob[:, 1] < ob[:, 0], ob[:, 1] + 360.0, ob[:, 1])
    wo = _overlap(ob[:, 0], ob[:, 1], pred_lon - 1.0, pred_lon + 1.0)
    return wl / wl.sum(1, keepdims=True), wo / wo.sum(1, keepdims=True)


def gcm_dir(exp: Experiment, period: str) -> Path:
    return WORK / "cache" / exp.domain / "extra_gcm" / period


def extract_gcm(exp: Experiment, periods=None) -> None:
    """Daily GCM fields (exp.extra['gcm']) regridded to the predictor grid: (T, h, w) per variable and period."""
    from .data import load_grid

    g = exp.extra["gcm"]
    grid = load_grid(exp)
    for period in periods or exp.extra["periods"]:
        y0, y1 = exp.extra["periods"][period]
        days = np.arange(f"{y0}-01-01", f"{y1 + 1}-01-01", dtype="datetime64[D]")
        out = gcm_dir(exp, period)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "time.npy", days.astype("datetime64[ns]") + np.timedelta64(12, "h"))
        for var in g["variables"]:
            parts = []
            for scen in ("historical", "rcp85"):
                base = Path(g["root"]) / scen / g["path"]
                if not base.exists():
                    continue
                version = sorted(p.name for p in base.iterdir() if p.is_dir())[-1]
                for f in sorted((base / version / var).glob(f"{var}_*.nc")):
                    a, b = _years_of(f)
                    if b < y0 or a > y1:
                        continue
                    with xr.open_dataset(f) as ds:
                        wl, wo = _conservative_weights(ds, grid.pred_lat, grid.pred_lon)
                        t = ds["time"].values.astype("datetime64[D]")
                        keep = np.isin(t, days)
                        a3 = ds[var].values[keep].astype(np.float64)
                        parts.append((t[keep], np.einsum("il,tlk,jk->tij", wl, a3, wo)))
            t = np.concatenate([p[0] for p in parts])
            a = np.concatenate([p[1] for p in parts])
            order = np.argsort(t)
            t, a = t[order], a[order]
            if not np.array_equal(t, days):
                raise ValueError(f"GCM {var} {period}: {len(t)} days, expected {len(days)}")
            scale = 86400.0 if var == "pr" else 1.0
            np.save(out / f"{var}.npy", (a * scale).astype(np.float32))
            print(f"[extract] GCM {period} {var}: {a.shape}, mean {a.mean() * scale:.3f}", flush=True)


def load_gcm(exp: Experiment, time: np.ndarray, var: str) -> np.ndarray:
    """The GCM's daily field on the predictor grid for the given days (T, h, w)."""
    days = time.astype("datetime64[D]")
    out = None
    for period in exp.extra["periods"]:
        d = gcm_dir(exp, period)
        if not (d / f"{var}.npy").exists():
            continue
        pt = np.load(d / "time.npy").astype("datetime64[D]")
        sel = np.isin(days, pt)
        if sel.any():
            a = np.load(d / f"{var}.npy", mmap_mode="r")
            out = np.full((len(days), *a.shape[1:]), np.nan, np.float32) if out is None else out
            out[sel] = a[np.searchsorted(pt, days[sel])]
    if out is None or np.isnan(out).all(axis=(1, 2)).any():
        raise FileNotFoundError(f"GCM {var} not extracted for all days; run `dslab extract --gcm`")
    return out
