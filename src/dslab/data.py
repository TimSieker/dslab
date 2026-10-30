"""Load predictors and targets into memory as numpy arrays.

The first load of a predictor/target file pair reads the NetCDF files and writes plain .npy
arrays to work/cache/<domain>/ (float32, predictors selected by name in the experiment's order,
lat/lon sorted ascending). Later loads memory-map the cache, which takes seconds.

Array layout used everywhere in this package:
    predictors  x: (time, channel, lat, lon)   raw units, coarse grid
    targets     y: (time, variable, y, x)      physical units (pr mm/day, tasmax K), target grid
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .experiment import WORK, Experiment, Source

DROP_VARS = ("time_bnds", "lat_bnds", "lon_bnds")


@dataclass(frozen=True)
class Grid:
    """Target grid and the coarse predictor grid of one experiment."""

    lat: np.ndarray  # (H, W) target-grid latitude
    lon: np.ndarray  # (H, W)
    y: np.ndarray  # (H,) projection coordinate (km for ALPS)
    x: np.ndarray  # (W,)
    dims: tuple[str, str]  # names of the target spatial dims, e.g. ("y", "x")
    orog: np.ndarray  # (H, W) model orography, m
    pred_lat: np.ndarray  # (h,) ascending
    pred_lon: np.ndarray  # (w,) ascending

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape


@dataclass(frozen=True)
class Split:
    """One split (train, val or a test set) held in memory."""

    name: str
    time: np.ndarray  # (T,) datetime64[ns]
    x: np.ndarray  # (T, C, h, w) float32
    y: np.ndarray | None  # (T, V, H, W) float32, None without ground truth
    variables: tuple[str, ...]
    predictors: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.time)

    @property
    def years(self) -> np.ndarray:
        return pd.DatetimeIndex(self.time).year.values

    @property
    def dayofyear(self) -> np.ndarray:
        return pd.DatetimeIndex(self.time).dayofyear.values

    def subset(self, index) -> Split:
        return replace(self, time=self.time[index], x=self.x[index],
                       y=None if self.y is None else self.y[index])


def _open(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    return ds.drop_vars([v for v in DROP_VARS if v in ds.variables])


def _cache_dir(exp: Experiment, src: Source) -> Path:
    key = {
        "predictors_file": str(src.predictors),
        "target_file": str(src.target),
        "predictors": list(exp.predictors),
        "variables": list(exp.variables),
        "stat": [(p.stat().st_size, int(p.stat().st_mtime)) for p in (src.predictors, src.target) if p],
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    stem = str(src.predictors.relative_to(exp.data_root)).removesuffix(".nc").replace("/", "__")
    return WORK / "cache" / exp.domain / f"{stem}-{digest}"


def _build_cache(exp: Experiment, src: Source, out: Path) -> None:
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    meta = {"predictors_file": str(src.predictors), "target_file": str(src.target)}

    with _open(src.predictors) as ds:
        missing = [v for v in exp.predictors if v not in ds]
        if missing:
            raise KeyError(f"{src.predictors}: missing predictors {missing}")
        ds = ds.sortby(["lat", "lon"])
        x = np.stack([ds[v].transpose("time", "lat", "lon").values.astype(np.float32)
                      for v in exp.predictors], axis=1)
        time = ds["time"].values.astype("datetime64[ns]")
        meta["pred_lat"] = ds["lat"].values.tolist()
        meta["pred_lon"] = ds["lon"].values.tolist()
    meta["predictor_nans"] = {v: int(np.isnan(x[:, i]).sum()) for i, v in enumerate(exp.predictors)}
    np.save(tmp / "x.npy", x)
    np.save(tmp / "time.npy", time)

    if src.target is not None:
        with _open(src.target) as ds:
            ty = ds["time"].values.astype("datetime64[D]")
            if not np.array_equal(ty, time.astype("datetime64[D]")):
                raise ValueError(f"time axes of {src.predictors.name} and {src.target.name} differ")
            dims = ds[exp.variables[0]].dims[-2:]
            y = np.stack([ds[v].transpose("time", *dims).values.astype(np.float32)
                          for v in exp.variables], axis=1)
            meta["target_dims"] = list(dims)
            meta["target_units"] = {v: ds[v].attrs.get("units", "") for v in exp.variables}
        meta["target_nans"] = {v: int(np.isnan(y[:, i]).sum()) for i, v in enumerate(exp.variables)}
        np.save(tmp / "y.npy", y)

    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def _load_source(exp: Experiment, src: Source):
    d = _cache_dir(exp, src)
    if not (d / "meta.json").exists():
        print(f"[data] building cache {d}", flush=True)
        _build_cache(exp, src, d)
    time = np.load(d / "time.npy")
    x = np.load(d / "x.npy", mmap_mode="r")
    y = np.load(d / "y.npy", mmap_mode="r") if (d / "y.npy").exists() else None
    return time, x, y


def load_split(exp: Experiment, split: str) -> Split:
    """Load a split into memory. 'train' and 'val' are the training file without / with only val_years."""
    time, x, y = _load_source(exp, exp.source(split))
    if split in ("train", "val"):
        is_val = np.isin(pd.DatetimeIndex(time).year.values, exp.val_years)
        idx = np.flatnonzero(is_val if split == "val" else ~is_val)
    else:
        idx = np.arange(len(time))
    return Split(
        name=split,
        time=time[idx],
        x=np.ascontiguousarray(x[idx]),
        y=None if y is None else np.ascontiguousarray(y[idx]),
        variables=exp.variables,
        predictors=exp.predictors,
    )


@lru_cache(maxsize=8)
def load_grid(exp: Experiment) -> Grid:
    """Target-grid coordinates (from the training target file), orography and predictor grid."""
    d = _cache_dir(exp, exp.train)
    if not (d / "meta.json").exists():
        _load_source(exp, exp.train)
    meta = json.loads((d / "meta.json").read_text())
    dims = tuple(meta["target_dims"])
    with _open(exp.train.target) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        if lat.ndim == 1:  # regular lat/lon target grid (NZ, SA)
            lon, lat = np.meshgrid(lon, lat)
        yc, xc = ds[dims[0]].values, ds[dims[1]].values
    with _open(exp.static) as ds:
        orog = ds["orog"].transpose(*dims).values.astype(np.float32)
    return Grid(lat=lat, lon=lon, y=yc, x=xc, dims=dims, orog=orog,
                pred_lat=np.asarray(meta["pred_lat"]), pred_lon=np.asarray(meta["pred_lon"]))
