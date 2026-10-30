"""Prediction files: one NetCDF per run and split, dims (member, time, y, x), on the target grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .data import Grid, Split

UNITS = {"pr": "mm day-1", "tasmax": "K"}
LONG_NAMES = {"pr": "Precipitation", "tasmax": "Daily Maximum Near-Surface Air Temperature"}


def to_dataset(pred: np.ndarray, split: Split, grid: Grid, attrs: dict | None = None) -> xr.Dataset:
    """pred: (M, T, V, H, W) physical units -> Dataset with one variable per target."""
    ydim, xdim = grid.dims
    coords = {
        "member": np.arange(pred.shape[0]),
        "time": split.time,
        ydim: grid.y,
        xdim: grid.x,
        "lat": ((ydim, xdim), grid.lat),
        "lon": ((ydim, xdim), grid.lon),
    }
    data = {v: (("member", "time", ydim, xdim), pred[:, :, i],
                {"units": UNITS.get(v, ""), "long_name": LONG_NAMES.get(v, v)})
            for i, v in enumerate(split.variables)}
    return xr.Dataset(data, coords=coords, attrs=attrs or {})


def write_prediction(path: Path, pred: np.ndarray, split: Split, grid: Grid, attrs: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = to_dataset(pred, split, grid, attrs)
    tmp = path.with_suffix(".tmp.nc")
    # Compression mostly pays off for pr (many dry grid points).
    enc = {v: {"zlib": True, "complevel": 1, "dtype": "float32",
               "chunksizes": (1, 1, *pred.shape[-2:])} for v in split.variables}
    ds.to_netcdf(tmp, encoding=enc)
    tmp.rename(path)
    return path


def open_prediction(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def truth_dataset(split: Split, grid: Grid) -> xr.Dataset:
    """Ground truth of a split in the same layout as a prediction (single 'member' dropped)."""
    if split.y is None:
        raise ValueError(f"split {split.name} has no ground truth")
    return to_dataset(split.y[None], split, grid).isel(member=0, drop=True)
