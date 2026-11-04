"""Linear physical laws on the target grid, for law-conditioned adaptive flow matching (LC-AFM; the plan
"pcAFM: from Burgers to 12 km climate downscaling", laws T and O, plus the unfiltered variant of T).

Each law is linear in one target variable x and reads, for day d in calendar month m,

    J x  ≈  J phi_d + c_m      with tolerance tau_m per grid point,

where phi_d is computed from that day's coarse predictors and the static orography only (no truth at
test time), c_m is the mean residual on training truth and tau_m the RMS of the remainder, so that the
truth scores 1 in tolerance units on the training data. J is the identity or the high-pass H = I - P that
removes the scales resolved by the 2° predictor grid. choice: P is the orthogonal projector onto the bilinear
interpolants of 2° fields (the least-squares version of U A, with A the mean over the 12 km cells in each 2°
cell and U bilinear interpolation back): H is then symmetric and idempotent, which turns the conditioning into
a small positive-definite (Cholesky) solve. Laws:

    T      lapse rate, sub-2° scales:     H tasmax ≈ H T_free + c    T_free: coarse t(z) profile at the
    Tfull  lapse rate, all scales:          tasmax ≈ T_free + c      grid point's elevation; c = monthly
                                                                     boundary-layer excess
    O      orographic precipitation:     H pr ≈ H P_SB(h') + c       Smith & Barstad (2004) linear theory on
                                                                     the sub-2° terrain h' = H orog

Choices not fixed by the plan are marked "choice:".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Grid, Split
from .experiment import WORK

VERSION = 2  # bump when a law's definition changes (invalidates the caches); 2: orthogonal projector P
LAWS = {"T": ("tasmax", True), "Tfull": ("tasmax", False), "O": ("pr", True)}  # name -> (variable, high-pass)
G, RD, RV, CP, LV = 9.81, 287.04, 461.5, 1004.6, 2.5e6
EPS = RD / RV


# ---- coarse operator
@dataclass(frozen=True)
class CoarseOperator:
    """A: (r, N) mean over the target points inside each 2° cell that contains any (r cells);
    U: (N, r) bilinear interpolation from those cells' centres, renormalised over cells with data (U A maps
    constants to themselves); Q: (N, r) orthonormal basis of span(U), so that P = Q Q^T is the orthogonal
    projector onto the 2°-resolved fields (constants included). N = H * W target points, row-major."""

    A: np.ndarray
    U: np.ndarray
    Q: np.ndarray
    B_full: np.ndarray  # (N, h * w) bilinear interpolation from the full predictor grid
    shape: tuple[int, int]

    @classmethod
    def from_grid(cls, grid: Grid) -> CoarseOperator:
        lat, lon = grid.lat.ravel(), grid.lon.ravel()
        plat, plon = grid.pred_lat, grid.pred_lon
        h, w = len(plat), len(plon)
        dlat, dlon = plat[1] - plat[0], plon[1] - plon[0]
        N = len(lat)
        cell = (np.rint((lat - plat[0]) / dlat).astype(int) * w + np.rint((lon - plon[0]) / dlon).astype(int))
        cells, inv, counts = np.unique(cell, return_inverse=True, return_counts=True)
        A = np.zeros((len(cells), N), np.float32)
        A[inv, np.arange(N)] = 1 / counts[inv]

        fi, fj = (lat - plat[0]) / dlat, (lon - plon[0]) / dlon
        i0, j0 = np.clip(np.floor(fi).astype(int), 0, h - 2), np.clip(np.floor(fj).astype(int), 0, w - 2)
        di, dj = fi - i0, fj - j0
        corners = [(i0, j0, (1 - di) * (1 - dj)), (i0 + 1, j0, di * (1 - dj)),
                   (i0, j0 + 1, (1 - di) * dj), (i0 + 1, j0 + 1, di * dj)]
        B_full = np.zeros((N, h * w), np.float32)
        U = np.zeros((N, len(cells)), np.float32)
        slot = np.full(h * w, -1)
        slot[cells] = np.arange(len(cells))
        for ii, jj, wt in corners:
            B_full[np.arange(N), ii * w + jj] += wt
            k = slot[ii * w + jj]
            ok = k >= 0
            U[np.arange(N)[ok], k[ok]] += wt[ok]
        U /= U.sum(axis=1, keepdims=True)
        Q = np.linalg.qr(U.astype(np.float64))[0].astype(np.float32)
        return cls(A=A, U=U, Q=Q, B_full=B_full, shape=grid.shape)

    def lowpass(self, x: np.ndarray) -> np.ndarray:
        """P x = Q Q^T x for x (..., H, W)."""
        flat = x.reshape(-1, self.Q.shape[0])
        return ((flat @ self.Q) @ self.Q.T).reshape(x.shape)

    def highpass(self, x: np.ndarray) -> np.ndarray:
        return x - self.lowpass(x)

    def interp(self, coarse: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of coarse fields (..., h, w) to the target grid (..., H, W)."""
        flat = coarse.reshape(-1, self.B_full.shape[1])
        return (flat @ self.B_full.T).reshape(*coarse.shape[:-2], *self.shape)


def _channel(x: np.ndarray, predictors, name: str) -> np.ndarray:
    return x[:, list(predictors).index(name)].astype(np.float64)


# ---- law T: free-atmosphere temperature at the grid point's elevation
def free_temperature(x: np.ndarray, predictors, op: CoarseOperator, orog: np.ndarray) -> np.ndarray:
    """Coarse t(z) profile (850, 700, 500 hPa) interpolated horizontally to each target point and
    vertically to its elevation; below 850 hPa extrapolated with the 850-700 hPa lapse rate (above 500 hPa
    with the 700-500 hPa one). x: (T, C, h, w) raw predictors -> (T, H, W) K."""
    t = {lev: op.interp(_channel(x, predictors, f"t_{lev}")) for lev in (850, 700, 500)}
    z = {lev: op.interp(_channel(x, predictors, f"z_{lev}")) for lev in (850, 700, 500)}
    zf = orog[None].astype(np.float64)
    low = t[850] + (t[700] - t[850]) * (zf - z[850]) / (z[700] - z[850])
    high = t[700] + (t[500] - t[700]) * (zf - z[700]) / (z[500] - z[700])
    return np.where(zf <= z[700], low, high).astype(np.float32)


# ---- law O: Smith & Barstad (2004) linear orographic precipitation
def _moist_coefficients(x: np.ndarray, predictors) -> dict[str, np.ndarray]:
    """Per day and coarse cell: 700 hPa wind, moist stability N_m^2, condensation coefficient C_w."""
    t850, t700 = _channel(x, predictors, "t_850"), _channel(x, predictors, "t_700")
    z850, z700 = _channel(x, predictors, "z_850"), _channel(x, predictors, "z_700")
    q850 = _channel(x, predictors, "q_850")
    T, p = 0.5 * (t850 + t700), 77500.0
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    rs = EPS * es / (p - es)
    gamma_m = G * (1 + LV * rs / (RD * T)) / (CP + LV**2 * rs * EPS / (RD * T**2))  # moist adiabat, K/m
    gamma = -(t700 - t850) / (z700 - z850)  # environmental lapse rate, K/m
    nm2 = np.maximum(G / T * (gamma_m - gamma), 1e-6)  # choice: floor N_m = 1e-3 1/s (no unstable branch)
    rho_v = q850 * 85000.0 / (RD * t850 * (1 + 0.61 * q850))
    # choice: actual 850 hPa vapour density instead of the saturation value (humidity enters), and the
    # lapse rate floored at 2 K/km so that C_w stays bounded in inversions
    cw = rho_v * gamma_m / np.maximum(gamma, 2e-3)
    return {"u": _channel(x, predictors, "u_700"), "v": _channel(x, predictors, "v_700"), "nm2": nm2, "cw": cw}


def _grid_spacing_m(grid: Grid) -> float:
    """Mean distance between neighbouring target points (m), from lat/lon."""
    lat, lon = np.radians(grid.lat), np.radians(grid.lon)

    def dist(a, b, c, d):
        h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
        return 2 * 6.371e6 * np.arcsin(np.sqrt(h))

    return float(0.5 * (dist(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:]).mean()
                        + dist(lat[:-1], lon[:-1], lat[1:], lon[1:]).mean()))


class SmithBarstad:
    """P_SB(h') on overlapping patches: core `patch` x `patch` points, each computed by FFT on the core plus a
    `margin` of terrain on every side (zero outside the domain, Tukey taper) with that patch's own coarse
    coefficients at its centre; cores blended with Hann weights. Output in mm/day, linear in the
    terrain (negative on lee slopes: the linear law is not clipped)."""

    def __init__(self, grid: Grid, op: CoarseOperator, patch: int = 32, margin: int = 16):
        H, W = grid.shape
        self.patch, self.margin, self.size = patch, margin, patch + 2 * margin
        terrain = op.highpass(grid.orog.astype(np.float64))  # h' = sub-2° terrain
        stride = patch // 2
        starts = lambda n: sorted(set(list(range(0, n - patch + 1, stride)) + [n - patch]))  # noqa: E731
        self.corners = [(i, j) for i in starts(H) for j in starts(W)]
        L = self.size
        taper = np.ones(L)
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(margin) + 0.5) / margin))
        taper[:margin], taper[-margin:] = ramp, ramp[::-1]
        taper2 = np.outer(taper, taper)
        hhat, centres = [], []
        for i, j in self.corners:
            ext = np.zeros((L, L))
            i0, j0 = i - margin, j - margin
            si, sj = slice(max(i0, 0), min(i0 + L, H)), slice(max(j0, 0), min(j0 + L, W))
            ext[si.start - i0:si.stop - i0, sj.start - j0:sj.stop - j0] = terrain[si, sj]
            hhat.append(np.fft.fft2(ext * taper2))
            centres.append((i + patch / 2 - 0.5, j + patch / 2 - 0.5))
        self.hhat = np.stack(hhat).astype(np.complex128)  # (P, L, L)
        dx = _grid_spacing_m(grid)
        k = 2 * np.pi * np.fft.fftfreq(L, dx)
        self.kx, self.ky = np.meshgrid(k, k)  # x = second (column) axis, y = first
        # patch-centre interpolation weights from the target grid's lat/lon to the coarse grid
        ci = np.array([c[0] for c in centres])
        cj = np.array([c[1] for c in centres])
        lat_c = np.array([grid.lat[int(round(a)), int(round(b))] for a, b in zip(ci, cj)])
        lon_c = np.array([grid.lon[int(round(a)), int(round(b))] for a, b in zip(ci, cj)])
        self.centre_op = CoarseOperator.from_grid(Grid(
            lat=lat_c[:, None], lon=lon_c[:, None], y=np.arange(len(ci)), x=np.arange(1), dims=grid.dims,
            orog=np.zeros((len(ci), 1)), pred_lat=grid.pred_lat, pred_lon=grid.pred_lon))
        win = np.sin(np.pi * (np.arange(patch) + 0.5) / patch) ** 2
        self.win = np.outer(win, win)
        wsum = np.zeros((H, W))
        for i, j in self.corners:
            wsum[i:i + patch, j:j + patch] += self.win
        self.wsum = wsum
        self.shape = (H, W)
        # map: grid x direction = increasing column index; the 700 hPa wind is (eastward, northward), which
        # we rotate into grid axes with the local orientation of the target grid at each patch centre
        self.rot = self._grid_rotation(grid, ci, cj)

    @staticmethod
    def _grid_rotation(grid: Grid, ci, cj):
        """(P, 2, 2) rotation taking (east, north) wind components to (column, row) grid components."""
        out = []
        for a, b in zip(ci, cj):
            i, j = int(np.clip(round(a), 1, grid.shape[0] - 2)), int(np.clip(round(b), 1, grid.shape[1] - 2))
            coslat = np.cos(np.radians(grid.lat[i, j]))
            ex = np.array([(grid.lon[i, j + 1] - grid.lon[i, j - 1]) * coslat, grid.lat[i, j + 1] - grid.lat[i, j - 1]])
            ey = np.array([(grid.lon[i + 1, j] - grid.lon[i - 1, j]) * coslat, grid.lat[i + 1, j] - grid.lat[i - 1, j]])
            ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)
            out.append(np.stack([ex, ey]))  # rows: unit vectors of the grid axes in (east, north)
        return np.stack(out)

    def coefficients(self, x: np.ndarray, predictors) -> dict[str, np.ndarray]:
        """Per day and patch: wind in grid axes (U, V), N_m^2, C_w. Returns arrays (T, P)."""
        c = _moist_coefficients(x, predictors)
        at = {k: self.centre_op.interp(v)[:, :, 0] for k, v in c.items()}  # (T, P)
        wind = np.stack([at["u"], at["v"]], axis=-1)  # (T, P, 2) east, north
        grid_wind = np.einsum("pkc,tpc->tpk", self.rot, wind)
        return {"U": grid_wind[..., 0], "V": grid_wind[..., 1], "nm2": at["nm2"], "cw": at["cw"]}

    def transfer_parts(self, coef: dict, sl) -> tuple[np.ndarray, np.ndarray]:
        """Intrinsic frequency sigma and vertical wavenumber m for days `sl`: (t, P, L, L)."""
        U, V, nm2 = (coef[k][sl][..., None, None] for k in ("U", "V", "nm2"))
        sig = U * self.kx + V * self.ky
        sig = np.where(np.abs(sig) < 1e-12, 1e-12, sig)
        k2 = self.kx**2 + self.ky**2
        m2 = (nm2 - sig**2) / sig**2 * k2
        m = np.where(m2 >= 0, np.sign(sig) * np.sqrt(np.abs(m2)), 1j * np.sqrt(np.abs(m2)))
        return sig, m

    def response(self, coef: dict, sl, sig, m, hw, tau_c, tau_f) -> np.ndarray:
        """P_SB (t, H, W) in mm/day for days `sl` and the given parameters."""
        cw = coef["cw"][sl][..., None, None]
        phat = (cw * 1j * sig * self.hhat[None]
                / ((1 - 1j * m * hw) * (1 + 1j * sig * tau_c) * (1 + 1j * sig * tau_f)))
        p = np.fft.ifft2(phat).real * 86400.0  # kg m-2 s-1 -> mm/day
        out = np.zeros((p.shape[0], *self.shape))
        a, b = self.margin, self.margin + self.patch
        for n, (i, j) in enumerate(self.corners):
            out[:, i:i + self.patch, j:j + self.patch] += p[:, n, a:b, a:b] * self.win
        return out / self.wsum

    def __call__(self, x: np.ndarray, predictors, params: dict, chunk: int = 16) -> np.ndarray:
        coef = self.coefficients(x, predictors)
        out = np.empty((len(x), *self.shape), np.float32)
        for i in range(0, len(x), chunk):
            sl = slice(i, i + chunk)
            sig, m = self.transfer_parts(coef, sl)
            out[sl] = params["gain"] * self.response(coef, sl, sig, m, params["hw"], params["tau_c"], params["tau_f"])
        return out


SB_SEARCH = {"hw": [1000.0, 2000.0, 3000.0, 4000.0], "tau_c": [500.0, 1000.0, 2000.0],
             "tau_f": [500.0, 1000.0, 2000.0]}


def fit_smith_barstad(sb: SmithBarstad, op: CoarseOperator, train: Split, n_days: int = 365,
                      seed: int = 0) -> dict:
    """Grid search of H_w, tau_c, tau_f on a random subset of training days, each with the least-squares
    gain of H P_SB against H pr (choice: a gain absorbs the uncertain amplitude of C_w)."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(len(train), size=min(n_days, len(train)), replace=False))
    x = train.x[days]
    hpr = op.highpass(train.y[days, list(train.variables).index("pr")].astype(np.float64))
    coef = sb.coefficients(x, train.predictors)
    combos = [(a, b, c) for a in SB_SEARCH["hw"] for b in SB_SEARCH["tau_c"] for c in SB_SEARCH["tau_f"]]
    num = np.zeros(len(combos))
    den = np.zeros(len(combos))
    for i in range(0, len(days), 16):
        sl = slice(i, i + 16)
        sig, m = sb.transfer_parts(coef, sl)
        for k, (hw, tc, tf) in enumerate(combos):
            hp = op.highpass(sb.response(coef, sl, sig, m, hw, tc, tf))
            num[k] += (hp * hpr[sl]).sum()
            den[k] += (hp * hp).sum()
    gain = num / np.maximum(den, 1e-30)
    ss_tot = (hpr**2).sum()
    explained = gain * num / ss_tot  # = 1 - SSE / SS for the least-squares gain
    best = int(np.argmax(explained))
    hw, tc, tf = combos[best]
    return {"hw": hw, "tau_c": tc, "tau_f": tf, "gain": float(gain[best]),
            "explained_variance_highpass": float(explained[best]), "search_days": len(days),
            "search": [{"hw": a, "tau_c": b, "tau_f": c, "gain": float(g), "explained": float(e)}
                       for (a, b, c), g, e in zip(combos, gain, explained)]}


# ---- the law set
def months(time: np.ndarray) -> np.ndarray:
    """Calendar month index 0..11 per day."""
    return pd.DatetimeIndex(time).month.values - 1


@dataclass
class LawSet:
    """Fitted laws: offset c and tolerance tau per calendar month and grid point (physical units), and the
    Smith-Barstad parameters. phi() gives the right-hand side J phi + c for any split."""

    names: tuple[str, ...]
    predictors: tuple[str, ...]
    offset: dict[str, np.ndarray]  # name -> (12, H, W)
    tol: dict[str, np.ndarray]  # name -> (12, H, W)
    sb_params: dict | None = None
    info: dict = field(default_factory=dict)
    _ops: dict = field(default_factory=dict, repr=False)

    # geometry is rebuilt from the grid (deterministic), not stored
    def _op(self, grid: Grid) -> CoarseOperator:
        if "op" not in self._ops:
            self._ops["op"] = CoarseOperator.from_grid(grid)
        return self._ops["op"]

    def _sb(self, grid: Grid) -> SmithBarstad:
        if "sb" not in self._ops:
            self._ops["sb"] = SmithBarstad(grid, self._op(grid))
        return self._ops["sb"]

    @staticmethod
    def variable(name: str) -> str:
        return LAWS[name][0]

    @staticmethod
    def filtered(name: str) -> bool:
        return LAWS[name][1]

    def _raw_phi(self, name: str, x: np.ndarray, grid: Grid, sb_params=None) -> np.ndarray:
        op = self._op(grid)
        if name in ("T", "Tfull"):
            return free_temperature(x, self.predictors, op, grid.orog)
        if name == "O":
            return self._sb(grid)(x, self.predictors, sb_params or self.sb_params)
        raise KeyError(name)

    def apply_J(self, name: str, field_: np.ndarray, grid: Grid) -> np.ndarray:
        return self._op(grid).highpass(field_) if self.filtered(name) else field_

    @classmethod
    def fit(cls, train: Split, grid: Grid, names) -> LawSet:
        names = tuple(names)
        law = cls(names=names, predictors=tuple(train.predictors), offset={}, tol={})
        mon = months(train.time)
        if "O" in names:
            law.sb_params = fit_smith_barstad(law._sb(grid), law._op(grid), train)
            law.info["O"] = {k: v for k, v in law.sb_params.items() if k != "search"}
        for name in names:
            v = list(train.variables).index(cls.variable(name))
            r = law.apply_J(name, train.y[:, v].astype(np.float64) - law._raw_phi(name, train.x, grid), grid)
            off = np.stack([r[mon == k].mean(axis=0) for k in range(12)])
            rem = r - off[mon]
            tol = np.stack([np.sqrt((rem[mon == k] ** 2).mean(axis=0)) for k in range(12)])
            floor = 1e-3 * float(np.sqrt((r**2).mean()))  # choice: no zero tolerances
            law.offset[name] = off.astype(np.float32)
            law.tol[name] = np.maximum(tol, floor).astype(np.float32)
            Jy = law.apply_J(name, train.y[:, v].astype(np.float64), grid)
            law.info.setdefault(name, {}).update(
                variable=cls.variable(name), filtered=cls.filtered(name),
                rms_J_truth=float(np.sqrt((Jy**2).mean())) if cls.filtered(name) else float(train.y[:, v].std()),
                rms_residual=float(np.sqrt((r**2).mean())), rms_tolerance=float(np.sqrt((tol**2).mean())))
        return law

    def phi(self, split: Split, grid: Grid) -> dict[str, np.ndarray]:
        """Right-hand side b = J phi + c (physical units, (T, H, W)) of every law for this split."""
        mon = months(split.time)
        return {n: (self.apply_J(n, self._raw_phi(n, split.x, grid).astype(np.float64), grid)
                    + self.offset[n][mon]).astype(np.float32) for n in self.names}

    def residual(self, name: str, field_: np.ndarray, b: np.ndarray, mon: np.ndarray, grid: Grid) -> float:
        """Mean squared law residual in tolerance units ((J x - b) / tau)^2; the truth scores 1 on training."""
        r = (self.apply_J(name, field_.astype(np.float64), grid) - b) / self.tol[name][mon]
        return float((r**2).mean())

    def subset(self, names) -> LawSet:
        names = tuple(names)
        return LawSet(names=names, predictors=self.predictors, offset={n: self.offset[n] for n in names},
                      tol={n: self.tol[n] for n in names}, sb_params=self.sb_params if "O" in names else None,
                      info={n: self.info[n] for n in names if n in self.info}, _ops=self._ops)

    # ---- persistence (arrays + JSON)
    def state(self) -> dict[str, np.ndarray]:
        out = {"meta": np.array(json.dumps({"names": self.names, "predictors": self.predictors,
                                            "sb_params": self.sb_params, "info": self.info}))}
        for n in self.names:
            out[f"offset_{n}"], out[f"tol_{n}"] = self.offset[n], self.tol[n]
        return out

    @classmethod
    def from_state(cls, st: dict) -> LawSet:
        meta = json.loads(str(st["meta"]))
        names = tuple(meta["names"])
        return cls(names=names, predictors=tuple(meta["predictors"]),
                   offset={n: np.asarray(st[f"offset_{n}"]) for n in names},
                   tol={n: np.asarray(st[f"tol_{n}"]) for n in names},
                   sb_params=meta["sb_params"], info=meta["info"])


# ---- caches (fitting O takes minutes on CPU; phi of a 20-year split about as long)
def _digest(*arrays, extra="") -> str:
    h = hashlib.sha1(f"{VERSION}|{extra}".encode())
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8).data)
    return h.hexdigest()[:12]


CACHE = WORK / "cache" / "laws"


def _save(path: Path, arrays: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def fit_cached(train: Split, grid: Grid, names=tuple(LAWS)) -> LawSet:
    """All laws fitted once per training set (cached on WORK), restricted to `names`."""
    key = _digest(train.x, train.y, grid.orog, grid.lat, extra=",".join(train.predictors))
    path = CACHE / f"lawset-{key}.npz"
    if path.exists():
        with np.load(path) as f:
            law = LawSet.from_state({k: f[k] for k in f.files})
    else:
        print(f"[laws] fitting {', '.join(LAWS)} on {train.name} ({len(train)} days)", flush=True)
        law = LawSet.fit(train, grid, tuple(LAWS))
        _save(path, law.state())
    return law.subset(names)


def phi_cached(law: LawSet, split: Split, grid: Grid) -> dict[str, np.ndarray]:
    """b = J phi + c for every law in `law` and every day of `split`, cached per law."""
    out = {}
    base = _digest(split.x, split.time.astype("int64"), grid.orog)
    for n in law.names:
        one = law.subset([n])
        st = one.state()
        key = _digest(*[st[k] for k in sorted(st) if k != "meta"], extra=f"{base}|{n}|{one.sb_params}")
        path = CACHE / f"phi-{n}-{key}.npz"
        if path.exists():
            with np.load(path) as f:
                out[n] = f["b"]
        else:
            out[n] = one.phi(split, grid)[n]
            _save(path, {"b": out[n]})
    return out
