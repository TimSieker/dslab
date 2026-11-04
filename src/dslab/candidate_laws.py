"""Candidate laws beyond T / Tfull / O, tested on the RCM truth before they are used for conditioning
(`dslab laws --exp alps_esd --candidates`; plotted in notebooks/laws.ipynb, part 2).

Fields that LC-AFM would generate alongside pr and tasmax (10 m wind, near-surface humidity, evaporation, ...)
are taken from the RCM itself (work/cache/<domain>/extra, `dslab extract`), so these are **oracle** tests: they
show what a law can contribute if those fields were right, an upper bound for joint downscaling. Coarse inputs
come from the benchmark's 2° predictors, as at test time.

Each law reads   target ≈ Σ_k a_k term_k + c_m   with tolerance τ_m (per grid point, calendar month m):
fixed terms have a_k = 1 (exact physics), the others have one coefficient fitted by least squares on the training
years' day-to-day anomalies; c_m and τ_m are fitted as for the other laws (truth scores 1 on training).
For each law LAWS[name]["exact"] states the law in a perfect environment and LAWS[name]["estimate"] how it is
estimated from the data we have.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
import xarray as xr

from .data import Grid, Split, load_grid, load_split
from .experiment import Experiment
from .extra import divergence, geometry, gradient, load_extra, to_grid_axes
from .law_diagnostics import CHUNK, SPLITS, SplitStats, assemble
from .laws import LV, RD, RV, CoarseOperator, _moist_coefficients, free_temperature, months

G0, OMEGA, R_EARTH = 9.81, 7.292e-5, 6.371e6
WET = 1.0  # mm/day: wet-day threshold of the Clausius-Clapeyron law

LAWS = {
    "Tfull_tas": dict(
        target="tas", needs=("tas",), fixed=("T_free",), fitted=(),
        title="Tfull_tas · lapse rate, daily mean",
        exact="Hydrostatic balance gives the free-atmosphere temperature profile T(z) between the pressure levels; "
              "the 2 m temperature equals that profile at the surface plus the boundary-layer departure "
              "T_2m - T_free(z_s), which is set by the surface energy balance.",
        estimate="T_free from the coarse t, z at 850/700/500 hPa (as Tfull); the boundary-layer departure is "
                 "not modelled, only its monthly mean per grid point (the offset) and its spread (the tolerance)."),
    "Tfull_tasmin": dict(
        target="tasmin", needs=("tasmin",), fixed=("T_free",), fitted=(),
        title="Tfull_tasmin · lapse rate, minimum",
        exact="As Tfull_tas for the daily minimum: T_min = T_free(z_s) + night-time boundary-layer departure "
              "(radiative cooling, inversions, cold-air pools).",
        estimate="As Tfull_tas; the night-time inversion is left to the offset and tolerance."),
    "D": dict(
        target="tas", needs=("tas", "tasmin"), fixed=("Tmid",), fitted=(),
        title="D · tas vs (tasmax + tasmin) / 2",
        exact="The daily mean is the time average of T(t) over the day; it equals (T_max + T_min) / 2 exactly only "
              "for a diurnal cycle symmetric about its mean.",
        estimate="tas ≈ (tasmax + tasmin) / 2 + c_m: the offset takes the mean asymmetry of the diurnal cycle; "
                 "all three are generated fields (oracle)."),
    "U_lin": dict(
        target="pr", needs=("pr", "uas", "vas"), fixed=(), fitted=("upslope_lin",),
        title="U_lin · upslope, linear",
        exact="Kinematic boundary condition w_s = V_s · ∇h (exact for flow over terrain); saturated ascent condenses "
              "C_w w with C_w = ρ_s Γ_m / γ (Smith & Barstad's condensation coefficient), all of which falls as "
              "precipitation (no drift, no delay, no evaporation) in the upslope limit; with the positive part "
              "(ascent only) it is the classical upslope model.",
        estimate="w_s from the daily-mean 10 m wind (rotated to the grid axes) and the 12 km orography; C_w from the "
                 "coarse 850/700 hPa t, z and q; no positive part, so the law stays linear in the wind; one fitted "
                 "gain absorbs the efficiency (precipitation / condensation) and the 10 m vs. steering-level wind."),
    "U_pos": dict(
        target="pr", needs=("pr", "upslope_pos"), fixed=(), fitted=("upslope_pos",),
        title="U_pos · upslope, 6-hourly",
        exact="As U_lin with the positive part (V_s · ∇h)_+, evaluated instantaneously and averaged over the day.",
        estimate="Mean of the four 6-hourly instantaneous (V_10m · ∇h)_+ times the daily C_w; one fitted gain. "
                 "Nonlinear in the wind (positive part): usable for conditioning only linearised around the base "
                 "wind or as sampler guidance."),
    "M": dict(
        target="pr", needs=("pr", "evspsbl", "quas", "qvas", "huss", "ua850", "va850"), fixed=("E",),
        fitted=("conv_sfc", "conv_850"),
        title="M · moisture budget",
        exact="Column water-vapour budget (exact, neglecting condensate storage): P - E = -∂W/∂t - ∇·Q with "
              "W = (1/g)∫ q dp and Q = (1/g)∫ q V dp from the surface to the top, Q the daily mean of the "
              "instantaneous flux.",
        estimate="No column water or flux is archived, so the integral is replaced by two layers with effective "
                 "masses a_s, a_850 (kg/m², fitted): Q ≈ a_s <q_s V_10m> + a_850 q_850 V_850 and W ≈ a_s q_s, with "
                 "<q_s V_10m> the daily mean of 6-hourly products (huss 3-hourly), q_850 from the coarse predictor "
                 "(no fine 850 hPa humidity archived) times the daily-mean fine 850 hPa wind (0 below ground), "
                 "∂W/∂t from centred differences of daily huss; E = evspsbl (fixed, a = 1)."),
    "Qfull": dict(
        target="huss", needs=("huss",), fixed=("q_free",), fitted=(),
        title="Qfull · humidity profile at the surface",
        exact="In a well-mixed lower troposphere the specific humidity varies smoothly with height, so the "
              "near-surface humidity at a grid point follows the free-atmosphere humidity profile q(z) at the "
              "point's elevation; departures come from surface evaporation and the boundary layer.",
        estimate="q_free from the coarse q, z at 850/700/500 hPa (linear in height, extrapolated below 850 hPa "
                 "with the 850-700 gradient, ≥ 0); the boundary-layer departure is left to the offset and "
                 "tolerance. Target: the 12 km 2 m specific humidity huss (generated)."),
    "Hyd": dict(
        target="zg500", needs=("zg500", "ps", "tas", "huss"), fixed=("z_hyd",), fitted=(),
        title="Hyd · hydrostatic, surface to 500 hPa",
        exact="Hydrostatic equation (primitive equations): ∂Φ/∂p = -R T_v / p; integrated from the surface: "
              "z_500 = z_s + (R_d/g) ∫ T_v d ln p from 500 hPa to p_s (hypsometric equation), exact.",
        estimate="z_s = orography; T_v(ln p) piecewise linear through the surface value (tas, huss, generated) "
                 "and the coarse t, q at the levels above the surface (850, 700, 500 hPa); p_s generated. Target: "
                 "the 12 km 500 hPa height zg500 (generated)."),
    "C_u850": dict(
        target="cellmean:ua850", needs=("ua850",), fixed=("coarse:u_850",), fitted=(),
        title="C · 850 hPa eastward wind",
        exact="The perfect predictors are the RCM's own fields regridded conservatively to 2°: the area mean of "
              "the 12 km field over each 2° cell equals the predictor (A x = y).",
        estimate="Area mean over the 12 km points of each 2° cell lying entirely inside the window (53 cells), "
                 "ignoring points where 850 hPa is below ground, against the predictor of that cell; shown "
                 "piecewise constant on the 12 km grid (NaN outside those cells)."),
    "C_v850": dict(
        target="cellmean:va850", needs=("va850",), fixed=("coarse:v_850",), fitted=(),
        title="C · 850 hPa northward wind", exact="As C_u850.", estimate="As C_u850."),
    "C_z500": dict(
        target="cellmean:zg500", needs=("zg500",), fixed=("coarse:z_500",), fitted=(),
        title="C · 500 hPa height", exact="As C_u850.", estimate="As C_u850."),
    # variants: is it a question of scale (the law at ~40 km instead of the 12 km grid point), or of the
    # evaporation coefficient fixed at its exact value 1?
    "U_pos_40km": dict(
        target="pr", needs=("pr", "upslope_pos"), fixed=(), fitted=("upslope_pos",), scale=3.0,
        title="U_pos at ~40 km",
        exact="As U_pos.",
        estimate="As U_pos, with target and term smoothed by a Gaussian of σ = 3 grid points (37 km): the law at "
                 "the scale of a few grid cells instead of the grid point."),
    "M_40km": dict(
        target="pr", needs=("pr", "evspsbl", "quas", "qvas", "huss", "ua850", "va850"), fixed=("E",),
        fitted=("conv_sfc", "conv_850"), scale=3.0,
        title="M at ~40 km",
        exact="As M.",
        estimate="As M, with target and terms smoothed by a Gaussian of σ = 3 grid points (37 km)."),
    "M_Efit": dict(
        target="pr", needs=("pr", "evspsbl", "quas", "qvas", "huss", "ua850", "va850"), fixed=(),
        fitted=("E", "conv_sfc", "conv_850"),
        title="M, evaporation fitted",
        exact="As M.",
        estimate="As M, but the evaporation coefficient is fitted instead of fixed at its exact value 1."),
    "CC": dict(
        target="lnpr_wet", physics_alone=False, needs=("pr",), fixed=("cc",), fitted=(),
        title="CC · Clausius–Clapeyron",
        exact="Clausius-Clapeyron: d ln e_s / dT = L / (R_v T²) ≈ 6.5-7 %/K (exact thermodynamics). Precipitation "
              "intensity follows it only if the condensing vertical motion is unchanged (thermodynamic scaling).",
        estimate="On wet days (pr ≥ 1 mm/day): ln pr ≈ c_m + β_m T_850, with T_850 the coarse 850 hPa temperature "
                 "at the grid point and β_m = L / (R_v T̄²) from its training-years monthly mean T̄: day-to-day "
                 "intensity, and in a warmer climate its shift, follow CC at the condensation level."),
    "CC0": dict(
        target="lnpr_wet", physics_alone=False, needs=("pr",), fixed=(), fitted=(),
        title="CC0 · intensity climatology",
        exact="Reference for CC: the training-years monthly climatology of ln pr on wet days.",
        estimate="Its residual in another period is the actual change of wet-day intensity, CC's residual the "
                 "departure from Clausius-Clapeyron scaling."),
}


def _levels(x, predictors, op, var):
    return {lev: op.interp(x[:, list(predictors).index(f"{var}_{lev}")].astype(np.float64)) for lev in (850, 700, 500)}


def _free_humidity(x, predictors, op, orog):
    """Coarse q(z) profile at the grid point's elevation (as free_temperature, for specific humidity)."""
    q, z = _levels(x, predictors, op, "q"), _levels(x, predictors, op, "z")
    zf = orog[None].astype(np.float64)
    low = q[850] + (q[700] - q[850]) * (zf - z[850]) / (z[700] - z[850])
    high = q[700] + (q[500] - q[700]) * (zf - z[700]) / (z[500] - z[700])
    return np.maximum(np.where(zf <= z[700], low, high), 0.0).astype(np.float32)


def _hypsometric_z500(x, predictors, op, orog, ps, tas, huss):
    """z_500 = z_s + (R_d/g) ∫_{500 hPa}^{p_s} T_v d ln p, T_v piecewise linear in ln p through the surface value
    and the coarse levels above the surface."""
    t, q = _levels(x, predictors, op, "t"), _levels(x, predictors, op, "q")
    tv = {lev: t[lev] * (1 + 0.608 * q[lev]) for lev in t}
    ps = ps.astype(np.float64)
    tvs = tas.astype(np.float64) * (1 + 0.608 * huss.astype(np.float64))
    above850, above700 = ps > 85000.0, ps > 70000.0
    seg = lambda a, b, pa, pb: 0.5 * (a + b) * np.log(pa / pb)  # noqa: E731
    integral = np.where(above850, seg(tvs, tv[850], ps, 85000.0) + seg(tv[850], tv[700], 85000.0, 70000.0),
                        np.where(above700, seg(tvs, tv[700], ps, 70000.0), 0.0))
    integral = integral + np.where(above700, seg(tv[700], tv[500], 70000.0, 50000.0), seg(tvs, tv[500], ps, 50000.0))
    return (orog[None] + RD / G0 * integral).astype(np.float32)


class _Ctx:
    """Per-split inputs and derived terms, computed once per split."""

    def __init__(self, exp: Experiment, split: Split, grid: Grid, op: CoarseOperator, needs: set):
        self.exp, self.s, self.grid, self.op = exp, split, grid, op
        self.geo = geometry(grid)
        aux_fields = sorted(needs - {"pr", "tasmax"})
        self.aux = load_extra(exp, split.time, aux_fields) if aux_fields else {}
        self._cache = {}
        self.noise: dict[str, float] = {}  # field -> white-noise std added to it (noise-shaping test)
        self.override: dict[str, np.ndarray] = {}  # field -> replacement (e.g. a model's white base)

    def variant(self, noise=None, override=None) -> _Ctx:
        """Same split and inputs, with white noise added to generated fields and/or fields replaced."""
        c = object.__new__(_Ctx)
        c.__dict__.update(exp=self.exp, s=self.s, grid=self.grid, op=self.op, geo=self.geo, aux=self.aux,
                          _cache={}, noise=dict(noise or {}), override=dict(override or {}))
        c._noisy = {}
        return c

    def truth(self, name):
        if name in ("pr", "tasmax"):
            return self.s.y[:, list(self.s.variables).index(name)].astype(np.float32)
        return self.aux[name]

    def field(self, name):
        if name in self.override:
            return self.override[name]
        if name in self.noise:
            if name not in self.__dict__.setdefault("_noisy", {}):
                rng = np.random.default_rng(zlib.crc32(name.encode()))
                a = self.truth(name)
                self._noisy[name] = (a + np.float32(self.noise[name]) * rng.standard_normal(a.shape, np.float32))
            return self._noisy[name]
        return self.truth(name)

    def cells(self):
        """(index of the full 2° cell of each target point or -1, predictor-grid (i, j) per full cell)."""
        if "cells" not in self._cache:
            full = _full_cells(self.grid, self.op)
            idx = np.full(self.grid.shape[0] * self.grid.shape[1], -1)
            for k, (_, _, pts) in enumerate(full):
                idx[pts] = k
            self._cache["cells"] = (idx, [(i, j) for i, j, _ in full])
        return self._cache["cells"]

    def cell_mean(self, a):
        """Area mean of a (t, H, W) over each full 2° cell (NaN-aware), broadcast back to its points."""
        idx, ij = self.cells()
        flat = a.reshape(len(a), -1).astype(np.float64)
        ok = (idx >= 0)[None] & np.isfinite(flat)
        num = np.zeros((len(a), len(ij)))
        den = np.zeros((len(a), len(ij)))
        for k in range(len(ij)):
            sel = idx == k
            num[:, k] = np.where(ok[:, sel], flat[:, sel], 0.0).sum(1)
            den[:, k] = ok[:, sel].sum(1)
        mean = num / np.maximum(den, 1)
        out = np.full(flat.shape, np.nan)
        out[:, idx >= 0] = mean[:, idx[idx >= 0]]
        return out.reshape(a.shape)

    def coarse_cell(self, pred, sl):
        """The predictor's value in each point's full 2° cell (NaN outside full cells)."""
        idx, ij = self.cells()
        x = self.s.x[sl, list(self.s.predictors).index(pred)]
        vals = np.stack([x[:, i, j] for i, j in ij], 1).astype(np.float64)
        out = np.full((len(x), idx.size), np.nan)
        out[:, idx >= 0] = vals[:, idx[idx >= 0]]
        return out.reshape(len(x), *self.grid.shape)

    def cw(self):  # condensation coefficient C_w on the target grid (kg m-3), per day
        if "cw" not in self._cache:
            self._cache["cw"] = self.op.interp(_moist_coefficients(self.s.x, self.s.predictors)["cw"]).astype(np.float32)
        return self._cache["cw"]

    def coarse(self, name):
        if name not in self._cache:
            self._cache[name] = self.op.interp(self.s.x[:, list(self.s.predictors).index(name)].astype(np.float64)
                                               ).astype(np.float32)
        return self._cache[name]

    def term(self, name, sl, beta=None) -> np.ndarray:
        g = self.grid
        if name == "T_free":
            return free_temperature(self.s.x[sl], self.s.predictors, self.op, g.orog)
        if name == "Tmid":
            return 0.5 * (self.field("tasmax")[sl] + self.field("tasmin")[sl])
        if name == "upslope_lin":
            ux, vy = to_grid_axes(self.field("uas")[sl], self.field("vas")[sl], self.geo)
            hx, hy = gradient(g.orog.astype(np.float64), self.geo)
            return (self.cw()[sl] * (ux * hx + vy * hy) * 86400.0).astype(np.float32)
        if name == "upslope_pos":
            return (self.cw()[sl] * self.field("upslope_pos")[sl] * 86400.0).astype(np.float32)
        if name == "E":
            return self.field("evspsbl")[sl]
        if name == "conv_sfc":  # -div <q_s V_10m> - dq_s/dt, per day (kg/kg/day)
            fx, fy = to_grid_axes(self.field("quas")[sl], self.field("qvas")[sl], self.geo)
            if "dqdt" not in self._cache:
                tdays = self.s.time.astype("datetime64[D]").astype(np.float64)
                self._cache["dqdt"] = np.gradient(self.field("huss"), tdays, axis=0).astype(np.float32)  # per day
            return (-divergence(fx, fy, self.geo) * 86400.0 - self._cache["dqdt"][sl]).astype(np.float32)
        if name == "conv_850":
            u = np.nan_to_num(self.field("ua850")[sl])  # below ground: no 850 hPa layer
            v = np.nan_to_num(self.field("va850")[sl])
            ux, vy = to_grid_axes(u, v, self.geo)
            q = self.coarse("q_850")[sl]
            return (-divergence(q * ux, q * vy, self.geo) * 86400.0).astype(np.float32)
        if name == "cc":
            return beta[months(self.s.time[sl])] * self.coarse("t_850")[sl]
        if name == "q_free":
            return _free_humidity(self.s.x[sl], self.s.predictors, self.op, g.orog)
        if name == "z_hyd":
            return _hypsometric_z500(self.s.x[sl], self.s.predictors, self.op, g.orog, self.field("ps")[sl],
                                     self.field("tas")[sl], self.field("huss")[sl])
        if name.startswith("coarse:"):
            return self.coarse_cell(name.split(":", 1)[1], sl)
        raise KeyError(name)

    def target(self, kind, sl):
        if kind.startswith("cellmean:"):
            return self.cell_mean(self.field(kind.split(":", 1)[1])[sl])
        if kind == "lnpr_wet":
            pr = self.field("pr")[sl]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(pr >= WET, np.log(np.maximum(pr, WET)), np.nan).astype(np.float64)
        return self.field(kind)[sl].astype(np.float64)


def _monthly_mean(arr_iter, H, W):
    """NaN-aware monthly mean per grid point from an iterator of (values (t, H, W), months (t,))."""
    acc, cnt = np.zeros((12, H, W)), np.zeros((12, H, W))
    for a, m in arr_iter:
        v = np.isfinite(a)
        np.add.at(acc, m, np.where(v, a, 0.0))
        np.add.at(cnt, m, v)
    with np.errstate(invalid="ignore"):
        return acc / cnt


def _smooth(a: np.ndarray, sigma: float | None) -> np.ndarray:
    """Gaussian smoothing over the two spatial axes (the law at a coarser scale); NaNs stay NaN."""
    if not sigma:
        return a
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, sigma=(0, sigma, sigma), mode="nearest")


def fit_candidate(name: str, ctx: _Ctx) -> dict:
    """Coefficients, offset and tolerance of a candidate law on the training split (ctx)."""
    spec = LAWS[name]
    H, W = ctx.grid.shape
    mon = months(ctx.s.time)
    chunks = [slice(i, i + CHUNK) for i in range(0, len(ctx.s), CHUNK)]
    beta = None
    if "cc" in spec["fixed"]:
        tbar = _monthly_mean(((ctx.coarse("t_850")[sl].astype(np.float64), mon[sl]) for sl in chunks), H, W)
        beta = (LV / (RV * tbar**2)).astype(np.float32)

    sc = spec.get("scale")

    def parts(sl):
        y = _smooth(ctx.target(spec["target"], sl), sc)
        fixed = sum((_smooth(ctx.term(t, sl, beta).astype(np.float64), sc) for t in spec["fixed"]), np.zeros_like(y))
        G = [_smooth(ctx.term(t, sl, beta).astype(np.float64), sc) for t in spec["fitted"]]
        valid = np.isfinite(y) & np.isfinite(fixed)
        for g in G:
            valid &= np.isfinite(g)
        return y, fixed, G, valid

    # climatologies of target, fixed sum and fitted terms (masked like the target)
    sums = {k: np.zeros((12, H, W)) for k in ["y", "f"] + [f"g{k}" for k in range(len(spec["fitted"]))]}
    cnt = np.zeros((12, H, W))
    for sl in chunks:
        y, f, G, v = parts(sl)
        m = mon[sl]
        np.add.at(cnt, m, v)
        np.add.at(sums["y"], m, np.where(v, y, 0.0))
        np.add.at(sums["f"], m, np.where(v, f, 0.0))
        for k, g in enumerate(G):
            np.add.at(sums[f"g{k}"], m, np.where(v, g, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = {k: a / cnt for k, a in sums.items()}
    # least-squares coefficients of the fitted terms on day-to-day anomalies
    K = len(spec["fitted"])
    coef = np.zeros(K)
    if K:
        ata, atb = np.zeros((K, K)), np.zeros(K)
        for sl in chunks:
            y, f, G, v = parts(sl)
            m = mon[sl]
            ya = np.where(v, (y - clim["y"][m]) - (f - clim["f"][m]), 0.0)
            ga = [np.where(v, g - clim[f"g{k}"][m], 0.0) for k, g in enumerate(G)]
            for i in range(K):
                atb[i] += (ga[i] * ya).sum()
                for j in range(K):
                    ata[i, j] += (ga[i] * ga[j]).sum()
        coef = np.linalg.solve(ata, atb)
    offset = clim["y"] - clim["f"] - sum((coef[k] * clim[f"g{k}"] for k in range(K)), np.zeros((12, H, W)))
    # tolerance: RMS of the residual about its monthly offset
    ss = np.zeros((12, H, W))
    for sl in chunks:
        y, f, G, v = parts(sl)
        m = mon[sl]
        r = y - f - sum((coef[k] * g for k, g in enumerate(G)), np.zeros_like(y)) - offset[m]
        np.add.at(ss, m, np.where(v, r**2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        tol = np.sqrt(ss / cnt)
    floor = 1e-3 * np.nanmean(tol)
    tol = np.where(np.isfinite(tol), np.maximum(tol, floor), np.nan)
    return {"coef": dict(zip(spec["fitted"], coef.tolist())), "offset": offset, "tol": tol, "clim": clim["y"],
            "beta": beta}


def _rhs(name: str, fit: dict, ctx: _Ctx, sl) -> tuple[np.ndarray, np.ndarray]:
    spec = LAWS[name]
    sc = spec.get("scale")
    y = _smooth(ctx.target(spec["target"], sl), sc)
    b = np.zeros_like(y)
    for t in spec["fixed"]:
        b += _smooth(ctx.term(t, sl, fit["beta"]).astype(np.float64), sc)
    for t, a in fit["coef"].items():
        b += a * _smooth(ctx.term(t, sl, fit["beta"]).astype(np.float64), sc)
    return y, b + fit["offset"][months(ctx.s.time[sl])]


# ---- noise shaping: how strongly does a white base violate the law?
WHITE_SPLITS = ("val", "historical_perfect", "end_century_perfect")
GENERATED = {"pr", "tasmax", "tas", "tasmin", "huss", "ps", "zg500", "ua850", "va850", "uas", "vas", "quas", "qvas",
             "upslope_pos", "evspsbl"}


def noise_levels(ctx: _Ctx, fields, step: int = 5) -> dict[str, float]:
    """Conservative prior-noise proxy per generated field: the RMS of the sub-2° part of its daily anomaly about
    the training-years monthly climatology, H (f - clim_m), i.e. what neither the predictors (large scales) nor the
    climatology (static pattern) supply and a generator must add. A lower bound: the trained AFM's noise sigma_max is
    larger (tasmax 2.38 K against about 1 K here), which the measured white base (afm-base) shows where it exists."""
    H, W = ctx.grid.shape
    mon = months(ctx.s.time)
    out = {}
    for f in fields:
        a = ctx.truth(f).astype(np.float64)
        clim = _monthly_mean(((a, mon),), H, W)
        d = (a - clim[mon])[::step]
        ok = np.isfinite(d)
        out[f] = float(np.sqrt(np.mean(ctx.op.highpass(np.where(ok, d, 0.0))[ok] ** 2)))
    return out


def _afm_base(exp: Experiment, split: str, variables=("pr", "tasmax")) -> dict[str, np.ndarray]:
    """Member 0 of the trained AFM's white base (predictions of `dslab predict --base-only`), if available."""
    path = exp.predictions_dir / "afm-base" / f"{split}.nc"
    if not path.exists():
        return {}
    with xr.open_dataset(path) as ds:
        return {v: ds[v].isel(member=0).transpose("time", ...).values.astype(np.float32) for v in variables}


def _uses_model_fields(name: str) -> bool:
    return LAWS[name]["target"] in ("pr", "lnpr_wet", "tasmax")


def compute(exp: Experiment, names=None, splits=None) -> tuple[xr.Dataset, dict]:
    grid = load_grid(exp)
    op = CoarseOperator.from_grid(grid)
    names = list(names or LAWS)
    needs = set().union(*(LAWS[n]["needs"] for n in names)) | {"tasmax"}
    splits = [s for s in (splits or SPLITS) if s in exp.splits]
    ctx = _Ctx(exp, load_split(exp, "train"), grid, op, needs)
    fits = {}
    for n in names:
        fits[n] = fit_candidate(n, ctx)
        print(f"[candidates] fitted {n}: coef {fits[n]['coef']}, τ RMS {np.sqrt(np.nanmean(fits[n]['tol'] ** 2)):.3f}",
              flush=True)
    s_noise = noise_levels(ctx, sorted(needs & GENERATED))
    print(f"[candidates] prior-noise proxies (RMS of the sub-2° part): {s_noise}", flush=True)

    def stats(n, c, mon, jja):
        st, f = SplitStats(grid.shape), fits[n]
        for i in range(0, len(c.s), CHUNK):
            sl = slice(i, i + CHUNK)
            y, b = _rhs(n, f, c, sl)
            st.add(y, b, f["tol"], f["offset"], f["clim"], mon[sl], jja[sl])
        return st.result()

    results, n_days = {}, np.zeros(len(splits), int)
    white = np.full((len(names), len(splits)), np.nan)
    afmbase = np.full((len(names), len(splits)), np.nan)
    for j, split in enumerate(splits):
        s = load_split(exp, split)
        if s.y is None:
            continue
        ctx = ctx if split == "train" else _Ctx(exp, s, grid, op, needs)
        mon = months(s.time)
        jja = np.isin(mon, [5, 6, 7])
        n_days[j] = len(s)
        for n in names:
            results[(n, split)] = stats(n, ctx, mon, jja)
        if split in WHITE_SPLITS:  # noise shaping: truth + white noise (a perfect encoder), and AFM's white base
            noisy = ctx.variant(noise=s_noise)
            base = _afm_base(exp, split)
            model = ctx.variant(override=base) if base else None
            for i_n, n in enumerate(names):
                white[i_n, j] = stats(n, noisy, mon, jja)["ms_z"]
                if model is not None and _uses_model_fields(n):
                    afmbase[i_n, j] = stats(n, model, mon, jja)["ms_z"]
            del noisy, model
        print(f"[candidates] {split}: " + ", ".join(
            f"{n} {results[(n, split)]['ms_z']:.3f}" + (f" (white {white[i, j]:.2f})" if np.isfinite(white[i, j]) else "")
            for i, n in enumerate(names)), flush=True)
    variables = {n: LAWS[n]["target"] for n in names}
    ds = assemble(names, splits, results, grid, variables, n_days, {n: fits[n]["tol"] for n in names},
                  {n: fits[n]["offset"] for n in names})
    ds["ms_z_white"] = (("law", "split"), white)
    ds["ms_z_afmbase"] = (("law", "split"), afmbase)
    ds.attrs["noise_levels"] = json.dumps(s_noise)
    info = {n: {"title": LAWS[n]["title"], "target": LAWS[n]["target"], "exact": LAWS[n]["exact"],
                "estimate": LAWS[n]["estimate"], "fixed": list(LAWS[n]["fixed"]), "coef": fits[n]["coef"],
                "physics_alone": LAWS[n].get("physics_alone", True),
                "rms_tolerance": float(np.sqrt(np.nanmean(fits[n]["tol"] ** 2)))} for n in names}
    ds.attrs.update(experiment=exp.name, law_info=json.dumps(info))
    ds.attrs["fits"] = "candidate_fits.npz"
    np.savez(exp.eval_dir / "laws" / "candidate_fits_new.npz",
             **{f"{n}__{k}": fits[n][k] for n in names for k in ("offset", "tol", "clim") if fits[n][k] is not None},
             **{f"{n}__beta": fits[n]["beta"] for n in names if fits[n]["beta"] is not None})
    return ds, info


# ---- coarse consistency (law C) and geostrophy (law G): tables, not fine-grid fields
def _full_cells(grid: Grid, op: CoarseOperator, min_cover: float = 0.98):
    """2° cells entirely inside the target window: (cell index on the predictor grid, point mask per cell)."""
    geo = geometry(grid)
    area = (geo["dx"] * geo["dy"]).ravel()
    plat, plon = grid.pred_lat, grid.pred_lon
    dlat, dlon = np.radians(plat[1] - plat[0]), np.radians(plon[1] - plon[0])
    cell = (np.rint((grid.lat.ravel() - plat[0]) / (plat[1] - plat[0])).astype(int) * len(plon)
            + np.rint((grid.lon.ravel() - plon[0]) / (plon[1] - plon[0])).astype(int))
    out = []
    for c in np.unique(cell):
        i, j = divmod(c, len(plon))
        phi = np.radians(plat[i])
        cell_area = R_EARTH**2 * dlon * (np.sin(phi + dlat / 2) - np.sin(phi - dlat / 2))
        pts = cell == c
        if area[pts].sum() / cell_area >= min_cover:
            out.append((i, j, pts))
    return out


C_PAIRS = {"ua850": "u_850", "va850": "v_850", "zg500": "z_500"}


def coarse_consistency(exp: Experiment, splits=None) -> pd.DataFrame:
    """Law C: area mean of the fine RCM field over each 2° cell inside the window vs the 2° predictor."""
    grid = load_grid(exp)
    op = CoarseOperator.from_grid(grid)
    cells = _full_cells(grid, op)
    rows = []
    for split in [s for s in (splits or SPLITS) if s in exp.splits]:
        s = load_split(exp, split)
        aux = load_extra(exp, s.time, list(C_PAIRS))
        mon = months(s.time)
        for fine, coarse in C_PAIRS.items():
            f = aux[fine].reshape(len(s), -1)
            y = s.x[:, list(s.predictors).index(coarse)]
            ax = np.stack([np.nanmean(f[:, pts], axis=1) for _, _, pts in cells], 1)
            ay = np.stack([y[:, i, j] for i, j, _ in cells], 1)
            nanfrac = np.array([np.isnan(f[:, pts]).mean() for _, _, pts in cells])
            clim = np.stack([ay[mon == k].mean(0) for k in range(12)])
            err, anom = ax - ay, ay - clim[mon]
            for subset, sel in (("all cells", np.ones(len(cells), bool)), ("cells without missing 850 hPa", nanfrac == 0)):
                if not sel.any():
                    continue
                e, a = err[:, sel], anom[:, sel]
                rows.append({"law": "C", "field": f"{fine} vs {coarse}", "split": split, "cells": subset,
                             "n_cells": int(sel.sum()), "bias": float(np.nanmean(e)),
                             "RMSE": float(np.sqrt(np.nanmean(e**2))),
                             "relative RMSE": float(np.sqrt(np.nanmean(e**2) / np.nanmean(a**2))),
                             "corr": float(np.corrcoef(ax[:, sel].ravel(), ay[:, sel].ravel())[0, 1])})
        print(f"[candidates] C {split} done", flush=True)
    return pd.DataFrame(rows)


COUNTERPARTS = ("pr", "tasmax", "tas", "tasmin", "huss", "ua850", "va850", "zg500")


def coarse_counterparts(exp: Experiment, splits=("train", "historical_perfect", "mid_century_perfect",
                                                  "end_century_perfect")) -> pd.DataFrame:
    """Coarse consistency for fields without a benchmark predictor. Perfect case: the counterpart is the RCM's
    own 2° area mean, so A x = y holds exactly by construction, and the question is how much of the 12 km field it
    pins: the share of the daily variance (anomalies about the split's own monthly climatology) held by the 2° cell
    means, and the share of the climate-change signal (the split's monthly climatology minus the training years') held
    by them. Imperfect case: the counterpart is the driving GCM's field on the 2° grid (pr, tas: `dslab extract
    --gcm`), compared with the RCM's 2° area mean."""
    from .extra import load_gcm

    grid = load_grid(exp)
    op = CoarseOperator.from_grid(grid)
    rows = []
    train_clim = {}

    def share(ctx, d, inside):  # 1 - |d - cell mean of d|^2 / |d|^2 over the points of full cells
        r = (d - ctx.cell_mean(d)).reshape(len(d), -1)[:, inside]
        d = d.reshape(len(d), -1)[:, inside]
        ok = np.isfinite(r) & np.isfinite(d)
        return float(1 - np.mean(r[ok] ** 2) / np.mean(d[ok] ** 2))

    for split in splits:
        s = load_split(exp, split)
        ctx = _Ctx(exp, s, grid, op, set(COUNTERPARTS))
        mon = months(s.time)
        idx, ij = ctx.cells()
        inside = idx >= 0
        for f in COUNTERPARTS:
            a = ctx.truth(f).astype(np.float64)
            clim = _monthly_mean(((a, mon),), *grid.shape)
            if f not in train_clim:  # monthly climatology of the training years (first split)
                train_clim[f] = clim
            row = {"field": f, "split": split, "counterpart": "RCM 2° mean (perfect)",
                   "share of daily variance at 2°": share(ctx, a - clim[mon], inside)}
            if clim is not train_clim[f]:
                row["share of the change at 2°"] = share(ctx, clim - train_clim[f], inside)
                row["mean change"] = float(np.nanmean((clim - train_clim[f]).reshape(12, -1)[:, inside]))
            rows.append(row)
            if f in ("pr", "tas"):
                g = load_gcm(exp, s.time, f)
                gc = np.stack([g[:, i, j] for i, j in ij], 1).astype(np.float64)
                rc = np.stack([np.nanmean(a.reshape(len(a), -1)[:, idx == k], axis=1) for k in range(len(ij))], 1)
                ca = np.stack([rc[mon == k].mean(0) for k in range(12)])
                e = gc - rc
                rows.append({"field": f, "split": split, "counterpart": "GCM on the 2° grid (imperfect)",
                             "bias": float(e.mean()), "RMSE": float(np.sqrt((e**2).mean())),
                             "relative RMSE": float(np.sqrt((e**2).mean() / ((rc - ca[mon]) ** 2).mean())),
                             "corr": float(np.corrcoef(gc.ravel(), rc.ravel())[0, 1]),
                             "corr of daily anomalies": float(np.corrcoef((gc - np.stack([gc[mon == k].mean(0) for k in range(12)])[mon]).ravel(),
                                                                          (rc - ca[mon]).ravel())[0, 1])})
        print(f"[candidates] counterparts {split} done", flush=True)
    return pd.DataFrame(rows)


def geostrophy(exp: Experiment, splits=None) -> pd.DataFrame:
    """Law G at 2° (the only scale with wind and height on the same level): V vs the geostrophic wind of z."""
    grid = load_grid(exp)
    lat = np.radians(grid.pred_lat)  # predictors are stored with latitude ascending (dslab.data)
    dlat, dlon = lat[1] - lat[0], np.radians(grid.pred_lon[1] - grid.pred_lon[0])
    f = 2 * OMEGA * np.sin(lat)[:, None]
    rows = []
    for split in [s for s in (splits or SPLITS) if s in exp.splits]:
        s = load_split(exp, split)
        for lev in (850, 700, 500):
            z = s.x[:, list(s.predictors).index(f"z_{lev}")].astype(np.float64)
            u = s.x[:, list(s.predictors).index(f"u_{lev}")].astype(np.float64)
            v = s.x[:, list(s.predictors).index(f"v_{lev}")].astype(np.float64)
            dzdy = np.gradient(z, axis=1) / (R_EARTH * dlat)
            dzdx = np.gradient(z, axis=2) / (R_EARTH * np.cos(lat)[:, None] * dlon)
            ug, vg = -G0 / f * dzdy, G0 / f * dzdx
            inner = (slice(None), slice(1, -1), slice(1, -1))
            du, dv = (u - ug)[inner], (v - vg)[inner]
            speed2 = (u[inner] ** 2 + v[inner] ** 2).mean()
            rows.append({"law": "G", "level": f"{lev} hPa", "split": split,
                         "ageostrophic RMS (m/s)": float(np.sqrt((du**2 + dv**2).mean())),
                         "wind RMS (m/s)": float(np.sqrt(speed2)),
                         "ageostrophic fraction": float(np.sqrt((du**2 + dv**2).mean() / speed2)),
                         "corr u": float(np.corrcoef(u[inner].ravel(), ug[inner].ravel())[0, 1]),
                         "corr v": float(np.corrcoef(v[inner].ravel(), vg[inner].ravel())[0, 1])})
    return pd.DataFrame(rows)


def run(exp: Experiment, names=None, splits=None) -> xr.Dataset:
    out = exp.eval_dir / "laws"
    out.mkdir(parents=True, exist_ok=True)
    ds, info = compute(exp, names, splits)
    path = out / "candidates.nc"
    fits_new, fits_path = out / "candidate_fits_new.npz", out / "candidate_fits.npz"
    if names and path.exists():  # a subset: replace those laws in the existing file, keep the others
        with xr.open_dataset(path) as old:
            old = old.load()
        keep = [n for n in old.law.values if n not in ds.law.values]
        if keep:
            merged_info = {**json.loads(old.attrs["law_info"]), **info}
            attrs = dict(ds.attrs)
            ds = xr.concat([old.sel(law=keep), ds], dim="law", data_vars="minimal", coords="minimal",
                           compat="override", join="override")
            ds.attrs.update(attrs, law_info=json.dumps({n: merged_info[n] for n in ds.law.values}))
            if fits_path.exists():  # keep the other laws' fits as well
                with np.load(fits_path) as f_old, np.load(fits_new) as f_new:
                    merged = {k: f_old[k] for k in f_old.files if k.split("__")[0] in keep}
                    merged.update({k: f_new[k] for k in f_new.files})
                np.savez(fits_new, **merged)
    ds.to_netcdf(path)
    fits_new.replace(fits_path)
    tables = pd.concat([coarse_consistency(exp, splits), geostrophy(exp, splits)], ignore_index=True)
    tables.to_csv(out / "consistency.csv", index=False)
    counterparts = coarse_counterparts(exp)
    counterparts.to_csv(out / "counterparts.csv", index=False)
    print(counterparts.round(3).to_string())
    from .law_diagnostics import summary

    print(summary(ds).round(3).to_string())
    print(tables.round(3).to_string())
    print(f"[candidates] written {out / 'candidates.nc'}, {out / 'consistency.csv'}")
    return ds
