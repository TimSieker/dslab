"""Plot style and figures for the evaluation notebook.

Colour rules: every run keeps one fixed categorical colour (pin them with RUN_COLORS); the reference
RCM ("truth") is drawn in primary ink, the climatology floor and published models in muted grey.
Magnitudes use one-hue sequential maps (pr blue, tasmax orange), biases a blue-grey-red diverging map
with the sign chosen so that red means "too warm" / "too dry".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm

from .experiment import WORK

# Chart chrome (light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Validated categorical palette (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRUTH_COLOR = INK
FLOOR_COLOR = MUTED  # climatology baseline
PUBLISHED_COLOR = "#c3c2b7"

# Canonical run -> slot, so colours never depend on which runs happen to be loaded.
CANONICAL_RUNS = ["linear", "deepesd", "unet", "flow", "flow_residual"]
RUN_COLORS: dict[str, str] = {}  # explicit overrides, e.g. RUN_COLORS["my_model"] = SERIES[5]

BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fcfcfb", "#fde3d6", "#f7b796", "#f08a5d", "#eb6834", "#c4501f", "#933a14", "#5f250c"]
GREY_RAMP = ["#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e"]
SEQ = {
    "pr": LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP),
    "tasmax": LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP),
    "orog": LinearSegmentedColormap.from_list("seq_grey", GREY_RAMP),
}
_DIV = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f6c9c4", "#e98783", "#e34948", "#8c1d1d"]
DIV_COLD_WARM = LinearSegmentedColormap.from_list("div_blue_red", _DIV)  # negative blue, positive red
DIV = {"tasmax": DIV_COLD_WARM, "pr": DIV_COLD_WARM.reversed()}  # pr: positive (wetter) blue

UNITS = {"pr": "mm/day", "tasmax": "K"}
METRIC_UNITS = {("pr", "RMSE"): "mm/day", ("pr", "Mean Bias"): "mm/day", ("pr", "SDII"): "mm/day",
                ("pr", "Rx1day"): "mm/day", ("pr", "RALSD"): "dB²", ("pr", "LHD"): "dB",
                ("tasmax", "RMSE"): "K", ("tasmax", "Mean Bias"): "K", ("tasmax", "TXx"): "K",
                ("tasmax", "RALSD"): "dB²", ("tasmax", "IAV"): "K"}
LABELS = {"pr": "Precipitation", "tasmax": "Daily max. temperature"}
DX_KM = 12.5  # target grid spacing


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
    try:  # keep cartopy's Natural Earth downloads out of ~/.local
        import cartopy
        cartopy.config["data_dir"] = str(WORK / "cache" / "cartopy")
    except ImportError:
        pass


def run_colors(runs) -> dict[str, str]:
    colors, free = {}, iter(SERIES[len(CANONICAL_RUNS):] + SERIES)
    for run in runs:
        if run in RUN_COLORS:
            colors[run] = RUN_COLORS[run]
        elif run == "climatology":
            colors[run] = FLOOR_COLOR
        elif run in CANONICAL_RUNS:
            colors[run] = SERIES[CANONICAL_RUNS.index(run)]
    for run in sorted(r for r in runs if r not in colors):
        colors[run] = next(free)
    return colors


def _savefig(fig, save: str | Path | None):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save)


# ---------------------------------------------------------------- maps

class MapGrid:
    """Where to draw: the target grid in its own projection (rectangular panels) if the experiment
    declares grid_crs, otherwise lat/lon on PlateCarree."""

    def __init__(self, ds, grid_crs: dict | None = None):
        import cartopy.crs as ccrs

        self.lon, self.lat = ds["lon"].values, ds["lat"].values
        if grid_crs and grid_crs.get("proj") == "lcc":
            scale = 1000.0 if grid_crs.get("coord_units", "m") == "km" else 1.0
            ydim, xdim = ds["lat"].dims
            self.x, self.y = ds[xdim].values * scale, ds[ydim].values * scale
            r = grid_crs["R"]
            self.proj = ccrs.LambertConformal(
                central_longitude=grid_crs["lon_0"], central_latitude=grid_crs["lat_0"],
                standard_parallels=(grid_crs["lat_1"], grid_crs["lat_2"]),
                false_easting=grid_crs["x_0"], false_northing=grid_crs["y_0"],
                globe=ccrs.Globe(ellipse=None, semimajor_axis=r, semiminor_axis=r))
            self.data_crs = self.proj
            dx, dy = abs(self.x[1] - self.x[0]) / 2, abs(self.y[1] - self.y[0]) / 2
            self.extent = [self.x.min() - dx, self.x.max() + dx, self.y.min() - dy, self.y.max() + dy]
        else:
            self.proj = self.data_crs = ccrs.PlateCarree()
            self.x, self.y = self.lon, self.lat
            self.extent = [float(self.lon.min()), float(self.lon.max()), float(self.lat.min()), float(self.lat.max())]
        x0, x1, y0, y1 = self.extent
        self.aspect = (x1 - x0) / (y1 - y0)

    @classmethod
    def from_experiment(cls, exp, ds):
        return cls(ds, exp.grid_crs)


def map_axes(mg: MapGrid, nrows: int, ncols: int, panel: float = 2.4, cbar_room: float = 0.9):
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * mg.aspect * ncols + cbar_room, panel * nrows + 0.5),
                             subplot_kw={"projection": mg.proj}, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.04, wspace=0.02, hspace=0.04)
    return fig, axes


def draw_map(ax, mg: MapGrid, field, cmap, norm, title: str | None = None):
    import cartopy.feature as cfeature

    m = ax.pcolormesh(mg.x, mg.y, field, transform=mg.data_crs, cmap=cmap, norm=norm,
                      shading="nearest", rasterized=True)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor=INK2)
    ax.coastlines("50m", linewidth=0.5, color=INK2)
    ax.set_extent(mg.extent, crs=mg.proj)
    ax.spines["geo"].set_edgecolor(AXIS)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
    return m


def robust_abs(arrays, q=98) -> float:
    v = np.concatenate([np.abs(np.asarray(a)).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else 1.0


def _cbar(fig, m, ax, label=None, location="bottom"):
    cb = fig.colorbar(m, ax=ax, location=location, shrink=0.85, aspect=25, pad=0.02)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(labelsize=7, color=AXIS)
    if label:
        cb.set_label(label, fontsize=8, color=INK2)
    return cb


def fig_bias_maps(diags: dict, var: str, fields: list[tuple[str, str]], mg: MapGrid, save=None):
    """Top row: truth of each field; one row per run: bias of that field (shared scale per column).
    fields: [(name, label)] with diagnostics <var>_<name>_true / <var>_<name>_bias
    (name "mean": <var>_mean_true / <var>_bias_mean)."""
    runs = list(diags)
    d0 = diags[runs[0]]
    fig, axes = map_axes(mg, len(runs) + 1, len(fields))
    tname = lambda n: f"{var}_mean_true" if n == "mean" else f"{var}_{n}_true"  # noqa: E731
    bname = lambda n: f"{var}_bias_mean" if n == "mean" else f"{var}_{n}_bias"  # noqa: E731
    for j, (name, label) in enumerate(fields):
        truth = d0[tname(name)].values
        lo, hi = np.nanpercentile(truth, [1, 99])
        m = draw_map(axes[0, j], mg, truth, SEQ[var], Normalize(lo, hi), f"RCM truth · {label}")
        _cbar(fig, m, axes[0, j])
        vmax = robust_abs([diags[r][bname(name)].values for r in runs])
        for i, run in enumerate(runs, start=1):
            m = draw_map(axes[i, j], mg, diags[run][bname(name)].values, DIV[var], Normalize(-vmax, vmax),
                         f"{run} · bias")
        _cbar(fig, m, axes[1:, j], "bias (prediction − truth)")
    fig.suptitle(f"{LABELS[var]}: RCM truth (top) and bias of each run", x=0.0, ha="left",
                 fontsize=11, color=INK)
    _savefig(fig, save)
    return fig


def fig_field_maps(diags: dict, var: str, name: str, label: str, mg: MapGrid, cmap=None,
                   vmin=None, vmax=None, save=None):
    """One map per run of a non-bias diagnostic, e.g. rmse_map or corr_map."""
    runs = list(diags)
    fig, axes = map_axes(mg, 1, len(runs))
    vals = [diags[r][f"{var}_{name}"].values for r in runs]
    lo, hi = np.nanpercentile(np.stack(vals), [1, 99])
    norm = Normalize(lo if vmin is None else vmin, hi if vmax is None else vmax)
    for j, (run, v) in enumerate(zip(runs, vals)):
        m = draw_map(axes[0, j], mg, v, cmap or SEQ["pr"], norm, run)
    _cbar(fig, m, axes[0, :], label, location="right")
    fig.suptitle(f"{LABELS[var]}: {label}", x=0.0, ha="left", fontsize=11, color=INK)
    _savefig(fig, save)
    return fig


def fig_example_days(truth, preds: dict, dates, var: str, mg: MapGrid, save=None):
    """truth: (D, H, W); preds: run -> (D, H, W) (first member); one row per date."""
    runs = list(preds)
    fig, axes = map_axes(mg, len(dates), len(runs) + 1, panel=2.2)
    for i, date in enumerate(dates):
        t = truth[i]
        cmap = SEQ[var].copy()
        if var == "pr":
            norm = LogNorm(0.5, max(np.nanpercentile(t, 99.5), 5))
            cmap.set_under(SURFACE)  # < 0.5 mm/day: background
        else:
            norm = Normalize(*np.nanpercentile(t, [1, 99]))
        draw_map(axes[i, 0], mg, t, cmap, norm, f"{str(date)[:10]} · RCM truth")
        for j, run in enumerate(runs, start=1):
            m = draw_map(axes[i, j], mg, preds[run][i], cmap, norm, run)
        _cbar(fig, m, axes[i, :], UNITS[var], location="right")
    _savefig(fig, save)
    return fig


def fig_change_maps(d_hist: dict, d_fut: dict, var: str, mg: MapGrid, save=None):
    """Climate change signal (future - historical) of the time mean: truth and each run (top),
    error of each run's signal (bottom). pr in %, tasmax in K."""
    runs = list(d_hist)
    d0h, d0f = d_hist[runs[0]], d_fut[runs[0]]

    def delta(mh, mf):
        return 100 * (mf - mh) / np.maximum(mh, 0.1) if var == "pr" else mf - mh

    th, tf = d0h[f"{var}_mean_true"].values, d0f[f"{var}_mean_true"].values
    true_delta = delta(th, tf)
    run_delta = {r: delta(th + d_hist[r][f"{var}_bias_mean"].values, tf + d_fut[r][f"{var}_bias_mean"].values)
                 for r in runs}
    unit = "%" if var == "pr" else "K"
    fig, axes = map_axes(mg, 2, len(runs) + 1)
    if var == "pr":
        v = robust_abs([true_delta, *run_delta.values()])
        norm, cmap = Normalize(-v, v), DIV["pr"]
    else:
        norm, cmap = Normalize(0, np.nanpercentile(np.stack([true_delta, *run_delta.values()]), 99)), SEQ["tasmax"]
    m = draw_map(axes[0, 0], mg, true_delta, cmap, norm, "RCM truth · change")
    for j, r in enumerate(runs, start=1):
        draw_map(axes[0, j], mg, run_delta[r], cmap, norm, f"{r} · change")
    _cbar(fig, m, axes[0, :], f"change ({unit})", location="right")
    err = {r: run_delta[r] - true_delta for r in runs}
    v = robust_abs(err.values())
    axes[1, 0].set_visible(False)
    for j, r in enumerate(runs, start=1):
        m = draw_map(axes[1, j], mg, err[r], DIV[var], Normalize(-v, v), f"{r} · change error")
    _cbar(fig, m, axes[1, :], f"error ({unit})", location="right")
    fig.suptitle(f"{LABELS[var]}: change of the mean, future − historical", x=0.0, ha="left",
                 fontsize=11, color=INK)
    _savefig(fig, save)
    return fig


# ---------------------------------------------------------------- line / dot charts

def _wavelength_km(nbins: int, n: int = 128) -> np.ndarray:
    k = np.arange(nbins, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(k > 0, n * DX_KM / k, np.nan)


def fig_spectra(diags: dict, var: str, colors: dict, save=None):
    runs = list(diags)
    d0 = diags[runs[0]]
    st = d0[f"{var}_spec_true"].values
    n = len(st)
    lam = _wavelength_km(n)
    sel = slice(1, 65)  # up to Nyquist (2 dx = 25 km); higher bins are grid corners
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.25})
    a1.loglog(lam[sel], st[sel], color=TRUTH_COLOR, lw=2.2, label="RCM truth")
    for r in runs:
        sp = diags[r][f"{var}_spec_pred"].values
        a1.loglog(lam[sel], sp[sel], color=colors[r], label=r)
        a2.semilogx(lam[sel], 10 * np.log10(sp[sel] / st[sel]), color=colors[r], label=r)
    a2.axhline(0, color=AXIS, lw=1)
    for a in (a1, a2):
        a.invert_xaxis()
        a.set_xlabel("wavelength (km)")
    a1.set_ylabel("power")
    a2.set_ylabel("prediction / truth (dB)")
    a1.set_title(f"{LABELS[var]}: mean radial power spectrum")
    a2.set_title("Spectral ratio (0 dB = right amount of small-scale variance)")
    a1.legend(loc="lower left")
    _savefig(fig, save)
    return fig


def fig_distributions(diags: dict, var: str, colors: dict, save=None):
    runs = list(diags)
    d0 = diags[runs[0]]
    lower, upper = d0[f"{var}_bin_lower"].values, d0[f"{var}_bin_upper"].values
    width = upper - lower
    ht = d0[f"{var}_hist_true"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.25})
    centres = d0[f"{var}_bin"].values
    dens = lambda h: h / h.sum() / width  # noqa: E731
    if var == "pr":
        a1.set_xscale("log")
        a1.set_yscale("log")
        keep = lower > 0  # first bin = dry days, shown in the legend instead
        dry_t = ht[0] / ht.sum()
    else:
        keep = slice(None)
        a1.set_yscale("log")
    a1.plot(centres[keep], dens(ht)[keep], color=TRUTH_COLOR, lw=2.2,
            label="RCM truth" + (f" (dry {dry_t:.0%})" if var == "pr" else ""))
    for r in runs:
        hp = diags[r][f"{var}_hist_pred"].values
        extra = f" (dry {hp[0] / hp.sum():.0%})" if var == "pr" else ""
        a1.plot(centres[keep], dens(hp)[keep], color=colors[r], label=r + extra)
    a1.set_xlabel(f"{var} ({'mm/day' if var == 'pr' else '°C'})")
    a1.set_ylabel("density (all days and grid points)")
    a1.set_title(f"{LABELS[var]}: distribution")
    a1.legend(loc="lower left" if var == "pr" else "upper left")

    q = d0["quantile"].values
    qt = d0[f"{var}_quant_true"].values
    lo, hi = qt.min(), qt.max()
    for r in runs:
        qp = diags[r][f"{var}_quant_pred"].values
        a2.plot(qt, qp, "o-", color=colors[r], label=r, ms=5)
        lo, hi = min(lo, qp.min()), max(hi, qp.max())
    a2.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=0)
    for qi, x in zip(q, qt):
        if qi >= 0.99:
            a2.annotate(f"q{qi:g}", (x, x), textcoords="offset points", xytext=(-6, 8), ha="right",
                        fontsize=7, color=MUTED)
    a2.set_xlabel(f"truth quantile ({UNITS[var]})")
    a2.set_ylabel(f"predicted quantile ({UNITS[var]})")
    a2.set_title("Quantile-quantile (pooled over space and time)")
    _savefig(fig, save)
    return fig


def fig_seasonal_bias(diags: dict, colors: dict, variables=("pr", "tasmax"), save=None):
    runs = list(diags)
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 3.2), squeeze=False,
                             gridspec_kw={"wspace": 0.25})
    months = np.arange(1, 13)
    for ax, var in zip(axes[0], variables):
        for r in runs:
            d = diags[r]
            ax.plot(months, d[f"{var}_season_pred"].values - d[f"{var}_season_true"].values,
                    "o-", color=colors[r], label=r, ms=4)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(months, list("JFMAMJJASOND"))
        ax.set_ylabel(f"domain-mean bias ({UNITS[var]})")
        ax.set_title(f"{LABELS[var]}: seasonal cycle of bias")
    axes[0, 0].legend()
    _savefig(fig, save)
    return fig


def fig_scorecard_strips(card, colors: dict, title: str = "", save=None):
    """One small panel per scorecard metric: published models as grey dots, our runs as coloured
    diamonds. card: output of results.scorecard (index (source, model), columns (variable, metric))."""
    cols = list(card.columns)
    ncols = 6
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 1.25 * nrows + 0.9), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.12, wspace=0.04, hspace=0.12)
    pub = card.xs("published", level="source") if "published" in card.index.get_level_values(0) else None
    ours = card.xs("this study", level="source") if "this study" in card.index.get_level_values(0) else None
    for ax, (var, metric) in zip(axes.ravel(), cols):
        vals = []
        if pub is not None:
            v = pub[(var, metric)].dropna().values
            ax.scatter(v, np.zeros_like(v), s=22, color=PUBLISHED_COLOR, zorder=2, linewidths=0)
            vals.append(v)
        if ours is not None:
            for run, v in ours[(var, metric)].dropna().items():
                ax.scatter([v], [0], s=60, marker="D", color=colors.get(run, INK), zorder=3,
                           edgecolors=SURFACE, linewidths=1.2)
                vals.append([v])
        allv = np.concatenate([np.asarray(x, float) for x in vals]) if vals else np.array([1.0])
        if metric == "RALSD" and np.all(allv > 0):
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{var} · {metric}", fontsize=8.5)
        unit = METRIC_UNITS.get((var, metric), "")
        better = "higher is better" if metric == "PSS" else "lower is better"
        ax.set_xlabel(f"{unit}{' · ' if unit else ''}{better}", fontsize=7, color=MUTED, labelpad=2)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", color=PUBLISHED_COLOR, label="published entries")]
    if ours is not None:
        handles += [plt.Line2D([], [], ls="", marker="D", color=colors.get(r, INK), label=r) for r in ours.index]
    fig.legend(handles=handles, loc="outside lower left", ncols=len(handles))
    if title:
        fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK)
    _savefig(fig, save)
    return fig


def fig_rank_histograms(diags: dict, colors: dict, variables=("pr", "tasmax"), save=None):
    runs = [r for r in diags if f"{variables[0]}_rank_hist" in diags[r]]
    if not runs:
        return None
    fig, axes = plt.subplots(len(runs), len(variables), figsize=(4.5 * len(variables), 2.4 * len(runs)),
                             squeeze=False)
    for i, r in enumerate(runs):
        for j, var in enumerate(variables):
            h = diags[r][f"{var}_rank_hist"].values
            ax = axes[i, j]
            ax.bar(np.arange(len(h)), h / h.sum(), color=colors[r], width=0.85)
            ax.axhline(1 / len(h), color=INK2, lw=1)
            ax.set_title(f"{r} · {var} rank histogram (flat = calibrated)")
            ax.set_xlabel("rank of truth among members")
    _savefig(fig, save)
    return fig
