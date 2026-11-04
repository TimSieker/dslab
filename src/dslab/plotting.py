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
import pandas as pd
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

# Canonical run -> slot, so colours never depend on which runs happen to be loaded. afm takes the purple
# slot because the green one next in line is hard to tell from the U-Net's teal; the LC-AFM arms take
# orange and pink, which stay apart from teal and purple also for red-green colour vision deficiency.
CANONICAL_RUNS = {"linear": 0, "deepesd": 1, "unet": 2, "flow": 3, "flow_residual": 4, "afm": 6,
                  "lcafm-T-O": 1, "lcafm-Tfull-O": 4}
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
    taken = set(CANONICAL_RUNS.values())
    colors, free = {}, iter([c for i, c in enumerate(SERIES) if i not in taken] + SERIES)
    for run in runs:
        if run in RUN_COLORS:
            colors[run] = RUN_COLORS[run]
        elif run == "climatology":
            colors[run] = FLOOR_COLOR
        elif run in CANONICAL_RUNS:
            colors[run] = SERIES[CANONICAL_RUNS[run]]
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


def _cbar(fig, m, ax, label=None, location="bottom", extend="neither"):
    cb = fig.colorbar(m, ax=ax, location=location, shrink=0.85, aspect=25, pad=0.02, extend=extend)
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
    extend = "min" if np.nanmin(np.stack(vals)) < norm.vmin else "neither"
    _cbar(fig, m, axes[0, :], label, location="right", extend=extend)
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
    """Mean radial spectra over all bins used by the published RALSD. Bins beyond the Nyquist
    wavenumber (shaded) are the grid's diagonal 'corners' (pixel-scale texture); 25 of the 90 bins."""
    runs = list(diags)
    d0 = diags[runs[0]]
    st = d0[f"{var}_spec_true"].values
    n = len(st)
    k = np.arange(n)
    nyq = int(d0.sizes.get(d0["lat"].dims[0], 128)) // 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.25})
    a1.loglog(k[1:], st[1:], color=TRUTH_COLOR, lw=2.2, label="RCM truth")
    for r in runs:
        sp = diags[r][f"{var}_spec_pred"].values
        a1.loglog(k[1:], sp[1:], color=colors[r], label=r)
        a2.semilogx(k[1:], 10 * np.log10(sp[1:] / st[1:]), color=colors[r], label=r)
    a2.axhline(0, color=AXIS, lw=1)
    for a in (a1, a2):
        a.axvspan(nyq, n, color=GRID, alpha=0.6, lw=0, zorder=0)
        a.set_xlim(1, n)
        a.set_xlabel("wavenumber bin  (wavelength = 1600 km / k)")
        sec = a.secondary_xaxis("top", functions=(lambda x: 128 * DX_KM / np.maximum(x, 1e-9),
                                                   lambda x: 128 * DX_KM / np.maximum(x, 1e-9)))
        sec.set_xlabel("wavelength (km)", fontsize=8, color=INK2)
        sec.tick_params(labelsize=7, colors=AXIS, labelcolor=INK2)
    a1.text(nyq * 0.95, a1.get_ylim()[1] * 0.5, "beyond Nyquist\n(grid corners) →", ha="right", va="top",
            fontsize=7, color=MUTED)
    a1.set_ylabel("power")
    a2.set_ylabel("prediction / truth (dB)")
    a1.set_title(f"{LABELS[var]}: mean radial power spectrum", pad=28)
    a2.set_title("Spectral ratio (RALSD = mean square of this curve)", pad=28)
    a1.legend(loc="lower left")
    _savefig(fig, save)
    return fig


def ralsd_split(diags: dict, var: str) -> dict:
    """Published RALSD of each run split into resolved scales (k <= Nyquist) and grid-corner bins."""
    from .metrics import ralsd

    out = {}
    for r, d in diags.items():
        st, sp = d[f"{var}_spec_true"].values, d[f"{var}_spec_pred"].values
        nyq = len(st) and int(d.sizes.get(d["lat"].dims[0], 128)) // 2
        out[r] = {"RALSD": ralsd(st, sp), "k <= Nyquist": ralsd(st[:nyq + 1], sp[:nyq + 1]),
                  "corner bins": ralsd(st[nyq + 1:], sp[nyq + 1:])}
    return out


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


def fig_training_curves(histories: dict, colors: dict, save=None):
    """One panel per run (the training losses of different models are different quantities): training
    loss (grey) and validation loss (run colour) against passes over the training data; the dot marks
    the checkpoint that is kept."""
    if not histories:
        return None
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 3.4), squeeze=False, layout="constrained")
    for ax, (run, h) in zip(axes[0], histories.items()):
        ax.plot(h.epochs_done, h.train_loss, color=MUTED, label="training")
        ax.plot(h.epochs_done, h.val_loss, color=colors[run], label="validation")
        best = h.val_loss.idxmin()
        ax.plot(h.epochs_done[best], h.val_loss[best], "o", color=colors[run], ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="kept")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Plain-number labels at 1-2-5 steps; the default minor labels collide on ranges below one decade
        for axis, subs in ((ax.xaxis, (1, 2, 5)), (ax.yaxis, (1, 1.5, 2, 3, 5, 7))):
            axis.set_major_locator(mpl.ticker.LogLocator(subs=subs))
            axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("passes over the training data")
        ax.set_ylabel(h.attrs.get("loss_name", "loss"))
        ax.set_title(run)
        ax.legend()
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


# ---- LC-AFM law diagnostics (notebooks/laws.ipynb; data from dslab.law_diagnostics)
SPLIT_LABELS = {"train": "training years 1961–80", "val": "validation years 1967, 1975",
                "historical_perfect": "1981–2000", "mid_century_perfect": "2041–2060",
                "end_century_perfect": "2080–2099", "historical_imperfect": "1981–2000, GCM predictors",
                "end_century_imperfect": "2080–2099, GCM predictors"}
# colour = period, dashes = imperfect (GCM) predictors; training in primary ink as the reference
SPLIT_STYLE = {"train": (INK, "-"), "val": (MUTED, "-"), "historical_perfect": (SERIES[0], "-"),
               "mid_century_perfect": (SERIES[3], "-"), "end_century_perfect": (SERIES[1], "-"),
               "historical_imperfect": (SERIES[0], "--"), "end_century_imperfect": (SERIES[1], "--")}
LAW_TITLES = {"T": "T · lapse rate, sub-2° scales", "Tfull": "Tfull · lapse rate, all scales",
              "O": "O · orographic pr, sub-2° scales"}
LAW_UNITS = {"T": "K", "Tfull": "K", "O": "mm/day"}


def _law_axes(n: int, panel_w: float, panel_h: float, ncols: int = 4, **kw):
    """Small multiples for n laws, at most `ncols` per row; unused panels hidden. Returns (fig, axes list)."""
    ncols = min(n, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False,
                             layout="constrained", **kw)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n], ncols


def _split_dots(ax, splits, values, filled=True, color=INK2, xmin=None, label=None):
    ypos = np.arange(len(splits))[::-1]
    v = np.asarray(values, float)
    shown = v if xmin is None else np.maximum(v, xmin)
    ax.plot(shown, ypos, "o", ms=7, color=color, mfc=color if filled else SURFACE, mew=1.5, label=label,
            zorder=3)
    if xmin is not None:
        for y, a in zip(ypos, v):
            if a < xmin:
                ax.annotate(f"◂ {a:.2g}", (xmin, y), xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=7, color=INK2)
    return ypos


def fig_law_skill(summary, save=None):
    """Per law: fraction of the day-to-day variance (about the training climatology of J x) that the law
    explains, per split; filled = law (physics + monthly offset), hollow = physics alone."""
    laws = list(dict.fromkeys(summary.index.get_level_values("law")))
    splits = list(dict.fromkeys(summary.index.get_level_values("split")))
    fig, axes, ncols = _law_axes(len(laws), 3.6, 0.32 * len(splits) + 1.4, sharey=True)
    for ax, n in zip(axes, laws):
        d = summary.loc[n].reindex(splits)
        lo = -1.0  # common scale; anything below is drawn at the edge with its value printed
        ypos = _split_dots(ax, splits, d["skill of physics alone"], filled=False, color=SERIES[0], xmin=lo,
                           label="physics alone")
        _split_dots(ax, splits, d["skill vs climatology"], filled=True, color=SERIES[0], xmin=lo,
                    label="law (physics + monthly offset)")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(lo - 0.05, 1)
        ax.set_title(LAW_TITLES.get(n, n))
        ax.set_xlabel("variance explained beyond climatology")
        ax.grid(axis="y", visible=False)
    for ax in axes[::ncols]:
        ax.set_yticks(ypos, [SPLIT_LABELS.get(s, s) for s in splits])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    _savefig(fig, save)
    return fig


def fig_law_calibration(summary, save=None, cap=4.0):
    """Mean squared normalised residual per split (1 = tolerance calibrated); label: inflation alpha. Values above
    `cap` are drawn at the cap as a triangle, with their alpha printed."""
    laws = list(dict.fromkeys(summary.index.get_level_values("law")))
    splits = list(dict.fromkeys(summary.index.get_level_values("split")))
    fig, axes, ncols = _law_axes(len(laws), 3.6, 0.32 * len(splits) + 1.4, sharey=True, sharex=True)
    hi = min(float(np.nanmax(summary["mean z²"])), cap)
    for ax, n in zip(axes, laws):
        d = summary.loc[n].reindex(splits)
        z2s = d["mean z²"].astype(float).values
        ypos = _split_dots(ax, splits, np.minimum(z2s, cap), color=SERIES[1])
        for y, z2 in zip(ypos, z2s):
            if z2 > cap:
                ax.plot(cap, y, ">", ms=9, color=SERIES[1], zorder=4)
            ax.annotate(f"α = {np.sqrt(z2):.3g}", (min(z2, cap), y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK2)
        ax.axvline(1, color=INK2, lw=1)
        ax.set_xlim(0, max(hi, 1.6) * 1.35)
        ax.set_title(LAW_TITLES.get(n, n))
        ax.set_xlabel("mean z²  (1 = tolerance calibrated)")
        ax.grid(axis="y", visible=False)
    for ax in axes[::ncols]:
        ax.set_yticks(ypos, [SPLIT_LABELS.get(s, s) for s in splits])
    _savefig(fig, save)
    return fig


def fig_law_z_hist(ds, splits, save=None):
    """Density of the normalised residual z (log scale) per split, against the standard normal."""
    laws = list(ds.law.values)
    fig, axes, ncols = _law_axes(len(laws), 3.6, 3.0, sharey=True)
    z = ds.z.values
    gauss = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    for ax, n in zip(axes, laws):
        for s in splits:
            col, ls = SPLIT_STYLE.get(s, (INK2, "-"))
            ax.plot(z, ds.z_hist.sel(law=n, split=s), color=col, ls=ls, label=SPLIT_LABELS.get(s, s))
        ax.plot(z, gauss, color=INK2, lw=1, ls=":", label="standard normal")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(-7, 7)
        ax.set_title(LAW_TITLES.get(n, n))
        ax.set_xlabel("normalised residual z = r / τ")
    for ax in axes[::ncols]:
        ax.set_ylabel("density")
    axes[0].legend(loc="lower center", fontsize=7)
    _savefig(fig, save)
    return fig


def fig_law_monthly(ds, name: str, splits, ylabel: str, ref: float | None = None, save=None):
    """Seasonal cycle of a per-month law diagnostic (e.g. ms_z_month, bias_month) for several splits."""
    laws = list(ds.law.values)
    fig, axes, _ = _law_axes(len(laws), 3.6, 2.9)
    for ax, n in zip(axes, laws):
        for s in splits:
            col, ls = SPLIT_STYLE.get(s, (INK2, "-"))
            ax.plot(ds.month, ds[name].sel(law=n, split=s), color=col, ls=ls, marker="o", ms=3.5,
                    label=SPLIT_LABELS.get(s, s))
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1)
        ax.set_xticks(ds.month.values, list("JFMAMJJASOND"))
        ax.set_title(LAW_TITLES.get(n, n))
        ax.set_ylabel(ylabel.format(units=LAW_UNITS.get(n, "")))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    _savefig(fig, save)
    return fig


def fig_law_sizes(ds, save=None):
    """Per law and month (training years, domain RMS): the day-to-day signal the law has to explain
    (J x about its monthly climatology), the tolerance tau (what the law leaves), and the prior noise s
    of the base it conditions (sigma_max of the reference run)."""
    laws = list(ds.law.values)
    fig, axes = plt.subplots(1, len(laws), figsize=(3.6 * len(laws), 2.9), layout="constrained")
    for ax, n in zip(np.atleast_1d(axes), laws):
        d = ds.sel(law=n, split="train")
        ax.plot(ds.month, np.sqrt(d.mse_clim_month), color=INK, marker="o", ms=3.5,
                label="day-to-day signal (truth − climatology)")
        ax.plot(ds.month, np.sqrt((ds.tol.sel(law=n) ** 2).mean(ds.tol.dims[-2:])), color=SERIES[0], marker="o",
                ms=3.5, label="tolerance τ (truth − law)")
        if "prior_noise" in ds:
            ax.axhline(float(ds.prior_noise.sel(law=n)), color=SERIES[1], lw=1.4, ls="--",
                       label=f"prior noise s ({ds.attrs.get('prior_noise_run', '')})")
        ax.set_ylim(0, None)
        ax.set_xticks(ds.month.values, list("JFMAMJJASOND"))
        ax.set_title(LAW_TITLES.get(n, n))
        ax.set_ylabel(f"domain RMS ({LAW_UNITS.get(n, '')})")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    _savefig(fig, save)
    return fig


def fig_law_maps(panels: list[dict], mg: MapGrid, ncols: int = 3, save=None):
    """Grid of maps, one colour bar each: panels = [{field, title, cmap, norm, label, extend?}]."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = map_axes(mg, nrows, ncols, panel=2.6, cbar_room=0.2)
    for ax, p in zip(axes.flat, panels):
        m = draw_map(ax, mg, p["field"], p["cmap"], p["norm"], p.get("title"))
        _cbar(fig, m, ax, p.get("label"), extend=p.get("extend", "neither"))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    _savefig(fig, save)
    return fig


def fig_law_scorecard(card, save=None):
    """One row per law (grouped), three panels for the three criteria: (a) does it hold: mean z² today, in
    2080-2099 and with GCM predictors (1 = calibrated); (b) does conditioning change the base: mean z² of a white
    base (truth + conservative noise proxy, and the trained AFM's white base where it exists; truth = 1); (c) daily
    information: variance explained beyond the training climatology, today and in 2080-2099.
    card: DataFrame indexed by law with columns group, title, a_today, a_future, a_gcm, b_proxy, b_afm, c_today,
    c_future."""
    groups = list(dict.fromkeys(card["group"]))
    rows, labels, ypos, heads, y = [], [], [], [], 0.0
    for g in groups:
        sub = card[card["group"] == g]
        heads.append((g, y + 0.8))
        for law, r in sub.iterrows():
            rows.append(r)
            labels.append(r["title"])
            ypos.append(y)
            y -= 1
        y -= 0.6
    ypos = np.array(ypos)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.3 * len(rows) + 1.6), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    df = pd.DataFrame(rows)

    def dots(ax, col, color, filled, label, lo=None, hi=None):
        v = df[col].astype(float).values
        shown = v if lo is None else np.clip(v, lo, hi)
        ax.plot(shown, ypos, "o", ms=6.5, color=color, mfc=color if filled else SURFACE, mew=1.4, label=label, zorder=3)
        if lo is not None:
            for yy, a, sv in zip(ypos, v, shown):
                if np.isfinite(a) and a != sv:
                    ax.annotate(f"{a:.3g}", (sv, yy), xytext=(5 if a > sv else -5, 5), textcoords="offset points",
                                fontsize=6.5, color=INK2, ha="left" if a > sv else "right")

    a, b, c = axes
    dots(a, "a_today", INK, True, "1981–2000", 0.5, 20)
    dots(a, "a_future", SERIES[1], True, "2080–2099", 0.5, 20)
    dots(a, "a_gcm", SERIES[1], False, "1981–2000, GCM predictors", 0.5, 20)
    a.axvline(1, color=INK2, lw=1)
    a.set_xscale("log")
    a.set_xlim(0.45, 25)
    a.set_title("(a) does it hold?")
    a.set_xlabel("mean z² of the truth (1 = calibrated)")
    dots(b, "b_proxy", SERIES[0], False, "truth + white noise (conservative proxy)", 0.3, 40)
    dots(b, "b_afm", SERIES[0], True, "trained AFM's white base", 0.3, 40)
    b.axvline(1, color=INK2, lw=1)
    b.set_xscale("log")
    b.set_xlim(0.28, 50)
    b.set_title("(b) does it shape the noise?")
    b.set_xlabel("mean z² of a white base (truth ≈ 1)")
    dots(c, "c_today", INK, True, "1981–2000", -0.5, 1)
    dots(c, "c_future", SERIES[1], True, "2080–2099", -0.5, 1)
    c.axvline(0, color=INK2, lw=1)
    c.set_xlim(-0.55, 1.05)
    c.set_title("(c) does it add daily information?")
    c.set_xlabel("variance explained beyond climatology")
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), fontsize=7, ncol=1, frameon=False)
        ax.set_ylim(ypos.min() - 0.7, 1.4)
    for g, yy in heads:  # group headings in the gaps
        a.text(0.01, yy, g, transform=a.get_yaxis_transform(), ha="left", va="center", fontsize=7.5,
               fontweight="bold", color=INK2)
    axes[0].set_yticks(ypos, labels)
    _savefig(fig, save)
    return fig
