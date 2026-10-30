"""Metric functions on plain numpy arrays.

truth: (T, H, W); pred: (M, T, H, W) with M ensemble members (M = 1 for deterministic models).
The "scorecard" metrics reproduce the published CORDEX-ML-Bench numbers exactly as the organisers'
code computes them (see docs/metric_definitions.md); the others are extra diagnostics for development.
"""

from __future__ import annotations

import numpy as np

SCORECARD = {
    "pr": ("RMSE", "Mean Bias", "SDII", "Rx1day", "RALSD", "LHD"),
    "tasmax": ("RMSE", "Mean Bias", "TXx", "RALSD", "PSS", "IAV"),
}
LHD_BINS = np.linspace(0, 600, 61)  # pr, mm/day
PSS_BINS = np.linspace(-50, 60, 111)  # tasmax, degC
N_INDEX_MEMBERS = 5  # extreme indices are averaged over (at most) 5 members


def rms_space(field: np.ndarray) -> float:
    """Published reduction of a bias map to one number: RMS over finite grid points, unweighted."""
    v = field[np.isfinite(field)].astype(np.float64)
    return float(np.sqrt(np.mean(v**2))) if v.size else np.nan


def time_mean(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=-3, dtype=np.float64)


def _by_year(a: np.ndarray, years: np.ndarray, fn) -> np.ndarray:
    """Apply fn over time within each calendar year: (..., T, H, W) -> (..., Y, H, W)."""
    return np.stack([fn(a[..., years == y, :, :]) for y in np.unique(years)], axis=-3)


def annual_max_mean(a: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Mean over years of the calendar-year maximum (Rx1day, TXx)."""
    return _by_year(a, years, lambda b: b.max(axis=-3)).mean(axis=-3, dtype=np.float64)


def interannual_std(a: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Standard deviation (ddof=0) of annual means (IAV)."""
    return _by_year(a, years, lambda b: b.mean(axis=-3, dtype=np.float64)).std(axis=-3)


def sdii(a: np.ndarray, wet: float = 1.0) -> np.ndarray:
    """Mean precipitation on wet days (>= wet mm/day) over the whole period; NaN where never wet."""
    is_wet = a >= wet
    total = np.where(is_wet, a, 0).sum(axis=-3, dtype=np.float64)
    n = is_wet.sum(axis=-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


def wet_fraction(a: np.ndarray, wet: float = 1.0) -> np.ndarray:
    return (a >= wet).mean(axis=-3)


def quantile_map(a: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(a, q, axis=-3)


def temporal_corr(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pearson correlation over time at every grid point."""
    p = pred - pred.mean(axis=0, dtype=np.float64)
    t = truth - truth.mean(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (p * t).sum(axis=0) / np.sqrt((p**2).sum(axis=0) * (t**2).sum(axis=0))


class RadialSpectrum:
    """Radially averaged power spectrum exactly as ml-benchmark evaluation/diagnostics.py::psd:
    fft2 without detrending or windowing, fftshift, |F|^2, mean over pixels with
    r = int(hypot(x - (nx-1)/2, y - (ny-1)/2))."""

    def __init__(self, shape: tuple[int, int]):
        ny, nx = shape
        yy, xx = np.indices(shape)
        self.r = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2).astype(np.int32).ravel()
        self.nbins = int(self.r.max()) + 1
        self.counts = np.bincount(self.r, minlength=self.nbins).astype(np.float64)

    def __call__(self, fields: np.ndarray) -> np.ndarray:
        """(B, H, W) -> (B, nbins)."""
        f = np.nan_to_num(fields.astype(np.float64))
        power = np.abs(np.fft.fftshift(np.fft.fft2(f), axes=(-2, -1))) ** 2
        power = power.reshape(len(f), -1)
        out = np.stack([np.bincount(self.r, weights=p, minlength=self.nbins) for p in power])
        return out / np.maximum(self.counts, 1)

    def time_mean(self, a: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Mean daily spectrum of a (T, H, W) array."""
        total = np.zeros(self.nbins)
        for i in range(0, len(a), chunk):
            total += self(a[i:i + chunk]).sum(axis=0)
        return total / len(a)


def ralsd(spec_true: np.ndarray, spec_pred: np.ndarray) -> float:
    """Published RALSD: mean squared dB difference of time-mean spectra (no sqrt, units dB^2)."""
    d = 10 * np.log10(np.maximum(spec_true / np.maximum(spec_pred, 1e-30), 1e-30))
    return float(np.mean(d**2))


def histogram(a: np.ndarray, bins: np.ndarray) -> np.ndarray:
    v = a[np.isfinite(a)]
    return np.histogram(v, bins)[0].astype(np.float64)


def lhd(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """Log-histogram distance (dB) over bins with > 10 counts in both raw-count histograms."""
    ok = (h_pred > 10) & (h_true > 10)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((10 * np.log10(h_pred[ok] / h_true[ok])) ** 2)))


def pss(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """Perkins skill score: overlap of the normalised histograms."""
    return float(np.minimum(h_pred / h_pred.sum(), h_true / h_true.sum()).sum())


def crps_ensemble(pred: np.ndarray, truth: np.ndarray, chunk: int = 256) -> float:
    """Mean fair CRPS of an M-member ensemble (equals MAE for M = 1)."""
    M, T = pred.shape[:2]
    total = 0.0
    for i in range(0, T, chunk):
        p = np.sort(pred[:, i:i + chunk].astype(np.float64), axis=0)
        t = truth[i:i + chunk].astype(np.float64)
        skill = np.abs(p - t).mean(axis=0)
        if M > 1:
            # sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - M + 1) x_(k) for sorted members
            w = (2 * np.arange(M) - M + 1).reshape(-1, *([1] * (p.ndim - 1)))
            spread = 2 * (w * p).sum(axis=0) / (2 * M * (M - 1))
        else:
            spread = 0.0
        total += (skill - spread).sum()
    return float(total / truth.size)


def spread_skill(pred: np.ndarray, truth: np.ndarray, chunk: int = 256) -> float:
    """Ratio of ensemble spread to ensemble-mean RMSE, with the (M+1)/M finite-ensemble factor (~1 if calibrated).
    Accumulated over time chunks to keep memory low for large ensembles."""
    M, T = pred.shape[:2]
    var_sum = sq_err = 0.0
    for i in range(0, T, chunk):
        p = pred[:, i:i + chunk].astype(np.float64)
        var_sum += p.var(axis=0, ddof=1).sum()
        sq_err += ((p.mean(axis=0) - truth[i:i + chunk]) ** 2).sum()
    spread = np.sqrt(var_sum / truth.size * (M + 1) / M)
    return float(spread / np.sqrt(sq_err / truth.size))


def miscalibration(rank_counts: np.ndarray) -> float:
    """MCB (Schillinger et al. 2025, used by Debeire et al. 2026): mean squared deviation of the relative
    rank frequencies from uniform; 0 = perfectly calibrated."""
    h = rank_counts / rank_counts.sum()
    return float(np.mean((h - 1 / len(h)) ** 2))


def rank_histogram(pred: np.ndarray, truth: np.ndarray, stride: int = 4, seed: int = 0) -> np.ndarray:
    """Counts of the truth's rank among the members (M + 1 bins), ties broken at random.
    Uses every `stride`-th grid point in y and x to keep it cheap."""
    rng = np.random.default_rng(seed)
    p = pred[:, :, ::stride, ::stride]
    t = truth[:, ::stride, ::stride]
    below = (p < t).sum(axis=0)
    equal = (p == t).sum(axis=0)
    ranks = below + np.floor(rng.random(below.shape) * (equal + 1)).astype(int)
    return np.bincount(ranks.ravel(), minlength=pred.shape[0] + 1)
