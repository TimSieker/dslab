# How the published CORDEX-ML-Bench scorecard metrics are computed

Source: reading of the organisers' code (2026-10-30), clones in `~/cordex-ml-bench/repos/`
(cordex-bench-eval @ `0fd7df6`, ml-benchmark @ `950e749`). These definitions are what
`src/dslab/metrics.py` reproduces ("scorecard" metrics). File abbreviations:

| short | file |
|---|---|
| SC | `cordex-bench-eval/manuscript_figures/Scorecard_figures_11_13/final_scorecard.py` |
| EV / DL / MP | `cordex-bench-eval/src/cordex_bench_eval/{evaluator,data_loader,mappings}.py` |
| MJ | `cordex-bench-eval/configs/metrics.json` |
| V2 | `cordex-bench-eval/diagnostics/compute_diagnostics_v2.py` |
| DG / IX | `ml-benchmark/evaluation/{diagnostics,indices}.py` |

## Common chain

- **Periods.** Upper-left triangle = `Perfect_cross_validation` (1981–2000). Lower-right =
  `Perfect_extrapolation` for ESD = **2041–2060 and 2080–2099 merged** (MP:40, DL:179-192, DL:369-379,
  EV:470 `merge_mid_end=True`), `Perfect_interpolation` for Emulator = 2041–2060 (MP:47).
  CNRM-CM5, perfect predictors. (The S12/S13 caption describes the triangles reversed; code = legend.)
- **Loading.** Truth and prediction inner-joined on x/y and time (EV:512-513); no unit conversion
  (pr mm/day, tasmax K), no mask, no area weighting.
- **Members** (EV:155-190, 663-735): `mean` = ensemble mean; `random` = one member chosen by a hash of
  the context; `member_mean` = up to 5 distinct members, metric map per member, maps averaged over
  members. *dslab uses members 0..4 and member 0 instead of the hash (members are exchangeable).*
- **Reduction to one number** (SC:246-262): map metrics → `sqrt(mean(v**2))` over finite grid points,
  unweighted. RALSD, LHD, PSS are scalars already. AV = mean over the three domains (SC:304).

## Per metric

| metric | members | definition |
|---|---|---|
| RMSE (pr, tasmax) | ensemble mean | `sqrt(mean((ens_mean - truth)**2))` over all t, y, x (DG.rmse per-gridpoint map, then RMS over space — identical) |
| Mean bias | ensemble mean | map `ens_mean.mean(time) - truth.mean(time)` (absolute: mm/day, K) → RMS over space |
| SDII (pr) | 5 members | map `pr.where(pr >= 1.0).mean(time)` (wet threshold ≥ 1 mm/day inclusive, pooled over the whole period); bias map per member, averaged → RMS over space |
| Rx1day / TXx | 5 members | map `resample(time="1YE").max().mean(time)` (mean of calendar-year maxima; empty years in a merged period are skipped); bias, averaged → RMS over space |
| RALSD (pr, tasmax) | 1 member | NaN→0, `fft2` per day (no detrending, window or normalisation), `fftshift`, `|F|²`; radial average with centre ((nx−1)/2, (ny−1)/2) and `r = int(hypot)` → 90 bins for 128×128; **daily radial spectra averaged over time**; `d = 10*log10(max(S_true / max(S_pred, 1e-30), 1e-30))`; **RALSD = mean_k(d²)**, no sqrt, units dB² (DG:40-59, 236-309) |
| LHD (pr) | all members, nanmean | `np.histogram` of all values pooled over t, y, x with bins `linspace(0, 600, 61)` (10 mm/day bins), raw counts; valid bins where pred > 10 and truth > 10 counts; `sqrt(mean((10*log10(pred/truth))**2))` (V2:437-475, 515-547) |
| PSS (tasmax) | all members, nanmean | °C (−273.15); bins `linspace(-50, 60, 111)`; `p = counts / in-range total`; `PSS = Σ min(p_pred, p_true)` (V2:383-394, 448-483) |
| IAV (tasmax) | 1 member | map `groupby("time.year").mean().std("year")` (ddof=0); bias pred − truth (K) → RMS over space (IX:361) |

Notes:
- The paper's RALSD (eq. 5, per-day sqrt) is **not** what produced the published numbers. Evidence:
  published values of 0.025–0.04 for generative models and 300–1000 for others are only reachable with
  time-averaged spectra in dB².
- ESD future period LHD/PSS: the code does not determine whether 2041–2060 only, 2080–2099 only, or the
  pooled 40 years was used (V2 overwrites; the older pipeline pools). dslab computes it on the merged
  period like the other metrics; treat the published ESD-future LHD/PSS values as uncertain.
- S13 TXx/SDII/Rx1day reflect the 5-member version (commit 2a8554c, 2026-05-13);
  `compute_vs_skill_master.csv` predates it (differs only for generative models).
- RMSE of generative models is the RMSE of the ensemble mean (favours smooth ensemble means).
- Display rounding in the figures (SC:150-157): |v| < 0.1 → 3 decimals, < 10 → 2, < 100 → 1, else integer.
