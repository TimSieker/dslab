# downscaling-lab: working notes

Running log of decisions, jobs and results. Newest at the bottom. Times CET.

## Ground rules (from the user, carried over from ~/cordex-ml-bench)

- Ask before any job or set of jobs above **5 GPU-hours**.
- Code in HOME, data / outputs on WORK, caches on scratch; Python env on HOME (DKRZ advice, see below).
- Heavy work as SLURM batch jobs (see README › Where to run what); login nodes for light work only.
- Never contact anyone / open issues / submit; never delete data or checkpoints without asking.
- Always `PYTHONNOUSERSITE=1` (`~/.local` holds unrelated user packages); don't modify `~/.local` without asking.

## 2026-10-30: project start (Phase A: data, CPU baselines, evaluation)

Context: the CorrDiff-TW1 replication in `~/cordex-ml-bench` was judged too expensive. Plan agreed with the user:
cheap experiments on the same ALPS data, a ladder of methods sharing one data / evaluation pipeline, then the
user's own model (adaptive flow matching). Start with **ESD pseudo-reality only**, keep it extensible to the
Emulator experiment and other domains; evaluation and plotting in a Jupyter notebook.

Method ladder: 0 climatology, 1 linear (PCA + ridge), 2 DeepESD, 3 small U-Net, 4 conditional flow matching
(plain and residual, same backbone), 5 the user's model. Phase A = 0 and 1 + all infrastructure (CPU only).

### Decisions

- **Own lightweight package** `dslab` (no PhysicsNeMo/hydra); experiments are YAML files; models implement one
  `Downscaler` interface; CLI `dslab prepare|fit|predict|evaluate`.
- **Splits.** Train on 1961–1980 without the validation years 1967 and 1975 (the benchmark's suggested
  validation years); model selection on `val` only. Test splits: perfect predictors 1981–2000, 2041–2060,
  2080–2099 (+ imperfect 1981–2000 configured, not run). ESD "future" is evaluated as the benchmark does:
  2041–2060 + 2080–2099 pooled (`mid_end_century_perfect`).
- **Metrics** reproduce the organisers' code (not the paper text where they differ), see
  `docs/metric_definitions.md` (from a full read of cordex-bench-eval / ml-benchmark): RALSD = mean squared dB
  difference of time-mean spectra (dB², no sqrt), extreme indices over ≤ 5 members, RMSE of the ensemble mean,
  map metrics reduced by RMS over space. Member selection: members 0..4 / member 0 instead of the organisers'
  hash (exchangeable members).
- **Environment** `~/envs/dslab` (Python 3.12, numpy 2.3, xarray, scikit-learn, cartopy, torch 2.10 cu128,
  ipykernel/nbconvert, cordex-bench-eval @ 0fd7df6). Built on HOME because DKRZ recommends Python envs on /home
  (many small files): imports 1.8 s (scientific stack) and 4.2 s (torch) vs > 2 min from the old WORK env.
  7.9 GB. A first build on WORK was stopped after 11 min and deleted (it also pulled Qt6/cloud libs via
  `matplotlib`/`dask`; the spec now uses `matplotlib-base`/`dask-core`).
- Raw data read from `~/cordex-ml-bench/work/data/benchmark` (Zenodo v7, record 20985924); not copied.

### DKRZ guidance on memory / compute-intensive work (checked at the user's request)

- Login nodes: editing, compiling, job handling, light serial tasks; ≤ 10% of node memory per user (killed above).
- Memory/compute-intensive interactive work: partition `interactive` via `salloc` (1940 MB/CPU, max 12 h,
  one node's worth per user) or JupyterHub *interactive* preset (5–100 GB); prefer batch jobs when no
  interactivity is needed.
- `shared`: max 940 MB per CPU; a larger `--mem` makes Slurm allocate more CPUs (charged as CPU-seconds).
  My first `srun` test (28577890/28577897, `--cpus-per-task=8 --mem=40G`) was therefore allocated **44 CPUs**
  for a 1.8 GB / 37 s job. All scripts now request `--cpus-per-task=N --mem-per-cpu=940M` explicitly.
- `compute`: exclusive nodes (whole node charged), 940 MB default per CPU, `--mem=0` for the full node.
- Kernels for JupyterHub: `python -m ipykernel install --user` → `~/.local/share/jupyter/kernels/` (asked the
  user before running `env/install_kernel.sh`).

### Jobs

| job | what | resources | wall | notes |
|---|---|---|---|---|
| 28577890 | prepare (failed: helper script in login-node /tmp) | shared, 44 CPUs (40G) | 1 s | |
| 28577897 | prepare: build .npy caches of all ESD splits | shared, 44 CPUs (40G) | 37 s | MaxRSS 1.8 GB |
| 28578201 | smoke test fit/predict/evaluate on val (old cdx env) | shared, 16 CPUs × 940M | ~3 min | |
| 28578292 | `jobs/baselines.sbatch alps_esd` (fit, predict 4 splits, evaluate 5 splits, tests) | shared, 16 CPUs × 940M | 14 min 33 s | MaxRSS 9.5 GB; evaluation ≈ 65 s per 20-year split; 4 tests passed |
| 28578762, 28578813, 28578863 | `jobs/notebook.sbatch` (execute evaluation.ipynb) | shared, 8 CPUs × 940M | 22–23 s | MaxRSS 0.4 GB |
| | **total** | CPU only | | ≈ 5.5 CPU-h (≈ 0.02 node-h), 0 GPU-h |

### Results, Phase A (ALPS ESD, perfect predictors)

Linear = ridge on all 3840 predictor PCs (selected on val: α = 10 for sqrt(pr), α = 100 for tasmax; val RMSE
4.41 mm/day / 1.55 K). Climatology = 31-day smoothed day-of-year mean of 1961–1980 (without val years).

| split | model | pr RMSE | pr Mean Bias | pr SDII | pr Rx1day | pr RALSD | pr LHD | tx RMSE | tx Mean Bias | tx TXx | tx RALSD | tx PSS | tx IAV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hist 1981–2000 | climatology | 6.85 | 0.17 | 5.27 | 49.3 | 90.5 | 24.1 | 3.33 | 0.36 | 6.21 | 6.76 | 0.938 | 0.707 |
| | linear | 4.42 | 0.45 | 2.99 | 20.6 | 10.0 | 5.22 | 1.55 | 0.10 | 0.65 | 3.97 | 0.978 | 0.114 |
| | *DeepESD-IFCAv1-orog (published)* | 4.32 | 0.25 | 1.42 | 15.9 | 12.3 | 1.81 | 1.64 | 0.16 | 0.92 | 8.53 | 0.98 | 0.091 |
| | *XGBoost_IDL (published)* | 4.57 | 0.16 | 2.82 | 23.9 | 9.73 | 2.56 | 1.51 | 0.10 | 1.02 | 1.75 | 0.97 | 0.12 |
| future 2041–60 + 2080–99 | climatology | 7.34 | 0.36 | 5.87 | 54.6 | 99.2 | 24.4 | 4.90 | 3.30 | 10.2 | 6.48 | 0.815 | 1.23 |
| | linear | 4.83 | 0.40 | 3.49 | 24.4 | 9.33 | 7.12 | 1.65 | 0.34 | 1.11 | 1.61 | 0.974 | 0.151 |
| | *DeepESD-IFCAv1-orog (published)* | 4.72 | 0.29 | 1.61 | 18.2 | 13.0 | 2.27 | 1.76 | 0.31 | 1.20 | 5.64 | 0.98 | 0.17 |

Observations:
- **Pipeline check passed.** Linear lands among the published deterministic entries on every metric
  (e.g. pr RALSD 10.0 vs DeepESD 12.3 / XGBoost 9.7; tasmax RMSE 1.55 vs 1.64 / 1.51). `tests/test_metrics.py`
  reproduces RMSE, mean bias, SDII, Rx1day, TXx, IAV and RALSD with the organisers' own functions on stored
  predictions (rel. tol. 1e-4). So the evaluation is consistent with the published scorecards.
- Linear extrapolates temperature well (future tasmax mean bias 0.34 K, change error ±0.5 K) but is too smooth
  (pr spectra 2–4 dB low at all scales), has too few dry days (31% vs 49%) and underestimates extremes
  (q0.9999 85 vs 109 mm/day): what deterministic regression should do; generative models should fix this.
- Climatology has no warming (future tasmax bias 3.3 K): the floor behaves as intended.

### Map projection of the ALPS target grid

The target x/y (km) are an exact spherical Lambert conformal projection (fitted, max error 1 mm):
`+proj=lcc +lat_0=49.5 +lon_0=10.5 +lat_1=49.5 +lat_2=49.5 +x_0=2825000 +y_0=2825000 +R=6371229`.
Stored as `grid_crs` in the experiment YAML; the notebook draws maps in it (rectangular domain).

## 2026-10-30: first commit, kernel, Phase B (DeepESD, U-Net)

User: "make the first git commit, then continue with the other steps".
- Commit `483ca82` (Phase A), repo-local identity `b384340 <timsieker@gmail.com>` as in ~/cordex-ml-bench.
- Kernel registered: `env/install_kernel.sh` → `~/.local/share/jupyter/kernels/dslab` (only that directory
  added; sets PYTHONNOUSERSITE, LD_PRELOAD, HDF5_USE_FILE_LOCKING, PATH).

### Phase B setup

- `models/torch_base.py`: shared training for neural regressors: targets transformed (sqrt pr) and
  standardised per grid point with training statistics, MSE, AdamW + ReduceLROnPlateau, early stopping on val
  loss (best epoch kept), whole training split on the GPU, bf16 autocast; `state.pt` holds weights and
  normalisation.
- `deepesd`: CORDEX-ML-Bench reference CNN (conv 50-25-1 on the 16x16 grid + dense to 2x128x128), joint
  pr/tasmax, 8.44 M parameters; original recipe (Adam 1e-4, batch 100, patience 30).
- `unet`: residual U-Net (base 32, mults 1-2-4-8, 2 blocks/level, GroupNorm/SiLU), 9.84 M parameters,
  14.2 GFLOP per forward pass (CorrDiff-TW1: 333). Input: 15 predictors bilinearly interpolated to the true
  lat/lon of every target point (`grid_sample`, buffers stored in the model) + standardised orography.
  The U-Net takes an optional time embedding (FiLM), to be reused as the flow-matching backbone in Phase C.
- Jobs: `jobs/train.sbatch` (1 GPU, ≤ 1 h, fit + predict), `jobs/submit_model.sh` (train → evaluate chain).
- Smoke test 28579769 (gpu-devel, A100-PCIE-40GB, 2 epochs each, val only): all steps OK; U-Net ≈ 12.5 s
  per epoch, DeepESD < 0.5 s per epoch; 2 min 17 s wall (≈ 0.04 GPU-h). Smoke runs deleted afterwards.

| job | what | resources | wall | notes |
|---|---|---|---|---|
| 28579769 | GPU smoke test | gpu-devel, 1 GPU | 2 min 17 s | |
| 28579806 → 28579807 | DeepESD train+predict → evaluate | A100-SXM4-80GB → shared 16 CPUs | 3 min 09 s → 5 min 50 s | stopped at epoch 899, kept 869 (fit 112 s) |
| 28579808 → 28579810 | U-Net train+predict → evaluate | A100-SXM4-80GB → shared 16 CPUs | 8 min 22 s → 6 min 49 s | stopped at epoch 35, kept 26 (fit 419 s, ~12 s/epoch) |
| 28579811 + 3 reruns | notebook refresh | shared 8 CPUs | ~30 s each | |
| | **Phase B GPU total** | | | **≈ 0.23 GPU-h** (incl. smoke test) |

### Results, Phase B (ALPS ESD, perfect predictors)

| split | model | pr RMSE | pr Mean Bias | pr SDII | pr Rx1day | pr RALSD | pr LHD | tx RMSE | tx Mean Bias | tx TXx | tx RALSD | tx PSS | tx IAV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hist 1981–2000 | linear | 4.42 | 0.45 | 2.99 | 20.6 | 10.0 | 5.22 | 1.55 | 0.10 | 0.65 | 3.97 | 0.978 | 0.114 |
| | deepesd | 4.33 | 0.60 | 2.63 | 21.6 | 16.3 | 6.64 | 1.78 | 0.21 | 1.29 | 0.56 | 0.974 | 0.123 |
| | unet | **3.75** | 0.43 | 1.27 | 12.0 | 19.7 | 3.18 | 1.40 | 0.24 | 0.70 | 13.1 | 0.979 | 0.098 |
| | *DeepESD-IFCAv1-orog (pub.)* | 4.32 | 0.25 | 1.42 | 15.9 | 12.3 | 1.81 | 1.64 | 0.16 | 0.92 | 8.53 | 0.98 | 0.091 |
| | *DetUNet-orog (pub.)* | 4.47 | 0.43 | 1.42 | 11.7 | 9.16 | 0.51 | 1.41 | 0.35 | 2.00 | 11.9 | 0.97 | 0.22 |
| | *CorrDiff-TW1-orog (pub.)* | 3.77 | 0.62 | 1.88 | 17.5 | 3.78 | 2.03 | 2.58 | 2.10 | 2.77 | 0.24 | 0.89 | 0.13 |
| future (pooled) | deepesd | 4.74 | 0.63 | 3.00 | 25.5 | 17.4 | 8.81 | 1.88 | 0.26 | 2.10 | 1.12 | 0.967 | 0.079 |
| | unet | **4.15** | 0.46 | 1.45 | 14.4 | 21.3 | 4.13 | 1.59 | 0.46 | 2.24 | 9.15 | 0.951 | 0.103 |
| | *CorrDiff-TW1-orog (pub.)* | 4.14 | 0.67 | 2.16 | 20.0 | 4.54 | 2.07 | 2.83 | 2.44 | 3.63 | 0.34 | 0.86 | 0.11 |

Observations:
- **DeepESD replicates the published entry on RMSE** (pr 4.33 vs 4.32 hist, 4.74 vs 4.72 future); other
  scores differ (pr bias/SDII/LHD worse, tasmax RALSD much better) — loss / normalisation choices (ours: MSE on
  per-grid-point standardised sqrt(pr), joint pr+tasmax) vs. theirs (unknown in detail).
- **U-Net is the best regression model so far**: pr RMSE 3.75 is the third lowest of the 40 published ALPS ESD
  entries (Prithvi-UNet-orog 3.63, RCMGEM-mv-orog 3.74; CorrDiff-TW1 3.77); tasmax RMSE 1.40 vs. published best
  1.26 (RCMGEM), 1.30 (RCMFlow). *(Corrected 2026-10-30: an earlier version said "lowest of all published
  entries" after comparing against a shortlist only.)* It overfits quickly (train MSE 0.05 vs val 0.15 at stop) → room for regularisation
  / augmentation. Extrapolation of hot extremes is weaker than linear (future TXx bias 2.24 vs 1.11 K; mean bias
  0.46 vs 0.34 K): the non-linear model saturates outside the training climate.
- **RALSD is dominated by pixel-scale texture.** The published RALSD averages 90 radial bins of which 25 lie
  beyond the Nyquist wavenumber (grid-diagonal "corners"). Up to Nyquist the U-Net has the best spectra
  (tasmax 0.58, pr 9.7 dB²) but it loses 6–8 dB in the corner bins (45 dB² there), hence total 13.1 / 19.7.
  DeepESD's per-grid-point dense layer gives slightly too much corner power (+1–2 dB). The notebook now shows
  the full spectrum with the corner range shaded and a RALSD split table. Relevant for generative models: the
  metric rewards correct grid-scale texture.
- All deterministic models are 3–4 dB too smooth at 30–100 km for pr and underestimate extremes; that is the
  gap the generative models (Phase C) should close.

### Notebook: metric guide, captions, findings (user request)

The notebook now has a data/splits section, a "Metrics in brief" table (definition, units, ideal value, published
ALPS ESD range, interpretation), a caption at the top of every figure/table cell generated from the experiment
config (`results.caption`: what is shown, trained on, model selection on, evaluated on, runs) and a dated findings
note after each figure (snapshot 2026-10-30, to be updated when runs change), plus a rank table "number of
published entries that score better". Every number in the findings was checked against the stored diagnostics.

New finding while writing them: **all trained models are ~15% too dry** (domain-mean pr bias −0.39 to −0.55 mm/day
of 3.0; every month, strongest in autumn and over high terrain). Likely cause: MSE on sqrt(pr) estimates
E[sqrt(pr)], and squaring underestimates E[pr] (Jensen). Climatology (raw pr) has none. Candidate fixes: train on
raw pr / log1p with a variance correction, or generative models.

### Notebook: experiments introduced (user request)

New "Experiments" section: the two CORDEX-ML-Bench experiments (ESD pseudo-reality vs. Emulator; training periods,
tests, perfect/imperfect predictors, cross-validation/extrapolation/interpolation, GCM transferability, which tests
enter the published scorecard) with a config-generated status table (`results.benchmark_tests`), and the model
runs as a method ladder (what each model is, input → output, training cost, why it is on the ladder) with a
generated table (`results.run_info`, model `summary` attributes). All official tests are now configured in both
experiment YAMLs (`benchmark_tests`; ESD Tests 2, 4, 5 and Emulator Tests 2–6 not evaluated yet); `default_splits`
unchanged, so nothing new is predicted by default.

### Next (awaiting the user)

Phase C: conditional flow matching with the same U-Net backbone (time-conditioned `UNet`, already in
`networks.py`), (a) plain, (b) residual on top of the `unet` run (CorrDiff-style); ensembles of 10 members.
Estimate ≈ 1–1.5 GPU-h training + ≈ 0.5 GPU-h sampling per variant (to be measured on gpu-devel first).
