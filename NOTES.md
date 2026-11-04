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

## 2026-10-30: Phase C start, adaptive flow matching (AFM)

User: commit (done: `e428f91`); implement the flow-matching algorithm of Debeire et al. (2026, arXiv:2604.03459)
with the same loss but **without the conservation loss** (their "AFM-baseline"); notebook: show only `unet`
(deterministic baseline) and the FM model; ask before heavy compute.

- `src/dslab/models/afm.py` (paper → code mapping in the docstring), `configs/models/afm.yaml`,
  `tests/test_afm.py` (loss identity w(σ)‖D−x1‖² = ‖F−target‖², tiny CPU training, sampling, save/load: pass).
- Details taken from the original AFM paper (Fotiadis et al. 2025, arXiv:2410.19814) where Debeire et al. defer
  to it: σ_max = EMA of the encoder's RMS error (ML noise estimate), per channel, init 1.0; flow gradients reach
  the encoder; SongUNet-L = 32 channels, mult [1,2,2,4,4], ~12 M parameters (ours: 11.9 M each), attention at
  the lowest resolution, dropout 0.13.
- Choices (not fixed by the papers): statics orography + latitude (no land-sea mask in the benchmark); scalar
  noise label ln(s·RMS(σ_max))/4 (channels share t); σ_min relative to RMS(σ_max); σ_max EMA factor 0.01 per
  micro-batch; batches drawn with replacement; EDM-style EMA ramp-up (0.05); pr clipped at 0 after sampling.
- Environment: + nvidia-physicsnemo 2.1.1 (SongUNet), torchvision 0.25.0+cu128; `pip check` clean.
- Evaluation: memory-safe for ensembles (pre-allocated loading, chunked spread/skill), + MCB.

| job | what | resources | wall | notes |
|---|---|---|---|---|
| 28582366 | AFM smoke test: 51,200 samples, val prediction 12 members × 50 steps, evaluation | 1 A100-80GB | 18 min | pipeline OK; 145 samples/s; 662 denoiser evals/s; 38.5 GB GPU |
| 28582579 | throughput benchmark (`scripts/bench_afm.py`) | 1 A100-80GB | 25 min | see below |

Benchmark (SongUNet-L ×2, 128×128, bf16): training 152 samples/s (micro-batch 64) → **273/s with
`torch.compile`** (batch 256 in one pass, 65 GB); sampling 714 → **2,300 denoiser evaluations/s** with
`torch.compile` + channels-last. Estimates with compile: 2 M training samples ≈ 2.1 h, 6 M ≈ 6.1 h; prediction of
val + 3 test splits with 12 members × 50 steps (13.6 M evaluations) ≈ 1.6–1.8 h. GPU used so far in Phase C ≈ 0.7 GPU-h.

### Status at end of day (2026-10-30) — next steps

Nothing running. Awaiting the user's choice of AFM run (estimates with `torch.compile`, measured on A100-80GB):
1. 2 M samples, 12 members everywhere ≈ 4 GPU-h (recommended); 2. 2 M, 5 members for the future splits ≈ 3.2 GPU-h;
3. 6 M samples (paper length) ≈ 8 GPU-h (> 5 GPU-h threshold). Also offered: checkpoint/resume to extend 2 M → 6 M.
To do before the real run: add `torch.compile` (+ channels-last for sampling) to `afm.py` and re-test briefly;
run with `TRAIN_TIME=... EVAL_CPUS=40 bash jobs/submit_model.sh alps_esd afm`. Then apply
`notebooks/build_evaluation_notebook.py` (runs unet + afm), execute, and rewrite the findings notes.
Smoke-test artefacts `afm-smoke` (runs/, predictions/, eval/; val only, 1.1 GB) can be deleted.
The AFM code and the env change (PhysicsNeMo) are not committed yet.

## 2026-11-02: AFM production run (user: 2 M samples, 12 members val/historical, 5 members future)

- `torch.compile` added (`AFMNet.use_compiled`; training NCHW, sampling channels-last, static shapes, last
  sampling batch padded), micro-batch 256 (whole batch, A100-80GB), `members_per_split` in `afm.yaml`,
  `jobs/gpu.sbatch` (generic dslab command on 1 A100-80GB, 63 GB host memory).
- Check 28651163 (13.5 min, 0.23 GPU-h): compile ≈ 6 min once per job; training 260 samples/s incl. validation;
  62 GB GPU; compiled vs. eager samples with the same seed differ by 0.017 mm/day (pr) / 0.013 K (tasmax) on
  average (field std 4.8 / 6.8): bf16 rounding amplified over 50 steps, equivalent.
- Chain: fit 28651465 (limit 3 h, expected ~2.3 h) → predict 28651466 (limit 2 h, expected ~1.4 h) →
  evaluate 28651467 (shared, 40 CPUs). GPU ≤ 5 GPU-h by limits, expected ≈ 3.6 GPU-h.

| job | what | resources | wall | notes |
|---|---|---|---|---|
| 28651465 | AFM fit, 2.0 M samples (304 passes) | 1 A100-80GB | 2 h 11 min | 268 samples/s; best val at 1.1 M samples (0.522), overfits after (2.0 M: 0.553) |
| 28651466 | AFM predict: val, historical (12 members), mid/end century (5 members) | 1 A100-80GB | 1 h 18 min | 50 Euler steps |
| 28651467 | evaluate afm, 5 splits | shared, 40 CPUs | 16 min | |
| 28658773, 28659014 | notebook (unet + afm) | shared, 8 CPUs | ~3 min each | |

GPU used in this run 3.5 GPU-h; Phase C total ≈ 4.4 GPU-h (smoke 0.3, benchmark 0.4, compile check 0.23, run 3.5).

### Results, AFM vs. U-Net (ALPS ESD, perfect predictors)

Historical test (1981–2000; published rank = number of the 40 entries that score better):

| | pr RMSE | pr Mean Bias | SDII | Rx1day | pr RALSD | LHD | tx RMSE | tx Mean Bias | TXx | tx RALSD | PSS | IAV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unet | 3.75 (2) | 0.43 (21) | 1.27 (15) | 12.0 (20) | 19.7 (26) | 3.18 (36) | 1.40 (6) | 0.24 (17) | 0.70 (4) | 13.1 (33) | 0.979 (16) | 0.098 (5) |
| afm | 3.83 (4) | 0.24 (14) | 0.52 (2) | 5.90 (1) | 0.81 (9) | 2.15 (31) | 1.46 (8) | 0.25 (17) | 0.52 (1) | 0.18 (2) | 0.974 (16) | 0.111 (7) |

- AFM better on 7/12 (all realism metrics + pr Mean Bias); U-Net keeps RMSE. Only RCMGEM-mv-orog beats the AFM on
  Rx1day and TXx; RCMGEM-mv-orog is ahead of the AFM on all 12 metrics (IAV tie). CRPS 23–24% lower than U-Net.
- Dry bias gone: domain-mean pr bias −0.09 (−3%) vs. −0.39 mm/day (U-Net, sqrt-MSE Jensen bias).
- Weak spots: (1) pr LHD: histogram within 0.5 dB up to 100 mm/day but too few values > 150 mm/day (member max 233
  vs. 368 mm/day); (2) ensemble under-dispersive, spread/skill ≈ 0.6, members differ in small-scale detail only;
  (3) +0.21 K warm bias (winter +0.6 K); (4) slight drizzle frame in the outermost 2 grid rows (edge effect);
  (5) more trace rain 0.1–1 mm/day (dry fraction 40% vs. 49%).
- **Future extrapolation (tasmax) is worse than the U-Net:** 2080–2099 RMSE 2.23 vs. 1.70 K; July/August −2.8/−3.2 K
  (U-Net −1.4/−1.6), hottest 10% of days −3.9 K (U-Net −1.9); warming 2.50 vs. 2.72 vs. 2.97 K (truth); truth above
  all 5 members in 33% of cases. Precipitation keeps its advantages in the future (Rx1day 7.3 mm/day: better than
  all 40 published entries), but the AFM over-dries the south (pr change +1.0% vs. +2.1% truth).
- Training: validation loss rises after 1.1 M samples, so the planned extension to 6 M samples is not worthwhile
  without more data/regularisation.

Notebook regenerated (`notebooks/build_evaluation_notebook.py`, runs unet + afm, findings rewritten and checked
against scores.json / diagnostics.nc). Plotting: `afm` gets the purple slot (green was hard to tell from the
U-Net's teal); log-axis tick labels of the training curves fixed.

Ideas, not started: anomaly/residual target for tasmax relative to the large-scale predictors (extrapolation);
spatially correlated noise or larger σ_max for spread at large scales; Emulator experiment (trains on 2080–2099).

### Notebook: predictors and downscaled variables (user request, 2026-11-02)

New subsection under "Experiments": `results.variables_table(exp)` lists predictors (grouped by quantity with
pressure levels and channel names), static fields and target variables with long names, units and grids, read from
the config and the training files' metadata (works for alps_emulator too). A short table below says how `unet` and
`afm` use them (input channels, target transform/normalisation).

## 2026-11-03: LC-AFM (law-conditioned AFM), from the user's plan "pcAFM: from Burgers to 12 km"

User decisions: keep joint training (encoder + denoiser, as AFM); add the unfiltered lapse-rate law;
implement the conditioning in dslab; new name (PC-AFM clashes with Debeire et al.) -> **LC-AFM**, law-conditioned
adaptive flow matching (no clash found in a web search). Stage 1 of the plan: laws T and O (benchmark data only).

- `src/dslab/laws.py`: coarse operator (A block means, U bilinear, Q orthonormal basis of span U, P = Q Q^T),
  laws T (H tasmax ≈ H T_free + c), Tfull (tasmax ≈ T_free + c), O (H pr ≈ H P_SB(h') + c; Smith & Barstad 2004 on
  49 overlapping 32x32 patches with 700 hPa wind, N_m from t/z 850-700, C_w from q/t 850; H_w, tau_c, tau_f and a
  gain fitted by grid search on 365 training days); offsets c and tolerances tau per calendar month and grid point
  from training truth (truth scores 1); caches in work/cache/laws/.
- `src/dslab/models/lcafm.py`: `LawConditioning` = exact Gaussian posterior (mean + Matheron noise) of the prior
  N(E, sigma_max^2 I) given the law; Tfull elementwise, T/O by Woodbury with an r x r (r = 92) positive-definite
  core, Cholesky per month; differentiable in E (joint training). `LCAFM(AFM)` registered as `lcafm`; AFM got
  hooks `_aux` / `_base` (white AFM numerically unchanged), `base_only` sampling.
- Fix on the way: MKL's multithreaded batched LU (indefinite 2r x 2r core with U A) returned invalid pivots on a
  24-CPU compute node (job 28686232 failed); switched H to the orthogonal projector (laws VERSION 2), which makes
  the core SPD. Test added (1 vs 24 threads identical).
- CLI: `dslab predict --base-only` (writes `<run>-base`); evaluate: `Law <n>` residuals (member 0 and truth) and
  `Elev. gradient bias` (pr). Config `configs/models/lcafm.yaml` (= afm.yaml + laws). Tests: tests/test_lcafm.py.
- First law fit (VERSION 1, train): T residual 1.08 K vs 2.58 K sub-2° tasmax; Tfull 2.70 K; O explains only
  6.4% of the sub-2° pr variance (gain 0.078, H_w 1 km, tau_c = tau_f = 2000 s).
- Go/no-go (job 28687871, CPU 19 min; `scripts/law_check.py`, results work/eval/alps_esd/laws/law_check.csv;
  VERSION 2 laws: T tolerance 0.99 K RMS vs 2.43 K sub-2° tasmax, Tfull 2.70 K, O explains 6.4% of sub-2° pr).
  Residuals in tolerance units (truth 1.03-1.13 on val, 1.23-1.34 in 2080-2099): white base T 22.1, Tfull 2.3,
  O 3.8 -> all "go". Conditioning the trained afm encoder on Tfull lowers the 2080-2099 tasmax RMSE 2.57 -> 2.23 K
  and the JJA bias -0.87 -> -0.28 K (val 2.39 -> 2.14 K): the extrapolation lever. T only fixes fine scales
  (residual 22 -> 0.89, bias unchanged). O is weak: mostly shrinks sub-2° noise, pr mean RMSE 4.62 -> 4.70.
  Base | Tfull still has white fine-scale noise (law T 13.2): combining T and Tfull would need a 2r indefinite core
  (LU on GPU) or CG; not implemented.

### Laws notebook (user request, 2026-11-03): do the laws hold on the truth?

`dslab laws --exp alps_esd` (src/dslab/law_diagnostics.py; CPU job 28688882, ~15 min) -> work/eval/alps_esd/laws/
diagnostics.nc; `notebooks/laws.ipynb` (builder notebooks/build_laws_notebook.py, findings inline) plots skill beyond
the training climatology, tolerance vs signal vs prior noise, calibration (mean z², α), z tails, drift, GCM predictors.
- T: no day-to-day skill beyond climatology (−2%); acts as a fine-scale climatological prior with weight 0.87.
- Tfull: 37–40% skill today, 70% in 2080–2099; α 1.03 / 1.17 / 1.84 (today / 2080–99 / GCM predictors);
  JJA 2080–99 drift +0.06 K over land, −1.56 K over sea; GCM predictors: −1.0 K bias (−2.5 K in July).
- O: 3% skill, heavy-tailed peaked residuals (|z|>4: 0.93% vs 0.006% Gaussian); w = 0.69.
- Recommendation in the notebook: keep Tfull (α = 1 for the perfect tests), T optional, drop O for the first arms.

### 2026-11-03 (cont.): GPU arms on hold; candidate laws for joint downscaling (user request)

User: hold the GPU training; drop T and O, keep Tfull; find alternatives, including laws coupling extra downscaled
variables (u, v, humidity, ...), as Debeire et al. downscale several variables; extract the data; describe each law's
exact form and its estimate from the available data. Deleted (user OK): v1 law caches (1.4 GB), afm-smoke and
lcafm-smoke outputs.
- `dslab extract --exp alps_esd` (src/dslab/extra.py; `extra:` section in the experiment YAMLs): same ALADIN63 run
  from the DKRZ CORDEX pool, native 453x453 Lambert grid, benchmark window x 165:293, y 113:241; periods 1961-80,
  1981-2000, 2041-60, 2080-99; 13 daily fields (ua850, va850, zg500, huss, ps, tas, tasmin, evspsbl; uas, vas, quas,
  qvas, upslope_pos from 6-hourly uas/vas and 3-hourly huss); 23 GB in work/cache/ALPS/extra/. Job 28690373,
  48 CPUs, 7.5 min. Pool pr == benchmark pr exactly (first year of each period). Pitfall: "latest" is a link to the
  version directory (files were read twice) -> one version directory only. Not in the pool: prw, ta850, hus850,
  any 700/500 hPa wind or humidity.
- `src/dslab/candidate_laws.py` (`dslab laws --candidates`): Tfull_tas, Tfull_tasmin, D (tas vs (tasmax+tasmin)/2),
  U_lin / U_pos (terrain-forced ascent, linear / 6-hourly upslope), M (two-layer moisture budget), CC / CC0
  (Clausius-Clapeyron scaling of wet-day ln P), plus tables for C (coarse consistency) and G (geostrophy at 2°).
  Oracle tests: co-generated fields from the RCM. law_diagnostics refactored (SplitStats, masks; reproduces the
  stored diagnostics exactly).
- First numbers (train/val): D skill 0.97 (τ 0.50 K); Tfull_tas 0.20; Tfull_tasmin -0.05; CC -0.05 day-to-day.
  C with perfect predictors exact (relative RMSE 0.3-0.7%, corr 1.000, missing 850 hPa ignored like the organisers'
  remap); with GCM predictors relative RMSE 64-79%, z500 bias -28 m. G at 2°: ageostrophic fraction 0.19 (700/500
  hPa), 0.28 (850 hPa), unchanged in 2080-99.
- Candidate results (job 28690739, 23 min; variants job 28691393): skill beyond monthly climatology, oracle fields:
  D 0.97-0.99; Tfull_tas 0.13-0.20 today, 0.64 in 2080-99; Tfull_tasmin -0.05/-0.12 today, 0.54 in 2080-99; U_lin
  0.015, U_pos 0.05, M -0.04 (fixed E = 1 hurts; fitted E coefficient 0.04), CC -0.05 to -0.11; at ~40 km (both
  sides Gaussian sigma 3 px): U_pos 0.18 today / 0.13 2080-99, M 0.14 / 0.10. Precip residuals strongly skewed
  (|z|>4 ~1%). JJA 2080-99: truth 0.4 mm/day (land) to 1 mm/day (>1000 m) drier than U/M. Mean wet-day intensity
  +7.4% (2080-99) vs +29.8% CC-implied at coarse t850 (ratio 0.27). Conclusion in the notebook: no linear
  precipitation law worth conditioning on; C and D exact for jointly generated fields; Tfull for temperature.
- Notebook laws.ipynb part 2 (sections 6-8 + summary); figures c_*.png in work/eval/alps_esd/laws/figures/.

### 2026-11-03 (cont.): law analysis reworked (user: three criteria; counterparts; moisture; primitive equations)

User points: (1) white noise destroys any structure, so conditioning the noise is valuable even for laws without daily
information -> criterion (b) "does it shape the noise" (white-base go/no-go of the plan) added next to (a) holds and
(c) daily information; (2) residuals need not be Gaussian (flow learns the rest; posterior mean = best linear estimate
given correct second moments) -> notebook corrected; (3) pr / temperature need 2° counterparts; (4) moisture;
(5) primitive equations (Coti Zelati et al. 2015, Nonlinearity 28, 625; arXiv 1406.3165). No model changes (user).
- Counterparts: the benchmark's predictors contain no coarse pr/tas/tasmax/huss (perfect prognosis). Constructed:
  RCM 2° area mean (perfect) and the GCM's daily pr, tas from /work/kd0956/CMIP5_waterpark (CNRM-CM5, only pr and tas
  daily), conservatively regridded to the 2° grid (`dslab extract --gcm`, work/cache/ALPS/extra_gcm/).
  `candidate_laws.coarse_counterparts` -> counterparts.csv.
- New candidate laws: Qfull (huss vs coarse q profile at the elevation), Hyd (hydrostatic z500 from ps, tas, huss and
  coarse t/q profile, 12 km), C_u850/C_v850/C_z500 in the generic framework (piecewise constant per full 2° cell).
- Criterion (b): ms_z_white (truth + white noise at a conservative proxy: RMS of the sub-2° daily anomaly per field)
  and ms_z_afmbase (trained AFM's white base, member 0) in diagnostics.nc and candidates.nc. Criterion (c):
  scripts/law_value.py (AFM encoder E conditioned on T/Tfull/O and on the pointwise pr laws) -> law_value.csv.
- `src/dslab/primitive.py` (`dslab laws --primitive`): at 2° hydrostatic thickness, momentum balance, 850-500 hPa
  humidity budget vs 2° precipitation -> primitive.csv. First numbers: hydrostatic residual 10-12% of thickness
  anomalies with T_v (18% with T); geostrophic residual 19% of the pressure gradient at 700/500 hPa, 27% at 850,
  15/25% with tendency + advection; humidity layer budget vs 2° pr: corr of daily anomalies 0.26 (perfect), -0.02 (GCM).
- Notebook laws.ipynb rebuilt from scratch around criteria (a)-(c), counterparts, primitive equations.
- Fix (analysis code only): `coarse_counterparts` subtracted the climatology per point instead of per cell, so the
  static sub-cell pattern counted as unpinned (tasmin came out negative). Now: share of the daily variability about
  the period's own climatology, plus share of the climate-change signal (period minus training climatology).
- Results (all in laws.ipynb findings, numbers from diagnostics.nc, candidates.nc, counterparts.csv, law_value.csv,
  primitive.csv):
  - Counterparts (exact 2° law, perfect): pins 61-62% of daily pr variability, 87-92% of the temperatures, 90% huss,
    93-94% 850 hPa wind, 99% z500; 74% (pr) / 99% (temperatures) of the 2080-2099 change. GCM counterpart poor: pr
    0.6-0.7 mm/day drier, relative RMSE 1.0, anomaly corr 0.38; tas +0.5-0.7 K, anomaly corr 0.71-0.74.
  - (a) all laws mean z² 0.99-1.13 on unseen years; 2080-2099: C ~1, D 1.05, Hyd 1.16, Tfull 1.36, Qfull 1.89, pr laws
    1.34-1.39. GCM predictors: C 7e4-1.3e6, Hyd 120, Qfull 4.3, Tfull 3.4. Tfull summer drift 2080-2099: land +0.10 K,
    sea -1.57 K.
  - (b) white base / truth (proxy): C_u/v 21, D 10.5, T 4.6, O 3.5, C_z500 3.3, Hyd 2.2, pr 12 km 1.7-2.0, Tfull 1.26;
    AFM base: T 22, Tfull 2.4, O 2.0, pointwise pr laws 0.87-0.89 (< 1: tolerance wider than the base noise).
  - (c) skill: Tfull 0.37 -> 0.70, Qfull 0.30 -> 0.56, pr laws <= 0.05 (12 km), <= 0.18 (40 km). Conditioned AFM
    encoder mean: Tfull tasmax RMSE -11% (-13% end century, JJA bias -0.87 -> -0.28 K), T -2.5%; every pr law worsens
    pr (O +2%, U/M +8%).
  - Primitive: hydrostatic at 2° 10-12% (T_v), momentum residual 14-16% / 24-25% of PGF with tendency + advection,
    humidity layer budget corr 0.25 (mean convergence only 14% of mean pr: moisture enters below 850 hPa). Only Hyd is
    usable at 12 km (nonlinear -> linearise); the archive has no 12 km humidity on levels, no prw, no ta850.
- Recommendations (to discuss with the user): Tfull + T for tasmax (needs J = [I; H] on one channel, small
  implementation extension); pr needs a coarse counterpart (leaves the benchmark protocol) or joint generation;
  co-generated ua850/va850/zg500 (C), tas/tasmin (D), zg500/ps/huss (Hyd); widen tolerances for GCM predictors.
- Plot fixes: scorecard (group headings, axis labels), calibration (capped axis). Stale figures from the earlier
  notebook version remain in work/eval/alps_esd/laws/figures/ (c_*.png, calibration_monthly.png).
