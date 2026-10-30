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

### Open / next (awaiting the user)

1. Register the JupyterHub kernel: `bash env/install_kernel.sh` (writes `~/.local/share/jupyter/kernels/dslab`).
2. Phase B: DeepESD and a small U-Net (GPU, ≈ 0.5 GPU-h), then flow-matching baselines (≈ 2–3 GPU-h).
3. First git commit (repo initialised, nothing committed yet).
