# downscaling-lab

Small, cheap downscaling experiments on CORDEX-ML-Bench data (ALPS, ESD pseudo-reality first), to compare
baseline methods with each other, with the published benchmark entries and, later, with our own model.
Scores are computed exactly as in the benchmark's published scorecards (`docs/metric_definitions.md`).

## Layout

```
CODE  ~/downscaling-lab/                 git repo
  configs/experiments/   one YAML per experiment (alps_esd.yaml active, alps_emulator.yaml ready)
  configs/models/        one YAML per model configuration
  src/dslab/             package: experiment, data, preprocessing, models/, io, metrics, evaluate,
                         results, plotting, cli
  jobs/                  SLURM scripts (env.sh, dslab.sbatch, baselines.sbatch, notebook.sbatch)
  notebooks/             evaluation.ipynb
  tests/                 metric tests (incl. cross-check against the organisers' functions)
  docs/                  metric definitions, published reference scores
  env/                   environment spec, build / activate / kernel scripts
  NOTES.md               running log (decisions, jobs, results)
  work -> /work/bb1596/b384340/downscaling-lab

ENV   ~/envs/dslab                       conda env on HOME (DKRZ: Python envs on /home for fast imports)

WORK  /work/bb1596/b384340/downscaling-lab/
  cache/<domain>/        .npy copies of the NetCDF inputs (built on first use), cartopy map data
  runs/<exp>/<run>/      model.yaml, state, fit_info.json, run.yaml; runs/logs/ = SLURM logs
  predictions/<exp>/<run>/<split>.nc     (member, time, y, x), physical units
  eval/<exp>/<run>/<split>/{scores.json, diagnostics.nc};  eval/<exp>/figures/
```

Raw benchmark data are read from `/work/bb1596/b384340/cordex-ml-bench/data/benchmark/` (not copied).

## Usage

```bash
source ~/downscaling-lab/env/activate.sh          # login node: light work only
sbatch jobs/baselines.sbatch alps_esd             # climatology + linear: fit, predict, evaluate, tests
sbatch jobs/dslab.sbatch fit --exp alps_esd --model linear --name linear-test --set n_pcs=[400]
sbatch jobs/dslab.sbatch predict --exp alps_esd --run linear-test
sbatch jobs/dslab.sbatch evaluate --exp alps_esd --run linear-test
sbatch --time=03:00:00 jobs/gpu.sbatch fit --exp alps_esd --model afm   # any dslab command on 1 A100-80GB
sbatch jobs/notebook.sbatch                        # execute notebooks/evaluation.ipynb headless
```

Splits: `train` / `val` (training file without / with only the `val_years`), the test splits listed in the
experiment YAML, and evaluation-only `combined_splits` (ESD future = 2041–2060 + 2080–2099, as scored by the
benchmark). `benchmark_tests` maps each official CORDEX-ML-Bench test to the split implementing it (all tests of
both experiments are configured; `default_splits` decides what is predicted by default). Another experiment =
another YAML in `configs/experiments/`; nothing else changes.

## Where to run what (DKRZ Levante)

- **Login nodes**: editing, submitting, light serial work only. One user may use at most 10% of a login node's
  memory; processes beyond that are killed.
- **Batch jobs** (preferred): `shared` partition for CPU work. Memory is at most 940 MB per CPU there, and Slurm
  raises the CPU count (and the charge, CPU-seconds) to cover a larger `--mem`, so the scripts request
  `--cpus-per-task=N --mem-per-cpu=940M` explicitly (16 CPUs = 15 GB). GPU work: partition `gpu`, account
  `bb1596_gpu`, `--gpus=1`.
- **Interactive work**: `salloc -p interactive -A bb1596 -n 1 --mem=10G -t 120` (1940 MB per CPU, max 12 h), or
  JupyterHub with the *interactive* preset (5–100 GB). Use the **dslab** kernel (`bash env/install_kernel.sh`
  once; it writes `~/.local/share/jupyter/kernels/dslab`).

Sources: docs.dkrz.de › Levante › Running Jobs › Partitions and Limits; Data Processing on Levante;
Access and Environment; JupyterHub › Spawner options and Kernels.

## Adding a model

1. `src/dslab/models/<name>.py`: subclass `Downscaler`, decorate with `@register("<name>")`, implement
   `fit(train, val, grid)` and `predict(split, grid, n_members, seed)` returning `(member, time, var, y, x)` in
   physical units; persistence via `state_dict()` (arrays) or override `save_state` / `load_state` (torch).
2. Import it in `src/dslab/models/__init__.py`, add `configs/models/<name>.yaml`.
3. `dslab fit`, `dslab predict`, `dslab evaluate`; the notebook picks the run up automatically.

## Environment

`bash env/build_env.sh` builds `~/envs/dslab` from `env/conda-base.yml` + torch 2.10 (cu128) +
`env/pip-requirements.txt` (cordex-bench-eval for cross-checks, this package editable) and writes the pinned
`environment.yml`. Always use it via `env/activate.sh` / `jobs/env.sh` (`PYTHONNOUSERSITE=1`: `~/.local` holds
unrelated packages; `LD_PRELOAD` of the env's libstdc++ for torch).
