#!/bin/bash
#SBATCH --job-name=dslab-baselines
#SBATCH --partition=shared
#SBATCH --account=bb1596
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=940M
#SBATCH --time=01:00:00
#SBATCH --output=/work/bb1596/b384340/downscaling-lab/runs/logs/%x-%j.out

# Phase A: CPU baselines (climatology, linear) end to end for one experiment:
# fit -> predict default splits -> evaluate default eval splits -> metric tests.
# Usage: sbatch jobs/baselines.sbatch [experiment]   (default alps_esd)
# 16 CPUs x 940 MB = 15 GB (shared partition max 940 MB/CPU, see jobs/dslab.sbatch).
set -euo pipefail
EXP=${1:-alps_esd}
source "$HOME/downscaling-lab/jobs/env.sh"
dslab prepare --exp "$EXP"
for model in climatology linear; do
    dslab fit --exp "$EXP" --model "$model" --overwrite
    dslab predict --exp "$EXP" --run "$model"
done
dslab evaluate --exp "$EXP" --run climatology linear --overwrite
python -m pytest -q tests/
