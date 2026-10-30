#!/bin/bash
#SBATCH --job-name=dslab-notebook
#SBATCH --partition=shared
#SBATCH --account=bb1596
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=940M
#SBATCH --time=00:30:00
#SBATCH --output=/work/bb1596/b384340/downscaling-lab/runs/logs/%x-%j.out

# Execute a notebook headless and store it with outputs (figures also go to work/eval/<exp>/figures/).
# Usage: sbatch jobs/notebook.sbatch [notebooks/evaluation.ipynb]
# 8 CPUs x 940 MB = 7.5 GB (the notebook only loads precomputed diagnostics).
set -euo pipefail
NB=${1:-notebooks/evaluation.ipynb}
source "$HOME/downscaling-lab/jobs/env.sh"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
    --ExecutePreprocessor.kernel_name=python3 "$NB"
echo "executed $NB"
