#!/bin/bash
#SBATCH --job-name=dslab-gpu
#SBATCH --partition=gpu
#SBATCH --account=bb1596_gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100_80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3940M
#SBATCH --time=01:00:00
#SBATCH --output=/work/bb1596/b384340/downscaling-lab/runs/logs/%x-%j.out

# Run any dslab command on 1 A100-80GB, e.g.
#   sbatch --time=03:00:00 jobs/gpu.sbatch fit --exp alps_esd --model afm
#   sbatch --time=02:00:00 jobs/gpu.sbatch predict --exp alps_esd --run afm
# Cost: at most 1 GPU x wall-time limit. Ask the user before anything above 5 GPU-h in total.
# 16 CPUs x 3940 MB (gpu partition maximum per CPU) = 63 GB host memory (12-member predictions: ~12 GB).
set -euo pipefail
source "$HOME/downscaling-lab/jobs/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun dslab "$@"
