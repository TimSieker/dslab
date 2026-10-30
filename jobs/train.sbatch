#!/bin/bash
#SBATCH --job-name=dslab-train
#SBATCH --partition=gpu
#SBATCH --account=bb1596_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/work/bb1596/b384340/downscaling-lab/runs/logs/%x-%j.out

# Fit one model on 1 GPU and write predictions for the experiment's default splits.
# Usage: sbatch jobs/train.sbatch <experiment> <model> [run_name] [fit options, e.g. --set max_epochs=10]
# Cost: at most 1 GPU x 1 h = 1 GPU-h (ask the user before anything above 5 GPU-h in total).
# 16 CPUs x 1940 MB (gpu partition default) = 31 GB host memory.
set -euo pipefail
EXP=$1; MODEL=$2; NAME=${3:-$MODEL}; shift $(( $# >= 3 ? 3 : $# ))
source "$HOME/downscaling-lab/jobs/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
dslab fit --exp "$EXP" --model "$MODEL" --name "$NAME" "$@"
dslab predict --exp "$EXP" --run "$NAME"
