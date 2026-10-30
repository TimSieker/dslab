#!/bin/bash
# Train a model (GPU job) and evaluate it (CPU job, runs after successful training).
# Usage: [TRAIN_TIME=01:00:00] [EVAL_CPUS=16] bash jobs/submit_model.sh <experiment> <model> [run_name] [fit options]
#   TRAIN_TIME  wall-time limit of the GPU job (GPU-hours charged at most = 1 GPU x TRAIN_TIME)
#   EVAL_CPUS   CPUs of the evaluation job on `shared` (memory = EVAL_CPUS x 940 MB); ensembles need more,
#               e.g. 40 for 12 members (pooled future split: ~11.5 GB per variable)
# Prints the evaluation job id last, so callers can chain on it (e.g. jobs/notebook.sbatch).
set -euo pipefail
EXP=$1; MODEL=$2; NAME=${3:-$MODEL}; shift $(( $# >= 3 ? 3 : $# ))
TRAIN_TIME=${TRAIN_TIME:-01:00:00}
EVAL_CPUS=${EVAL_CPUS:-16}
cd "$HOME/downscaling-lab"
T=$(sbatch --parsable --job-name="train-$NAME" --time="$TRAIN_TIME" jobs/train.sbatch "$EXP" "$MODEL" "$NAME" "$@")
E=$(sbatch --parsable --job-name="eval-$NAME" --cpus-per-task="$EVAL_CPUS" --time=02:00:00 \
    --dependency=afterok:"$T" jobs/dslab.sbatch evaluate --exp "$EXP" --run "$NAME" --overwrite)
echo "$NAME: train $T (limit $TRAIN_TIME) -> evaluate $E ($EVAL_CPUS CPUs)" >&2
echo "$E"
