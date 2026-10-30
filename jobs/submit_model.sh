#!/bin/bash
# Train a model (GPU job) and evaluate it (CPU job, runs after successful training).
# Usage: bash jobs/submit_model.sh <experiment> <model> [run_name] [fit options]
# Prints the evaluation job id last, so callers can chain on it (e.g. jobs/notebook.sbatch).
set -euo pipefail
EXP=$1; MODEL=$2; NAME=${3:-$MODEL}; shift $(( $# >= 3 ? 3 : $# ))
cd "$HOME/downscaling-lab"
T=$(sbatch --parsable --job-name="train-$NAME" jobs/train.sbatch "$EXP" "$MODEL" "$NAME" "$@")
E=$(sbatch --parsable --job-name="eval-$NAME" --dependency=afterok:"$T" \
    jobs/dslab.sbatch evaluate --exp "$EXP" --run "$NAME" --overwrite)
echo "$NAME: train $T -> evaluate $E" >&2
echo "$E"
