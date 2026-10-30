#!/bin/bash
# Write the fully pinned environment (conda + pip parts) to environment.yml.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=/home/b/b384340/envs/dslab
export MAMBA_ROOT_PREFIX=/scratch/b/b384340/caches/micromamba
micromamba env export -p "$PREFIX" --no-build > "$CODE/environment.yml"
echo "wrote $CODE/environment.yml"
