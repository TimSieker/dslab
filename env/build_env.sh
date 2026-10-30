#!/bin/bash
# Build the project environment in $HOME/envs (run on the login node). DKRZ recommends Python envs on
# /home rather than /work: many small files, much faster imports (docs.dkrz.de, JupyterHub > Kernels).
#   bash env/build_env.sh
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=/home/b/b384340/envs/dslab
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/scratch/b/b384340/caches/pip
export MAMBA_ROOT_PREFIX=/scratch/b/b384340/caches/micromamba
module load git

micromamba create -y -p "$PREFIX" -f "$CODE/env/conda-base.yml"
PY="$PREFIX/bin/python"
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
cd "$CODE"
"$PY" -m pip install -c env/pip-constraints.txt -r env/pip-requirements.txt
"$PY" -m pip check
bash "$CODE/env/export_env.sh"
