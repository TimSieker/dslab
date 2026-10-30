#!/bin/bash
# Register the dslab environment as Jupyter kernel "dslab" for DKRZ JupyterHub
# (writes ~/.local/share/jupyter/kernels/dslab/kernel.json, as described in the DKRZ JupyterHub docs).
# The kernel sets the same variables as env/activate.sh.
set -euo pipefail
PREFIX=/home/b/b384340/envs/dslab
PYTHONNOUSERSITE=1 "$PREFIX/bin/python" -m ipykernel install --user --name dslab \
    --display-name "dslab (downscaling-lab)" \
    --env PYTHONNOUSERSITE 1 \
    --env HDF5_USE_FILE_LOCKING FALSE \
    --env LD_PRELOAD "$PREFIX/lib/libstdc++.so.6" \
    --env PATH "$PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
cat ~/.local/share/jupyter/kernels/dslab/kernel.json
