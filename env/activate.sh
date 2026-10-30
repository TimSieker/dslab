# Interactive/login-node use: source "$HOME/downscaling-lab/env/activate.sh"
# (Batch jobs use jobs/env.sh, which sets the same variables.)
# PYTHONNOUSERSITE=1 is essential: ~/.local/lib/python3.12/site-packages holds unrelated user packages
# that pip/python would otherwise see and even uninstall/upgrade.
export ENV_PREFIX="/home/b/b384340/envs/dslab"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/scratch/b/b384340/caches/pip
export MAMBA_ROOT_PREFIX=/scratch/b/b384340/caches/micromamba
export HDF5_USE_FILE_LOCKING=FALSE
# Login node: torch/numpy would otherwise use every core.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# pip's torch wheels resolve libstdc++.so.6 to the old system one (/lib64, RHEL 8, no GLIBCXX_3.4.29);
# if torch is imported before numpy this breaks conda-forge numpy. Preload the env's (backward-compatible) one.
export LD_PRELOAD="$ENV_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
