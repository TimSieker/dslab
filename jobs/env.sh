# Common environment for all job scripts: source "$HOME/downscaling-lab/jobs/env.sh"
export CODE="$HOME/downscaling-lab"                       # code (HOME, backed up)
export WORK="/work/bb1596/b384340/downscaling-lab"        # cache, runs, predictions, eval (WORK)
export ENV_PREFIX="$HOME/envs/dslab"                      # HOME: fast imports (DKRZ advice for Python envs)
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1                                 # ignore ~/.local site-packages
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE                        # Lustre + netCDF reads from several processes
# pip's torch wheels resolve libstdc++.so.6 to the old system one (/lib64, RHEL 8, no GLIBCXX_3.4.29);
# if torch is imported before numpy this breaks conda-forge numpy. Preload the env's (backward-compatible) one.
export LD_PRELOAD="$ENV_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
cd "$CODE"
