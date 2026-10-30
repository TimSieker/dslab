#!/bin/bash
#SBATCH --job-name=dslab
#SBATCH --partition=shared
#SBATCH --account=bb1596
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=940M
#SBATCH --time=01:00:00
#SBATCH --output=/work/bb1596/b384340/downscaling-lab/runs/logs/%x-%j.out

# Run any dslab command as a CPU batch job, e.g.
#   sbatch jobs/dslab.sbatch fit --exp alps_esd --model linear
# Sizing (DKRZ "Partitions and Limits"): the shared partition gives at most 940 MB per CPU and
# Slurm raises the CPU count (and the charge) to cover larger --mem requests, so memory is
# requested as CPUs x 940 MB: 16 CPUs = 15 GB. Override on the command line, e.g.
#   sbatch --cpus-per-task=32 jobs/dslab.sbatch evaluate --exp alps_esd
set -euo pipefail
source "$HOME/downscaling-lab/jobs/env.sh"
srun dslab "$@"
