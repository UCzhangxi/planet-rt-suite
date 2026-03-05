#!/bin/bash
#SBATCH --job-name=1D_U_CPU
#SBATCH --partition=cpuq
#SBATCH --account=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

echo "=========================================================="
echo "Starting 1D CPU Uranus local dry run on Date: $(date)"
echo "Nodes assigned by Slurm: $SLURM_JOB_NODELIST"
echo "=========================================================="

# Optional restart file name inside ./output (leave empty for fresh run)
#RESTART_FILE="uranus_rt_local_dry_1d.00010.restart"
#RESTART_FILE=""

if [ -z "$RESTART_FILE" ]; then
    RESTART_ARG=""
else
    RESTART_ARG="--restart-name $RESTART_FILE"
fi

srun -N 1 --ntasks=1 --ntasks-per-node=1 \
  udocker run \
  --entrypoint="" \
  -v /home/xiz/data:/data \
  -v "$PWD":/workspace \
  -w /workspace \
  -e PYTHONPATH=/data/paddle \
  canoe_118 \
  bash -c "
    python ./run_planet_rt.py \
      $RESTART_ARG \
      --config=uranus_rt_local_dry_1d.yaml \
      --output-dir=./output
  "

echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="
