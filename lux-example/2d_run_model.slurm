#!/bin/bash
#SBATCH --job-name=2D_U
#SBATCH --partition=gpuq
#SBATCH --account=gpuq
#SBATCH --nodes=1               # Request 3 physical machines
#SBATCH --ntasks=2              # Request 6 total tasks (forces Lux to give 6 GPUs)
#SBATCH --cpus-per-task=20      # Give 20 CPUs per task (Lots of CPU power for NCCL/MPI)
#SBATCH --mem=0                 # Request all available RAM on the node
#SBATCH --exclusive             # Guarantee no other users are on these nodes!
#SBATCH --time=24:00:00         # Max time allowed in gpuq
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Exit if anything fails
set -e

echo "=========================================================="
echo "Starting 6-GPU Sub-Neptune Job on Date: $(date)"
echo "Nodes assigned by Slurm: $SLURM_JOB_NODELIST"
echo "=========================================================="

# 1. Dynamically find the Master Node IP/Hostname
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Master Node is: $MASTER_ADDR"

# 2. Launch the container across all nodes simultaneously
# Toggle this at the top of your Slurm script
#RESTART_FILE="sub_neptune_tidallock.day110_uniform_cooling.restart"
# To disable restart, just uncomment the line below:
# RESTART_FILE=""

# Build the restart argument string
if [ -z "$RESTART_FILE" ]; then
    RESTART_ARG=""
else
    RESTART_ARG="--restart-name $RESTART_FILE"
fi

srun -N 1 --ntasks=2 --ntasks-per-node=2 \
  udocker run \
  --entrypoint="" \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  --device=/dev/nvidia0 \
  --device=/dev/nvidia1 \
  -v /home/xiz/data:/data \
  -v "$PWD":/workspace \
  -w /workspace \
  -e SLURM_NODEID \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e PYTHONPATH=/data/paddle \
  canoe_118 \
  bash -c "
    echo 'Booting container on node rank: \$SLURM_NODEID'

    torchrun \
      --nnodes=1 \
      --nproc_per_node=2 \
      --node_rank=\$SLURM_NODEID \
      --master_addr=$MASTER_ADDR \
      --master_port=29501 \
      ./run_planet_rt.py \
        $RESTART_ARG \
        --config=uranus_rt_local_dry.yaml \
        --output-dir=./output > /workspace/node_\${SLURM_NODEID}.log 2>&1
  "

echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="
