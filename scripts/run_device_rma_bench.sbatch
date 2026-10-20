#!/bin/bash
#SBATCH --job-name=device_rma_bench
#SBATCH --partition=debug-h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/nvwulf/home/vchotalia/mpiwrap/logs/device_rma_bench_%j.out

# Overhead of MPIX_Put/MPIX_Get over raw pointer access, on the same device
# (np=1, so every rank targets its own window). See the header of
# mpi-intercept/bench_device_rma.cu for what raw / ct / rt mean.
#
# Two GPUs, target = the neighbour over NVLink:
#   sbatch -p h200x4 --ntasks=2 --gpus=2 scripts/run_device_rma_bench.sbatch

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

REPO=/lustre/nvwulf/home/vchotalia/mpiwrap
cd "$REPO"
export UCX_LOG_LEVEL=error

echo "node=$(hostname)  date=$(date -Is)  commit=$(git log -1 --format='%h %s')"
nvidia-smi -L
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv

source scripts/build_mpiwrap.sh || exit 1
MPIWRAP=$MPIWRAP_LIB

echo "--- building bench_device_rma ---"
nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
     -I mpi-intercept $(mpicc --showme:incdirs | xargs -n1 printf -- '-I%s ') \
     mpi-intercept/bench_device_rma.cu -o build/bench_device_rma \
     "$MPIWRAP" -Xlinker -rpath="$(dirname "$MPIWRAP")" \
     $(mpicc --showme:libdirs | xargs -n1 printf -- '-L%s ') -lmpi -lcuda || {
    echo "FATAL: bench build failed"; exit 1; }

LD_PRELOAD=$MPIWRAP mpirun -np "$SLURM_NTASKS" ./build/bench_device_rma
