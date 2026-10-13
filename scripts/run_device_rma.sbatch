#!/bin/bash
#SBATCH --job-name=device_rma
#SBATCH --partition=debug-h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/nvwulf/home/vchotalia/mpiwrap/logs/device_rma_%j.out

# Device-initiated Put/Get through MPIX_Win_get_device_handle
# (mpi-intercept/test_device_rma.cu). Builds the interposer and the test
# first, so a compile error shows up in the first minute.
#
# Default: one H200 node, np=2 and np=4.
#   sbatch scripts/run_device_rma.sbatch
# NVLink islands (GPUs 0-3 | 4-7): ranks 3 and 7 cannot reach their target,
# so their puts must come back MPI_ERR_OTHER (reported as SKIP):
#   sbatch -p h200x8 -w h200x8-04 --ntasks=8 --gpus=8 scripts/run_device_rma.sbatch

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

REPO=/lustre/nvwulf/home/vchotalia/mpiwrap
cd "$REPO"
export UCX_LOG_LEVEL=error   # quieter logs; does not change transport choice

echo "node=$(hostname)  date=$(date -Is)  commit=$(git log -1 --format='%h %s')"
nvidia-smi -L

source scripts/build_mpiwrap.sh || exit 1
MPIWRAP=$MPIWRAP_LIB

echo "--- building test_device_rma ---"
nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
     -I mpi-intercept $(mpicc --showme:incdirs | xargs -n1 printf -- '-I%s ') \
     mpi-intercept/test_device_rma.cu -o build/test_device_rma \
     "$MPIWRAP" -Xlinker -rpath="$(dirname "$MPIWRAP")" \
     $(mpicc --showme:libdirs | xargs -n1 printf -- '-L%s ') -lmpi -lcuda || {
    echo "FATAL: test build failed"; exit 1; }

for NP in $(echo 2 "$SLURM_NTASKS" | tr ' ' '\n' | sort -nu); do
    echo ""
    echo "=== np=$NP ==="
    LD_PRELOAD=$MPIWRAP mpirun -np "$NP" ./build/test_device_rma
done
