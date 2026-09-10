#!/bin/bash
#SBATCH --job-name=island_p2p_test
#SBATCH --partition=h200x8
#SBATCH --nodelist=h200x8-04
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=00:10:00
#SBATCH --output=island_p2p_test_%j.out

# Minimal reproducer for the cross-island silent write loss. See the header of
# mpi-intercept/test_island_p2p.cu. Raw CUDA IPC only -- no interposer -- so a
# failure here is CUDA behaviour on this topology rather than a WinIPC bug.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ==="
echo ""
echo "### NVLink topology ###"
nvidia-smi topo -m 2>&1 | head -12

nvcc -O3 -gencode arch=compute_90,code=sm_90 mpi-intercept/test_island_p2p.cu \
     -o /tmp/island_p2p -lmpi -I$MPI_HOME/include -L$MPI_HOME/lib \
     || { echo "BUILD FAILED"; exit 1; }

echo ""
echo "### pair matrix, 8 ranks ###"
mpirun -np 8 /tmp/island_p2p 2>&1 | sort

echo ""
echo "=== READ THIS ==="
echo "=== Pairs within an island (both dev 0-3, or both dev 4-7) should be"
echo "===   OPEN=1 CAN=1 LAND=1."
echo "=== Pairs across the boundary decide it:"
echo "===   OPEN=1 CAN=1 LAND=0 -> the write is silently lost while every"
echo "===     available check says the peer is fine. cudaDeviceCanAccessPeer"
echo "===     cannot be the gate; we need a delivery probe at window setup."
echo "===   OPEN=0 -> the open does fail after all, and the interposer should"
echo "===     simply be reporting that failure."
echo "===   LAND=1 -> raw IPC is fine and the fault is ours, in the stencil"
echo "===     or the interposer, not in the platform."
