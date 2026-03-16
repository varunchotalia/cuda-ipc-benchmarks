#!/bin/bash
#SBATCH --job-name=ipc_noaccum
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=ipc_noaccum_%j.out

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

SRC=transpose_cuda_ipc_noaccum.cu
MPI_INC=-I${MPI_HOME}/include
MPI_LIB="-L${MPI_HOME}/lib -lmpi"
ARCH="-gencode arch=compute_90,code=sm_90"

nvcc -O3 -lineinfo $ARCH -DCOMM_MODE=0 $MPI_INC $SRC -o ipc_direct_noaccum $MPI_LIB
nvcc -O3 -lineinfo $ARCH -DCOMM_MODE=1 $MPI_INC $SRC -o ipc_buffered_noaccum $MPI_LIB
nvcc -O3 -lineinfo $ARCH -DCOMM_MODE=2 $MPI_INC $SRC -o ipc_mpigpu_noaccum $MPI_LIB
nvcc -O3 -lineinfo $ARCH -DCOMM_MODE=3 $MPI_INC $SRC -o ipc_staged_noaccum $MPI_LIB

echo "=========================================="
echo "IPC No-Accumulation Benchmark - ${SLURM_NTASKS} GPUs"
echo "=========================================="

SIZES="1024 2048 4096 8192 16384"
ITERS=100

for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"

    echo "IPC Direct (no accum):"
    mpirun -np ${SLURM_NTASKS} ./ipc_direct_noaccum $ITERS $SIZE

    echo "IPC Buffered+Stream (no accum):"
    mpirun -np ${SLURM_NTASKS} ./ipc_buffered_noaccum $ITERS $SIZE

    echo "GPU-aware MPI (no accum):"
    mpirun -np ${SLURM_NTASKS} ./ipc_mpigpu_noaccum $ITERS $SIZE

    echo "Staged MPI (no accum):"
    mpirun -np ${SLURM_NTASKS} ./ipc_staged_noaccum $ITERS $SIZE
done

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
