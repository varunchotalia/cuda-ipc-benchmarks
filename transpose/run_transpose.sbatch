#!/bin/bash
#SBATCH --job-name=transpose_bench
#SBATCH --partition=debug-h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=transpose_bench_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0

make clean
make all GPU_ARCH="-gencode arch=compute_90,code=sm_90"

echo "=========================================="
echo "Transpose Benchmark - ${SLURM_NTASKS} GPUs"
echo "=========================================="

SIZES="1024 2048 4096 8192 16384"
ITERS=100

for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"

    echo "IPC Direct:"
    mpirun -np ${SLURM_NTASKS} ./transpose_direct $ITERS $SIZE

    echo "IPC Buffered+Stream:"
    mpirun -np ${SLURM_NTASKS} ./transpose_buffered $ITERS $SIZE

    echo "GPU-aware MPI:"
    mpirun -np ${SLURM_NTASKS} ./transpose_mpigpu $ITERS $SIZE

    echo "Staged MPI:"
    mpirun -np ${SLURM_NTASKS} ./transpose_staged $ITERS $SIZE
done

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
