#!/bin/bash
#SBATCH --job-name=nvshmem_bench
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=nvshmem_bench_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0

export NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem

# Add NVSHMEM libs to runtime path
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# NVSHMEM environment variables
export NVSHMEM_BOOTSTRAP=MPI

# Use the nvshmem-specific makefile
cp Makefile_nvshmem Makefile_nvshmem_build
make -f Makefile_nvshmem_build clean
make -f Makefile_nvshmem_build all

echo "=========================================="
echo "NVSHMEM Transpose Benchmark - ${SLURM_NTASKS} GPUs"
echo "=========================================="

SIZES="1024 2048 4096 8192 16384"
ITERS=100

for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"

    echo "NVSHMEM Direct:"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_direct $ITERS $SIZE

    echo "NVSHMEM Buffered:"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_buffered $ITERS $SIZE
done

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
