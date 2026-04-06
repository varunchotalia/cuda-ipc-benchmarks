#!/bin/bash
#SBATCH --job-name=nvshmem_all
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=nvshmem_all_%j.out

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI

SRC=transpose/transpose_nvshmem.cu
INC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
LIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
ARCH="-gencode arch=compute_90,code=sm_90"

# Build all variants
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=0 -DACCUMULATE=1 $INC $SRC $LIB -o nvshmem_direct
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=0 -DACCUMULATE=1 -DSINGLE_KERNEL=1 $INC $SRC $LIB -o nvshmem_direct_single
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=1 -DACCUMULATE=1 $INC $SRC $LIB -o nvshmem_buffered
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=0 -DACCUMULATE=0 $INC $SRC $LIB -o nvshmem_direct_noaccum
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=0 -DACCUMULATE=0 -DSINGLE_KERNEL=1 $INC $SRC $LIB -o nvshmem_direct_single_noaccum
nvcc -O3 -lineinfo -rdc=true $ARCH -DCOMM_MODE=1 -DACCUMULATE=0 $INC $SRC $LIB -o nvshmem_buffered_noaccum

ITERS=100
SIZES="1024 2048 4096 8192 16384"

echo "=========================================="
echo "NVSHMEM Transpose — All Modes — ${SLURM_NTASKS} GPUs"
echo "=========================================="

echo ""
echo "############## WITH ACCUMULATION ##############"
for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"
    echo "NVSHMEM Direct (per-phase):"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_direct $ITERS $SIZE
    echo "NVSHMEM Direct (single-kernel):"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_direct_single $ITERS $SIZE
    echo "NVSHMEM Buffered:"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_buffered $ITERS $SIZE
done

echo ""
echo "############## WITHOUT ACCUMULATION ##############"
for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"
    echo "NVSHMEM Direct (per-phase, no accum):"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_direct_noaccum $ITERS $SIZE
    echo "NVSHMEM Direct (single-kernel, no accum):"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_direct_single_noaccum $ITERS $SIZE
    echo "NVSHMEM Buffered (no accum):"
    mpirun -np ${SLURM_NTASKS} ./nvshmem_buffered_noaccum $ITERS $SIZE
done

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
