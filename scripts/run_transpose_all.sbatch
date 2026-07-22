#!/bin/bash
#SBATCH --job-name=transpose_all
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=transpose_all_%j.out

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap/transpose

MPIWRAP=~/mpiwrap/mpi-intercept/libmpiwrap.so
(cd ~/mpiwrap/mpi-intercept && mpicxx -O2 -fPIC -shared mpiwrap_ipc.cc \
    -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcudart -o libmpiwrap.so)

SRC=transpose_ipc.cu
INC="-I${MPI_HOME}/include"
LIB="-L${MPI_HOME}/lib -lmpi"
ARCH="-gencode arch=compute_90,code=sm_90"

nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=1                    $INC $SRC -o direct $LIB
nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=1 -DSINGLE_KERNEL=1  $INC $SRC -o direct_single $LIB
nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=1                    $INC $SRC -o buffered $LIB
nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=1                    $INC $SRC -o gpumpi $LIB
nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=1                    $INC $SRC -o staged $LIB
nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=0                    $INC $SRC -o direct_noaccum $LIB
nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=0 -DSINGLE_KERNEL=1  $INC $SRC -o direct_single_noaccum $LIB
nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=0                    $INC $SRC -o buffered_noaccum $LIB
nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=0                    $INC $SRC -o gpumpi_noaccum $LIB
nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=0                    $INC $SRC -o staged_noaccum $LIB

ITERS=100
SIZES="1024 2048 4096 8192 16384"

echo "=========================================="
echo "IPC Transpose — All Modes — ${SLURM_NTASKS} GPUs"
echo "=========================================="

echo ""
echo "############## WITH ACCUMULATION ##############"
for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"
    echo "Direct (per-phase):";        LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./direct $ITERS $SIZE
    echo "Direct (single-kernel):";    LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./direct_single $ITERS $SIZE
    echo "Buffered:";                  LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./buffered $ITERS $SIZE
    echo "GPU-aware MPI:";             mpirun -np ${SLURM_NTASKS} ./gpumpi $ITERS $SIZE
    echo "Staged MPI:";               mpirun -np ${SLURM_NTASKS} ./staged $ITERS $SIZE
done

echo ""
echo "############## WITHOUT ACCUMULATION ##############"
for SIZE in $SIZES; do
    echo ""
    echo "--- Matrix size: ${SIZE} x ${SIZE} ---"
    echo "Direct (per-phase, no accum):";     LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./direct_noaccum $ITERS $SIZE
    echo "Direct (single-kernel, no accum):"; LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./direct_single_noaccum $ITERS $SIZE
    echo "Buffered (no accum):";              LD_PRELOAD=$MPIWRAP mpirun -np ${SLURM_NTASKS} ./buffered_noaccum $ITERS $SIZE
    echo "GPU-aware MPI (no accum):";         mpirun -np ${SLURM_NTASKS} ./gpumpi_noaccum $ITERS $SIZE
    echo "Staged MPI (no accum):";            mpirun -np ${SLURM_NTASKS} ./staged_noaccum $ITERS $SIZE
done

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
