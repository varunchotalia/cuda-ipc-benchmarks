#!/bin/bash
#SBATCH --job-name=nvshmem_debug
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=nvshmem_debug_%j.out

# Why is NVSHMEM ~93% slower in job 90161 than in job 63328?
#
# Same transpose_nvshmem.cu md5 (aeff8603...), same build flags, same launch,
# same node class, no NVSHMEM warning anywhere in the 90161 log, and every run
# still reports "Solution validates". So this is not a correctness or config
# difference that reading the two harnesses can find -- we checked.
#
# What 90161 does show is that the loss is SIZE-DEPENDENT, which is the useful
# clue. Of its nv* runs at order >= 24576, 60 of 72 exceed 200 GB/s; below
# 24576, only 9 of 162 do. That is the shape of a large FIXED per-iteration
# cost that only amortises once the problem is big enough -- not random
# flakiness, and not a slow link.
#
# This job runs the two extremes with NVSHMEM_DEBUG=INFO so the library says
# which transport and which path it picked. If the small order selects a
# different remote-access path than the large one, that is the answer.
#
# Reference points from 63328 (accum=0, GB/s): nvsingle 415.1 at order 1024,
# 1295.0 at 16384; nvbuffered 120.0 at 1024, 1167.0 at 16384. Job 90161 gives
# roughly 16-24 at 1024.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
export UCX_LOG_LEVEL=error
unset UCX_TLS
ARCH="-gencode arch=compute_90,code=sm_90"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ==="
echo "=== transpose_nvshmem.cu md5=$(md5sum transpose/transpose_nvshmem.cu | cut -d' ' -f1) ==="
echo "=== NVSHMEM_HOME=$NVSHMEM_HOME ==="

NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=0 \
     transpose/transpose_nvshmem.cu $NVLIB -o /tmp/nvd_single || { echo BUILD FAILED; exit 1; }
nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=1 -DACCUMULATE=0 \
     transpose/transpose_nvshmem.cu $NVLIB -o /tmp/nvd_buffered || { echo BUILD FAILED; exit 1; }
echo "build done"

for BIN in nvd_single nvd_buffered; do
    for ORDER in 1024 32768; do
        echo ""
        echo "########## $BIN order=$ORDER  NVSHMEM_DEBUG=INFO ##########"
        TRANSPOSE_WARMUP=20 NVSHMEM_DEBUG=INFO NVSHMEM_INFO=1 \
          mpirun -np 4 -x TRANSPOSE_WARMUP -x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH \
                 -x NVSHMEM_DEBUG -x NVSHMEM_INFO -x UCX_LOG_LEVEL \
                 /tmp/$BIN 100 $ORDER 2>&1 | grep -viE '^\[' | head -60
        echo "########## end $BIN order=$ORDER ##########"
    done
done

echo ""
echo "=== READ THIS ==="
echo "=== Compare the transport/path lines NVSHMEM prints at order 1024 against"
echo "===   order 32768. A different remote-access path at the small order is"
echo "===   the answer. Identical paths mean the cost is per-iteration"
echo "===   synchronisation rather than transport selection, and the next step"
echo "===   is the device-timer breakdown, not the NVSHMEM log."
echo "=== 63328 reference (accum=0): nvsingle 415.1 GB/s at 1024, 1295.0 at 16384."
