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

# NVSHMEM_DEBUG has already answered its question (see the READ THIS block):
# job 92230 showed "P2P list: 0 1 2 3" on every rank with IBRC explicitly
# skipped, so all four GPUs use P2P and transport selection is NOT the cause.
# The remaining question is purely how fast it is, so the timed runs below are
# clean -- no debug output to drown them.
#
# Capture into a variable and grep afterwards. Two previous attempts lost every
# result line to a `head` applied to the live stream: first to NVSHMEM_INFO's
# 200-line config dump (92224), then to INFO lines that matched the filter's own
# keywords (92230). Nothing is truncated before it is searched now.
run_one() {   # $1=binary $2=order $3=rep
    local BIN=$1 ORDER=$2 REP=$3 OUT R V
    OUT=$(TRANSPOSE_WARMUP=20 \
          mpirun -np 4 -x TRANSPOSE_WARMUP -x NVSHMEM_BOOTSTRAP \
                 -x LD_LIBRARY_PATH -x UCX_LOG_LEVEL \
                 /tmp/$BIN 100 $ORDER 2>&1)
    R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
    V=$(echo "$OUT" | grep -c 'Solution validates')
    echo "NVRESULT bin=$BIN order=$ORDER rep=$REP mbps=${R:-FAIL} validates=$V"
    if [ -z "$R" ]; then
        echo "---- no rate, full output ----"; echo "$OUT" | tail -40
        echo "---- end ----"
    fi
}

echo ""
echo "### clean timed runs: is NVSHMEM slow on THIS node at small orders? ###"
for BIN in nvd_single nvd_buffered; do
    for ORDER in 1024 2048 4096 16384 32768; do
        for REP in 1 2 3; do run_one $BIN $ORDER $REP; done
    done
done

echo ""
echo "### one debug run kept for the record ###"
DBG=$(TRANSPOSE_WARMUP=20 NVSHMEM_DEBUG=INFO \
      mpirun -np 4 -x TRANSPOSE_WARMUP -x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH \
             -x NVSHMEM_DEBUG -x UCX_LOG_LEVEL /tmp/nvd_single 100 1024 2>&1)
echo "$DBG" | grep -oE 'P2P list:.*|Skipping transport|GDRCopy library not found|IBGDA [A-Za-z]+' | sort -u

echo ""
echo "=== READ THIS ==="
echo "=== 63328 reference on this node class, accum=0, GB/s:"
echo "===   nvsingle   415.1 @1024   1295.0 @16384"
echo "===   nvbuffered 120.0 @1024   1167.0 @16384"
echo "=== Job 90161 gave roughly 16-24 GB/s at order 1024."
echo "=== If the rates here match 63328, the loss is specific to h200x4-04 or"
echo "===   to something about job 90161, not to NVSHMEM on this cluster."
echo "=== If they match 90161, NVSHMEM really has regressed and transport"
echo "===   selection is not why -- P2P is being chosen correctly."
