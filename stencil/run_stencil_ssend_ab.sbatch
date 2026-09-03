#!/bin/bash
#SBATCH --job-name=stencil_ssend_np8
#SBATCH --partition=h200x8
#SBATCH --nodelist=h200x8-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=01:00:00
#SBATCH --output=stencil_ssend_np8_%j.out

# Three-way handshake comparison: barrier vs Isend token vs Ssend token.
#
# WHY: the advisor's original suggestion was to drop the global barrier and
# synchronise by Ssend-ing empty messages to neighbours. What we actually built
# uses zero-byte MPI_Isend plus double-buffered receive slots. Both are correct
# (job 63329: 144/144 matching L2), but Ssend has never been measured, so we
# cannot say which is faster. This job measures it.
#
# The difference: MPI_Isend of zero bytes completes locally and proves only that
# the sender's IPC write landed. MPI_Ssend does not return until the receiver
# posts its matching receive, so it additionally proves the neighbour has
# reached this iteration. Ssend therefore gives a stronger guarantee and would
# in principle allow single buffering -- but it stalls both sides, which may
# cost more than it saves.
#
# Everything except the handshake is identical across the three arms, including
# the double buffering, so this isolates the handshake cost alone.
#
# EXPECT: all three arms must report the same L2 norm at every np/size. Any
# mismatch means the handshake under test does not order the exchange safely.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20
REPS=5
SIZES="1024 2048 4096 8192 16384"

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ($(git log -1 --pretty=%s)) ==="
echo "=== stencil_ipc.cu md5=$(md5sum stencil/stencil_ipc.cu | cut -d' ' -f1) ==="
echo "=== warmup=$WU reps=$REPS ==="

for SIZE in $SIZES; do
    sed "s/const int N = [0-9]*/const int N = $SIZE/; \
         s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" \
        stencil/stencil_ipc.cu > /tmp/ssab_ipc_$SIZE.cu
    nvcc -O3 $ARCH /tmp/ssab_ipc_$SIZE.cu -o /tmp/ssab_ipc_$SIZE -lmpi \
        || { echo "BUILD FAILED $SIZE"; exit 1; }
done
echo "build done"

run_one() {   # $1=np $2=sync $3=size $4=rep
    local NP=$1 SYNC=$2 SIZE=$3 REP=$4 OUT T G L H
    OUT=$(STENCIL_WARMUP=$WU STENCIL_SYNC=$SYNC \
          mpirun -np $NP -x STENCIL_WARMUP -x STENCIL_SYNC -x UCX_LOG_LEVEL \
          env LD_PRELOAD=$MPIWRAP /tmp/ssab_ipc_$SIZE 2>&1)
    T=$(echo "$OUT" | grep -oP 'Total time: \K[0-9.]+')
    G=$(echo "$OUT" | grep -oP 'Throughput: \K[0-9.]+')
    L=$(echo "$OUT" | grep -oP 'Global L2 norm: \K[0-9.]+')
    H=$(echo "$OUT" | grep -oP 'handshake: \K.*' | head -1)
    echo "RESULT stencil np=$NP size=$SIZE sync=$SYNC rep=$REP time_ms=${T:-FAIL} gcells=${G:-FAIL} l2=${L:-FAIL} chose=\"${H:-unreported}\""
    if [ -z "$L" ]; then
        echo "---- FAILED CELL np=$NP size=$SIZE sync=$SYNC rep=$REP ----"
        echo "$OUT"
        echo "---- end ----"
    fi
}

# Job 83186 wasted its np=8 rows: the first ~8 minutes ran ~150x slow (every
# size pinned near 450 ms regardless of work) and only recovered by the last
# size. We had all 8 GPUs to ourselves -- the previous job released the node
# 60 s before we started -- so this was the node settling, not contention.
# Discard a full sweep before timing anything, and run np=8 alone since that
# is the only rank count whose timings 83186 lost.
echo "### settling run (untimed, discarded) ###"
for S in 1024 4096 16384; do
    STENCIL_WARMUP=20 STENCIL_SYNC=neighbor \
      mpirun -np 8 -x STENCIL_WARMUP -x STENCIL_SYNC -x UCX_LOG_LEVEL \
      env LD_PRELOAD=$MPIWRAP /tmp/ssab_ipc_$S >/dev/null 2>&1
done
echo "settled"

for NP in 8; do
    for SIZE in $SIZES; do
        for SYNC in barrier neighbor ssend; do
            for R in $(seq 1 $REPS); do
                run_one $NP $SYNC $SIZE $R
            done
        done
    done
done

echo "=== done $(date -Is) ==="
echo "=== CHECK 1: every l2 at a given np/size must match across all three"
echo "===          sync modes. A mismatch means that handshake is unsafe."
echo "=== CHECK 2: compare median time_ms per sync mode. Ssend gives a stronger"
echo "===          ordering guarantee; this says what it costs."
