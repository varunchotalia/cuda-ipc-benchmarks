#!/bin/bash
#SBATCH --job-name=stencil_sync_ab
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=02:00:00
#SBATCH --output=stencil_sync_ab_%j.out

# Validate the SECOND attempt at neighbour-only stencil synchronisation
# (STENCIL_SYNC, committed OFF by default in 03f23c8).
#
# ---- WHAT IS BEING TESTED ------------------------------------------------
# stencil_ipc.cu and stencil_large_nvshmem.cu ship a per-iteration global
# barrier as the completion handshake. The MPI variants do not carry it, so
# the one-sided variants pay a synchronisation the comparison baseline does
# not -- that asymmetry is the reason to want a neighbour-only handshake.
#
# The FIRST attempt was reverted in 445c39a as RACY. At 8 ranks it produced
#     L2 3.7550293292 (1024^2) and 5.0563202266 (4096^2)
# against the correct 5.1449605829, while every barrier run matched. A token
# proves the neighbour finished WRITING, not UNPACKING; ranks then drift and
# one begins iteration i+1's write while its neighbour still reads iteration i.
#
# The second attempt adds double-buffered slots (slot = iter & 1): iteration
# i+1 writes the other slot, so it cannot clobber iteration i while a
# neighbour is still reading it. Plausible, but as of this job it has NEVER
# BEEN RUN. This job is the "job log showing matching L2 at 8 ranks" that the
# in-code comment demands before the default may be flipped.
#
# ---- WHY 8 RANKS, AND WHY REPEATS ----------------------------------------
# h200x8, not h200x4: the v1 failure appeared AT 8 RANKS. Drift is largest and
# compute slack smallest there. Validating only at 2/4 would be testing
# exactly where the bug did not show. 2 and 4 are included as controls.
#
# The barrier arm is deterministic, so it runs once per cell as the reference.
# The neighbour arm is the one under suspicion and a race is nondeterministic,
# so it runs 5 times per cell. ONE matching run would prove nothing.
#
# stencil_mpi runs once per cell as an INDEPENDENT ground truth -- it shares no
# handshake code with either arm, so if barrier and mpi agree, the reference is
# not itself wrong.
#
# ---- GO / NO-GO ----------------------------------------------------------
#   PASS: all 5 neighbour reps report an L2 identical to the barrier reference
#         at every np and every size, AND barrier == mpi.
#         -> the default in stencil_ipc.cu / stencil_large_nvshmem.cu may be
#            flipped to neighbour, citing this job.
#   FAIL: any neighbour rep differs at any cell.
#         -> the double-buffered slots do not close the race either. Leave the
#            default at barrier and record the observed L2 in-code next to the
#            v1 values, so a third attempt does not repeat this one.
#
# ---- ON THE EXPECTED PAYOFF ----------------------------------------------
# Do not expect a speed win. 445c39a measured the barrier as marginally
# CHEAPER than tokens: 4 ranks, 16384^2 36.89 vs 37.04 ms and 32768^2 135.29
# vs 135.45 ms, because OpenMPI's intra-node barrier beats four Isend/Irecv
# plus a Waitall. The value here is a synchronisation-matched comparison
# against the MPI baselines, not throughput. Timing is recorded anyway so the
# claim is measured rather than inherited.
#
# ---- CONFIG --------------------------------------------------------------
# Matches run_rebaseline.sbatch (job 59853, the source of every committed
# stencil number) so this is comparable to it: UCX defaults (UCX_TLS
# deliberately unset -- exporting it fakes a CUDA-aware-MPI setup cost),
# STENCIL_WARMUP=20, -O3 sm_90, grid size baked in per binary via sed.

set -uo pipefail

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20
REPS=5
SIZES="1024 2048 4096 8192 16384 32768"

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== UCX_TLS=${UCX_TLS:-<unset>} warmup=$WU reps=$REPS ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ($(git log -1 --pretty=%s)) ==="
echo "=== stencil_ipc.cu           md5=$(md5sum stencil/stencil_ipc.cu | cut -d' ' -f1) ==="
echo "=== stencil_large_nvshmem.cu md5=$(md5sum stencil/stencil_large_nvshmem.cu | cut -d' ' -f1) ==="
echo "=== v1 reference failure (445c39a): 8 ranks gave L2 3.7550293292 (1024^2)"
echo "===   and 5.0563202266 (4096^2) against the correct 5.1449605829 ==="

# ---- build ---------------------------------------------------------------
for SIZE in $SIZES; do
    for MODE in ipc mpi; do
        sed "s/const int N = [0-9]*/const int N = $SIZE/; \
             s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" \
            stencil/stencil_${MODE}.cu > /tmp/sab_${MODE}_$SIZE.cu
        nvcc -O3 $ARCH /tmp/sab_${MODE}_$SIZE.cu -o /tmp/sab_${MODE}_$SIZE -lmpi \
            || echo "BUILD FAILED stencil $MODE $SIZE"
    done
done
echo "IPC/MPI stencil binaries built"

# NVSHMEM stencil at its native 16384 only -- it is the large-stencil variant
# and the handshake under test is size-independent.
NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
nvcc -O3 -rdc=true $ARCH $NVINC stencil/stencil_large_nvshmem.cu $NVLIB \
    -o /tmp/sab_nvshmem_16384 || echo "BUILD FAILED stencil nvshmem"
echo "build done"

# ---- run helper ----------------------------------------------------------
# $1=binary $2=np $3=label $4=sync-mode-or-NA $5=size $6=rep
run_one() {
    local BIN=$1 NP=$2 LABEL=$3 SYNC=$4 SIZE=$5 REP=$6 PRE="" EXTRA="" ENVS=""
    case $LABEL in
        ipc)     PRE="env LD_PRELOAD=$MPIWRAP" ;;
        nvshmem) EXTRA="-x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH" ;;
    esac
    [ "$SYNC" != "NA" ] && ENVS="-x STENCIL_SYNC"
    local OUT
    OUT=$(STENCIL_WARMUP=$WU STENCIL_SYNC=$SYNC \
          mpirun -np $NP -x STENCIL_WARMUP -x UCX_LOG_LEVEL $ENVS $EXTRA $PRE \
          $BIN 2>&1)
    local T G L H
    T=$(echo "$OUT" | grep -oP 'Total time: \K[0-9.]+')
    G=$(echo "$OUT" | grep -oP 'Throughput: \K[0-9.]+')
    L=$(echo "$OUT" | grep -oP 'Global L2 norm: \K[0-9.]+')
    # Echo back which handshake the binary says it chose, so a typo'd
    # STENCIL_SYNC cannot silently make this an A/A test.
    H=$(echo "$OUT" | grep -oP 'handshake: \K.*' | head -1)
    echo "RESULT stencil np=$NP size=$SIZE variant=$LABEL sync=$SYNC rep=$REP time_ms=${T:-FAIL} gcells=${G:-FAIL} l2=${L:-FAIL} chose=\"${H:-unreported}\""
    if [ -z "$L" ]; then
        echo "---- FAILED CELL $LABEL np=$NP size=$SIZE sync=$SYNC rep=$REP ----"
        echo "$OUT"
        echo "---- end failed cell ----"
    fi
}

# ---- section 1: IPC stencil, barrier vs neighbour ------------------------
echo ""
echo "### SECTION 1: stencil_ipc -- np 2/4/8 x 6 sizes, barrier x1 + neighbor x$REPS ###"
for NP in 8 4 2; do          # 8 first: that is where v1 failed
    for SIZE in $SIZES; do
        run_one /tmp/sab_mpi_$SIZE  $NP mpi NA       $SIZE 1   # independent ground truth
        run_one /tmp/sab_ipc_$SIZE  $NP ipc barrier  $SIZE 1   # reference
        for R in $(seq 1 $REPS); do
            run_one /tmp/sab_ipc_$SIZE $NP ipc neighbor $SIZE $R
        done
    done
done

# ---- section 2: NVSHMEM large stencil, barrier vs neighbour --------------
echo ""
echo "### SECTION 2: stencil_large_nvshmem 16384^2 -- np 2/4/8 ###"
for NP in 8 4 2; do
    run_one /tmp/sab_nvshmem_16384 $NP nvshmem barrier 16384 1
    for R in $(seq 1 $REPS); do
        run_one /tmp/sab_nvshmem_16384 $NP nvshmem neighbor 16384 $R
    done
done

echo ""
echo "=== done $(date -Is) ==="
echo "=== VERDICT: compare every sync=neighbor l2 against the sync=barrier l2"
echo "===   at the same np/size/variant. Any mismatch = the race is not closed."
