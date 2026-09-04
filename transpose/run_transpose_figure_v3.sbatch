#!/bin/bash
#SBATCH --job-name=transpose_fig_v3
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=03:00:00
#SBATCH --output=transpose_fig_v3_%j.out

# Denser order sweep for the 4-GPU transpose figure, with repetitions on every
# series. Supersedes job 63328 (run_transpose_figure_v2.sbatch).
#
# ---- WHY: THE PUBLISHED SWEEP IS FIVE POWERS OF TWO ----------------------
# 63328 measured 1024/2048/4096/8192/16384 and nothing else. Two problems.
#
# 1. EVERY ORDER IS A POWER OF TWO, so every derived dimension is too:
#    Bo = order/P is a power of two, blk_elems = Bo^2 is a power of two, and
#    the 32x32 tiling divides all of them exactly. If alignment or tiling
#    contributes anything to the measured rate, the whole published curve sits
#    on the favourable case and nothing in the data would reveal it. This job
#    interleaves 1.5x steps (1536, 3072, 6144, 12288, 24576, 49152), which are
#    divisible by 4 and 8 but are not powers of two, so a tiling artifact
#    would show up as a sawtooth rather than hide inside a smooth curve.
#
# 2. THE ONE MODE INVERSION WE REPORT IS BRACKETED BY A SINGLE OCTAVE.
#    In 63328, accum=1: direct beats buffered at 1024^2 (230.7 vs 188.1 GB/s)
#    and has already lost by 2048^2 (499.0 vs 526.6), never recovering
#    (863.3 vs 1062.4 at 16384^2). In accum=0 direct wins at every size and
#    there is no crossover at all. So the paper reports an inversion that
#    exists in one accumulate setting, with zero measured points inside the
#    interval where it happens. 1536 and 3072 put two points there.
#
# The sweep also extends 4x past the old top end, to 65536^2. Headroom, at
# P=4: col_elems = order^2/P doubles, so A and B together are 17.2 GB/rank of
# device memory at 65536^2 (H200 has 140 GB), the host verification buffer
# B_h is another 8.6 GB/rank, and the staged path's two pinned blk_elems
# buffers add 4.3 GB/rank -- about 52 GB of host memory per node. The hard
# ceiling is transpose_ipc.cu:307, which aborts when blk_elems = (order/P)^2
# exceeds INT_MAX: at P=4 that is order > 185363, far above anything here.
# TRANSPOSE_ORDERS can push to 98304 (38.7 GB/rank device) if wanted, but it
# alone costs more wall clock than the entire list below.
#
# ---- WHAT CHANGED FROM v2 ------------------------------------------------
# DROPPED: directC / singleC, the -DDEVICE_TIMING=0 controls. They existed to
# test whether the CUDA-event instrumentation is free, and 63328 answered it
# (direct == directC and single == singleC across the sweep). Re-running a
# settled control at 13 orders would cost a quarter of this job for nothing.
# If that question ever reopens, v2 is the harness for it.
#
# ADDED: 3 repetitions on ALL eight series, not just the four COMM_MODE=0
# ones. 63328 ran buffered/gpumpi/staged/nv* once each, so the published
# figure has no spread to draw on those series and cannot carry error bars.
#
# ---- CONFIG (unchanged from 63328 except where stated) -------------------
# UCX defaults -- UCX_TLS deliberately unset. Exporting it fakes a CUDA-aware
# MPI setup cost and cuts staged 2.6x at large messages; see the appendix in
# results/transpose_results.md. TRANSPOSE_WARMUP=20, -O3 sm_90, 4 ranks, 100
# timed iterations, both ACCUMULATE settings, h200x4 (NOT h200x8 -- job
# 59853's h200x8 run is what made its staged column disagree with the
# recorded 4-GPU table).
#
# SINGLE JOB, ON PURPOSE. results/transpose_results.md carries a standing
# provenance correction because its IPC/MPI and NVSHMEM halves were measured
# three months apart and stitched. Adding orders in a new job while keeping
# 63328's five would recreate exactly that. Every cell in this log comes from
# one allocation, one node, one build.
#
# ---- BUDGET --------------------------------------------------------------
# 8 modes x 13 orders x 2 accumulate x 3 reps = 624 launches. Compute is
# ~22 min (cost scales as order^2; the order list sums to 33.3 "16384^2
# units"), mpirun startup is ~26 min, the build is ~10 min, and host-side
# verification at the top two orders adds ~15. Roughly 75 minutes; the 3-hour
# request is headroom, because overrunning wastes the queue wait rather than
# the walltime. Trim via TRANSPOSE_ORDERS or REPS below.
#
# No --mem line, deliberately: nvwulf's QOS rejects jobs whose memory request
# trips QOSMaxMemoryPerJob, and the default per-node allocation is sufficient.

set -uo pipefail

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
# The nvwulf openmpi module only prepends PATH/CPATH/LIBRARY_PATH -- it never
# exports MPI_HOME (only scripts/env_ptyche.sh does, for the GB200 side). Under
# `set -u` a bare $MPI_HOME aborts the job in seconds: that is exactly how 61945
# and 61948 died. Derive the prefix instead.
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20
ITERS=${ITERS:-100}
REPS=${REPS:-3}
# 1.5x interleaved with the powers of two. Every entry is divisible by 4 (the
# rank count) and by 8, so the same list works if this is ever re-run at np=8.
TRANSPOSE_ORDERS=${TRANSPOSE_ORDERS:-"1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536"}
MODES=${MODES:-"direct single buffered gpumpi staged nvdirect nvsingle nvbuffered"}

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== UCX_TLS=${UCX_TLS:-<unset>} warmup=$WU iters=$ITERS reps=$REPS ==="
echo "=== orders: $TRANSPOSE_ORDERS ==="
echo "=== modes:  $MODES ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ($(git log -1 --pretty=%s)) ==="
echo "=== transpose_ipc.cu md5=$(md5sum transpose/transpose_ipc.cu | cut -d' ' -f1) ==="
echo "=== transpose_nvshmem.cu md5=$(md5sum transpose/transpose_nvshmem.cu | cut -d' ' -f1) ==="

# ---- build ---------------------------------------------------------------
# 5 IPC/MPI modes + 3 NVSHMEM modes, x2 accumulate. The DEVICE_TIMING=0
# controls that v2 built are gone; see the header.
for ACC in 0 1; do
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf3_direct_$ACC -lmpi                     || echo "BUILD FAILED direct $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf3_single_$ACC -lmpi                     || echo "BUILD FAILED single $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf3_buffered_$ACC -lmpi                   || echo "BUILD FAILED buffered $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf3_gpumpi_$ACC -lmpi                     || echo "BUILD FAILED gpumpi $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf3_staged_$ACC -lmpi                     || echo "BUILD FAILED staged $ACC"

    NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
    NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf3_nvdirect_$ACC   || echo "BUILD FAILED nvdirect $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf3_nvsingle_$ACC   || echo "BUILD FAILED nvsingle $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf3_nvbuffered_$ACC || echo "BUILD FAILED nvbuffered $ACC"
done

# Abort now rather than after the queue wait if a mode did not build: an
# absent binary would otherwise show up as 13 x 2 x 3 FAIL cells.
MISSING=""
for ACC in 0 1; do
    for TAG in $MODES; do
        [ -x /tmp/tf3_${TAG}_$ACC ] || MISSING="$MISSING ${TAG}_$ACC"
    done
done
if [ -n "$MISSING" ]; then
    echo "ERROR: these binaries did not build:$MISSING" >&2
    exit 1
fi
echo "build done, all $(echo $MODES | wc -w) modes x 2 accumulate present"

# direct/single/buffered ride the interposer; gpumpi/staged do not; the
# NVSHMEM variants use their own transport.
run_one() {   # $1=tag $2=accum $3=order
    local TAG=$1 ACC=$2 ORDER=$3 PRE="" EXTRA=""
    case $TAG in
        direct|single|buffered) PRE="env LD_PRELOAD=$MPIWRAP" ;;
        nv*) EXTRA="-x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH" ;;
    esac
    local OUT
    OUT=$(TRANSPOSE_WARMUP=$WU mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL \
            $EXTRA $PRE /tmp/tf3_${TAG}_$ACC $ITERS $ORDER 2>&1)
    local R V D W O
    R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
    V=$(echo "$OUT" | grep -c "Solution validates")
    # CUDA-event breakdown: instrumented COMM_MODE=0 binaries only.
    # Absent everywhere else -> NA, which is "not reported", not "zero".
    D=$(echo "$OUT" | grep -oP 'comm-kernel GPU time/iter: \K[0-9.]+')
    W=$(echo "$OUT" | grep -oP 'wall/iter: \K[0-9.]+')
    O=$(echo "$OUT" | grep -oP 'barrier\+host overhead: \K[-0-9.]+')
    echo "RESULT transpose np=4 order=$ORDER accum=$ACC mode=$TAG mbps=${R:-FAIL} validates=$V dev_us=${D:-NA} wall_us=${W:-NA} ovh_us=${O:-NA}"
    # Unfiltered on failure, so a build/run failure is diagnosable from this
    # log alone -- job 59853's grep-to-three-fields is why that run's fallback
    # state is still unknowable. The likeliest failure here is an allocation
    # failure at the top orders, which prints to stderr and would otherwise be
    # swallowed by the Rate grep.
    if [ -z "$R" ]; then
        echo "---- FAILED CELL $TAG acc=$ACC order=$ORDER, full output ----"
        echo "$OUT"
        echo "---- end failed cell ----"
    fi
}

# Ascending order within each accumulate setting, so a memory ceiling at the
# top end costs the largest cells and not the whole sweep.
for ACC in 1 0; do
    for ORDER in $TRANSPOSE_ORDERS; do
        for REP in $(seq 1 $REPS); do
            for TAG in $MODES; do
                echo -n "rep=$REP "
                run_one $TAG $ACC $ORDER
            done
        done
    done
done

echo "=== done $(date -Is) ==="
echo "=== CHECK 1: every cell must report validates=1. A 0 at the large"
echo "===          orders is most likely an allocation failure, not a"
echo "===          numerical one -- read the FAILED CELL block."
echo "=== CHECK 2: plot mbps against order per mode. The 1.5x points"
echo "===          (1536 3072 6144 12288 24576 49152) must sit ON the curve"
echo "===          through the powers of two. A sawtooth means the published"
echo "===          power-of-two sweep was measuring a tiling artifact."
echo "=== CHECK 3: accum=1 direct vs buffered between 1024 and 2048 -- 63328"
echo "===          has direct ahead at 1024 and behind at 2048, with nothing"
echo "===          in between. 1536 should say which side it crosses on."