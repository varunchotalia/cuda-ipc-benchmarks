#!/bin/bash
#SBATCH --job-name=transpose_fig_v2
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=01:30:00
#SBATCH --output=transpose_fig_v2_%j.out

# Regenerates every series behind plots/transpose_ipc_*.pdf on top of the
# device-timing fix, and proves in-job that the instrumentation is now free.
#
# ---- WHY THIS SUPERSEDES JOB 61541 ---------------------------------------
# 61541 regenerated all eight series on top of Jeff's timed-region merge
# (7c77c37) and looked like it showed direct/single regressing badly at small
# matrices: vs job 60636, direct 0.750 and single 0.821 at 1024^2, converging
# to ~1.00 by 16384^2. Job 61344 (stopsync_ab) A/B'd PRE(6f674cb) vs
# POST(7c77c37) in a single job and reproduced it: 0.778 / 0.821 at 1024^2.
#
# That was NOT the stop-side barrier and NOT a real bandwidth change. The
# merge called cudaEventElapsedTime() INSIDE the timed loop, once per timed
# iteration per comm kernel. It is a blocking driver call costing ~6.2 us, and
# it was being charged to the wall clock it was reporting against. Converting
# 61344's ratios to absolute time gives a constant per-iteration cost,
# independent of matrix size:
#
#     direct  17.3 / 18.4 / 18.1 us per iter at 1024 / 4096 / 16384
#     single   6.19 / 6.19 / 6.9  us per iter at 1024 / 4096 / 16384
#
# Constant absolute cost is why it reads as -22% at 1024^2 and -0.4% at
# 16384^2. The discriminator against the stop-barrier hypothesis is the ratio:
# direct's overhead is 2.9x single's, and direct records one event pair per
# remote phase (P-1 = 3 at np=4) where single records one per iteration.
# 2.9 ~ 3. A once-per-region stop barrier would cost both modes the same.
# 18.1/3 = 6.0 us per call, matching single's 6.19 directly.
#
# The fix (transpose_ipc.cu): the timed loop now only calls cudaEventRecord --
# asynchronous, enqueued on the stream -- into a pool of one event pair per
# comm kernel per timed iteration. Every blocking cudaEventElapsedTime read
# happens after the timed region closes, past the stop barrier.
#
# ---- WHAT THIS JOB ADDS: A SELF-VALIDATING CONTROL -----------------------
# "The instrumentation is now free" is a claim, so this job measures it rather
# than asserting it. direct and single are each built TWICE:
#
#   direct  / single         -DDEVICE_TIMING=1 (default) -- reports dev_us
#   directC / singleC        -DDEVICE_TIMING=0           -- provably zero
#                                                          instrumentation
#
# DEVICE_TIMING=0 compiles out every cudaEventRecord and cudaEventElapsedTime
# in our source; verified with `nvcc -E`, the clean build's CUDA-event call
# profile is identical to a COMM_MODE=2 build's.
#
#   Expected: direct == directC and single == singleC within run-to-run noise,
#     at EVERY order including 1024^2 where the old artifact was -22%. Then
#     the instrumented binaries are safe to use for the figure and dev_us is
#     a free extra column.
#   If instead the C variants are faster at 1024^2: cudaEventRecord itself is
#     not free either, and the figure must be built from the C binaries with
#     device timing reported from a separate pass.
#
# The other six series are unchanged from 61541 and are re-run only so the
# whole figure has single provenance.
#
# ---- CONFIG --------------------------------------------------------------
# Matches 60636/61541 exactly so the three are comparable: UCX defaults
# (UCX_TLS deliberately unset -- exporting it fakes a CUDA-aware-MPI setup
# cost and cuts staged 2.6x), TRANSPOSE_WARMUP=20, -O3 sm_90, 4 ranks, 100
# timed iterations, both ACCUMULATE settings.
#
# h200x4, not h200x8: the figure and results/transpose_results.md are 4-GPU
# numbers taken on the 4-GPU node type, and job 59853's h200x8 run is what
# made its staged column disagree with the recorded table.
#
# NOTE ON transpose_nvshmem.cu: unchanged by this job. 61541 showed the
# post-unpack nvshmem_barrier_all() moves ONLY the buffered NVSHMEM series
# (nvbuffered 0.80 at 1024^2 -> 0.99 at 16384^2); nvdirect and nvsingle are
# 0.996-1.006, i.e. unchanged. The barrier sits inside the recv_buf/unpack
# path, which only the buffered variant executes. It also fixed a real
# validation FAIL that 60636 hit at order=4096 accum=1 nvbuffered.

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
# and 61948 died. v1 (61541) has no `set -u`, so it expanded to empty and linked
# by accident off CPATH/LIBRARY_PATH. Derive the prefix instead.
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== UCX_TLS=${UCX_TLS:-<unset>} warmup=$WU iters=100 ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ($(git log -1 --pretty=%s)) ==="
echo "=== transpose_ipc.cu md5=$(md5sum transpose/transpose_ipc.cu | cut -d' ' -f1) ==="
echo "=== transpose_nvshmem.cu md5=$(md5sum transpose/transpose_nvshmem.cu | cut -d' ' -f1) ==="

# ---- build ---------------------------------------------------------------
# 5 IPC/MPI modes + 2 uninstrumented controls + 3 NVSHMEM modes, x2 accumulate
for ACC in 0 1; do
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_direct_$ACC -lmpi                    || echo "BUILD FAILED direct $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_single_$ACC -lmpi                    || echo "BUILD FAILED single $ACC"

    # --- uninstrumented controls (the point of this job) ---
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DDEVICE_TIMING=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_directC_$ACC -lmpi                   || echo "BUILD FAILED directC $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DDEVICE_TIMING=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_singleC_$ACC -lmpi                   || echo "BUILD FAILED singleC $ACC"

    nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_buffered_$ACC -lmpi                  || echo "BUILD FAILED buffered $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_gpumpi_$ACC -lmpi                    || echo "BUILD FAILED gpumpi $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf2_staged_$ACC -lmpi                    || echo "BUILD FAILED staged $ACC"

    NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
    NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf2_nvdirect_$ACC   || echo "BUILD FAILED nvdirect $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf2_nvsingle_$ACC   || echo "BUILD FAILED nvsingle $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf2_nvbuffered_$ACC || echo "BUILD FAILED nvbuffered $ACC"
done

# Record which binaries actually carry event calls, so the log proves the
# control is a control rather than relying on the -D flag having been typed.
echo "--- instrumentation check (md5 of each binary) ---"
for TAG in direct single directC singleC; do
    echo "  $TAG md5=$(md5sum /tmp/tf2_${TAG}_1 2>/dev/null | cut -d' ' -f1)"
done
echo "build done"

# direct/single/buffered ride the interposer; gpumpi/staged do not;
# the NVSHMEM variants use their own transport. The C controls are the same
# COMM_MODE=0 code path, so they ride it too.
run_one() {   # $1=tag $2=accum $3=order
    local TAG=$1 ACC=$2 ORDER=$3 PRE="" EXTRA=""
    case $TAG in
        direct|single|directC|singleC|buffered) PRE="env LD_PRELOAD=$MPIWRAP" ;;
        nv*) EXTRA="-x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH" ;;
    esac
    local OUT
    OUT=$(TRANSPOSE_WARMUP=$WU mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL \
            $EXTRA $PRE /tmp/tf2_${TAG}_$ACC 100 $ORDER 2>&1)
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
    # state is still unknowable.
    if [ -z "$R" ]; then
        echo "---- FAILED CELL $TAG acc=$ACC order=$ORDER, full output ----"
        echo "$OUT"
        echo "---- end failed cell ----"
    fi
}

# 3 repeats for the four COMM_MODE=0 series: the instrumented-vs-control
# comparison is the point of the job, and a null result needs noise bars.
# The other five series run once, as in 61541.
for ACC in 1 0; do
    for ORDER in 1024 2048 4096 8192 16384; do
        for REP in 1 2 3; do
            for TAG in direct directC single singleC; do
                echo -n "rep=$REP "
                run_one $TAG $ACC $ORDER
            done
        done
        for TAG in buffered gpumpi staged nvdirect nvsingle nvbuffered; do
            echo -n "rep=1 "
            run_one $TAG $ACC $ORDER
        done
    done
done

echo "=== done $(date -Is) ==="
