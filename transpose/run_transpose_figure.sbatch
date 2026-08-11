#!/bin/bash
#SBATCH --job-name=transpose_figure
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=01:30:00
#SBATCH --output=transpose_figure_%j.out

# Regenerates every series behind plots/transpose_ipc_*.pdf in ONE
# job so the whole figure has single provenance.
#
# Job 59853 already re-baselined the five transpose_ipc.cu modes, but it did
# NOT build the NVSHMEM variants, which make up the figure's entire bottom row.
# Rather than splice 59853's IPC data together with fresh NVSHMEM data, this
# re-runs all eight series together.
#
# Config matches 59853: UCX defaults (UCX_TLS deliberately unset),
# TRANSPOSE_WARMUP=20, -O3 -gencode arch=compute_90,code=sm_90, 4 ranks,
# 100 timed iterations, both ACCUMULATE settings.
#
# h200x4 (not h200x8) because the figure and results/transpose_results.md are
# 4-GPU numbers taken on the 4-GPU node type; job 59853's h200x8 run is what
# made its staged column disagree with the recorded table.
#
# ---- RERUN AFTER THE TIMING MERGE (7c77c37) ------------------------------
# Job 60636 produced the current figure from 6f674cb, i.e. BEFORE Jeff's
# timed-region fix. This rerun regenerates all eight series on top of the
# merge, so three things differ from 60636:
#
#  1. The stop of the timed region now mirrors the start (cudaDeviceSynchronize
#     + MPI_Barrier before the stop clock). Only SINGLE_KERNEL was actually
#     unsynchronized at loop exit; the per-phase modes were incidentally
#     covered by their own per-phase barriers. So `single` is the series that
#     can move here, and the direction is not assumed either way -- job 61344
#     (stopsync_ab) A/Bs PRE vs POST directly to answer that.
#  2. COMM_MODE=0 binaries (direct, single) now also report CUDA-event GPU
#     time per iteration. Captured below as dev_us/wall_us/ovh_us so the
#     barrier+host overhead is separable from real device work. The other
#     modes and all NVSHMEM variants do not emit it -> NA.
#  3. transpose_nvshmem.cu carries the post-unpack nvshmem_barrier_all() that
#     fixes the 32-PE race. It is a per-phase barrier on the critical path,
#     so the nvdirect/nvsingle/nvbuffered series are expected to move down
#     relative to 60636. That is a correctness cost, not a regression.
#
# Also fixed here: this job used to LD_PRELOAD a hardcoded ~/mpiwrap/
# libmpiwrap.so, which is the stale 2026-05-04 hand-built copy -- three months
# behind the WinIPC source, mpi-intercept/mpiwrap_ipc.cc. Now built in-job via
# scripts/build_mpiwrap.sh, same as the LULESH harnesses, with its md5 in the
# log. See that script's header for why this is not optional.

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
# The nvwulf openmpi module never exports MPI_HOME; without this the NVINC/NVLIB
# lines below expand to -I/include -L/lib and link only by accident off CPATH.
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) date=$(date -Is) UCX_TLS=${UCX_TLS:-<unset>} warmup=$WU ==="

# ---- build: 5 IPC/MPI modes + 3 NVSHMEM modes, x2 accumulate -------------
for ACC in 0 1; do
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf_direct_$ACC -lmpi                     || echo "BUILD FAILED direct $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf_single_$ACC -lmpi                     || echo "BUILD FAILED single $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf_buffered_$ACC -lmpi                   || echo "BUILD FAILED buffered $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf_gpumpi_$ACC -lmpi                     || echo "BUILD FAILED gpumpi $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/tf_staged_$ACC -lmpi                     || echo "BUILD FAILED staged $ACC"

    NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
    NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf_nvdirect_$ACC   || echo "BUILD FAILED nvdirect $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf_nvsingle_$ACC   || echo "BUILD FAILED nvsingle $ACC"
    nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=1 -DACCUMULATE=$ACC \
        transpose/transpose_nvshmem.cu $NVLIB -o /tmp/tf_nvbuffered_$ACC || echo "BUILD FAILED nvbuffered $ACC"
done
echo "build done"

# direct/single/buffered ride the interposer; gpumpi/staged do not;
# the NVSHMEM variants use their own transport.
run_one() {   # $1=tag $2=accum $3=order
    local TAG=$1 ACC=$2 ORDER=$3 PRE="" EXTRA=""
    case $TAG in
        direct|single|buffered) PRE="env LD_PRELOAD=$MPIWRAP" ;;
        nv*) EXTRA="-x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH" ;;
    esac
    local OUT
    OUT=$(TRANSPOSE_WARMUP=$WU mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL \
            $EXTRA $PRE /tmp/tf_${TAG}_$ACC 100 $ORDER 2>&1)
    local R V D W O
    R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
    V=$(echo "$OUT" | grep -c "Solution validates")
    # CUDA-event breakdown, COMM_MODE=0 binaries only (direct, single).
    # Absent everywhere else -> NA, which is "not reported", not "zero".
    D=$(echo "$OUT" | grep -oP 'comm-kernel GPU time/iter: \K[0-9.]+')
    W=$(echo "$OUT" | grep -oP 'wall/iter: \K[0-9.]+')
    O=$(echo "$OUT" | grep -oP 'barrier\+host overhead: \K[-0-9.]+')
    echo "RESULT transpose np=4 order=$ORDER accum=$ACC mode=$TAG mbps=${R:-FAIL} validates=$V dev_us=${D:-NA} wall_us=${W:-NA} ovh_us=${O:-NA}"
    # Unfiltered, so a build/run failure is diagnosable from this log alone --
    # job 59853's grep-to-three-fields is why that run's fallback state is
    # still unknowable.
    if [ -z "$R" ]; then
        echo "---- FAILED CELL $TAG acc=$ACC order=$ORDER, full output ----"
        echo "$OUT"
        echo "---- end failed cell ----"
    fi
}

for ACC in 1 0; do
    for ORDER in 1024 2048 4096 8192 16384; do
        for TAG in direct single buffered gpumpi staged nvdirect nvsingle nvbuffered; do
            run_one $TAG $ACC $ORDER
        done
    done
done

echo "=== done $(date -Is) ==="
rm -f /tmp/tf_*
