#!/bin/bash
# Inter-node benchmark suite for multi-node NVLink systems (GB200 NVL72).
# Scheduler-agnostic: set LAUNCH to your launcher prefix, e.g.
#   LAUNCH="mpirun -np"            (default if no Slurm allocation detected)
#   LAUNCH="srun --mpi=pmix -n"    (auto-selected inside a Slurm allocation)
# Auto-detected if left unset -- see the launcher-detection block below.
# Run from the repo root after building:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 && cmake --build build -j
#   NVSHMEM_HOME=/path/to/nvshmem bash scripts/run_nvl72.sh
#
# Optional first step, before this full suite: a fast INTERPOSER sanity
# check (raw IPC-window bandwidth/latency, no application code):
#   LD_PRELOAD=$PWD/build/libmpiwrap.so mpirun -np 2 ./build/test_ipc_win
#   (Slurm sites that discourage bare mpirun: swap in
#    "srun --mpi=pmix -n 2" for "mpirun -np 2" above)
# NOTE: this only proves the interposer + same-node CUDA IPC work. It uses
# MPI_Win_create over an app pointer, which never takes the fabric-handle
# branch (only MPI_Win_allocate on a cross-node communicator does, i.e.
# lulesh_mpiwrap/lulesh_mpiwrap_rp below) -- it is NOT a fabric proof.
# CHECK THE PRELOAD ACTUALLY TOOK. ld.so treats an unloadable LD_PRELOAD as a
# warning, not an error: "object '...libmpiwrap.so' from LD_PRELOAD cannot be
# preloaded ... ignored", and the run proceeds uninstrumented while still
# printing bandwidth. Job 2528116 shipped a bandwidth table produced this way
# (stale absolute path). Use $PWD/build/libmpiwrap.so, not a hardcoded path,
# and grep the output for that ld.so line before believing the numbers.
#
# What to look for:
#  - interposer log line "fabric window: N ranks ..." during
#    lulesh_mpiwrap/lulesh_mpiwrap_rp is the actual proof the cross-node
#    CUDA fabric-handle path (multi-node NVLink) is active; "N of M peers
#    not IPC-reachable" means it fell back to per-peer hybrid MPI instead
#  - LULESH: Final Origin Energy must match staged at every rank count
#  - section 4/4 repeats the LULESH WinIPC variants with the fabric path
#    forced off (WINIPC_DISABLE_FABRIC=1), for a fabric-vs-hybrid-MPI
#    apples-to-apples comparison at the same rank counts
#
# Rank counts need one MPI rank per GPU, and the job allocation must
# actually cover them: LULESH defaults to 8/27/64 ranks, transpose and stencil
# to 8/16/32/64. LULESH needs cubic counts. Request at least
# 64 GPUs (however many NVL72 trays that spans) for the defaults to run
# to completion; override via RANKS_LIST / TRANSPOSE_RANKS_LIST /
# STENCIL_RANKS_LIST if your allocation is smaller. One rank per GPU is set by
# --ntasks-per-node in the launcher, not by these lists -- see GPUS_PER_NODE.
#
# SIZING THE ALLOCATION. The defaults are much bigger than the 2026-08-06 run,
# which fitted in `-t 0:25:00`. Rough estimate for the current defaults:
#   transpose  ~60-90 min  (7 orders x 4 rank counts x 7 variants; cost scales
#                           as order^2, so 131072 alone is ~50% of the section)
#   stencil    ~20-30 min  (4 rank counts x 3 reps x 6 runs)
#   LULESH     ~45-60 min  (3 rank counts x 5 reps x 7 variants, plus 4/4)
# Call it 3 hours wall clock on 16 nodes, and pass -t accordingly. To trim,
# the cheapest cuts in order are: TRANSPOSE_ITERS=50 (halves the transpose),
# then dropping 98304/131072 from TRANSPOSE_ORDERS_LIST.
#
# HOST memory is the likeliest thing to blow up, not device memory. Every
# transpose rank mallocs order^2/P doubles on the host for verification: at
# order 131072 and 8 ranks that is 17.2 GB per rank, i.e. ~69 GB per node at
# 4 ranks per node, on top of ~4 GB of pinned staging buffers per rank.
#
# Requirements on the target system: CUDA >= 12.4 driver stack with the
# IMEX daemon running (for fabric handles), CUDA-aware MPI (for the ipc/
# ipc_rp hybrid fallback paths), NVSHMEM with a working bootstrap.
#
# On Ptyche: `source scripts/env_ptyche.sh` in the same shell as BOTH the
# cmake build and this script. It pins one HPC-X tree (2.22.1) for compile
# and runtime; mixing trees aborts every rank in MPI_Init, and 2.21's UCX
# has a cuda_ipc cache ucs_fatal that kills gpumpi/ipc/nvshmem past 8 ranks.

set -uo pipefail

BUILD=${BUILD:-build}

# --- MPI consistency guard ------------------------------------------------
# The MPI linked into the binaries must be the MPI the runtime resolves.
# Job 2527976 built against HPC-X 2.22.1 and ran against 2.21: mca_pml_ucx.so
# would not load, every rank aborted in MPI_Init, and all four sections
# reported n/a. That looks like a benchmark failure and is not one, so catch
# it here instead of after the queue wait. On Ptyche, `source
# scripts/env_ptyche.sh` before both the build and this script.
BUILD_MPI=$(ldd "./$BUILD/lulesh_staged" 2>/dev/null | awk '/libmpi\.so/{print $3}')
if [ -n "$BUILD_MPI" ] && command -v ompi_info >/dev/null 2>&1; then
    RUN_MPI="$(dirname "$(dirname "$(command -v ompi_info)")")/lib/libmpi.so"
    if [ "$(readlink -f "$BUILD_MPI")" != "$(readlink -f "$RUN_MPI")" ]; then
        echo "ERROR: MPI mismatch between build and runtime." >&2
        echo "       linked : $BUILD_MPI" >&2
        echo "       runtime: $RUN_MPI" >&2
        echo "       Load one MPI tree and rebuild -- mixing them aborts every" >&2
        echo "       rank in MPI_Init with 'pml/ucx component not found'." >&2
        exit 1
    fi
fi

# --- Launcher auto-detection (skipped if LAUNCH is already set) -----------
# GPUS_PER_NODE feeds srun's --ntasks-per-node. A bare `srun --mpi=pmix -n N`
# does NOT place one rank per GPU: in job 2528116 it spread 8 ranks over 8
# nodes, so lulesh_ipc and lulesh_ipc_rp reported "7 of 7 peers not IPC
# reachable" and ran entirely on the two-sided MPI fallback. Those rows were
# labelled IPC and measured MPI. srun derives the node count from -n once
# --ntasks-per-node is set, so this is the only geometry knob needed.
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
if [ -z "${LAUNCH:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
        LAUNCH="srun --mpi=pmix --ntasks-per-node=$GPUS_PER_NODE -n"
        echo "LAUNCH not set: detected a Slurm allocation, using '$LAUNCH'"
    elif command -v mpirun >/dev/null 2>&1; then
        LAUNCH="mpirun -np"
        echo "LAUNCH not set: no Slurm allocation detected, using '$LAUNCH'"
    elif command -v mpiexec >/dev/null 2>&1; then
        LAUNCH="mpiexec -n"
        echo "LAUNCH not set: no mpirun found, using '$LAUNCH'"
    else
        echo "ERROR: could not auto-detect an MPI launcher (no srun/mpirun/mpiexec" >&2
        echo "       on PATH). Set LAUNCH explicitly, e.g.:" >&2
        echo "       LAUNCH=\"your-launcher -n\" bash scripts/run_nvl72.sh" >&2
        exit 1
    fi
fi

RANKS_LIST=${RANKS_LIST:-8 27 64}        # LULESH needs cubic rank counts
TRANSPOSE_RANKS_LIST=${TRANSPOSE_RANKS_LIST:-8 16 32 64}
# 16 and 32 are NOT optional here. The stencil result this rerun exists to fix
# -- GPU-aware MPI at 121.85 ms (16 GPUs) and 169.51 ms (32), both measured at
# warmup 0 -- was taken at those two rank counts and nowhere else. A sweep of
# 8 and 64 alone produces new numbers that cannot replace them, so the open
# question (fixed cross-node wireup, or a genuinely slow inter-node device
# path?) would stay open. 8 and 64 are kept as the low and high anchors.
STENCIL_RANKS_LIST=${STENCIL_RANKS_LIST:-8 16 32 64}
TRANSPOSE_ITERS=${TRANSPOSE_ITERS:-100}
STENCIL_REPS=${STENCIL_REPS:-3}
LULESH_REPS=${LULESH_REPS:-5}
# The largest default order is intended to remain within GB200 device memory at
# the low rank counts. Each order must be divisible by its rank count because
# the transpose partitions the matrix into equal columns. The host-side
# verification buffer is order^2/P doubles per rank.
# The previous NVL72 run used one order, 6912. The default sweep extends the
# range so the scaling trend is not determined by one small matrix. Set
# TRANSPOSE_ORDER to run one order, or TRANSPOSE_ORDERS_LIST to replace this
# sweep without changing the script.
TRANSPOSE_ORDERS_LIST=${TRANSPOSE_ORDERS_LIST:-"8192 16384 32768 49152 65536 98304 131072"}
if [ -n "${TRANSPOSE_ORDER:-}" ]; then
    TRANSPOSE_ORDERS_LIST="$TRANSPOSE_ORDER"
fi
LULESH_SIZE=${LULESH_SIZE:-45}           # per-rank problem size
MPIWRAP_LIB=$PWD/$BUILD/libmpiwrap.so
export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}

# NVSHMEM's symmetric heap defaults to 1 GiB. The largest default transpose
# order needs about 32 GiB per PE at 8 ranks for its two distributed matrices,
# so reserve 40 GiB by default. Override this when running a smaller sweep.
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-42949672960}   # 40 GiB
# ...and passed per launch as well, for the same reason as the warmup vars
# below: exporting is enough for srun, not for Open MPI's mpirun.
NVSHMEM_ENV="env NVSHMEM_BOOTSTRAP=$NVSHMEM_BOOTSTRAP NVSHMEM_SYMMETRIC_SIZE=$NVSHMEM_SYMMETRIC_SIZE"

# --- Untimed warmup iterations -------------------------------------------
# MUST be non-zero. UCX establishes CUDA-aware connections lazily, on first
# message. The stencil timer starts immediately before the first exchange, so
# at warmup=0 that one-time wireup is charged to the GPU-aware MPI column and
# to nothing else: stencil_ipc and stencil_nvshmem do their handle / symmetric
# -heap setup before the loop by construction, and stencil_mpi sends host
# buffers so it has no CUDA-aware connection to establish. Job 57012 on nvwulf
# measured the resulting artifact at ~265 ms, independent of grid size:
# GPU-aware 1024^2 went 272 ms (warmup=0) -> 128 (warmup=1) -> 4.88 (warmup=5),
# while IPC and staged moved <1%. Job 2528116 on Ptyche shipped with warmup=0
# and reproduced it (RESULTS_NVL16_NVL32.md section 3). 20 is well past the
# convergence knee measured in results/stencil_results.txt.
#
# Passed as an `env` prefix on each launch rather than exported: srun forwards
# the environment but Open MPI's mpirun does not, and a silently-unforwarded
# warmup is the bug this block exists to prevent. Both launchers exec the
# `env` wrapper per task, so one form works for both -- same trick the
# interposer runs below already use for LD_PRELOAD.
STENCIL_WARMUP=${STENCIL_WARMUP:-20}
TRANSPOSE_WARMUP=${TRANSPOSE_WARMUP:-20}
WARMUP_S="env STENCIL_WARMUP=$STENCIL_WARMUP"
WARMUP_T="env TRANSPOSE_WARMUP=$TRANSPOSE_WARMUP"
STENCIL_BARRIER="env STENCIL_WARMUP=$STENCIL_WARMUP STENCIL_SYNC=barrier"
STENCIL_NEIGHBOR="env STENCIL_WARMUP=$STENCIL_WARMUP STENCIL_SYNC=neighbor"

# --- Interposer preflight -------------------------------------------------
# LD_PRELOAD failures are non-fatal by design: ld.so prints "cannot be
# preloaded ... ignored" and the binary runs anyway, without the interposer.
# Job 2528116 hit exactly this on its sanity check (a stale absolute path) and
# the run continued, producing a bandwidth number that reads as an interposer
# result and is not one. Fail here instead.
for _v in stencil_ipc transpose_buffered transpose_direct; do
    if [ -x "./$BUILD/$_v" ] && [ ! -r "$MPIWRAP_LIB" ]; then
        echo "ERROR: $MPIWRAP_LIB is missing or unreadable, but LD_PRELOAD runs" >&2
        echo "       are about to use it. LD_PRELOAD would be silently ignored" >&2
        echo "       and the ipc/buffered/direct variants would measure the" >&2
        echo "       uninstrumented path while still being labelled IPC." >&2
        echo "       Build it first (scripts/build_mpiwrap.sh), or set BUILD." >&2
        exit 1
    fi
done
unset _v

echo "launcher: '$LAUNCH', build dir: $BUILD"
echo "transpose ranks: ${TRANSPOSE_RANKS_LIST}"
echo "stencil ranks: ${STENCIL_RANKS_LIST}, repetitions: ${STENCIL_REPS}"
echo "LULESH ranks: ${RANKS_LIST}, repetitions: ${LULESH_REPS}"
echo "======================================================================"
echo "1/4 transpose; direct/buffered ride the interposer windows"
echo "======================================================================"
for N in $TRANSPOSE_RANKS_LIST; do
    for ORDER in $TRANSPOSE_ORDERS_LIST; do
        if [ $((ORDER % N)) -ne 0 ]; then
            echo "SKIP: transpose at $N ranks, order $ORDER is not divisible by $N"
            continue
        fi
        echo "########## transpose, $N ranks, order $ORDER ##########"
        for V in staged gpumpi buffered direct direct_single; do
            case $V in
                buffered|direct*) PRE="env LD_PRELOAD=$MPIWRAP_LIB TRANSPOSE_WARMUP=$TRANSPOSE_WARMUP" ;;
                *)                PRE="$WARMUP_T" ;;
            esac
            echo "--- transpose_$V ---"
            $LAUNCH $N $PRE ./$BUILD/transpose_$V $TRANSPOSE_ITERS $ORDER \
                || echo "FAILED: transpose_$V at $N ranks, order $ORDER"
        done
        if [ -x ./$BUILD/transpose_nvshmem_direct ]; then
            for V in transpose_nvshmem_direct transpose_nvshmem_buffered; do
                echo "--- $V ---"
                $LAUNCH $N $WARMUP_T $NVSHMEM_ENV ./$BUILD/$V $TRANSPOSE_ITERS $ORDER \
                    || echo "FAILED: $V at $N ranks, order $ORDER"
            done
        fi
    done
done

echo "======================================================================"
echo "2/4 stencil; ipc rides the interposer windows"
echo "  barrier and barrierless configurations run at STENCIL_WARMUP=$STENCIL_WARMUP"
echo "  with $STENCIL_REPS repetitions per timed point -- each prints"
echo "  'Warmup iterations (untimed): N'; if that line says 0, or is absent"
echo "  from the nvshmem variant, the binary predates the warmup support and"
echo "  its GPU-aware column is setup-dominated, not a throughput measurement"
echo "======================================================================"
for N in $STENCIL_RANKS_LIST; do
    for REP in $(seq 1 $STENCIL_REPS); do
        echo "########## stencil, $N ranks, repetition $REP ##########"
        $LAUNCH $N $STENCIL_BARRIER env LD_PRELOAD=$MPIWRAP_LIB ./$BUILD/stencil_ipc \
            || echo "FAILED: stencil_ipc barrier at $N ranks, repetition $REP"
        $LAUNCH $N $WARMUP_S ./$BUILD/stencil_mpi \
            || echo "FAILED: stencil_mpi at $N ranks, repetition $REP"
        $LAUNCH $N $WARMUP_S ./$BUILD/stencil_gpumpi \
            || echo "FAILED: stencil_gpumpi at $N ranks, repetition $REP"
        if [ -x ./$BUILD/stencil_nvshmem ]; then
            $LAUNCH $N $STENCIL_BARRIER $NVSHMEM_ENV ./$BUILD/stencil_nvshmem \
                || echo "FAILED: stencil_nvshmem barrier at $N ranks, repetition $REP"
        fi

        echo "########## stencil barrierless, $N ranks, repetition $REP ##########"
        $LAUNCH $N $STENCIL_NEIGHBOR env LD_PRELOAD=$MPIWRAP_LIB ./$BUILD/stencil_ipc \
            || echo "FAILED: stencil_ipc neighbor at $N ranks, repetition $REP"
        if [ -x ./$BUILD/stencil_nvshmem ]; then
            $LAUNCH $N $STENCIL_NEIGHBOR $NVSHMEM_ENV ./$BUILD/stencil_nvshmem \
                || echo "FAILED: stencil_nvshmem neighbor at $N ranks, repetition $REP"
        fi
    done
done

echo "======================================================================"
echo "3/4 LULESH at ${RANKS_LIST} ranks, -s $LULESH_SIZE per rank"
echo "  inter-node capable: staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp nvshmem"
echo "  (mpiwrap/mpiwrap_rp are the WinIPC variants; targets keep the old name)"
echo "  (shmwin and direct are single-node by construction and are skipped)"
echo "======================================================================"
VARIANTS="staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp"
[ -x ./$BUILD/lulesh_nvshmem ] && VARIANTS="$VARIANTS nvshmem"

declare -A FABRIC_ELAPSED   # keyed "$V_$N", for the section 4 comparison

for N in $RANKS_LIST; do
    for REP in $(seq 1 $LULESH_REPS); do
        declare -A ENERGY ELAPSED
        echo ""
        echo "########## $N ranks, repetition $REP ##########"
        for V in $VARIANTS; do
            case $V in
                mpiwrap*) PRE="env LD_PRELOAD=$MPIWRAP_LIB" ;;
                nvshmem)  PRE="$NVSHMEM_ENV" ;;
                *)        PRE="" ;;
            esac
            echo "--- lulesh_$V, $N ranks, repetition $REP ---"
            $LAUNCH $N $PRE ./$BUILD/lulesh_$V -s $LULESH_SIZE 2>&1 | tee run.tmp \
                || echo "FAILED: lulesh_$V at $N ranks, repetition $REP"
            ENERGY[$V]=$(awk '/Final Origin Energy/{print $5}' run.tmp)
            ELAPSED[$V]=$(awk '/Elapsed time/{print $4}' run.tmp)
            case $V in mpiwrap*) FABRIC_ELAPSED[${V}_${N}]=${ELAPSED[$V]:-} ;; esac
        done
        echo ""
        echo "Summary, $N ranks, repetition $REP:"
        printf "  %-12s %-16s %s\n" "variant" "energy" "elapsed(s)"
        for V in $VARIANTS; do
            CHECK="MATCH"
            [ -z "${ENERGY[$V]:-}" ] && CHECK="MISSING"
            [ -n "${ENERGY[$V]:-}" ] && [ "${ENERGY[$V]}" != "${ENERGY[staged]:-}" ] && CHECK="MISMATCH"
            printf "  %-12s %-16s %-10s %s\n" "$V" "${ENERGY[$V]:-}" "${ELAPSED[$V]:-}" "$CHECK"
        done
        unset ENERGY ELAPSED
    done
done

echo ""
echo "======================================================================"
echo "4/4 LULESH WinIPC (mpiwrap/mpiwrap_rp) with the fabric path forced OFF"
echo "  (WINIPC_DISABLE_FABRIC=1: cross-node peers fall back to real MPI"
echo "   send/recv, same as running on hardware with no fabric support)"
echo "======================================================================"
declare -A NOFABRIC_ELAPSED
for N in $RANKS_LIST; do
    echo ""
    echo "########## $N ranks, fabric disabled ##########"
    for V in mpiwrap mpiwrap_rp; do
        echo "--- lulesh_$V, $N ranks, WINIPC_DISABLE_FABRIC=1 ---"
        # Both spellings: an interposer built before the WinIPC rebrand only
        # knows MPIWRAP_DISABLE_FABRIC, and an unrecognised name fails
        # SILENTLY -- the run would keep the fabric path on and the
        # comparison would be vacuous rather than obviously broken.
        $LAUNCH $N env LD_PRELOAD=$MPIWRAP_LIB \
            WINIPC_DISABLE_FABRIC=1 MPIWRAP_DISABLE_FABRIC=1 \
            ./$BUILD/lulesh_$V -s $LULESH_SIZE 2>&1 | tee run.tmp \
            || echo "FAILED: lulesh_$V (no fabric) at $N ranks"
        NOFABRIC_ELAPSED[${V}_${N}]=$(awk '/Elapsed time/{print $4}' run.tmp)
    done
done

echo ""
echo "Fabric vs. no-fabric comparison (elapsed seconds, lower is better):"
printf "  %-12s %-8s %-14s %-14s\n" "variant" "ranks" "fabric" "no-fabric (MPI)"
for N in $RANKS_LIST; do
    for V in mpiwrap mpiwrap_rp; do
        printf "  %-12s %-8s %-14s %-14s\n" "$V" "$N" \
            "${FABRIC_ELAPSED[${V}_${N}]:-n/a}" "${NOFABRIC_ELAPSED[${V}_${N}]:-n/a}"
    done
done

rm -f run.tmp
echo "Done."
