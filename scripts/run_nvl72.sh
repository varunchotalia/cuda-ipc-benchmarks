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
# What to look for:
#  - interposer log line "fabric window: N ranks ..." confirms the CUDA
#    fabric-handle path (multi-node NVLink) is active; "N of M peers not
#    IPC-reachable" means it fell back to per-peer hybrid MPI instead
#  - LULESH: Final Origin Energy must match staged at every rank count
#  - section 4/4 repeats LULESH mpiwrap/mpiwrap_rp with the fabric path
#    forced off (MPIWRAP_DISABLE_FABRIC=1), for a fabric-vs-hybrid-MPI
#    apples-to-apples comparison at the same rank counts
#
# Requirements on the target system: CUDA >= 12.4 driver stack with the
# IMEX daemon running (for fabric handles), CUDA-aware MPI (for the ipc/
# ipc_rp hybrid fallback paths), NVSHMEM with a working bootstrap.

set -uo pipefail

BUILD=${BUILD:-build}

# --- Launcher auto-detection (skipped if LAUNCH is already set) -----------
if [ -z "${LAUNCH:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
        LAUNCH="srun --mpi=pmix -n"
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
TRANSPOSE_RANKS_LIST=${TRANSPOSE_RANKS_LIST:-$RANKS_LIST}
STENCIL_RANKS_LIST=${STENCIL_RANKS_LIST:-8 64}
TRANSPOSE_ITERS=${TRANSPOSE_ITERS:-100}
TRANSPOSE_ORDER=${TRANSPOSE_ORDER:-6912} # divisible by 8, 27, and 64
LULESH_SIZE=${LULESH_SIZE:-45}           # per-rank problem size
MPIWRAP_LIB=$PWD/$BUILD/libmpiwrap.so
export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}

echo "launcher: '$LAUNCH', build dir: $BUILD"
echo "transpose ranks: ${TRANSPOSE_RANKS_LIST}; stencil ranks: ${STENCIL_RANKS_LIST}; LULESH ranks: ${RANKS_LIST}"
echo "======================================================================"
echo "1/4 transpose; direct/buffered ride the interposer windows"
echo "======================================================================"
for N in $TRANSPOSE_RANKS_LIST; do
    if [ $((TRANSPOSE_ORDER % N)) -ne 0 ]; then
        echo "SKIP: transpose at $N ranks needs TRANSPOSE_ORDER divisible by $N"
        continue
    fi
    echo "########## transpose, $N ranks, order $TRANSPOSE_ORDER ##########"
    for V in staged gpumpi buffered direct direct_single; do
        case $V in buffered|direct*) PRE="env LD_PRELOAD=$MPIWRAP_LIB" ;; *) PRE="" ;; esac
        echo "--- transpose_$V ---"
        $LAUNCH $N $PRE ./$BUILD/transpose_$V $TRANSPOSE_ITERS $TRANSPOSE_ORDER \
            || echo "FAILED: transpose_$V at $N ranks"
    done
    if [ -x ./$BUILD/transpose_nvshmem_direct ]; then
        for V in nvshmem_direct nvshmem_buffered; do
            echo "--- transpose_$V ---"
            $LAUNCH $N ./$BUILD/transpose_$V $TRANSPOSE_ITERS $TRANSPOSE_ORDER \
                || echo "FAILED: transpose_$V at $N ranks"
        done
    fi
done

echo "======================================================================"
echo "2/4 stencil; ipc rides the interposer windows"
echo "======================================================================"
for N in $STENCIL_RANKS_LIST; do
    echo "########## stencil, $N ranks ##########"
    $LAUNCH $N env LD_PRELOAD=$MPIWRAP_LIB ./$BUILD/stencil_ipc \
        || echo "FAILED: stencil_ipc at $N ranks"
    $LAUNCH $N ./$BUILD/stencil_mpi || echo "FAILED: stencil_mpi at $N ranks"
    $LAUNCH $N ./$BUILD/stencil_gpumpi || echo "FAILED: stencil_gpumpi at $N ranks"
    if [ -x ./$BUILD/stencil_nvshmem ]; then
        $LAUNCH $N ./$BUILD/stencil_nvshmem || echo "FAILED: stencil_nvshmem at $N ranks"
    fi
done

echo "======================================================================"
echo "3/4 LULESH at ${RANKS_LIST} ranks, -s $LULESH_SIZE per rank"
echo "  inter-node capable: staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp nvshmem"
echo "  (shmwin and direct are single-node by construction and are skipped)"
echo "======================================================================"
VARIANTS="staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp"
[ -x ./$BUILD/lulesh_nvshmem ] && VARIANTS="$VARIANTS nvshmem"

declare -A FABRIC_ELAPSED   # keyed "$V_$N", for the section 4 comparison

for N in $RANKS_LIST; do
    declare -A ENERGY ELAPSED
    echo ""
    echo "########## $N ranks ##########"
    for V in $VARIANTS; do
        case $V in mpiwrap*) PRE="env LD_PRELOAD=$MPIWRAP_LIB" ;; *) PRE="" ;; esac
        echo "--- lulesh_$V, $N ranks ---"
        $LAUNCH $N $PRE ./$BUILD/lulesh_$V -s $LULESH_SIZE 2>&1 | tee run.tmp \
            || echo "FAILED: lulesh_$V at $N ranks"
        ENERGY[$V]=$(awk '/Final Origin Energy/{print $5}' run.tmp)
        ELAPSED[$V]=$(awk '/Elapsed time/{print $4}' run.tmp)
        case $V in mpiwrap*) FABRIC_ELAPSED[${V}_${N}]=${ELAPSED[$V]:-} ;; esac
    done
    echo ""
    echo "Summary, $N ranks:"
    printf "  %-12s %-16s %s\n" "variant" "energy" "elapsed(s)"
    for V in $VARIANTS; do
        CHECK="MATCH"
        [ -z "${ENERGY[$V]:-}" ] && CHECK="MISSING"
        [ -n "${ENERGY[$V]:-}" ] && [ "${ENERGY[$V]}" != "${ENERGY[staged]:-}" ] && CHECK="MISMATCH"
        printf "  %-12s %-16s %-10s %s\n" "$V" "${ENERGY[$V]:-}" "${ELAPSED[$V]:-}" "$CHECK"
    done
    unset ENERGY ELAPSED
done

echo ""
echo "======================================================================"
echo "4/4 LULESH mpiwrap/mpiwrap_rp with the fabric path forced OFF"
echo "  (MPIWRAP_DISABLE_FABRIC=1: cross-node peers fall back to real MPI"
echo "   send/recv, same as running on hardware with no fabric support)"
echo "======================================================================"
declare -A NOFABRIC_ELAPSED
for N in $RANKS_LIST; do
    echo ""
    echo "########## $N ranks, fabric disabled ##########"
    for V in mpiwrap mpiwrap_rp; do
        echo "--- lulesh_$V, $N ranks, MPIWRAP_DISABLE_FABRIC=1 ---"
        $LAUNCH $N env LD_PRELOAD=$MPIWRAP_LIB MPIWRAP_DISABLE_FABRIC=1 \
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
