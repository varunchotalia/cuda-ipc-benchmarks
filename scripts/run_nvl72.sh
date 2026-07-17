#!/bin/bash
# Inter-node benchmark suite for multi-node NVLink systems (GB200 NVL72).
# Scheduler-agnostic: set LAUNCH to your launcher prefix, e.g.
#   LAUNCH="mpirun -np"            (default)
#   LAUNCH="srun --mpi=pmix -n"
# Run from the repo root after building:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 && cmake --build build -j
#   NVSHMEM_HOME=/path/to/nvshmem bash scripts/run_nvl72.sh
#
# What to look for:
#  - interposer log line "fabric window: N ranks ..." confirms the CUDA
#    fabric-handle path (multi-node NVLink) is active; "N of M peers not
#    IPC-reachable" means it fell back to per-peer hybrid MPI instead
#  - LULESH: Final Origin Energy must match staged at every rank count
#
# Requirements on the target system: CUDA >= 12.4 driver stack with the
# IMEX daemon running (for fabric handles), CUDA-aware MPI (for the ipc/
# ipc_rp hybrid fallback paths), NVSHMEM with a working bootstrap.

set -uo pipefail

BUILD=${BUILD:-build}
LAUNCH=${LAUNCH:-mpirun -np}
RANKS_LIST=${RANKS_LIST:-8 27 64}        # LULESH needs cubic rank counts
LULESH_SIZE=${LULESH_SIZE:-45}           # per-rank problem size
MPIWRAP_LIB=$PWD/$BUILD/libmpiwrap.so
export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}

echo "launcher: '$LAUNCH', build dir: $BUILD"
echo "======================================================================"
echo "1/3 transpose (4 GPUs; direct/buffered ride the interposer windows)"
echo "======================================================================"
for V in staged gpumpi buffered direct direct_single; do
    case $V in buffered|direct*) PRE="env LD_PRELOAD=$MPIWRAP_LIB" ;; *) PRE="" ;; esac
    echo "--- transpose_$V ---"
    $LAUNCH 4 $PRE ./$BUILD/transpose_$V 100 8192 || echo "FAILED: transpose_$V"
done
if [ -x ./$BUILD/transpose_nvshmem_direct ]; then
    for V in nvshmem_direct nvshmem_buffered; do
        echo "--- transpose_$V ---"
        $LAUNCH 4 ./$BUILD/transpose_$V 100 8192 || echo "FAILED: transpose_$V"
    done
fi

echo "======================================================================"
echo "2/3 stencil (4 GPUs; ipc rides the interposer windows)"
echo "======================================================================"
$LAUNCH 4 env LD_PRELOAD=$MPIWRAP_LIB ./$BUILD/stencil_ipc || echo "FAILED: stencil_ipc"
$LAUNCH 4 ./$BUILD/stencil_mpi || echo "FAILED: stencil_mpi"
if [ -x ./$BUILD/stencil_nvshmem ]; then
    $LAUNCH 4 ./$BUILD/stencil_nvshmem || echo "FAILED: stencil_nvshmem"
fi

echo "======================================================================"
echo "3/3 LULESH at ${RANKS_LIST} ranks, -s $LULESH_SIZE per rank"
echo "  inter-node capable: staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp nvshmem"
echo "  (shmwin and direct are single-node by construction and are skipped)"
echo "======================================================================"
VARIANTS="staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp"
[ -x ./$BUILD/lulesh_nvshmem ] && VARIANTS="$VARIANTS nvshmem"

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
rm -f run.tmp
echo "Done."
