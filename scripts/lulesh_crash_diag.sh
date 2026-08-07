#!/bin/bash
# Diagnose the LULESH 27/64-rank failures reported on Ptyche (GB200 NVL72).
#
# WHAT WE ALREADY KNOW, and why this script is shaped the way it is.
#
# At 8 ranks (tp=2) every LULESH domain sits in a corner of the 2x2x2 rank
# grid, so it has 7 neighbours.  At 27 ranks (tp=3) an interior domain
# appears with the full 26 neighbours, and 64 (tp=4) has eight of them.  The
# 8-vs-27 boundary is therefore a NEIGHBOUR-COUNT boundary (7 -> 26), not a
# node-count boundary -- 8 ranks already spans 2 nodes on this machine.
#
# The variants that failed (gpumpi, ipc, ipc_rp) are exactly those that hand
# DEVICE pointers to MPI: gpumpi always, and ipc/ipc_rp for every peer whose
# cudaIpcOpenMemHandle could not be opened -- which on 4-GPU nodes is ~23 of
# an interior domain's 26 neighbours, since cudaIpc handles never cross a
# node.  The variants that survived (staged, mpiwrap) are exactly those that
# never do: staged uses host buffers, mpiwrap serves every peer including
# cross-node ones from the fabric window.
#
# staged vs gpumpi is already a controlled single-variable comparison --
# identical message pattern, counts and rank geometry, differing only in
# whether the recv buffer is host or device memory (comm/comm_staged.h vs
# comm/comm_gpumpi.h).  staged passes at 27 and gpumpi does not.  So the
# leading hypothesis is the CUDA-aware path of this HPC-X/UCX build, scaling
# with peers x outstanding requests, and NOT the halo-exchange code.
#
# This script tries to falsify that.  Total runtime is a few minutes; it is
# meant for a short interactive allocation, not a queue round-trip.
#
# Usage, from the repo root inside an allocation that covers 64 GPUs:
#   bash scripts/lulesh_crash_diag.sh 2>&1 | tee lulesh_diag.log
# Override the rank list if the allocation is smaller (27 is the cheapest
# reproducer, so RANKS="8 27" is a useful short form):
#   RANKS="8 27" bash scripts/lulesh_crash_diag.sh
#
# SEND BACK: the whole lulesh_diag.log plus the per-run files in $OUTDIR.
# The three questions it answers are listed at the bottom of the script.

set -uo pipefail

BUILD=${BUILD:-build}
RANKS=${RANKS:-8 27 64}
SIZE=${SIZE:-45}
OUTDIR=${OUTDIR:-lulesh_diag_out}
MPIWRAP_LIB=$PWD/$BUILD/libmpiwrap.so
mkdir -p "$OUTDIR"

if [ -z "${LAUNCH:-}" ]; then
    # Ptyche needs --ntasks-per-node so srun pins one rank per GPU; srun
    # derives the node count from -n.  See scripts/ptyche_nvl_run.sh.
    LAUNCH=${LAUNCH:-"srun --mpi=pmix --ntasks-per-node=4 -n"}
fi
echo "launcher: '$LAUNCH'   build: $BUILD   size: -s $SIZE   ranks: $RANKS"
echo "output dir: $OUTDIR"

# run <tag> <ranks> <variant> [env assignments...]
# Captures stdout+stderr per run, records the exit status, and echoes the
# tail on failure.  Everything here hinges on seeing the actual message.
run() {
    local tag=$1 n=$2 v=$3 ; shift 3
    local pre="" log="$OUTDIR/${tag}_${v}_${n}.log"
    case $v in mpiwrap*) pre="env LD_PRELOAD=$MPIWRAP_LIB" ;; *) pre="env" ;; esac
    echo "--- [$tag] lulesh_$v, $n ranks ${*:-} ---"
    # shellcheck disable=SC2086
    $LAUNCH "$n" $pre "$@" ./"$BUILD"/lulesh_"$v" -s "$SIZE" >"$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "    rc=0  $(awk '/Final Origin Energy/{print "energy="$5}' "$log")" \
             "$(awk '/Elapsed time/{print "elapsed="$4}' "$log")"
    else
        echo "    rc=$rc  FAILED -- last 25 lines of $log:"
        tail -25 "$log" | sed 's/^/      | /'
    fi
    # The transport mix decides how to label the run; both IPC backends now
    # print it unconditionally.
    grep -h -E "peers not IPC-reachable|fabric window" "$log" | sort -u | sed 's/^/    > /'
    return $rc
}

echo
echo "======================================================================"
echo "A. Is it immediate or does it accumulate?  (-i 1 = one timestep)"
echo "   Immediate failure => deterministic bug in setup or first exchange."
echo "   Runs a while then dies => resource growth over thousands of steps,"
echo "   which is the UCX-side signature."
echo "======================================================================"
for N in $RANKS; do
    for V in staged gpumpi ipc mpiwrap; do
        echo "--- [oneiter] lulesh_$V, $N ranks, -i 1 ---"
        log="$OUTDIR/oneiter_${V}_${N}.log"
        case $V in mpiwrap*) pre="env LD_PRELOAD=$MPIWRAP_LIB" ;; *) pre="env" ;; esac
        # shellcheck disable=SC2086
        $LAUNCH "$N" $pre ./"$BUILD"/lulesh_"$V" -s "$SIZE" -i 1 >"$log" 2>&1
        rc=$?
        echo "    rc=$rc"
        [ $rc -ne 0 ] && tail -20 "$log" | sed 's/^/      | /'
        grep -h -E "peers not IPC-reachable|fabric window" "$log" | sort -u | sed 's/^/    > /'
    done
done

echo
echo "======================================================================"
echo "B. Baseline reproduction with per-run logs kept."
echo "   Confirms the reported pattern and, unlike the summary table, keeps"
echo "   the error text.  staged is the control that must pass everywhere."
echo "======================================================================"
for N in $RANKS; do
    for V in staged gpumpi ipc ipc_rp mpiwrap mpiwrap_rp; do
        run base "$N" "$V"
    done
    [ -x ./"$BUILD"/lulesh_nvshmem ] && run base "$N" nvshmem
done

echo
echo "======================================================================"
echo "C. Push gpumpi off the UCX CUDA fast paths, at the smallest failing"
echo "   rank count.  If ANY of these three survives where the plain run"
echo "   died, the fault is UCX configuration and the halo-exchange code is"
echo "   exonerated -- that is the single most informative bit here."
echo
echo "   NB: these settings distort timing badly (pinning UCX_TLS fabricates"
echo "   a lazy-connection-setup cost -- see results/transpose_results.md)."
echo "   Use them to decide crash-vs-no-crash ONLY.  Do not take numbers."
echo "======================================================================"
DIAG_N=""
for N in $RANKS; do [ "$N" != 8 ] && { DIAG_N=$N ; break ; } ; done
if [ -z "$DIAG_N" ]; then
    echo "SKIP: RANKS has no count above 8; nothing failing to probe."
else
    echo "using $DIAG_N ranks (smallest failing count in RANKS)"
    # Eager for every size: skips the rendezvous protocol and its per-peer
    # GPU rkey/registration state, the most likely thing to exhaust.
    run rndv    "$DIAG_N" gpumpi UCX_RNDV_THRESH=inf
    # The memtype cache is a recurring source of wrong-memory-type decisions.
    run memtype "$DIAG_N" gpumpi UCX_MEMTYPE_CACHE=n
    # Exclude the cuda_ipc transport, keeping cuda_copy: isolates whether the
    # intra-node GPU transport is what falls over under 26 peers.
    run nocudaipc "$DIAG_N" gpumpi UCX_TLS=^cuda_ipc
    # Same three against ipc, which reaches the identical code path through
    # its cross-node fallback.
    run rndv    "$DIAG_N" ipc UCX_RNDV_THRESH=inf
fi

echo
echo "======================================================================"
echo "D. Version skew check: HPC-X is a cuda12 build, the toolkit is 13.0."
echo "======================================================================"
echo "MPI_HOME=${MPI_HOME:-unset}"
command -v ucx_info >/dev/null && ucx_info -v
"${BUILD}"/lulesh_staged --help >/dev/null 2>&1
for b in lulesh_gpumpi lulesh_ipc lulesh_mpiwrap; do
    [ -x ./"$BUILD"/$b ] && echo "$b -> $(ldd ./"$BUILD"/$b | grep -E 'libucp|libcudart|libmpi' | tr -s ' ')"
done

echo
echo "======================================================================"
echo "What this answers"
echo "======================================================================"
cat <<'EOF'
 1. Section A: does the failure need thousands of timesteps, or is it there
    at step one?  This separates a deterministic exchange bug from resource
    growth in the MPI layer.
 2. Section B: the error text itself, per rank count and variant, which the
    summary table did not carry.  Specifically: is it a segfault (and in
    whose stack -- libucp, libcudart, or lulesh), an MPI_Abort with a
    message, a CUDA error now named by COMM_CUDA_OK, or a hang killed by
    the time limit?
 3. Section C: whether any UCX setting makes gpumpi survive at the failing
    rank count.  A single survivor here settles it.
Also worth reporting: the "N of M peers not IPC-reachable" lines.  If ipc
reports ~23 of 26 at 27 ranks, then lulesh_ipc at multi-node scale is mostly
GPU-aware MPI regardless of whether it runs, and that column cannot be
labelled CUDA IPC in the paper.
EOF
echo "Done.  Send back this log plus $OUTDIR/."
