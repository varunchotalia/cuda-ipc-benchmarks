#!/bin/bash
#SBATCH --job-name=transpose_val_ctrl
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --output=transpose_val_ctrl_%j.out

# Does the transpose verifier actually check remotely delivered data?
#
# ---- WHY THIS JOB EXISTS -------------------------------------------------
# Every transpose run ever recorded prints "Solution validates". That single
# observation is consistent with TWO different worlds:
#
#   (a) the verifier inspects the whole destination matrix, including the
#       sub-blocks that arrived from peers, and the data really is correct; or
#   (b) the verifier is blind to remote arrivals and would print "validates"
#       no matter what showed up.
#
# A test that has never failed cannot distinguish them. NVL_SUMMARY.md §6 Q4
# raised exactly this doubt about the single-kernel variant ("reports
# validation success on work it does not appear to perform"), and that doubt
# is currently load-bearing: the abstract's headline 3.0x over GPU-aware MPI
# is IPC single-kernel at 1024^2.
#
# Static reading of transpose_ipc.cu says we are in world (a):
#   - The verifier copies the rank's ENTIRE B block to the host and checks
#     every element against the closed form order*i + (j+colstart). Rank j's B
#     block is filled by sub-blocks from every other rank, so checking it
#     checks remote arrivals.
#   - The normalisation is right: each rank checks order*Bo elements, and
#     across P ranks that is exactly order^2, which is the divisor.
#   - Single-kernel mode MPI_Aborts if any peer fails to map (it has no MPI
#     fallback), so it cannot silently skip a peer in the field.
# This job converts that reading into a measurement.
#
# ---- THE CONTROL ---------------------------------------------------------
# TRANSPOSE_SKIP_PEER=<rank> suppresses the fused kernel's WRITE to one peer
# and nothing else. The tile load and, under ACCUMULATE, the src[idx] += 1.0
# bookkeeping still run, so the ONLY thing that changes is that one peer's
# block never arrives -- a clean single-variable perturbation. Skipping any
# earlier would also disturb the local accumulate accounting and the verifier
# would fail for the wrong reason, telling us nothing.
#
#   BASELINE (no skip)  -> must print "Solution validates"
#   CONTROL  (skip set) -> must print "NEGATIVE CONTROL PASSED"
#
# Any "NEGATIVE CONTROL FAILED" line means the verifier does not inspect
# remote data, and every "Solution validates" in the record -- including the
# numbers behind Fig. 3 and Table V -- is unproven. That is the finding this
# job exists to surface, so it is deliberately loud.

set -uo pipefail

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap
export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
ARCH="-gencode arch=compute_90,code=sm_90"

export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ($(git log -1 --pretty=%s)) ==="
echo "=== transpose_ipc.cu md5=$(md5sum transpose/transpose_ipc.cu | cut -d' ' -f1) ==="

# ---- build: single-kernel only (the control lives in that path) ----------
for ACC in 0 1; do
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC \
        transpose/transpose_ipc.cu -o /tmp/tvc_single_$ACC -lmpi \
        -I$MPI_HOME/include -L$MPI_HOME/lib || { echo "BUILD FAILED acc=$ACC"; exit 1; }
done
echo "build done"

# NB: the binary takes <iterations> <matrix_order> in that order, not the
# reverse. Warmup is left at its default of 1; the ACCUMULATE expected-value
# formula derives from warmup+iterations, so every rank agrees without -x.
# Baseline arms must run with TRANSPOSE_SKIP_PEER genuinely ABSENT, not set to
# a sentinel. Job 66045 died on exactly that: it set TRANSPOSE_SKIP_PEER=NA and
# relied on withholding `-x` to keep it from the ranks. That reasoning is wrong
# on a single node -- mpirun forks the ranks locally, so they inherit its whole
# environment and `-x` is irrelevant. Every rank saw "NA", the binary's
# validation guard correctly rejected it, and all 16 cells aborted. `env -u`
# removes the variable outright and does not depend on launcher semantics.
run_one() {   # $1=accum $2=order $3=skip-or-NA
    local ACC=$1 ORD=$2 SKIP=$3 OUT
    if [ "$SKIP" = "NA" ]; then
        OUT=$(env -u TRANSPOSE_SKIP_PEER mpirun -np 4 -x UCX_LOG_LEVEL \
              env LD_PRELOAD=$MPIWRAP /tmp/tvc_single_$ACC 50 $ORD 2>&1)
    else
        OUT=$(TRANSPOSE_SKIP_PEER=$SKIP mpirun -np 4 -x UCX_LOG_LEVEL \
              -x TRANSPOSE_SKIP_PEER \
              env LD_PRELOAD=$MPIWRAP /tmp/tvc_single_$ACC 50 $ORD 2>&1)
    fi
    local VERDICT
    VERDICT=$(echo "$OUT" | grep -oE 'Solution validates|NEGATIVE CONTROL (PASSED|FAILED)|ERROR: Per-element error [0-9.e+-]+' | head -1)
    echo "CONTROL accum=$ACC order=$ORD skip=$SKIP -> ${VERDICT:-NO-VERDICT-LINE}"
    if [ -z "$VERDICT" ]; then
        echo "---- no verdict, full output ----"; echo "$OUT"; echo "---- end ----"
    fi
}

echo ""
echo "### BASELINE -- no skip. Every line must read 'Solution validates'. ###"
for ACC in 0 1; do for ORD in 1024 4096; do run_one $ACC $ORD NA; done; done

echo ""
echo "### CONTROL -- one peer withheld. Every line must read 'NEGATIVE CONTROL PASSED'. ###"
# Rank 0 reports; skip each of its three peers in turn so the result cannot
# hinge on one particular block position.
for ACC in 0 1; do for ORD in 1024 4096; do for SKIP in 1 2 3; do
    run_one $ACC $ORD $SKIP
done; done; done

echo ""
echo "=== done $(date -Is) ==="
echo "=== READ THIS ==="
echo "=== PASS: all BASELINE lines say 'Solution validates' AND all CONTROL"
echo "===       lines say 'NEGATIVE CONTROL PASSED'. The verifier has teeth,"
echo "===       and NVL_SUMMARY 6.Q4 can be closed for the validation half."
echo "=== FAIL: any CONTROL line says 'NEGATIVE CONTROL FAILED'. The verifier"
echo "===       does not check remote data; Fig. 3 / Table V are unproven and"
echo "===       the 3.0x abstract claim must be re-established before"
echo "===       camera-ready."
