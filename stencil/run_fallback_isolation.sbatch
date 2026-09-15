#!/bin/bash
#SBATCH --job-name=stencil_fallback_iso
#SBATCH --partition=h200x8
#SBATCH --nodelist=h200x8-04
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=8
#SBATCH --time=00:15:00
#SBATCH --output=stencil_fallback_iso_%j.out

# Is the stencil's MPI fallback path itself broken?
#
# WHERE WE ARE. Job 92211 proved the platform lies: on h200x8-04 every one of
# the 32 cross-island pairs opens its IPC handle, reports
# cudaDeviceCanAccessPeer=true, and then delivers NOTHING -- while all 24
# same-island pairs deliver correctly. Commit 63df376 replaced the reachability
# gate with a delivery probe, and job 92654 shows it works: 4 of 7 peers are
# correctly reported unreachable per rank, and both boundary ranks print
# "not IPC-reachable ... using MPI send/recv fallback".
#
# And np=8 STILL returns L2 0.6600931148 -- byte-identical to the value before
# the gate existed. Per-rank norms localise it exactly: only rank 4 is nonzero,
# and rank 4 is where the heat source sits. Its left edge never reaches rank 3.
# That single exchange is the cross-island pair, and it is now supposed to be
# going over MPI.
#
# THE GAP IN OUR TESTING. Every previous run either used one island (np=2, np=4
# -- no fallback needed) or spanned both (np=8 -- fallback needed and the answer
# wrong). The fallback path has therefore never once been observed to work. It
# may be broken in general rather than at the boundary specifically, and nothing
# we have run so far can tell those apart.
#
# THIS JOB. CUDA_VISIBLE_DEVICES=3,4 maps the two ranks onto physical GPUs 3 and
# 4, one per island. With np=2 there is exactly one neighbour pair, it straddles
# the boundary, and it MUST use the MPI fallback. Nothing else is in the way.
#
#   correct L2 (5.1449605829) -> the fallback works; the np=8 failure is
#     something else, e.g. a pair where only one side detects the problem.
#   wrong L2                  -> the fallback is broken outright, which is the
#     simpler explanation and makes the np=8 result a symptom rather than a bug
#     of its own.
#
# The np=2 same-island control runs first, so a wrong answer cannot be blamed on
# np=2 being untested.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
ARCH="-gencode arch=compute_90,code=sm_90"
export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ==="
echo "=== stencil_ipc.cu md5=$(md5sum stencil/stencil_ipc.cu | cut -d' ' -f1) ==="

SIZE=1024
sed "s/const int N = [0-9]*/const int N = $SIZE/; \
     s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" \
    stencil/stencil_ipc.cu > /tmp/fbiso_$SIZE.cu
nvcc -O3 $ARCH /tmp/fbiso_$SIZE.cu -o /tmp/fbiso_$SIZE -lmpi || { echo BUILD FAILED; exit 1; }
echo "build done"

run() {   # $1=label $2=CUDA_VISIBLE_DEVICES
    echo ""
    echo "########## np=2  GPUs=$2  ($1) ##########"
    STENCIL_WARMUP=20 STENCIL_SYNC=barrier CUDA_VISIBLE_DEVICES=$2 \
      mpirun -np 2 -x STENCIL_WARMUP -x STENCIL_SYNC -x UCX_LOG_LEVEL \
             -x CUDA_VISIBLE_DEVICES \
             env LD_PRELOAD=$MPIWRAP /tmp/fbiso_$SIZE 2>&1
    echo "########## end ($1) ##########"
}

run "control: same island, IPC path"  "0,1"
run "TEST: straddles boundary, MPI fallback path" "3,4"

echo ""
echo "=== READ THIS ==="
echo "=== Both runs must give Global L2 norm 5.1449605829."
echo "=== The control exercises the IPC path and should pass; if it does not,"
echo "===   nothing else here means anything."
echo "=== The TEST forces the MPI fallback and is the question. A wrong L2"
echo "===   there means the fallback itself does not deliver, which would"
echo "===   explain np=8 without needing a second bug."
