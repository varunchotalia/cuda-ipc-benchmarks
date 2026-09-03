#!/bin/bash
#SBATCH --job-name=stencil_island_diag
#SBATCH --partition=h200x8
#SBATCH --nodelist=h200x8-04
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=00:20:00
#SBATCH --output=stencil_island_diag_%j.out

# Why does the IPC stencil give the wrong answer at 8 ranks on h200x8-04?
#
# h200x8-04 is a two-island node: GPUs 0-3 are NVLink peers, GPUs 4-7 are
# NVLink peers, and across the 3/4 boundary it is PCIe only (SYS), so no CUDA
# IPC. h200x8-03 is NVSwitch all-to-all and has no such boundary.
#
# Job 70454 on -04 gave L2 0.6600931148 at np=8 for ALL THREE handshakes
# (barrier, neighbour token, ssend) while np=2 and np=4 gave the correct
# 5.1449605829. np=2 and np=4 fit inside one island; np=8 does not. The
# handshake is therefore not the variable -- crossing the island boundary is.
#
# Two candidate causes, which this job separates:
#
#   (A) The fallback fires but is wrong. Ranks 3 and 4 correctly detect that
#       the peer is unreachable, switch to MPI send/recv, and STILL produce
#       bad ghost data.
#   (B) The fallback never fires. MPI_Win_shared_query wrongly reports the
#       across-island peer as reachable, so we write through a pointer that
#       does not map, and the data silently never arrives.
#
# (B) is the worse outcome: the paper states in Section III-C that a failed
# peer query is reported so the application can fall back to MPI. If the query
# does not fail where it should, that claim needs qualifying.
#
# ANSWERED by job 83189: case (B). Ranks 3 and 4 both printed "IPC setup
# complete", never "not IPC-reachable", and np=8 returned 0.6600931148
# deterministically at full speed. The peer query claimed a cross-island peer
# was reachable. Root cause: cudaIpcOpenMemHandle is asked for LAZY peer
# enablement, so it returns success across the island boundary; the interposer
# inferred reachability from that success alone.
#
# FIXED by gating on cudaDeviceCanAccessPeer in MPI_Win_shared_query
# (mpi-intercept/mpiwrap_ipc.cc). This job is now the VERIFICATION run.
#
# HOW TO READ THE OUTPUT: this job prints the FULL per-rank output.
#   PASS -> ranks 3 and 4 print "not IPC-reachable", take the MPI fallback,
#           and np=8 gives L2 5.1449605829.
#   FAIL -> np=8 still gives 0.6600931148, or the message is still absent.
# np=4 must stay correct and must NOT print the message (one island).
#
# np=4 is included as the control: it fits in one island and must be correct.

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

echo ""
echo "### NVLink topology of this node ###"
nvidia-smi topo -m 2>&1 | head -14

SIZE=1024
sed "s/const int N = [0-9]*/const int N = $SIZE/; \
     s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" \
    stencil/stencil_ipc.cu > /tmp/isl_ipc_$SIZE.cu
nvcc -O3 $ARCH /tmp/isl_ipc_$SIZE.cu -o /tmp/isl_ipc_$SIZE -lmpi \
    || { echo "BUILD FAILED"; exit 1; }
echo "build done"

for NP in 4 8; do
    echo ""
    echo "############ np=$NP size=$SIZE barrier -- FULL OUTPUT ############"
    STENCIL_WARMUP=20 STENCIL_SYNC=barrier \
      mpirun -np $NP -x STENCIL_WARMUP -x STENCIL_SYNC -x UCX_LOG_LEVEL \
      env LD_PRELOAD=$MPIWRAP /tmp/isl_ipc_$SIZE 2>&1
    echo "############ end np=$NP ############"
done

echo ""
echo "=== done $(date -Is) ==="
echo "=== np=4 must give L2 5.1449605829 (one island)."
echo "=== np=8 gave 0.6600931148 in job 70454. Check whether ranks 3 and 4"
echo "===   printed 'not IPC-reachable'. Present = broken fallback (A);"
echo "===   absent = query wrongly claims the peer is reachable (B)."
