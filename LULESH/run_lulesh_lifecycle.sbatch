#!/bin/bash
# E2a -- LULESH lifecycle / setup-cost experiment.
#
# Closes the largest standing limitation in the LULESH case study: the reported
# elapsed time and FOM cover only the solve loop, so no number says what the
# halo-exchange abstraction costs to SET UP or TEAR DOWN.
#
# Measures, per independent launch (see lulesh.cu PHASE_* report):
#   PHASE_COMM_SETUP_ms    whole commAllocRecv(). Uniform across ALL nine
#                          backends -- this is the only cross-backend
#                          comparable setup number.
#   PHASE_WIN_ALLOCATE_ms  breakdown: MPI_Win_allocate only. Window-based
#                          backends only; 0.0000 means NO BREAKDOWN
#                          AVAILABLE, not "free". Currently only mpiwrap*.
#   PHASE_PEER_QUERY_ms    breakdown: the MPI_Win_shared_query loop (each
#                          query triggers at most one cudaIpcOpenMemHandle).
#   PHASE_COMM_FREE_ms     whole commTeardown().
#   PHASE_APP_TOTAL_ms     end-to-end, MPI_Init to just before MPI_Finalize.
#   Elapsed/FOM            the existing solve-loop numbers, for amortisation.
#
# RANKS: LULESH requires a perfect cube (lulesh.cu:4598 -- "Num processors must
# be a cube of an integer"). So the sweep is 1 and 8 ONLY. 2 and 4 abort; 27
# would need 27 GPUs. The 1-rank point is the floor: no peers, so it isolates
# window construction from peer mapping.
#
# REPS independent mpirun launches per configuration -- not repeated setup
# inside one process -- so driver/first-touch state is re-paid each time and
# launch-to-launch variation is reportable.
#
# Output is NOT filtered. Job 59853 grepped stdout down to three fields and
# silently discarded the interposer's IPC-fallback notices, which is why we
# still cannot say whether that run's 8-rank series was pure IPC or hybrid.
# Everything goes to stdout; RESULT lines are additive, not a replacement.
#
#SBATCH --job-name=lulesh_lifecycle
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=lulesh_lifecycle_%j.out
# Pinned to h200x8-03 (the only H200 SXM / NVSwitch all-to-all node) so these
# numbers are comparable with the authoritative baseline in
# results/lulesh_results.csv (job 60150, same node).
#SBATCH --nodelist=h200x8-03

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0 2>/dev/null || \
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

set -euo pipefail

export NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
# UCX_TLS deliberately NOT pinned -- UCX defaults are authoritative. Pinning it
# inflated LULESH staged 1.96 -> 3.56 s and cost transpose staged 2.6x.
export UCX_LOG_LEVEL=error

export REPO=/lustre/nvwulf/home/vchotalia/mpiwrap

echo "=================================================================="
echo "NODE: $(hostname)  date=$(date -Is)  UCX_TLS=${UCX_TLS:-<unset>}"
echo "=================================================================="

# Build the interposer from source first and record its md5 -- sets
# MPIWRAP_LIB. See scripts/build_mpiwrap.sh for why this is not optional.
source $REPO/scripts/build_mpiwrap.sh

cd $REPO/LULESH/cuda/src
make -j8 all
echo "build done"

echo "--- NVLink topology ---"
nvidia-smi topo -m 2>&1

SIZE=45
REPS=5
RANKS="1 8"
VARIANTS="staged gpumpi shmwin ipc ipc_rp mpiwrap mpiwrap_rp nvshmem direct"

for NP in $RANKS; do
  for V in $VARIANTS; do
    case $V in
        mpiwrap|mpiwrap_rp) PRELOAD="-x LD_PRELOAD=$MPIWRAP_LIB" ;;
        *)                  PRELOAD="" ;;
    esac
    for r in $(seq 1 $REPS); do
        echo ""
        echo "########## np=$NP variant=$V rep=$r -s $SIZE (FULL OUTPUT) ##########"
        OUT=$(mpirun -np $NP $PRELOAD -x UCX_LOG_LEVEL -x NVSHMEM_BOOTSTRAP \
              ./lulesh_$V -s $SIZE 2>&1) || OUT="$OUT
RUN FAILED: $V np=$NP rep=$r"
        # full, unfiltered
        echo "$OUT"
        S=$(echo "$OUT"  | grep -oP 'PHASE_COMM_SETUP_ms\s+\K[0-9.]+'   || true)
        W=$(echo "$OUT"  | grep -oP 'PHASE_WIN_ALLOCATE_ms\s+\K[0-9.]+' || true)
        Q=$(echo "$OUT"  | grep -oP 'PHASE_PEER_QUERY_ms\s+\K[0-9.]+'   || true)
        F=$(echo "$OUT"  | grep -oP 'PHASE_COMM_FREE_ms\s+\K[0-9.]+'    || true)
        A=$(echo "$OUT"  | grep -oP 'PHASE_APP_TOTAL_ms\s+\K[0-9.]+'    || true)
        E=$(echo "$OUT"  | awk '/Final Origin Energy/{print $5}'        || true)
        T=$(echo "$OUT"  | awk '/Elapsed time/{print $4}'               || true)
        M=$(echo "$OUT"  | awk '/FOM/{print $3}'                        || true)
        # how many peers fell back to MPI send/recv, if the backend said so
        NF=$(echo "$OUT" | grep -oP '\K[0-9]+(?= of [0-9]+ peers not IPC-reachable)' || true)
        echo "RESULT np=$NP variant=$V rep=$r setup_ms=${S:-FAIL} win_alloc_ms=${W:-NA} peer_query_ms=${Q:-NA} free_ms=${F:-FAIL} app_ms=${A:-FAIL} elapsed_s=${T:-FAIL} fom=${M:-FAIL} energy=${E:-FAIL} fallback_peers=${NF:-0}"
    done
  done
done

echo ""
echo "=================================================================="
echo "Correctness: every energy above must read 1.482403e+06"
echo "Fallback:    grep 'not IPC-reachable' in this file"
echo "=================================================================="
echo "=== done $(date -Is) ==="
