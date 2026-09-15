#!/bin/bash
#SBATCH --job-name=nvshmem_dist
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=nvshmem_dist_%j.out

# Characterise the NVSHMEM distribution so Fig. 3c can report one.
#
# WHY. Job 92232 measured 30 NVSHMEM runs on one node spanning 0.2 to 1085
# GB/s -- a 5421x range -- with every run validating and only 1 of 24 landing
# within 20% of the job 63328 reference. NVSHMEM here is bimodal, not slow.
# Ruled out by measurement: not node-specific (reproduces on h200x4-02 and
# -04), not transport selection (92230 shows "P2P list: 0 1 2 3" with IBRC
# skipped), not our source (md5 unchanged), not the repetition structure
# (rep-1-only is still -89% to -94%), not correctness.
#
# The paper's Fig. 3c NVSHMEM curves come from SINGLE runs (job 61541). A
# single sample cannot represent a distribution spanning two orders of
# magnitude -- 63328 simply landed in the fast mode. This job takes enough
# samples to state a median and a spread instead of a point.
#
# 10 reps rather than 3: at a roughly 1-in-8 hit rate for the fast mode, 3 reps
# can easily miss it entirely and would understate the upper mode. 10 gives a
# usable median and an honest min/max whisker.
#
# All seven Fig. 3c series in ONE job, so the NVSHMEM and IPC curves share a
# node, a build and an allocation -- the comparison is internally consistent
# even though the NVSHMEM half is unstable. Overwrite only (accum=0), which is
# what Fig. 3c plots.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
export REPO=~/mpiwrap
source $REPO/scripts/build_mpiwrap.sh
MPIWRAP=$MPIWRAP_LIB
NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
export UCX_LOG_LEVEL=error
unset UCX_TLS
ARCH="-gencode arch=compute_90,code=sm_90"
REPS=10
ORDERS="1024 2048 4096 8192 16384"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ==="
echo "=== transpose_ipc.cu     md5=$(md5sum transpose/transpose_ipc.cu | cut -d' ' -f1) ==="
echo "=== transpose_nvshmem.cu md5=$(md5sum transpose/transpose_nvshmem.cu | cut -d' ' -f1) ==="
echo "=== reps=$REPS accum=0 (overwrite, as Fig. 3c) ==="

nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=0 \
     transpose/transpose_ipc.cu -o /tmp/nvd_single_ipc -lmpi \
     -I$MPI_HOME/include -L$MPI_HOME/lib || { echo BUILD FAILED; exit 1; }
nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=0 \
     transpose/transpose_ipc.cu -o /tmp/nvd_perphase_ipc -lmpi \
     -I$MPI_HOME/include -L$MPI_HOME/lib || { echo BUILD FAILED; exit 1; }
nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=0 \
     transpose/transpose_ipc.cu -o /tmp/nvd_buffered_ipc -lmpi \
     -I$MPI_HOME/include -L$MPI_HOME/lib || { echo BUILD FAILED; exit 1; }
nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=0 \
     transpose/transpose_ipc.cu -o /tmp/nvd_gpumpi -lmpi \
     -I$MPI_HOME/include -L$MPI_HOME/lib || { echo BUILD FAILED; exit 1; }
NVINC="-I$NVSHMEM_HOME/include -I$MPI_HOME/include"
NVLIB="$NVSHMEM_HOME/lib/libnvshmem_device.a -L$NVSHMEM_HOME/lib -lnvshmem_host -L$MPI_HOME/lib -lmpi"
nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=0 \
     transpose/transpose_nvshmem.cu $NVLIB -o /tmp/nvd_nvsingle || { echo BUILD FAILED; exit 1; }
nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=0 -DACCUMULATE=0 \
     transpose/transpose_nvshmem.cu $NVLIB -o /tmp/nvd_nvdirect || { echo BUILD FAILED; exit 1; }
nvcc -O3 -rdc=true $ARCH $NVINC -DCOMM_MODE=1 -DACCUMULATE=0 \
     transpose/transpose_nvshmem.cu $NVLIB -o /tmp/nvd_nvbuffered || { echo BUILD FAILED; exit 1; }
echo "build done"

# Capture then grep. Three earlier diagnostics lost every result line to a
# `head` on the live stream; nothing is truncated before it is searched.
run_one() {   # $1=tag $2=order $3=rep
    local TAG=$1 ORDER=$2 REP=$3 PRE="" EXTRA="" OUT R V
    case $TAG in
        single_ipc|perphase_ipc|buffered_ipc) PRE="env LD_PRELOAD=$MPIWRAP" ;;
        nv*) EXTRA="-x NVSHMEM_BOOTSTRAP -x LD_LIBRARY_PATH" ;;
    esac
    OUT=$(TRANSPOSE_WARMUP=20 mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL \
          $EXTRA $PRE /tmp/nvd_$TAG 100 $ORDER 2>&1)
    R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
    V=$(echo "$OUT" | grep -c 'Solution validates')
    echo "DIST tag=$TAG order=$ORDER rep=$REP mbps=${R:-FAIL} validates=$V"
    [ -z "$R" ] && { echo "---- no rate ----"; echo "$OUT" | tail -25; echo "---- end ----"; }
}

for ORDER in $ORDERS; do
    for TAG in single_ipc perphase_ipc buffered_ipc gpumpi nvsingle nvdirect nvbuffered; do
        for REP in $(seq 1 $REPS); do run_one $TAG $ORDER $REP; done
    done
done

echo ""
echo "=== done $(date -Is) ==="
echo "=== Report NVSHMEM as median with min/max over $REPS reps, not as a point."
echo "=== 63328 single-run reference (accum=0, GB/s): nvsingle 415.1 @1024,"
echo "===   1061.4 @2048, 1215.6 @4096, 1295.0 @16384."
echo "=== If the upper mode reproduces at roughly those values, the paper's"
echo "===   numbers are the fast mode of a bimodal distribution and should be"
echo "===   reported with their spread rather than replaced."
