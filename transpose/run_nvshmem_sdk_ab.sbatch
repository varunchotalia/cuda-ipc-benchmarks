#!/bin/bash
#SBATCH --job-name=nvshmem_sdk_ab
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=nvshmem_sdk_ab_%j.out

# Does NVSHMEM recover when built against the CUDA version the driver is?
#
# THE CASE. Job 96770 could not reproduce the paper's NVSHMEM numbers: 4 of 90
# runs reached 80% of published, medians sat at 8-10%, and at orders 4096 and
# 16384 the published values were never approached at all (best of ten: 27% and
# 20%). Our own WinIPC and MPI series in the same job were rock solid, 0-4% CoV.
#
# NVSHMEM reports CUDA Driver 13020 under a runtime and library built for
# 12080. And the dates line up: HPC SDK 25.9, carrying a CUDA 13.0 toolchain and
# a matching NVSHMEM, was installed 2026-08-27. Job 61541 took the published
# NVSHMEM numbers on 2026-08-07, twenty days BEFORE that. Every run that fails
# to reproduce them is from September.
#
# That is circumstantial, not proof -- no log before 2026-09-10 records a driver
# version, so we cannot show what changed. This job tests it directly.
#
#   A: 25.3 / CUDA 12.8 NVSHMEM, built with the 12.8 nvcc. What we have been
#      running, and what job 96770 measured.
#   B: 25.9 / CUDA 13.0 NVSHMEM, built with the 13.0 nvcc. Matched to the
#      driver.
#
# Same source, same node, same allocation, same orders, interleaved so neither
# arm gets a systematically warmer machine.
#
#   B recovers, A does not -> the mismatch is the cause. Rebuild and re-measure
#     Fig. 3c; the August numbers were taken on a matched stack and stand.
#   both fail -> not the toolchain. The published numbers remain unreproducible
#     and Section V-D needs the caveat.
#   both pass -> job 96770 caught the cluster in a bad state; re-run before
#     drawing any conclusion.

set -uo pipefail
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
cd ~/mpiwrap
SDK=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64
MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
NV3=$SDK/25.3/comm_libs/12.8/nvshmem
NV9=$SDK/25.9/comm_libs/13.0/nvshmem
NVCC3=nvcc
NVCC9=$SDK/25.9/cuda/13.0/bin/nvcc
ARCH="-gencode arch=compute_90,code=sm_90"
export NVSHMEM_BOOTSTRAP=MPI
export UCX_LOG_LEVEL=error
unset UCX_TLS
REPS=6
ORDERS="1024 4096 16384"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) ==="
echo "=== HEAD=$(git rev-parse --short HEAD) ==="
echo "=== transpose_nvshmem.cu md5=$(md5sum transpose/transpose_nvshmem.cu | cut -d' ' -f1) ==="
# The question this job exists to answer was unanswerable because no earlier log
# recorded this. Record it everywhere from now on.
echo "=== driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) ==="
echo "=== nvcc A: $($NVCC3 --version | tail -2 | head -1) ==="
echo "=== nvcc B: $($NVCC9 --version | tail -2 | head -1) ==="

for MODE in single buffered; do
    [ "$MODE" = single ] && FLAGS="-DCOMM_MODE=0 -DSINGLE_KERNEL=1" || FLAGS="-DCOMM_MODE=1"
    $NVCC3 -O3 -rdc=true $ARCH -I$NV3/include -I$MPI_HOME/include $FLAGS -DACCUMULATE=0 \
        transpose/transpose_nvshmem.cu $NV3/lib/libnvshmem_device.a -L$NV3/lib \
        -lnvshmem_host -L$MPI_HOME/lib -lmpi -o /tmp/ab_A_$MODE 2>/dev/null \
        || { echo "BUILD FAILED A $MODE"; exit 1; }
    $NVCC9 -O3 -rdc=true $ARCH -I$NV9/include -I$MPI_HOME/include $FLAGS -DACCUMULATE=0 \
        transpose/transpose_nvshmem.cu $NV9/lib/libnvshmem_device.a -L$NV9/lib \
        -lnvshmem_host -L$MPI_HOME/lib -lmpi -o /tmp/ab_B_$MODE 2>/dev/null \
        || { echo "BUILD FAILED B $MODE"; exit 1; }
done
echo "build done"

run_one() {   # $1=arm $2=mode $3=order $4=rep
    local ARM=$1 MODE=$2 ORDER=$3 REP=$4 LIB OUT R V
    [ "$ARM" = A ] && LIB=$NV3/lib || LIB=$NV9/lib
    OUT=$(TRANSPOSE_WARMUP=20 LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH \
          mpirun -np 4 -x TRANSPOSE_WARMUP -x NVSHMEM_BOOTSTRAP \
                 -x LD_LIBRARY_PATH -x UCX_LOG_LEVEL \
                 /tmp/ab_${ARM}_$MODE 100 $ORDER 2>&1)
    R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
    V=$(echo "$OUT" | grep -c 'Solution validates')
    echo "SDKAB arm=$ARM sdk=$([ "$ARM" = A ] && echo 25.3/cuda12.8 || echo 25.9/cuda13.0) mode=$MODE order=$ORDER rep=$REP mbps=${R:-FAIL} validates=$V"
    [ -z "$R" ] && { echo "---- no rate ----"; echo "$OUT" | tail -25; echo "---- end ----"; }
}

# Interleaved A/B so a drifting machine cannot favour one arm.
for ORDER in $ORDERS; do
    for MODE in single buffered; do
        for REP in $(seq 1 $REPS); do
            run_one A $MODE $ORDER $REP
            run_one B $MODE $ORDER $REP
        done
    done
done

echo ""
echo "=== done $(date -Is) ==="
echo "=== Published single-run reference (job 61541/63328, accum=0, GB/s):"
echo "===   nvsingle   415.1 @1024   1215.6 @4096   1295.0 @16384"
echo "===   nvbuffered 120.0 @1024    751.5 @4096   1167.0 @16384"
echo "=== Compare arm B against those. Arm A should reproduce job 96770's"
echo "===   medians of roughly 8-10% of published."
