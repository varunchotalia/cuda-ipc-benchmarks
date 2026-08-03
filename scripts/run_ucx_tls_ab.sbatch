#!/bin/bash
#SBATCH --job-name=transpose_ucx_tls
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:45:00
#SBATCH --output=transpose_ucx_tls_%j.out

# Resolve the staged-transpose discrepancy.
#
# results/transpose_results.md (2026-05-05) records staged MPI at
#   4096^2 = 131.7 GB/s, 16384^2 = 114.8 GB/s
# My job 58445 measured the same binary/flags at
#   4096^2 =  37.0 GB/s, 16384^2 =  37.2 GB/s   (~3x low, uniformly)
#
# The COMM_MODE==3 staged path is UNCHANGED since 9b9e6eb (verified by diff),
# and the Makefile flags I used match (-O3 -gencode arch=compute_90). The one
# environmental difference: I exported UCX_TLS=self,sm,cuda_copy,cuda_ipc,
# a pinning introduced later for LULESH. The May 5 runs used UCX defaults.
#
# Hypothesis: with cuda_copy/cuda_ipc in UCX_TLS, UCX may route the staged
# path's cudaMallocHost (pinned, CUDA-registered) host buffers through a CUDA
# transport instead of plain shared memory, costing ~3x. Tested by varying
# UCX_TLS only, everything else fixed.

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

MPIWRAP=~/mpiwrap/libmpiwrap.so
ARCH="-gencode arch=compute_90,code=sm_90"

export UCX_LOG_LEVEL=error

for M in 1 2 3; do
    nvcc -O3 $ARCH -DCOMM_MODE=$M transpose/transpose_ipc.cu -o /tmp/tls_$M -lmpi \
        || echo "COMPILE FAILED: mode $M"
done

# warmup=20: past the convergence knee established in job 58445, so GPU-aware
# numbers here are steady-state. Irrelevant for staged but kept uniform.
WU=20

run_one() {   # $1=mode  $2=tls-label  $3=tls-value ("" = UCX default)
    local M=$1 LABEL=$2 TLS=$3 NAME PRE="" XARG=""
    case $M in
        1) NAME=ipc_buffered  ; PRE="env LD_PRELOAD=$MPIWRAP" ;;
        2) NAME=gpu_aware_mpi ;;
        3) NAME=staged_mpi ;;
    esac
    [ -n "$TLS" ] && XARG="-x UCX_TLS=$TLS"
    printf '  %-16s ' "$NAME"
    TRANSPOSE_WARMUP=$WU mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL \
        $XARG $PRE /tmp/tls_$M 100 $ORDER 2>&1 \
        | grep -oE "Rate \(MB/s\): [0-9.]+|ERROR[^,]*|abort" | tr '\n' ' '
    echo ""
}

for ORDER in 4096 16384; do
    echo "############################################################"
    echo "ORDER=$ORDER  (target: staged should hit ~131.7 / ~114.8 GB/s)"
    echo "############################################################"

    echo ""
    echo "--- UCX_TLS unset (UCX default selection; matches May 5 conditions) ---"
    for M in 1 2 3; do run_one $M default ""; done

    echo ""
    echo "--- UCX_TLS=self,sm,cuda_copy,cuda_ipc (my pinning; the suspect) ---"
    for M in 1 2 3; do run_one $M pinned "self,sm,cuda_copy,cuda_ipc"; done

    echo ""
    echo "--- UCX_TLS=self,sm (host transports only, no CUDA) ---"
    for M in 1 2 3; do run_one $M hostonly "self,sm"; done

    echo ""
    echo "--- UCX_TLS=self,sm,cuda_ipc (cuda_ipc but NO cuda_copy) ---"
    for M in 1 2 3; do run_one $M no_cudacopy "self,sm,cuda_ipc"; done
done

echo ""
echo "############################################################"
echo "Note: 1 GB/s = 1000 MB/s as printed. Divide printed MB/s by 1000."
echo "UCX transport actually chosen, staged, default vs pinned:"
echo "############################################################"
for TLS in "" "self,sm,cuda_copy,cuda_ipc"; do
    XARG=""; LBL="default"
    [ -n "$TLS" ] && { XARG="-x UCX_TLS=$TLS"; LBL="pinned"; }
    echo "--- staged, UCX_TLS=$LBL ---"
    TRANSPOSE_WARMUP=1 mpirun -np 4 -x TRANSPOSE_WARMUP -x UCX_LOG_LEVEL=info \
        $XARG /tmp/tls_3 5 4096 2>&1 \
        | grep -E "ucp_worker.c.*cfg#|selected|intra-node" | sort -u | head -8
done

rm -f /tmp/tls_1 /tmp/tls_2 /tmp/tls_3
echo "Done."
