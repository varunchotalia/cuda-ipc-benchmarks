#!/bin/bash
#SBATCH --job-name=rebaseline
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=02:30:00
#SBATCH --output=rebaseline_%j.out

# CONSOLIDATED RE-BASELINE.
#
# Purpose: give every number in the paper identical provenance. The existing
# tables were accumulated over months across different scripts, build flags,
# UCX settings, and warmup behavior, which is what made every spot-check
# disagree with something. This runs everything in ONE job on ONE node under
# ONE configuration:
#
#   * UCX DEFAULTS -- UCX_TLS is deliberately NOT exported. Job 59070 showed
#     any explicit UCX_TLS restriction costs staged MPI ~2.6x on large
#     messages (96.0 -> 37.3 GB/s at transpose 16384^2). The pinning was only
#     ever introduced to work around LULESH's "broken gpumpi", which is now
#     known to be one-time setup cost, not a transport fault.
#   * warmup=20 everywhere -- past the convergence knee measured in job 58445
#     (stencil plateaus at 5; 20 gives margin). Under-warmed GPU-aware MPI
#     reads 1.5-7x slow (job 59067).
#   * Uniform flags: -O3 -gencode arch=compute_90,code=sm_90 for every binary,
#     which also removes the unresolved build-flag confound in the old
#     stencil tables.
#   * 8 GPUs allocated so rank counts 2/4/8 all come from the same node.
#
# Output lines are tagged "RESULT ..." for easy tabulation.
# LULESH is deliberately NOT included: the paper does not use it, and its
# fixed/marginal decomposition (job 46508) is reproducible from a matched
# UCX-config pair already on disk. See note at end.

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

MPIWRAP=~/mpiwrap/libmpiwrap.so
ARCH="-gencode arch=compute_90,code=sm_90"
WU=20

# Keep logs readable without touching transport selection. UCX_LOG_LEVEL does
# not influence which transports UCX picks.
export UCX_LOG_LEVEL=error
unset UCX_TLS

echo "=================================================================="
echo "RE-BASELINE  node=$(hostname)  date=$(date -Is)"
echo "UCX_TLS: ${UCX_TLS:-<unset, UCX default selection>}"
echo "warmup=$WU   flags: -O3 $ARCH"
nvidia-smi -L
echo "=================================================================="

# ---------------------------------------------------------------- build ------
echo ""
echo "### BUILD ###"

STENCIL_SIZES="1024 2048 4096 8192 16384 32768"
for SIZE in $STENCIL_SIZES; do
    for MODE in ipc mpi gpu_mpi; do
        sed "s/const int N = [0-9]*/const int N = $SIZE/; \
             s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" \
            stencil/stencil_${MODE}.cu > /tmp/rb_st_${MODE}_$SIZE.cu
        nvcc -O3 $ARCH /tmp/rb_st_${MODE}_$SIZE.cu -o /tmp/rb_st_${MODE}_$SIZE -lmpi \
            || echo "BUILD FAILED stencil $MODE $SIZE"
    done
done
echo "stencil binaries built"

for ACC in 0 1; do
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/rb_tr_direct_$ACC        -lmpi || echo "BUILD FAILED tr direct $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=0 -DSINGLE_KERNEL=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/rb_tr_direct_single_$ACC -lmpi || echo "BUILD FAILED tr direct_single $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=1 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/rb_tr_buffered_$ACC      -lmpi || echo "BUILD FAILED tr buffered $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=2 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/rb_tr_gpumpi_$ACC        -lmpi || echo "BUILD FAILED tr gpumpi $ACC"
    nvcc -O3 $ARCH -DCOMM_MODE=3 -DACCUMULATE=$ACC transpose/transpose_ipc.cu \
        -o /tmp/rb_tr_staged_$ACC        -lmpi || echo "BUILD FAILED tr staged $ACC"
done
echo "transpose binaries built"

# ------------------------------------------------------------- stencil -------
echo ""
echo "### SECTION 1: stencil -- ranks 2/4/8 x 6 sizes x 3 modes ###"

for NP in 2 4 8; do
    for SIZE in $STENCIL_SIZES; do
        for MODE in ipc mpi gpu_mpi; do
            PRE=""; [ "$MODE" = "ipc" ] && PRE="env LD_PRELOAD=$MPIWRAP"
            OUT=$(STENCIL_WARMUP=$WU mpirun -np $NP -x STENCIL_WARMUP \
                    -x UCX_LOG_LEVEL $PRE /tmp/rb_st_${MODE}_$SIZE 2>&1)
            T=$(echo "$OUT" | grep -oP 'Total time: \K[0-9.]+')
            G=$(echo "$OUT" | grep -oP 'Throughput: \K[0-9.]+')
            L=$(echo "$OUT" | grep -oP 'Global L2 norm: \K[0-9.]+')
            echo "RESULT stencil np=$NP size=$SIZE mode=$MODE time_ms=${T:-FAIL} gcells=${G:-FAIL} l2=${L:-FAIL}"
        done
    done
done

# ----------------------------------------------------------- transpose -------
echo ""
echo "### SECTION 2: transpose -- ranks 2/4/8 x 5 orders x 5 modes x accum 0/1 ###"

TR_ORDERS="1024 2048 4096 8192 16384"
for NP in 2 4 8; do
    for ACC in 1 0; do
        for ORDER in $TR_ORDERS; do
            if [ $((ORDER % NP)) -ne 0 ]; then
                echo "RESULT transpose np=$NP order=$ORDER SKIPPED (order not divisible by np)"
                continue
            fi
            for MODE in direct direct_single buffered gpumpi staged; do
                PRE=""
                case $MODE in direct|direct_single|buffered) PRE="env LD_PRELOAD=$MPIWRAP" ;; esac
                OUT=$(TRANSPOSE_WARMUP=$WU mpirun -np $NP -x TRANSPOSE_WARMUP \
                        -x UCX_LOG_LEVEL $PRE /tmp/rb_tr_${MODE}_$ACC 100 $ORDER 2>&1)
                R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
                V=$(echo "$OUT" | grep -c "Solution validates")
                echo "RESULT transpose np=$NP order=$ORDER accum=$ACC mode=$MODE mbps=${R:-FAIL} validates=$V"
            done
        done
    done
done

# --------------------------------------------- warmup sensitivity control ----
echo ""
echo "### SECTION 3: does warmup=1 suffice under UCX DEFAULTS? ###"
echo "# The one cell left open by job 59070. If GPU-aware at warmup=1 matches"
echo "# warmup=20 here, then the 2026-05-05 tables (taken under defaults with"
echo "# the built-in single warmup) were never affected, and the warmup"
echo "# sensitivity is purely an artifact of the UCX_TLS pinning."

for W in 1 2 5 20; do
    for ORDER in 4096 16384; do
        OUT=$(TRANSPOSE_WARMUP=$W mpirun -np 4 -x TRANSPOSE_WARMUP \
                -x UCX_LOG_LEVEL /tmp/rb_tr_gpumpi_1 100 $ORDER 2>&1)
        R=$(echo "$OUT" | grep -oP 'Rate \(MB/s\): \K[0-9.]+')
        echo "RESULT warmupctl transpose_gpumpi np=4 order=$ORDER warmup=$W mbps=${R:-FAIL}"
    done
    for SIZE in 1024 16384; do
        OUT=$(STENCIL_WARMUP=$W mpirun -np 4 -x STENCIL_WARMUP \
                -x UCX_LOG_LEVEL /tmp/rb_st_gpu_mpi_$SIZE 2>&1)
        T=$(echo "$OUT" | grep -oP 'Total time: \K[0-9.]+')
        echo "RESULT warmupctl stencil_gpumpi np=4 size=$SIZE warmup=$W time_ms=${T:-FAIL}"
    done
done

echo ""
echo "=================================================================="
echo "RE-BASELINE COMPLETE  date=$(date -Is)"
echo "NOTE on LULESH: not re-run here. Its fixed/marginal decomposition in"
echo "LULESH/cuda/README.md cited gpumpi_default_45_200 paired with"
echo "gpumpi_ucx_memcache_off_45_full -- a MIXED UCX-config pair. The matched"
echo "pair (memcache_off 200 = 3.79s, memcache_off full = 5.82s) gives"
echo "marginal 0.689 ms/iter, fixed 3.65s, vs the cited 0.672 / 3.71s -- so"
echo "the conclusion is unchanged, but the citation should use the matched"
echo "pair. All 46508 gpumpi 200-iter runs fall in 3.79-4.01s regardless of"
echo "UCX config, which is why the decomposition is insensitive to this."
echo "=================================================================="

rm -f /tmp/rb_st_*.cu /tmp/rb_st_ipc_* /tmp/rb_st_mpi_* /tmp/rb_st_gpu_mpi_* /tmp/rb_tr_*
