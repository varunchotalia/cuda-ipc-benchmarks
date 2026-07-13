#!/bin/bash
#SBATCH --job-name=lulesh_bench
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=lulesh_%j.out

module load cuda12.8/toolkit/12.8.1
module load openmpi/gcc14.3/4.1.8 gcc/14.3.0

# After module loads: modules reference unset vars and would trip -u.
# From here on, a failed build aborts instead of running stale binaries.
set -euo pipefail

SRCDIR="/lustre/nvwulf/home/vchotalia/mpiwrap/LULESH/cuda/src"
MPIWRAP_LIB="/lustre/nvwulf/home/vchotalia/mpiwrap/mpi-intercept/libmpiwrap.so"

export NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
# Pin UCX away from its broken CUDA-introspection path on these nodes
export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_LOG_LEVEL=error

cd "$SRCDIR"

# Rebuild the interceptor and all six variants (no-op if already built)
(cd /lustre/nvwulf/home/vchotalia/mpiwrap/mpi-intercept && \
 mpicxx -O2 -fPIC -shared mpiwrap_ipc.cc -I$CUDA_HOME/include \
        -L$CUDA_HOME/lib64 -lcudart -o libmpiwrap.so)
make -j8 all 2>&1 | tee build_lulesh.log
VARIANTS="staged gpumpi shmwin ipc ipc_rp mpiwrap mpiwrap_rp nvshmem direct"
for V in $VARIANTS; do
    if [ ! -f ./lulesh_$V ]; then
        echo "Build failed for lulesh_$V — check build_lulesh.log" && exit 1
    fi
done

echo "======================================================================"
echo "LULESH halo-exchange benchmark: 8 ranks (2x2x2), h200x8 partition"
echo "NOTE: h200x8-03 is H200 SXM (NVSwitch all-to-all); -01/02/04 are"
echo "      H200 NVL (bridged NVLink groups) -- record which node ran"
echo "staged | gpumpi | shmwin | ipc(+rp) | mpiwrap(+rp) | nvshmem | direct"
echo "======================================================================"
echo "Node: $(hostname)"

# Problem sizes (edge elements per rank per side), iteration count
SIZES="45 60 80 100"
ITERS=200

run_mpi() {
    mpirun -np 8 --bind-to none \
        -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        "$@"
}

declare -A ENERGY ELAPSED FOM

for SZ in $SIZES; do
    echo ""
    echo "--- Problem size: ${SZ}^3 per rank ---"

    for V in $VARIANTS; do
        echo "[${V^^}]"
        # window-over-device-memory paths need the LD_PRELOADed interceptor
        case $V in
            mpiwrap|mpiwrap_rp) PRELOAD="-x LD_PRELOAD=$MPIWRAP_LIB" ;;
            *)                  PRELOAD="" ;;
        esac
        # a crashed variant records empty metrics instead of killing the sweep
        run_mpi $PRELOAD ./lulesh_$V -s ${SZ} 0 ${ITERS} 2>&1 | tee run.tmp \
            || echo "RUN FAILED: $V at size $SZ"
        ENERGY[$V]=$(awk '/Final Origin Energy/{print $5}' run.tmp)
        ELAPSED[$V]=$(awk '/Elapsed time/{print $4}' run.tmp)
        FOM[$V]=$(awk '/FOM/{print $3}' run.tmp)
    done

    echo ""
    echo "  Summary for ${SZ}^3:"
    printf "  %-10s %-16s %-12s %-12s %s\n" "variant" "energy" "elapsed(s)" "FOM(z/s)" "check"
    for V in $VARIANTS; do
        CHECK="MATCH"
        if [ -z "${ENERGY[$V]}" ]; then CHECK="MISSING";
        elif [ "${ENERGY[$V]}" != "${ENERGY[staged]}" ]; then CHECK="MISMATCH"; fi
        printf "  %-10s %-16s %-12s %-12s %s\n" "$V" "${ENERGY[$V]}" "${ELAPSED[$V]}" "${FOM[$V]}" "$CHECK"
    done
done
rm -f run.tmp

echo ""
echo "======================================================================"
echo "Done"
