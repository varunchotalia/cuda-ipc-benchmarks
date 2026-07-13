#!/bin/bash
#SBATCH --job-name=lulesh_all
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=lulesh_all_8gpu_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0 2>/dev/null || \
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

set -euo pipefail

export NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
# Pin UCX so its broken CUDA-introspection path stops flooding stderr and
# distorting the two-sided timings
export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_LOG_LEVEL=error

MPIWRAP_LIB=/lustre/nvwulf/home/vchotalia/mpiwrap/mpi-intercept/libmpiwrap.so

cd ~/mpiwrap/LULESH/cuda/src
make staged gpumpi shmwin ipc mpiwrap nvshmem

SIZE=45
VARIANTS="staged gpumpi shmwin ipc mpiwrap nvshmem"
declare -A ENERGY ELAPSED FOM

for V in $VARIANTS; do
    echo ""
    echo "=== LULESH $V: 8 ranks, 8 GPUs, -s $SIZE, full run to t=0.01 ==="
    if [ "$V" == "mpiwrap" ]; then
        mpirun -np 8 -x LD_PRELOAD=$MPIWRAP_LIB -x UCX_TLS -x UCX_LOG_LEVEL \
            ./lulesh_$V -s $SIZE 2>&1 | tee run_$V.tmp \
            || echo "RUN FAILED: $V"
    else
        mpirun -np 8 -x UCX_TLS -x UCX_LOG_LEVEL \
            ./lulesh_$V -s $SIZE 2>&1 | tee run_$V.tmp \
            || echo "RUN FAILED: $V"
    fi
    ENERGY[$V]=$(awk '/Final Origin Energy/{print $5}' run_$V.tmp)
    ELAPSED[$V]=$(awk '/Elapsed time/{print $4}' run_$V.tmp)
    FOM[$V]=$(awk '/FOM/{print $3}' run_$V.tmp)
done

echo ""
echo "==================================================================="
echo "LULESH halo-exchange comparison: 8 ranks / 8x H200, -s $SIZE, full run"
echo "==================================================================="
printf "%-10s %-16s %-12s %-12s %s\n" "variant" "energy" "elapsed(s)" "FOM(z/s)" "check"
PASS=1
for V in $VARIANTS; do
    CHECK="MATCH"
    if [ -z "${ENERGY[$V]}" ]; then CHECK="MISSING"; PASS=0;
    elif [ "${ENERGY[$V]}" != "${ENERGY[staged]}" ]; then CHECK="MISMATCH"; PASS=0; fi
    printf "%-10s %-16s %-12s %-12s %s\n" "$V" "${ENERGY[$V]}" "${ELAPSED[$V]}" "${FOM[$V]}" "$CHECK"
done
echo "==================================================================="
[ $PASS -eq 1 ] && echo "ALL VARIANTS: PASS" || echo "SOME VARIANTS FAILED: CHECK OUTPUT"
rm -f run_staged.tmp run_gpumpi.tmp run_shmwin.tmp run_ipc.tmp run_mpiwrap.tmp run_nvshmem.tmp
