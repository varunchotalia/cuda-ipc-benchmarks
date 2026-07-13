#!/bin/bash
#SBATCH --job-name=lulesh_modes
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=lulesh_modes_8gpu_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0 2>/dev/null || \
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0

set -euo pipefail

export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_LOG_LEVEL=error

MPIWRAP_LIB=/lustre/nvwulf/home/vchotalia/mpiwrap/mpi-intercept/libmpiwrap.so

cd ~/mpiwrap/LULESH/cuda/src
make staged ipc ipc_rp mpiwrap mpiwrap_rp direct

SIZE=45
# A = pack + copy + unpack, C = remote-pack + unpack, B = direct field writes
VARIANTS="staged ipc ipc_rp mpiwrap mpiwrap_rp direct"
declare -A ENERGY ELAPSED FOM

for V in $VARIANTS; do
    echo ""
    echo "=== LULESH $V: 8 ranks, 8 GPUs, -s $SIZE, full run to t=0.01 ==="
    case $V in
        mpiwrap|mpiwrap_rp) PRELOAD="-x LD_PRELOAD=$MPIWRAP_LIB" ;;
        *)                  PRELOAD="" ;;
    esac
    mpirun -np 8 $PRELOAD -x UCX_TLS -x UCX_LOG_LEVEL \
        ./lulesh_$V -s $SIZE 2>&1 | tee run_$V.tmp \
        || echo "RUN FAILED: $V"
    ENERGY[$V]=$(awk '/Final Origin Energy/{print $5}' run_$V.tmp)
    ELAPSED[$V]=$(awk '/Elapsed time/{print $4}' run_$V.tmp)
    FOM[$V]=$(awk '/FOM/{print $3}' run_$V.tmp)
done

echo ""
echo "==================================================================="
echo "LULESH comm-mode comparison: 8 ranks / 8x H200, -s $SIZE, full run"
echo "A: ipc, mpiwrap | C (remote-pack): ipc_rp, mpiwrap_rp | B: direct"
echo "==================================================================="
printf "%-12s %-16s %-12s %-12s %s\n" "variant" "energy" "elapsed(s)" "FOM(z/s)" "check"
for V in $VARIANTS; do
    CHECK="MATCH"
    if [ -z "${ENERGY[$V]}" ]; then CHECK="MISSING";
    elif [ "${ENERGY[$V]}" != "${ENERGY[staged]}" ]; then CHECK="MISMATCH"; fi
    # direct's atomicAdd changes summation order; small deviation expected
    [ "$V" == "direct" ] && [ "$CHECK" == "MISMATCH" ] && CHECK="MISMATCH(atomics?)"
    printf "%-12s %-16s %-12s %-12s %s\n" "$V" "${ENERGY[$V]}" "${ELAPSED[$V]}" "${FOM[$V]}" "$CHECK"
done
echo "==================================================================="
rm -f run_staged.tmp run_ipc.tmp run_ipc_rp.tmp run_mpiwrap.tmp run_mpiwrap_rp.tmp run_direct.tmp
