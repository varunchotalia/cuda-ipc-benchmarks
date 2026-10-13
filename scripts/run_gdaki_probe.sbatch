#!/bin/bash
#SBATCH --job-name=gdaki_probe
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/nvwulf/home/vchotalia/mpiwrap/logs/gdaki_probe_%j.out

# Does GDAKI (NVSHMEM calls it IBGDA: a GPU thread posting work straight to
# the NIC) work on an nvwulf H200 node at all? Answers that before we write
# any GDAKI code. Uses NVSHMEM's own put bandwidth test, twice:
#   A. defaults: the two GPUs talk over NVLink (baseline)
#   B. NVLink disabled and IBGDA forced, so the same traffic has to go
#      GPU -> NIC -> NIC -> GPU
# If B runs and NVSHMEM_DEBUG reports the ibgda transport, GDAKI works here.

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
set -o pipefail   # a failed launch must not hide behind | tail

N=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
export LD_LIBRARY_PATH=$N/lib:$LD_LIBRARY_PATH
PT=$N/bin/perftest/device/pt-to-pt/shmem_put_bw

echo "node=$(hostname)  date=$(date -Is)"
echo ""
echo "### node capabilities ###"
nvidia-smi -L
cat /proc/driver/nvidia/version | head -1
echo "--- GPU/NIC topology:"; nvidia-smi topo -m
echo "--- IB devices:"
for d in /sys/class/infiniband/*; do
    [ -e "$d" ] || { echo "none"; break; }
    echo "$(basename "$d"): link=$(cat "$d"/ports/1/link_layer) state=$(cat "$d"/ports/1/state) rate=$(cat "$d"/ports/1/rate)"
done
echo "--- GPUDirect RDMA kernel modules (nvidia_peermem / gdrdrv):"
lsmod | grep -E 'nvidia_peermem|nv_peer_mem|gdrdrv' || echo "none loaded"
echo "--- nvidia driver RegistryDwords (IBGDA may need PeerMappingOverride=1):"
grep -i RegistryDwords /proc/driver/nvidia/params || echo "not set"
echo "--- nvshmem-info (build config):"
$N/bin/nvshmem-info -a 2>&1 | grep -iE 'ibgda|version' | head -8

run() {
    echo ""
    echo "### $1 ###"
    shift
    # nvshmrun is NVSHMEM's own launcher (hydra); falls back to mpirun + PMIx
    env "$@" timeout 180 $N/bin/nvshmrun -n 2 "$PT" 2>&1 | tail -40 \
        || { echo "(nvshmrun failed; retrying with mpirun + PMIx bootstrap)";
             env "$@" NVSHMEM_BOOTSTRAP=PMIX timeout 180 mpirun -np 2 "$PT" 2>&1 | tail -40; }
}

run "A. defaults (NVLink)" NVSHMEM_DEBUG=WARN
run "B. IBGDA forced, NVLink off" NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_DISABLE_P2P=1 NVSHMEM_DEBUG=INFO
