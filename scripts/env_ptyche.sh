#!/bin/bash
# Ptyche (GB200 NVL72, coreai_libraries_nccl) toolchain -- source before BOTH
# the cmake build and scripts/run_nvl72.sh, in the same shell:
#
#   source scripts/env_ptyche.sh
#   srun ... cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 && cmake --build build -j
#   LAUNCH="srun --mpi=pmix --ntasks-per-node=4 -n" bash scripts/run_nvl72.sh
#
# WHY THIS FILE EXISTS. Job 2527976 built against HPC-X 2.22.1 (what cmake
# found on PATH) and ran against HPC-X 2.21 (what the job script put on
# LD_LIBRARY_PATH). Every rank died in MPI_Init with
#   mca_pml_base_open() failed / Framework: pml Component: ucx not found
# and the whole suite reported n/a. Nothing was wrong with the benchmarks.
# Everything below derives from HPCX_HOME so the two halves cannot drift.
#
# 2.22.1, not 2.21, is also deliberate: 2.21's UCX kills gpumpi/ipc/ipc_rp/
# nvshmem at 27 and 64 ranks with
#   cuda_ipc_cache.c:549 Fatal: failed to open ipc mem handle ...
#     (Element already exists)          [job 2528116]
# a ucs_fatal in UCX's cuda_ipc remote-memhandle cache, reached from
# uct_cuda_ipc_rkey_unpack under MPI_Waitall. It fires only once enough nodes
# are in play for two peers to present the same device VA, which is why 8
# ranks survived and 27/64 did not. If 2.22.1 still hits it, see
# UCX_CUDA_IPC_CACHE below.

CUDA_HOME=${CUDA_HOME:-/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-13.0}
HPCX_HOME=${HPCX_HOME:-/lustre/fsw/coreai_libraries_nccl/toolkits/hpcx-v2.22.1-gcc-inbox-ubuntu24.04-cuda12-aarch64}
NVSHMEM_HOME=${NVSHMEM_HOME:-/lustre/fsw/coreai_libraries_nccl/toolkits/nvshmem}

for d in "$CUDA_HOME" "$HPCX_HOME" "$NVSHMEM_HOME"; do
    [ -d "$d" ] || { echo "env_ptyche.sh: no such directory: $d" >&2; return 1 2>/dev/null || exit 1; }
done

# Single MPI prefix. MPI_HOME is what the sbatch banners and the CMake
# hint should both read; nothing else may name an hpcx tree.
export CUDA_HOME NVSHMEM_HOME HPCX_HOME
export MPI_HOME="$HPCX_HOME/ompi"
export OPAL_PREFIX="$MPI_HOME"
export PATH="$MPI_HOME/bin:$CUDA_HOME/bin:$PATH"

# HPC-X's mca_pml_ucx.so / mca_osc_ucx.so dlopen libucp/libucs/libuct from
# <hpcx>/ucx/lib, a sibling of ompi/ that is not on the default path. Without
# these two entries the pml/osc components are silently unloadable and
# MPI_Init aborts -- the 2527976 failure mode.
export LD_LIBRARY_PATH="$HPCX_HOME/ucx/lib:$HPCX_HOME/hcoll/lib:$MPI_HOME/lib:$NVSHMEM_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll_hcoll_enable=0

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=None
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_MAX_CTAS=2
export NVSHMEM_MAX_TEAMS=1024

# Workaround, only if 2.22.1 still hits the cuda_ipc_cache ucs_fatal above.
# Disables UCX's remote-memhandle mapping cache -- the exact structure that
# asserts -- at the cost of a re-map per transfer. Confirm the knob exists on
# the node first: ucx_info -f -c | grep CUDA_IPC
#   export UCX_CUDA_IPC_CACHE=n
# Do NOT reach for UCX_TLS=^cuda_ipc instead: that removes the transport the
# gpumpi/ipc variants exist to measure, so the numbers stop being comparable.

echo "env_ptyche.sh: CUDA=$CUDA_HOME"
echo "env_ptyche.sh: MPI =$MPI_HOME"
echo "env_ptyche.sh: UCX =$HPCX_HOME/ucx/lib"
echo "env_ptyche.sh: NVSHMEM=$NVSHMEM_HOME"
