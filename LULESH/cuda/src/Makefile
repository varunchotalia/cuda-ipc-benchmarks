NVCC     = nvcc
ARCH     = -arch=sm_90

MPI_INC   = $(shell mpicc --showme:incdirs | xargs -I{} echo -I{})
MPI_LDIRS = $(shell mpicc --showme:libdirs  | xargs -I{} echo -L{})
MPI_LIBS  = $(shell mpicc --showme:libs     | xargs -I{} echo -l{})

BASE_CFLAGS = $(ARCH) -O3 -DNDEBUG -DUSE_MPI -I. $(MPI_INC)
LDFLAGS     = $(MPI_LDIRS) $(MPI_LIBS)

SRCS_BASE = allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu

NVSHMEM_HOME ?= /lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem

.PHONY: all staged ipc shmwin nvshmem gpumpi mpiwrap ipc_rp mpiwrap_rp direct clean

all: staged ipc shmwin gpumpi nvshmem mpiwrap ipc_rp mpiwrap_rp direct

# ---- Mode C: remote-pack -- pack kernels write straight into the peer's ----
# ---- packed recv buffer (no local staging, no separate copy)            ----
OBJS_IPC_RP = $(patsubst %.cu,build/ipc_rp/%.o,$(SRCS_BASE))
ipc_rp: lulesh_ipc_rp
build/ipc_rp:
	mkdir -p $@
build/ipc_rp/%.o: %.cu | build/ipc_rp
	$(NVCC) $(BASE_CFLAGS) -DCOMM_IPC -DIPC_REMOTE_PACK -c $< -o $@
lulesh_ipc_rp: $(OBJS_IPC_RP)
	$(NVCC) $(ARCH) $(OBJS_IPC_RP) $(LDFLAGS) -o $@

OBJS_MPIWRAP_RP = $(patsubst %.cu,build/mpiwrap_rp/%.o,$(SRCS_BASE))
mpiwrap_rp: lulesh_mpiwrap_rp
build/mpiwrap_rp:
	mkdir -p $@
build/mpiwrap_rp/%.o: %.cu | build/mpiwrap_rp
	$(NVCC) $(BASE_CFLAGS) -DCOMM_IPC -DIPC_VIA_MPIWRAP -DIPC_REMOTE_PACK -c $< -o $@
lulesh_mpiwrap_rp: $(OBJS_MPIWRAP_RP)
	$(NVCC) $(ARCH) $(OBJS_MPIWRAP_RP) $(LDFLAGS) -o $@

# ---- Mode B: direct -- sender kernels write straight into the receiver's ---
# ---- field arrays (atomicAdd for SBN); MonoQ remote-packs               ----
SRCS_DIRECT = allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-direct.cu
OBJS_DIRECT = $(patsubst %.cu,build/direct/%.o,$(SRCS_DIRECT))
direct: lulesh_direct
build/direct:
	mkdir -p $@
build/direct/%.o: %.cu | build/direct
	$(NVCC) $(BASE_CFLAGS) -DCOMM_IPC -DCOMM_DIRECT -c $< -o $@
lulesh_direct: $(OBJS_DIRECT)
	$(NVCC) $(ARCH) $(OBJS_DIRECT) $(LDFLAGS) -o $@

# ---- mpiwrap variant (MPI windows over device memory; CUDA IPC supplied ----
# ---- by LD_PRELOAD=libmpiwrap.so at run time)                           ----
OBJS_MPIWRAP = $(patsubst %.cu,build/mpiwrap/%.o,$(SRCS_BASE))
mpiwrap: lulesh_mpiwrap
build/mpiwrap:
	mkdir -p $@
build/mpiwrap/%.o: %.cu | build/mpiwrap
	$(NVCC) $(BASE_CFLAGS) -DCOMM_IPC -DIPC_VIA_MPIWRAP -c $< -o $@
lulesh_mpiwrap: $(OBJS_MPIWRAP)
	$(NVCC) $(ARCH) $(OBJS_MPIWRAP) $(LDFLAGS) -o $@

# ---- GPU-aware MPI variant (device pointers passed to MPI) ----
OBJS_GPUMPI = $(patsubst %.cu,build/gpumpi/%.o,$(SRCS_BASE))
gpumpi: lulesh_gpumpi
build/gpumpi:
	mkdir -p $@
build/gpumpi/%.o: %.cu | build/gpumpi
	$(NVCC) $(BASE_CFLAGS) -DCOMM_GPUMPI -c $< -o $@
lulesh_gpumpi: $(OBJS_GPUMPI)
	$(NVCC) $(ARCH) $(OBJS_GPUMPI) $(LDFLAGS) -o $@

# ---- Shared-memory window variant (MPI_Win_allocate_shared + direct copies) ----
OBJS_SHMWIN = $(patsubst %.cu,build/shmwin/%.o,$(SRCS_BASE))
shmwin: lulesh_shmwin
build/shmwin:
	mkdir -p $@
build/shmwin/%.o: %.cu | build/shmwin
	$(NVCC) $(BASE_CFLAGS) -DCOMM_SHMWIN -c $< -o $@
lulesh_shmwin: $(OBJS_SHMWIN)
	$(NVCC) $(ARCH) $(OBJS_SHMWIN) $(LDFLAGS) -o $@

# ---- NVSHMEM variant (symmetric-heap recv buffers + GPU-direct puts) ----
OBJS_NVSHMEM = $(patsubst %.cu,build/nvshmem/%.o,$(SRCS_BASE))
nvshmem: lulesh_nvshmem
build/nvshmem:
	mkdir -p $@
build/nvshmem/%.o: %.cu | build/nvshmem
	$(NVCC) $(BASE_CFLAGS) -rdc=true -DCOMM_NVSHMEM -I$(NVSHMEM_HOME)/include -c $< -o $@
lulesh_nvshmem: $(OBJS_NVSHMEM)
	$(NVCC) $(ARCH) -rdc=true $(OBJS_NVSHMEM) $(NVSHMEM_HOME)/lib/libnvshmem_device.a -L$(NVSHMEM_HOME)/lib -lnvshmem_host $(LDFLAGS) -o $@

# ---- Staged MPI baseline ----
OBJS_STAGED = $(patsubst %.cu,build/staged/%.o,$(SRCS_BASE))
staged: lulesh_staged
build/staged:
	mkdir -p $@
build/staged/%.o: %.cu | build/staged
	$(NVCC) $(BASE_CFLAGS) -c $< -o $@
lulesh_staged: $(OBJS_STAGED)
	$(NVCC) $(ARCH) $(OBJS_STAGED) $(LDFLAGS) -o $@

# ---- CUDA IPC variant (peer GPU buffers mapped via cudaIpcOpenMemHandle) ----
OBJS_IPC = $(patsubst %.cu,build/ipc/%.o,$(SRCS_BASE))
ipc: lulesh_ipc
build/ipc:
	mkdir -p $@
build/ipc/%.o: %.cu | build/ipc
	$(NVCC) $(BASE_CFLAGS) -DCOMM_IPC -c $< -o $@
lulesh_ipc: $(OBJS_IPC)
	$(NVCC) $(ARCH) $(OBJS_IPC) $(LDFLAGS) -o $@

clean:
	rm -rf build lulesh_staged lulesh_ipc lulesh_shmwin lulesh_nvshmem lulesh_gpumpi lulesh_mpiwrap lulesh_ipc_rp lulesh_mpiwrap_rp lulesh_direct
