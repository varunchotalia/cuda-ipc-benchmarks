# Makefile for transpose_cuda_ipc
#
# Builds three variants:
#   transpose_ipc     — CUDA IPC (GPU-to-GPU via NVLink)
#   transpose_mpigpu  — GPU-aware MPI (device pointers in MPI calls)
#   transpose_staged  — Staged MPI (host staging buffers)
#
# Adjust MPI_HOME and CUDA_HOME as needed for your cluster.

NVCC       ?= nvcc
MPI_HOME   ?= $(shell dirname $(shell dirname $(shell which mpicc)))
CUDA_HOME  ?= $(shell dirname $(shell dirname $(shell which nvcc)))

NVCCFLAGS  = -O3 -lineinfo
INCLUDES   = -I$(MPI_HOME)/include
LDFLAGS    = -L$(MPI_HOME)/lib -lmpi

# Detect GPU architecture (adjust for your hardware)
# H200 = sm_90, Blackwell RTX Pro 6000 = sm_100 (or sm_120)
# Use -gencode for multiple targets if running on mixed hardware
GPU_ARCH   ?= -gencode arch=compute_90,code=sm_90

SRC = transpose_cuda_ipc.cu

all: transpose_ipc transpose_mpigpu transpose_staged

transpose_ipc: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=0 $(INCLUDES) $< -o $@ $(LDFLAGS)

transpose_mpigpu: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=1 $(INCLUDES) $< -o $@ $(LDFLAGS)

transpose_staged: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=2 $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f transpose_ipc transpose_mpigpu transpose_staged

# Quick benchmark: 2 GPUs, 100 iterations, 4096×4096 matrix
bench: all
	@echo "=== CUDA IPC ==="
	mpirun -np 2 ./transpose_ipc 100 4096
	@echo ""
	@echo "=== GPU-aware MPI ==="
	mpirun -np 2 ./transpose_mpigpu 100 4096
	@echo ""
	@echo "=== Staged MPI ==="
	mpirun -np 2 ./transpose_staged 100 4096

.PHONY: all clean bench
