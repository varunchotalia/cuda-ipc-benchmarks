NVCC       ?= nvcc
MPI_HOME   ?= $(shell dirname $(shell dirname $(shell which mpicc)))
NVCCFLAGS  = -O3 -lineinfo
INCLUDES   = -I$(MPI_HOME)/include
LDFLAGS    = -L$(MPI_HOME)/lib -lmpi
GPU_ARCH   ?= -gencode arch=compute_90,code=sm_90

SRC = transpose_ipc.cu

all: transpose_direct transpose_buffered transpose_mpigpu transpose_staged

transpose_direct: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=0 $(INCLUDES) $< -o $@ $(LDFLAGS)

transpose_buffered: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=1 $(INCLUDES) $< -o $@ $(LDFLAGS)

transpose_mpigpu: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=2 $(INCLUDES) $< -o $@ $(LDFLAGS)

transpose_staged: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_ARCH) -DCOMM_MODE=3 $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f transpose_direct transpose_buffered transpose_mpigpu transpose_staged

.PHONY: all clean
