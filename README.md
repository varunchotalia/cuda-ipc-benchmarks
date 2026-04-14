# CUDA IPC vs MPI — Multi-GPU Communication Benchmarks

Benchmarking CUDA IPC against MPI and NVSHMEM for multi-GPU communication on NVIDIA H200 GPUs (NVLink).

## Structure

```
├── transpose/                      # Matrix transpose benchmark (B = A^T or B += A^T)
│   ├── transpose_ipc.cu               # IPC + MPI modes (COMM_MODE 0–3, SINGLE_KERNEL)
│   ├── transpose_nvshmem.cu           # NVSHMEM modes (direct, buffered, single-kernel)
│   ├── Makefile                       # Builds all IPC/MPI variants
│   └── Makefile_nvshmem               # Builds all NVSHMEM variants
│
├── stencil/                        # 2D 5-point stencil benchmark
│   ├── stencil_ipc.cu                 # Ghost row exchange via CUDA IPC
│   └── stencil_mpi.cu                 # Ghost row exchange via MPI
│
├── mpi-intercept/                  # MPI interposer library for transparent CUDA IPC
│   ├── mpiwrap_ipc.cc                 # Intercepts MPI_Win_create/shared_query/free
│   ├── test_ipc_win.cu                # Test program for the interposer
│   └── CMakeLists.txt
│
├── plots/                          # Benchmark visualizations
│   ├── transpose_benchmark_ipc_nvshmem.png
│   └── transpose_stencil_comparison.png
│
├── results/                        # Tabulated benchmark data
│   ├── transpose_results.md
│   └── stencil_results.txt
│
├── run_transpose_all.sbatch        # Builds and runs all IPC/MPI transpose variants
├── run_nvshmem_all.sbatch          # Builds and runs all NVSHMEM transpose variants
└── libmpiwrap.so                   # Pre-built MPI interposer (see Build below)
```

## Transpose Communication Modes

| Mode | Flag | How it works |
|------|------|-------------|
| IPC direct (per-phase) | `COMM_MODE=0` | Kernel writes directly to peer's B matrix via IPC pointer, one phase at a time |
| IPC direct (single-kernel) | `COMM_MODE=0 SINGLE_KERNEL=1` | All peers handled in one kernel launch (blockIdx.z = peer), one barrier before and after |
| IPC buffered | `COMM_MODE=1` | Pack → cudaMemcpyAsync to peer's recv buffer → unpack |
| GPU-aware MPI | `COMM_MODE=2` | Pack → MPI_Sendrecv (GPU buffers) → unpack |
| Staged MPI | `COMM_MODE=3` | Pack → D2H memcpy → MPI_Sendrecv (host buffers) → H2D memcpy → unpack |

Set `-DACCUMULATE=0` for `B = A^T` (overwrite) or `-DACCUMULATE=1` (default) for `B += A^T` (PRK-style, A incremented each iteration).

## Key Findings

**IPC direct single-kernel** wins at small–medium matrices (up to 4096²) by eliminating P−1 barriers and overlapping all peer transfers in a single kernel launch.

**Buffered IPC ≈ GPU-aware MPI** at large matrices (~1060–1077 GB/s at 16384²). Both saturate NVLink bandwidth via bulk DMA transfers.

**NVSHMEM direct** is 3–4× slower than IPC direct at small sizes due to per-element `nvshmem_double_p/g` protocol overhead, but converges with IPC at large sizes. NVSHMEM single-kernel is 2× faster than NVSHMEM per-phase at small sizes for the same reason as IPC.

**Staged MPI** flatlines at ~100–120 GB/s (PCIe bottleneck from D2H/H2D transfers).

## Hardware

- NVIDIA H200 GPUs (141 GB HBM3e)
- NVLink interconnect
- Stony Brook University IACS cluster

## Dependencies

- CUDA Toolkit 12.8+
- OpenMPI 4.1+ with GPU-aware support (`openmpi/gcc14.3/4.1.8`)
- GCC 14.3+
- NVSHMEM 3.x (for NVSHMEM variants, set `NVSHMEM_HOME`)

On the IACS cluster, load modules:
```bash
module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0
```

## Build & Run

### Build the MPI interposer library

The IPC modes (`COMM_MODE=0,1`) require `libmpiwrap.so` at runtime for the `MPI_Win_create`/`MPI_Win_shared_query` intercept. A pre-built copy is in the repo root. To rebuild:

```bash
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
MPI_HOME=$(dirname $(dirname $(which mpicc)))
g++ -O2 -fPIC -shared \
    -I${CUDA_HOME}/include -I${MPI_HOME}/include \
    mpi-intercept/mpiwrap_ipc.cc \
    -o libmpiwrap.so \
    -L${CUDA_HOME}/lib64 -lcudart \
    -L${MPI_HOME}/lib -lmpi
```

> **Important:** build with `g++`, not `nvcc`. Building with `nvcc` introduces NVIDIA OpenMP runtime dependencies (`libnvomp.so`) that cause `LD_PRELOAD` to fail.

### Transpose (IPC/MPI modes)

```bash
cd transpose
make all                          # builds: direct, direct_single, buffered, gpumpi, staged
                                  #         + _noaccum variants

# Run with LD_PRELOAD for IPC modes (COMM_MODE=0 and 1)
MPIWRAP=~/mpiwrap/libmpiwrap.so
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./direct 100 4096
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./direct_single 100 4096
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./buffered 100 4096

# MPI modes do not need LD_PRELOAD
mpirun -np 4 ./gpumpi 100 4096
mpirun -np 4 ./staged 100 4096

# Or submit everything via sbatch (handles LD_PRELOAD automatically)
sbatch ~/mpiwrap/run_transpose_all.sbatch
```

### Transpose (NVSHMEM modes)

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export NVSHMEM_BOOTSTRAP=MPI

cd transpose
make -f Makefile_nvshmem all

mpirun -np 4 ./nvshmem_direct 100 4096
mpirun -np 4 ./nvshmem_direct_single 100 4096
mpirun -np 4 ./nvshmem_buffered 100 4096

# Or via sbatch
sbatch ~/mpiwrap/run_nvshmem_all.sbatch
```

### Stencil

```bash
cd stencil
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
MPI_HOME=$(dirname $(dirname $(which mpicc)))

# IPC version (requires LD_PRELOAD)
nvcc -O3 -gencode arch=compute_90,code=sm_90 \
    -I${MPI_HOME}/include stencil_ipc.cu \
    -o stencil_ipc -L${MPI_HOME}/lib -lmpi
MPIWRAP=~/mpiwrap/libmpiwrap.so
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./stencil_ipc

# MPI version
nvcc -O3 -gencode arch=compute_90,code=sm_90 \
    -I${MPI_HOME}/include stencil_mpi.cu \
    -o stencil_mpi -L${MPI_HOME}/lib -lmpi
mpirun -np 4 ./stencil_mpi
```
