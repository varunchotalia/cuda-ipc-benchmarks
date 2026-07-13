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
├── LULESH/                         # LLNL LULESH (CUDA) with pluggable halo-exchange backends
│   ├── cuda/src/                      # solver + comm layer (see LULESH section below)
│   ├── run_lulesh.sbatch              # benchmark: all 6 variants × sizes 45–100, energy cross-check
│   ├── run_lulesh_all_8gpu.sbatch     # full-length (t=0.01) verification of all 6 variants
│   ├── run_lulesh_modes_8gpu.sbatch   # send-mode comparison: pack/copy vs remote-pack vs direct
│   └── omp_4.0/ openacc/ stdpar/      # unmodified upstream LULESH variants
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

## LULESH Halo-Exchange Variants

`LULESH/cuda` is LLNL's CUDA LULESH with its 26-neighbor halo exchange
refactored into a compile-time backend API: the pack/unpack logic in
`lulesh-comms*.cu` is identical for every variant, and each backend defines
only *how bytes move* in one header under `cuda/src/comm/`. All variants are
built from `cuda/src/Makefile` (`make all`) and reproduce the staged
baseline's reported Final Origin Energy on the full sedov run
(8 ranks / 8× H200, verified by the sbatch harnesses).

### The six variants

| Variant | Binary | Build flags | Halo data path | LD_PRELOAD |
|---------|--------|-------------|----------------|------------|
| staged | `lulesh_staged` | *(none)* | GPU → host staging → `MPI_Isend/Irecv` → host → GPU | no |
| gpumpi | `lulesh_gpumpi` | `COMM_GPUMPI` | device pointers passed straight to CUDA-aware MPI | no |
| shmwin | `lulesh_shmwin` | `COMM_SHMWIN` | GPU → peer's slice of an `MPI_Win_allocate_shared` host window → GPU; `Win_sync` + barriers | no |
| ipc | `lulesh_ipc` | `COMM_IPC` | one D2D copy into the peer GPU's recv buffer via explicit `cudaIpcOpenMemHandle` mapping | no |
| mpiwrap | `lulesh_mpiwrap` | `COMM_IPC IPC_VIA_MPIWRAP` | same data path as ipc, but the app only writes portable `MPI_Win_create` + `MPI_Win_shared_query`; CUDA IPC is supplied by the interposer | **yes** |
| nvshmem | `lulesh_nvshmem` | `COMM_NVSHMEM` | `nvshmemx_putmem_on_stream` into symmetric-heap recv buffers | no |

The one-sided variants (shmwin/ipc/mpiwrap/nvshmem) post no receives: the
sender computes the destination offset inside the *receiver's* recv buffer
with `shmRecvOffset()` (which mirrors `CommRecv`'s message-ordering
bookkeeping) and writes it there directly. The init-time nodalMass exchange
stays host-packed plain MPI in every variant.

### The three send modes (IPC/mpiwrap family)

| Mode | Binaries | Build flags | What happens per message |
|------|----------|-------------|--------------------------|
| A — pack + copy | `lulesh_ipc`, `lulesh_mpiwrap` | *(default)* | pack kernel → local staging buffer → one D2D copy into the peer's packed recv buffer → receiver unpacks |
| C — remote-pack | `lulesh_ipc_rp`, `lulesh_mpiwrap_rp` | `+ IPC_REMOTE_PACK` | pack kernel writes **directly into the peer's packed recv buffer** (no staging, no copy); receiver unpack unchanged |
| B — direct | `lulesh_direct` | `COMM_IPC COMM_DIRECT` | **no pack, no unpack**: fused kernels write into the receiver's field arrays at the mirrored halo positions (`atomicAdd` for force summation, plain stores for pos/vel sync); MonoQ remote-packs since its targets are per-step pool allocations |

Mode B swaps `lulesh-comms-gpu.cu` for `lulesh-comms-direct.cu` at build
time and premaps all nine persistent nodal fields (`x,y,z,xd,yd,zd,fx,fy,fz`)
of every peer via CUDA IPC. It needs stronger synchronization (device-sync +
barrier on entry to every send, stream-sync + barrier on exit) and only
supports the structured `-s` path.

### LULESH file map

```
LULESH/
├── run_lulesh.sbatch              # all 6 variants × sizes 45–100 × 200 iters, energy table per size
├── run_lulesh_all_8gpu.sbatch     # all 6 variants, full run to t=0.01, energy cross-check
├── run_lulesh_modes_8gpu.sbatch   # modes A vs B vs C (staged/ipc/ipc_rp/mpiwrap/mpiwrap_rp/direct)
├── README                         # upstream LLNL readme
├── cuda/src/
│   ├── lulesh.cu                  # solver; calls commAllocRecv/commTeardown + COMM_RUNTIME_* hooks
│   ├── lulesh.h                   # Domain (incl. per-backend comm state); includes comm/comm_backend.h
│   ├── lulesh-comms.cu            # host-side CommRecv/CommSend/CommSBN (init exchange), backend-agnostic
│   ├── lulesh-comms-gpu.cu        # GPU pack/unpack for all packed-buffer backends, backend-agnostic
│   ├── lulesh-comms-direct.cu     # Mode B: fused remote-write kernels (replaces lulesh-comms-gpu.cu)
│   ├── comm/
│   │   ├── comm_backend.h         # compile-time dispatch, shmRecvOffset(), host-send hook defaults
│   │   ├── comm_staged.h          # baseline two-sided MPI with host staging
│   │   ├── comm_gpumpi.h          # CUDA-aware MPI (device pointers)
│   │   ├── comm_shmwin.h          # MPI shared-memory window + Win_sync barriers
│   │   ├── comm_ipc_common.h      # packed-buffer mapping + transfer macros shared by the IPC family (modes A/C)
│   │   ├── comm_ipc.h             # explicit cudaIpc handle exchange
│   │   ├── comm_mpiwrap.h         # MPI_Win_create + shared_query, backed by LD_PRELOADed libmpiwrap.so
│   │   ├── comm_direct.h          # Mode B setup: peer field mappings (+ packed buffer for MonoQ)
│   │   └── comm_nvshmem.h         # NVSHMEM symmetric heap + putmem_on_stream
│   ├── Makefile                   # targets: staged gpumpi shmwin ipc mpiwrap nvshmem ipc_rp mpiwrap_rp direct
│   ├── allocator.cu / allocator.h # upstream device-memory pool
│   ├── vector.h / util.h / sm_utils.inl   # upstream support code
│   └── sedov*.lmesh               # upstream sample meshes (-u path, untested with these backends)
└── omp_4.0/ openacc/ stdpar/      # unmodified upstream LULESH programming-model variants
```

### Build & run LULESH

```bash
cd LULESH/cuda/src
make all              # or any single target, e.g. make mpiwrap_rp

# mpiwrap flavors need the interposer at runtime; everything else runs plain
MPIWRAP=~/mpiwrap/mpi-intercept/libmpiwrap.so
mpirun -np 8 ./lulesh_staged  -s 45
LD_PRELOAD=$MPIWRAP mpirun -np 8 ./lulesh_mpiwrap -s 45

# or the full harnesses (build + run + energy cross-check):
sbatch LULESH/run_lulesh.sbatch
sbatch LULESH/run_lulesh_modes_8gpu.sbatch
```

Note: on nodes where UCX's default transport selection floods stderr with
`cuCtxGetApiVersion` errors and cripples two-sided MPI, pin the transports
(the sbatch harnesses do this): `export UCX_TLS=self,sm,cuda_copy,cuda_ipc`.

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
