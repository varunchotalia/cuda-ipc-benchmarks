# CUDA IPC vs MPI — Multi-GPU Communication Benchmarks

Code and data for the paper *MPI Windows as an Abstraction for CUDA
Interprocess Communication* (Chotalia, Schuchart, Hammond). WinIPC, an MPI
profiling-interface library, gives applications CUDA IPC peer pointers through
`MPI_Win_allocate` / `MPI_Win_shared_query` / `MPI_Win_free`. The repository
also holds the three case studies (distributed transpose, five-point stencil,
CUDA LULESH) and the comparison baselines (GPU-aware MPI, host-staged MPI,
NVSHMEM, a host shared window).

### Paper figures and their sources

| Paper | File | Generator | Data |
|---|---|---|---|
| Fig. 3, Table III | `plots/transpose_ipc_accum.pdf`, `plots/transpose_ipc_noaccum.pdf` | `plots/make_transpose_plots.py` | `results/transpose_90161.csv` |
| Fig. 4, Table IV | `plots/stencil_speedup.pdf` | `plots/make_stencil_plots.py` | `results/stencil_results.txt` |
| Fig. 5 | `plots/lulesh_variants_sxm.pdf` | `plots/make_lulesh_plots.py` | `results/lulesh_variance.csv` |
| Table I | — | `scripts/count_mechanism_lines.sh`, `scripts/check_no_ipc_calls.sh` | `results/table1_provenance.md` |
| Fig. 6, Table V | `plots/gb200_transpose.pdf`, `plots/gb200_stencil.pdf` | not public | collaborator runs on a GB200 system; the per-run records are not public |

Commit hashes from before 2026-09-23 that no longer resolve are listed in
`results/commit_hash_map_2026-09-23.md`.

> **Naming.** The MPI-window interposer and its halo-exchange backend are now
> called **WinIPC**; they were previously called *mpiwrap*. The rename is
> complete in prose, figures and comments. It is **not** yet applied to
> filenames or build identifiers — `libmpiwrap.so`, `mpi-intercept/mpiwrap_ipc.cc`,
> `LULESH/cuda/src/comm/comm_mpiwrap.h`, `scripts/build_mpiwrap.sh`, the
> `mpiwrap`/`mpiwrap_rp` make targets, the `lulesh_mpiwrap*` binaries, the
> `IPC_VIA_MPIWRAP` macro and the `$MPIWRAP_LIB` variable all keep their old
> spelling, because Slurm freezes a job's script text at submit time and queued
> jobs still reference them. Read `mpiwrap` as WinIPC wherever it appears as an
> identifier.

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
│   ├── stencil_mpi.cu                  # Ghost row exchange via staged MPI (D2H/H2D)
│   └── stencil_gpu_mpi.cu              # Ghost row exchange via GPU-aware MPI (no staging)
│
├── LULESH/                         # LLNL LULESH (CUDA) with pluggable halo-exchange backends
│   ├── cuda/src/                      # solver + comm layer (see LULESH section below)
│   ├── run_lulesh.sbatch              # benchmark sweep: all 9 binaries × sizes 45–100
│   ├── run_lulesh_verify.sbatch       # all 9 binaries, full run, energy cross-check
│   └── omp_4.0/ openacc/ stdpar/      # unmodified upstream LULESH variants
│
├── mpi-intercept/                  # WinIPC: MPI interposer for transparent CUDA IPC
│   ├── mpiwrap_ipc.cc                 # Intercepts MPI_Win_create/allocate/shared_query/free;
│   │                                  #   CUDA IPC on-node, CUDA fabric handles across NVLink nodes
│   ├── test_ipc_win.cu                # Standalone interposer sanity check (see root CMakeLists.txt)
│   ├── bench_ipc.cu                    #   raw IPC-window bandwidth/latency/atomics microbenchmark
│   └── bench_kernels.cu                #   GPU kernels used by bench_ipc.cu
│
├── plots/                          # Paper figures (PDF) and their generators
│   ├── make_transpose_plots.py        #   Fig. 3
│   ├── make_stencil_plots.py          #   Fig. 4
│   ├── make_lulesh_plots.py           #   Fig. 5
│   └── gb200_*.pdf                    #   Fig. 6
│
├── results/                        # Committed records behind the paper (raw *.out logs are not tracked)
│   ├── transpose_90161.csv            #   every transpose run behind Fig. 3 / Table III
│   ├── stencil_results.txt            #   Table IV / Fig. 4
│   ├── lulesh_variance.csv            #   Fig. 5
│   └── *.md                           #   investigation notes; each says what it supports
│
├── scripts/
│   ├── run_transpose_all.sbatch       # Builds and runs all IPC/MPI transpose variants
│   ├── run_nvshmem_all.sbatch         # Builds and runs all NVSHMEM transpose variants
│   └── run_nvl72.sh                   # Multi-node NVLink suite (see Build & Run below)
└── CMakeLists.txt                  # One-command build for everything (see Build & Run below)
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

**Per-peer IPC fallback** (`IPC direct` and `IPC buffered`): every
`MPI_Win_shared_query` is checked, and a peer that fails it (cross-node
without fabric handles, or a same-node GPU-island boundary with no P2P)
falls back to a real `MPI_Isend`/`Irecv` of the same tile instead of
writing through a stale/NULL pointer. `IPC direct (single-kernel)` can't
mix transports inside one kernel launch, so it aborts with a clear
message if any peer is unreachable, instead.

Known limitation: the fallback assumes IPC reachability is *symmetric*
per GPU pair, and (like the rest of the IPC/WinIPC family) it requires
a working CUDA-aware MPI to actually send the fallback messages — see
[LULESH/cuda/README.md](LULESH/cuda/README.md#the-six-variants) for the
full explanation.

## LULESH

`LULESH/cuda` is LLNL's CUDA LULESH with the halo exchange refactored into
six pluggable backends (staged / gpumpi / shmwin / ipc / WinIPC / nvshmem)
plus three send modes for the IPC/WinIPC family (pack+copy, remote-pack,
direct field writes). One binary per variant; all reproduce the staged
baseline's reported Final Origin Energy on the full sedov run. See
[LULESH/cuda/README.md](LULESH/cuda/README.md) for the variant/mode tables,
results, file map, and build instructions.

## Key Findings

These are the paper's results. Transpose rates follow the paper's Eq. 1,
2N²·sizeof(double)/t, which is an application-rate convention, not measured
link bandwidth.

**Transpose, 4× H200 in one NVLink domain** (job 90161; median of 3 runs, each
100 timed iterations after 20 untimed). With accumulation at 1024², the
single-kernel direct variant reaches 563.8 GB/s against 231.1 GB/s for
GPU-aware MPI (2.44×), and 1.9× at 2048². The lead is gone by 4096² (878.1 vs
883.1). At 16384², buffered WinIPC and GPU-aware MPI are within 1.5% (1064.1
vs 1080.3 GB/s). Host-staged MPI stays at 66–103 GB/s.

**Stencil, 4× H200.** WinIPC takes 3.95 ms against 4.71 ms for GPU-aware MPI
and 7.80 ms for host-staged MPI at 1024² (100 timed iterations). The gap
closes with grid size: 143.04 vs 143.73 ms at 32768².

**LULESH, 8× H200 SXM, `-s 45`.** Matched WinIPC and handwritten CUDA IPC
backends differ by at most 0.46% across 5 runs, with the sign changing, so
pointer acquisition through the window has no measurable steady-state cost.
Direct field writes (mode C) reach 1.832 Gzones/s, 55.7% above host-staged
MPI. GPU-aware MPI is +6.5% and NVSHMEM +11.7% over host-staged MPI.

**GB200 multi-node NVLink, 16 and 32 GPUs** (one run; 100 timed iterations
after 20 untimed). The per-phase direct WinIPC transpose is 1.42× and 1.41×
GPU-aware MPI at order 6912, and single-kernel direct is 8.1× and 19.8×. The
paper gives the placement and caveats.

**Not reported.** The intra-node NVSHMEM transpose comparison does not
reproduce on this cluster; see `results/nvshmem_not_reproducible.md`.

## Hardware

nvwulf cluster, Stony Brook University IACS. GPU interconnect differs per
node type (details in [LULESH/cuda/README.md](LULESH/cuda/README.md)):

- `h200x8-03`: 8× H200 SXM (141 GB HBM3e) — NVSwitch, all-to-all NVLink
- `h200x8-01/02/04`, `h200x4`: H200 NVL — two 4-GPU NVLink islands per
  8-GPU node (all-to-all NV6 within an island); cross-island P2P is
  PCIe + UPI
- `b40x4`: 4× RTX PRO 6000 Blackwell — no NVLink, P2P is PCIe only

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

### Quick start: CMake (one build for everything)

```bash
# GB200 / NVL72: use -DCMAKE_CUDA_ARCHITECTURES=100 (default is 90 = H200)
export NVSHMEM_HOME=/path/to/nvshmem        # optional: enables nvshmem targets
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100
cmake --build build -j

# scheduler-agnostic runner: transpose + stencil + LULESH, with LD_PRELOAD
# and energy cross-checks handled for you. Inside a Slurm allocation it picks
# `srun --mpi=pmix --ntasks-per-node=$GPUS_PER_NODE -n` itself. If you set
# LAUNCH by hand, keep --ntasks-per-node, or ranks will not land one per GPU.
bash scripts/run_nvl72.sh
# With an HPC-X toolchain, `source scripts/env_gb200.sh` first (it needs
# CUDA_HOME, HPCX_HOME and NVSHMEM_HOME set), in the same shell as the build.
```

On a multi-node NVLink system the interposer logs `fabric window: N ranks
...` when the CUDA fabric-handle path (requires the IMEX daemon) is active;
`N of M peers not IPC-reachable` means it degraded to per-peer hybrid
IPC+MPI instead. transpose and stencil use `MPI_Win_allocate`, so their IPC
modes ride the same interposer transports as LULESH's WinIPC variants.
The Makefile instructions below remain valid for per-benchmark builds.

### Build the MPI interposer library

The IPC modes (`COMM_MODE=0,1`) require `libmpiwrap.so` at runtime for the `MPI_Win_create`/`MPI_Win_shared_query` intercept. It's not committed (it links local CUDA/MPI install paths, so a shipped binary wouldn't be portable) — build it once:

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
MPIWRAP=$(realpath ../libmpiwrap.so)   # built in the step above, at the repo root
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./direct 100 4096
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./direct_single 100 4096
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./buffered 100 4096

# MPI modes do not need LD_PRELOAD
mpirun -np 4 ./gpumpi 100 4096
mpirun -np 4 ./staged 100 4096

# Or submit everything via sbatch (handles LD_PRELOAD automatically).
# The sbatch scripts assume the repository is checked out at ~/mpiwrap;
# edit their `cd` lines if yours is elsewhere.
sbatch scripts/run_transpose_all.sbatch
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

# Or via sbatch (same ~/mpiwrap assumption as above)
sbatch scripts/run_nvshmem_all.sbatch
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
MPIWRAP=$(realpath ../libmpiwrap.so)
LD_PRELOAD=$MPIWRAP mpirun -np 4 ./stencil_ipc

# Staged MPI version (D2H/H2D)
nvcc -O3 -gencode arch=compute_90,code=sm_90 \
    -I${MPI_HOME}/include stencil_mpi.cu \
    -o stencil_mpi -L${MPI_HOME}/lib -lmpi
mpirun -np 4 ./stencil_mpi

# GPU-aware MPI version (device pointers straight to MPI_Sendrecv, no staging)
nvcc -O3 -gencode arch=compute_90,code=sm_90 \
    -I${MPI_HOME}/include stencil_gpu_mpi.cu \
    -o stencil_gpumpi -L${MPI_HOME}/lib -lmpi
mpirun -np 4 ./stencil_gpumpi
```
