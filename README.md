# CUDA IPC vs MPI — Multi-GPU Communication Benchmarks

Benchmarking CUDA IPC against MPI for multi-GPU communication on NVIDIA H200 GPUs (NVLink).

## Structure

```
├── stencil/                        # 2D 5-point stencil benchmark
│   ├── stencil_large_contiguous-3.cu   # IPC version (ghost row exchange via IPC)
│   ├── stencil_large_mpi-3.cu          # MPI version (ghost row exchange via MPI)
│   ├── run_benchmark.sh                # Full benchmark across grid sizes
│   ├── run_4gpu.sh                     # Quick 4-GPU test
│   └── run_8gpu.sh                     # Quick 8-GPU test
│
├── transpose/                      # Matrix transpose benchmark (B = A^T)
│   ├── transpose_cuda_ipc.cu          # Final: 4 comm modes in one file
│   ├── transpose_gpu_ipc.cu           # Earlier standalone IPC version
│   ├── transpose_gpu_mpi.cu           # Earlier standalone MPI version
│   ├── Makefile
│   └── run_transpose.sbatch
│
├── mpi-intercept/                  # MPI interposer library for transparent IPC
│   ├── mpiwrap_ipc.cc                 # Intercepts MPI_Win_create/shared_query/free
│   ├── test_ipc_win.cu                # Test program for the interposer
│   └── CMakeLists.txt                 # Build system (needs bench_kernels.cu, bench_ipc.cu)
│
├── plots/                          # Benchmark visualizations
│   ├── transpose_stencil_comparison.png
│   ├── transpose_direct_vs_buffered.png
│   └── benchmark_dashboard.jsx        # Interactive React dashboard
│
└── results/                        # Tabulated benchmark data
    ├── stencil_results.txt
    └── transpose_results.txt
```

## Transpose Communication Modes

| Mode | How it works |
|------|-------------|
| IPC direct | Kernel writes directly to peer's B matrix via IPC pointer |
| IPC buffered | Pack → cudaMemcpyAsync (same stream) → unpack |
| GPU-aware MPI | Pack → MPI_Sendrecv → unpack |
| Staged MPI | Pack → GPU→CPU → MPI → CPU→GPU → unpack |

## Key Findings

**Stencil (latency-sensitive, nearest-neighbor):** IPC wins by 10–29% at 4 GPUs.
Small ghost row messages make latency dominant. IPC bypasses MPI stack overhead.

**Transpose (bandwidth-sensitive, all-to-all):** Buffered IPC ≈ GPU-aware MPI (~1065–1077 GB/s at 4 GPUs).
Direct IPC is slower at large sizes — scattered NVLink writes can't match bulk DMA transfers.
Staged MPI flatlines at ~100–120 GB/s (PCIe bottleneck).

**Direct vs Buffered IPC:** Direct wins at small matrices (less overhead). Buffered wins at large matrices (DMA engine saturates NVLink). The tradeoff is latency vs bandwidth.

## Hardware

- NVIDIA H200 GPUs (141 GB HBM3e)
- NVLink interconnect
- Stony Brook University IACS cluster

## Build & Run

```bash
# Transpose (builds all 4 modes)
cd transpose && make all
sbatch run_transpose.sbatch

# Stencil
cd stencil
sbatch run_benchmark.sh

# MPI intercept
cd mpi-intercept
mkdir build && cd build
cmake .. && make
mpirun -np 2 ./test_ipc_win
```
