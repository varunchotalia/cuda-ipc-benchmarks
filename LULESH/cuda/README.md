# LULESH-CUDA with Pluggable Halo-Exchange Backends

LLNL's CUDA LULESH with its 26-neighbor halo exchange refactored into a
compile-time backend API. The pack/unpack logic in `src/lulesh-comms*.cu`
is identical for every variant; each backend defines only *how bytes move*
in one header under `src/comm/`. One binary per variant, all built from
`src/Makefile` (`make all`), all reproducing the staged baseline's reported
Final Origin Energy on the full sedov run.

## The six variants

| Variant | Binary | Build flags | Halo data path | LD_PRELOAD |
|---------|--------|-------------|----------------|------------|
| staged | `lulesh_staged` | *(none)* | GPU → host staging → `MPI_Isend/Irecv` → host → GPU | no |
| gpumpi | `lulesh_gpumpi` | `COMM_GPUMPI` | device pointers passed straight to CUDA-aware MPI | no |
| shmwin | `lulesh_shmwin` | `COMM_SHMWIN` | GPU → peer's slice of an `MPI_Win_allocate_shared` host window → GPU; `Win_sync` + barriers | no |
| ipc | `lulesh_ipc` | `COMM_IPC` | one D2D copy into the peer GPU's recv buffer via explicit `cudaIpcOpenMemHandle` mapping | no |
| mpiwrap | `lulesh_mpiwrap` | `COMM_IPC IPC_VIA_MPIWRAP` | same data path as ipc, but the app only writes portable `MPI_Win_create` + `MPI_Win_shared_query`; CUDA IPC is supplied by the LD_PRELOADed `libmpiwrap.so` interposer | **yes** |
| nvshmem | `lulesh_nvshmem` | `COMM_NVSHMEM` | `nvshmemx_putmem_on_stream` into symmetric-heap recv buffers | no |

The one-sided variants (shmwin / ipc / mpiwrap / nvshmem) post no receives:
the sender computes the destination offset inside the *receiver's* recv
buffer with `shmRecvOffset()` (which replays `CommRecv`'s message-ordering
bookkeeping for the receiver's boundary booleans) and writes the data there
directly. Completion is a barrier-based epoch instead of recv completion.
The init-time nodalMass exchange stays host-packed plain MPI in every
variant; one-sided backends activate afterwards (`g_commActive`).

## The three send modes (IPC/mpiwrap family)

| Mode | Binaries | Build flags | What happens per message |
|------|----------|-------------|--------------------------|
| A — pack + copy | `lulesh_ipc`, `lulesh_mpiwrap` | *(default)* | pack kernel → local staging buffer → one D2D copy into the peer's packed recv buffer → receiver unpacks |
| C — remote-pack | `lulesh_ipc_rp`, `lulesh_mpiwrap_rp` | `+ IPC_REMOTE_PACK` | pack kernel writes **directly into the peer's packed recv buffer** (no local staging, no separate copy); receiver unpack unchanged |
| B — direct | `lulesh_direct` | `COMM_IPC COMM_DIRECT` | **no pack, no unpack**: one fused kernel per message reads the sender's strided boundary values and writes them into the receiver's field arrays at the mirrored halo positions |

Mode B details:
- Swaps `lulesh-comms-gpu.cu` for `lulesh-comms-direct.cu` at build time and
  premaps all nine persistent nodal fields (`x,y,z,xd,yd,zd,fx,fy,fz`) of
  every peer via CUDA IPC at setup.
- Force summation (SBN) uses `atomicAdd`: up to seven neighbors legitimately
  contribute to a shared edge/corner node, and non-atomic cross-GPU `+=`
  would lose updates. Summation order changes, so the last digits of the
  energy may deviate from the packed variants.
- Position/velocity sync uses plain stores (overlapping writers carry the
  value of the same physical node).
- MonoQ cannot go direct: its destinations (`delv_xi/eta/zeta`) are per-step
  pool allocations whose addresses cannot be premapped, so MonoQ remote-packs
  into the peer's packed recv buffer and unpacks locally.
- Needs stronger synchronization (device-sync + barrier on entry to every
  send, stream-sync + barrier on exit) and supports only the structured
  `-s` path (the `-u` path never calls `SetupCommBuffers`).

## Hardware: the nvwulf cluster (Stony Brook IACS)

GPU-to-GPU interconnect differs per node type, which matters for every
one-sided variant here ([cluster page](https://rci.stonybrook.edu/HPC/nvwulf/about)
lists nodes but not interconnect; the table below is from Slurm node
records, `nvidia-smi` device names in run logs, and `nvidia-smi topo -m`):

| Partition | Nodes | GPUs per node | GPU-to-GPU interconnect |
|-----------|-------|---------------|--------------------------|
| `h200x8` | h200x8-03 | 8× H200 **SXM** (192 CPUs, ~2.2 TB) | NVSwitch: all-to-all NVLink |
| `h200x8` | h200x8-01/02/04 | 8× H200 **NVL** (64 CPUs, ~1.4 TB) | two 4-GPU NVLink islands (GPUs 0–3 and 4–7, all-to-all NV6 within an island, one island per socket); every cross-island pair is `SYS` = PCIe + UPI, no NVLink |
| `h200x4` | h200x4-[01-04] | 4× H200 NVL | single socket; likely one 4-GPU NV6 island (not probed) |
| `b40x4` | b40x4-[01-09] | 4× RTX PRO 6000 Blackwell | **no NVLink** — P2P is PCIe only (`NODE`/`SYS` in `nvidia-smi topo -m`) |

Two consequences:

- The `h200x8` partition is **heterogeneous**: a job may land on the SXM
  node or an NVL node, and `nvidia-smi topo -m` differs between them.
  Record the node (or GPU name: "H200" = SXM, "H200 NVL" = NVL) with any
  number you intend to compare.
- On `b40x4` the IPC/NVSHMEM variants still run, but all peer traffic is
  PCIe — expect very different ratios than the H200 results below.
- On the NVL nodes with the 2×2×2 rank decomposition, the plane-direction
  halos — the largest messages — connect rank i to rank i+4, i.e. GPU
  islands 0–3 to 4–7: **the biggest transfers ride PCIe + UPI, not
  NVLink**. The results below were measured under that constraint.

## Results

Full sedov run (`-s 45`, 3145 iterations to t=0.01), 8 ranks on
**h200x8-03 (8× H200 SXM, NVSwitch all-to-all)**, OpenMPI 4.1.8 + UCX
pinned to `self,sm,cuda_copy,cuda_ipc`. All nine binaries reported the
identical Final Origin Energy (`1.482403e+06` at the log's `%12.6e`
precision) — including `direct`, whose atomicAdd reordering stayed below
printed precision:

| Variant | Mode | Elapsed (s) | ms/iter | FOM (z/s) | vs staged |
|---------|------|------------:|--------:|----------:|----------:|
| direct     | B | 1.28 | 0.407 | 1,796,653 | 2.78× |
| ipc_rp     | C | 1.47 | 0.467 | 1,558,925 | 2.42× |
| mpiwrap_rp | C | 1.48 | 0.470 | 1,552,615 | 2.41× |
| ipc        | A | 1.75 | 0.556 | 1,307,411 | 2.03× |
| nvshmem    | A | 1.76 | 0.560 | 1,304,443 | 2.02× |
| mpiwrap    | A | 1.76 | 0.560 | 1,304,145 | 2.02× |
| shmwin     | — | 2.09 | 0.665 | 1,095,799 | 1.70× |
| gpumpi     | — | 2.47 | 0.785 |   928,940 | 1.44× |
| staged     | — | 3.56 | 1.132 |   644,302 | 1.00× |

Takeaways (single-run numbers at one size — quote with that caveat):

- **Each mode step pays off**: remote-pack (C) removes the local staging
  copy and gains 16% over pack+copy (A); direct field writes (B) also
  remove the unpack and gain another 13%. End to end, `direct` runs the
  halo exchange 2.8× faster than staged MPI.
- **mpiwrap ≡ ipc in every mode** (1.76 vs 1.75, 1.48 vs 1.47): the
  interposer's portable-MPI-window abstraction costs nothing over
  hand-written CUDA IPC.
- **Node type matters**: mode A ipc measured 1.75 s here (all-to-all SXM)
  vs 1.94 s on an NVL node, where the plane-direction halos cross the
  4-GPU-island boundary over PCIe + UPI (~10% penalty).
- **gpumpi and staged trail** — per-message two-sided MPI is a bad fit for
  LULESH's 26 mostly-tiny messages × 3 phases per iteration.
- Without UCX transport pinning, these nodes' default UCX selection floods
  stderr with `cuCtxGetApiVersion` errors and can slow two-sided MPI by up
  to ~40×; the errors are steered around, not root-caused. One staged run
  also aborted with a Volume Error on freshly rebooted h200x8-03 and passed
  on rerun — treat isolated failures there with suspicion.

## File map

```
LULESH/
├── run_lulesh.sbatch              # benchmark sweep: all 9 binaries × sizes 45–100 × 200 iters,
│                                  #   energy/elapsed/FOM summary per size
├── run_lulesh_verify.sbatch       # verification: all 9 binaries, full run to t=0.01,
│                                  #   energy cross-check against staged
├── README                         # upstream LLNL readme
├── cuda/
│   ├── README.md                  # this file
│   └── src/
│       ├── lulesh.cu              # solver; calls commAllocRecv/commTeardown + COMM_RUNTIME_* hooks
│       ├── lulesh.h               # Domain (incl. per-backend comm state); includes comm/comm_backend.h
│       ├── lulesh-comms.cu        # host-side CommRecv/CommSend/CommSBN (init exchange), backend-agnostic
│       ├── lulesh-comms-gpu.cu    # GPU pack/unpack for all packed-buffer backends, backend-agnostic
│       ├── lulesh-comms-direct.cu # Mode B: fused remote-write kernels (replaces lulesh-comms-gpu.cu)
│       ├── comm/
│       │   ├── comm_backend.h     # compile-time dispatch, shmRecvOffset(), host-send hook defaults
│       │   ├── comm_staged.h      # baseline two-sided MPI with host staging
│       │   ├── comm_gpumpi.h      # CUDA-aware MPI (device pointers)
│       │   ├── comm_shmwin.h      # MPI shared-memory window + Win_sync barriers
│       │   ├── comm_ipc_common.h  # packed-buffer mapping + transfer macros shared by the
│       │   │                      #   IPC family; modes A and C live here
│       │   ├── comm_ipc.h         # explicit cudaIpc handle exchange
│       │   ├── comm_mpiwrap.h     # MPI_Win_create + shared_query, backed by libmpiwrap.so
│       │   ├── comm_direct.h      # Mode B setup: peer field mappings (+ packed buffer for MonoQ)
│       │   └── comm_nvshmem.h     # NVSHMEM symmetric heap + putmem_on_stream
│       ├── Makefile               # targets: staged gpumpi shmwin ipc mpiwrap nvshmem
│       │                          #          ipc_rp mpiwrap_rp direct   (make all builds all 9)
│       ├── allocator.cu/.h        # upstream device-memory pool
│       ├── vector.h util.h sm_utils.inl   # upstream support code
│       └── sedov*.lmesh           # upstream sample meshes (-u path, untested with these backends)
└── omp_4.0/ openacc/ stdpar/      # unmodified upstream LULESH programming-model variants
```

## Build & run

```bash
cd LULESH/cuda/src
make all                # or any single target, e.g. make mpiwrap_rp

# mpiwrap flavors need the interposer at runtime; everything else runs plain
MPIWRAP=~/mpiwrap/mpi-intercept/libmpiwrap.so
mpirun -np 8 ./lulesh_staged -s 45
LD_PRELOAD=$MPIWRAP mpirun -np 8 ./lulesh_mpiwrap -s 45

# or the harnesses (build + run + energy cross-check, UCX pinned):
sbatch LULESH/run_lulesh_verify.sbatch   # correctness, full-length run
sbatch LULESH/run_lulesh.sbatch          # size sweep benchmark
```

Rank count must be a cube (8 = 2×2×2). NVSHMEM requires one rank per GPU;
the other variants also run with ranks sharing a GPU.
