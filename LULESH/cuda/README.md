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

## Results

Full sedov run (`-s 45`, 3145 iterations to t=0.01), 8 ranks / 8× H200 SXM
(NVSwitch), OpenMPI 4.1.8 + UCX pinned to `self,sm,cuda_copy,cuda_ipc`.
All six variants reported the identical Final Origin Energy
(`1.482403e+06` at the log's `%12.6e` precision):

| Variant | Elapsed (s) | ms/iter | FOM (z/s) | vs staged |
|---------|------------:|--------:|----------:|----------:|
| ipc     | 1.94 | 0.617 | 1,184,149 | 1.38× |
| mpiwrap | 1.94 | 0.617 | 1,181,271 | 1.38× |
| nvshmem | 1.96 | 0.623 | 1,166,871 | 1.36× |
| shmwin  | 2.17 | 0.690 | 1,054,471 | 1.23× |
| staged  | 2.67 | 0.849 |   857,386 | 1.00× |
| gpumpi  | 6.16 | 1.959 |   372,261 | 0.43× |

Takeaways (single-run numbers at one size — quote with that caveat):

- **mpiwrap ≡ ipc**: the interposer's portable-MPI-window abstraction costs
  nothing over hand-written CUDA IPC.
- **One-sided beats two-sided by ~1.4×** at this size; the problem is
  kernel-launch-bound (~0.55 ms/iter compute floor), so this is a ~4×
  reduction of the communication share.
- **gpumpi is slowest**: per-message CUDA-aware MPI is a bad fit for
  LULESH's 26 mostly-tiny messages × 3 phases.
- Without UCX transport pinning, these nodes' default UCX selection floods
  stderr with `cuCtxGetApiVersion` errors and slows two-sided MPI by up to
  ~40×; the errors are steered around, not root-caused. Pin transports for
  any number you intend to quote.

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
