# WinIPC

WinIPC lets an MPI program get direct pointers to other GPUs' memory through
standard MPI window calls. It is a PMPI interposer: load it with `LD_PRELOAD`
in front of any MPI library. Windows over GPU memory are backed by CUDA IPC
(same node) or CUDA fabric handles (multi-node NVLink).
`MPI_Win_shared_query` then returns a device pointer to the target's memory,
which kernels can load from and store to.

> The file is still `mpiwrap_ipc.cc` and the library is `libmpiwrap.so`. Queued
> Slurm jobs use those names, so the rename waits until they finish.

## What it changes

| Call | Behavior |
|---|---|
| `MPI_Win_allocate` | With info `cuda_ipc=1`, allocates device memory and returns it in `baseptr`. Without that key, it passes straight through. |
| `MPI_Win_create` | If `base` is a device pointer, exchanges IPC handles. Host memory passes straight through. |
| `MPI_Win_shared_query` | Returns the target's device pointer and size. |
| `MPI_Win_free` | Unmaps peers and frees memory WinIPC allocated. |

All other MPI calls go to the real MPI library unchanged.

## Peers that can't be reached

`MPI_Win_shared_query` returns `MPI_ERR_OTHER` for such a peer, and the
application sends to it with ordinary MPI. (MPI 4.1 §13.2.3 says to return
success with size 0 instead; that change is in progress.) A peer is
unavailable when:

- it is on another node and the system has no fabric handles (or `WINIPC_DISABLE_FABRIC=1`);
- `cudaDeviceCanAccessPeer` says no;
- the delivery probe fails (below).

**Delivery probe.** On H200 NVL nodes (two 4-GPU NVLink islands),
`cudaDeviceCanAccessPeer` returns true across islands and the IPC open
succeeds, but stores silently never arrive. So at window creation each rank
writes a marker into every peer's memory, and each owner checks which markers
arrived. A peer whose writes did not arrive is reported unavailable. This
costs two barriers and one `Alltoall` per window. Any built-in version needs
the same check or an equivalent.

## Usage

```c
MPI_Info info;
MPI_Info_create(&info);
MPI_Info_set(info, "cuda_ipc", "1");
double *mine;
MPI_Win_allocate(bytes, sizeof(double), info, comm, &mine, &win);

for (int p = 0; p < nprocs; p++) {
    MPI_Aint sz; int disp;
    int rc = MPI_Win_shared_query(win, p, &sz, &disp, &peer[p]);
    use_mpi[p] = (rc != MPI_SUCCESS);   // unavailable: use MPI_Isend/Irecv
}
// kernels write through peer[p]; then cudaDeviceSynchronize + MPI_Barrier
```

```bash
LD_PRELOAD=/path/to/libmpiwrap.so mpirun -np 4 ./app
```

## Limits

- **Synchronization is the application's job.** WinIPC does not intercept
  fence, lock or flush. Complete GPU stores (`cudaDeviceSynchronize` or a
  device fence), then synchronize ranks (a barrier or a message).
- **Multi-node windows only support pointer access.** When a window spans nodes, the real MPI
  window has size 0. `MPI_Put`/`MPI_Get` on it do not reach the GPU memory.
- **Fabric handles require `MPI_Win_allocate`** (VMM memory) and a running
  IMEX daemon. `MPI_Win_create` over a `cudaMalloc` pointer is same-node IPC only.
- `MPI_Win_shared_query` returns the caller's `disp_unit`, not the target's,
  so all ranks should use the same `disp_unit`.
- At window creation, a failed IPC handle export or fabric allocation calls
  `MPI_Abort`. It is not returned as an error code.
- It is not thread-safe: window state is in a global map with no lock.

## Environment variables

| Variable | Effect |
|---|---|
| `WINIPC_DISABLE_FABRIC=1` | Use per-peer IPC with MPI fallback even on fabric-capable hardware. `MPIWRAP_DISABLE_FABRIC` still works. |
| `WINIPC_VERIFY_PEERS=0` | Skip the delivery probe. Only use it to reproduce results from before the probe was added. |

Log lines are written to stderr and start with `[winipc]`.

## Build and test

Build with a C++ compiler, **not nvcc**. nvcc links `libnvomp`, which makes
`LD_PRELOAD` fail. From the repository root:

```bash
cmake -B build && cmake --build build --target mpiwrap test_ipc_win
LD_PRELOAD=$PWD/build/libmpiwrap.so mpirun -np 2 ./build/test_ipc_win
```

Without CMake:

```bash
mpicxx -O2 -fPIC -shared -I$CUDA_HOME/include mpi-intercept/mpiwrap_ipc.cc \
    -o libmpiwrap.so -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda
```

The `stubs` directory is only needed on machines without a GPU driver, such as login nodes.

| File | Contents |
|---|---|
| `mpiwrap_ipc.cc` | The interposer |
| `test_ipc_win.cu`, `bench_ipc.cu` | Sanity check: bandwidth, latency and atomics between 2 GPUs |
| `test_island_p2p.cu` | Shows the cross-island store failure on H200 NVL |
