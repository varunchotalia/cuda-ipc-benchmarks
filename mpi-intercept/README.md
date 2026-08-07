# WinIPC — MPI Intercept Library

Intercepts `MPI_Win_create`, `MPI_Win_shared_query`, and `MPI_Win_free` to transparently add CUDA IPC support. Existing MPI programs get GPU-direct communication without code changes.

> Formerly *mpiwrap*. The source file (`mpiwrap_ipc.cc`), the built library
> (`libmpiwrap.so`) and the CMake target (`mpiwrap`) keep their old names for
> now, because queued Slurm jobs reference them by name — Slurm freezes a job's
> script text at submit time.

## Usage
```bash
LD_PRELOAD=./libmpiwrap.so mpirun -np 4 ./your_mpi_program
```

`WINIPC_DISABLE_FABRIC=1` forces cross-node windows onto the per-peer hybrid-MPI
fallback even on fabric-capable (NVL72-class) hardware. `MPIWRAP_DISABLE_FABRIC`
is honoured as a deprecated alias.

## Build
```bash
mkdir build && cd build
cmake .. && make
```
