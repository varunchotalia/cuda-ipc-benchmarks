# MPI Intercept Library

Intercepts `MPI_Win_create`, `MPI_Win_shared_query`, and `MPI_Win_free` to transparently add CUDA IPC support. Existing MPI programs get GPU-direct communication without code changes.

## Usage
```bash
LD_PRELOAD=./libmpiwrap.so mpirun -np 4 ./your_mpi_program
```

## Build
```bash
mkdir build && cd build
cmake .. && make
```
