#!/bin/bash
#SBATCH --job-name=stencil_test
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=00:10:00
#SBATCH --output=stencil_8gpu_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

nvcc -o stencil_large_ipc stencil_large_contiguous-2.cu -lmpi
nvcc -o stencil_large_mpi stencil_large_mpi-2.cu -lmpi

echo "=== IPC VERSION (8 GPUs) ==="
mpirun -np 8 ./stencil_large_ipc

echo ""
echo "=== MPI VERSION (8 GPUs) ==="
mpirun -np 8 ./stencil_large_mpi