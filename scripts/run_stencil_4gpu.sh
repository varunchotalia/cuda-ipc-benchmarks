#!/bin/bash
#SBATCH --job-name=stencil_test
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --output=stencil_4gpu_%j.out

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

nvcc -o stencil_large_ipc stencil_ipc.cu -lmpi
nvcc -o stencil_large_mpi stencil_mpi.cu -lmpi

echo "=== IPC VERSION (4 GPUs) ==="
mpirun -np 4 ./stencil_large_ipc

echo ""
echo "=== MPI VERSION (4 GPUs) ==="
mpirun -np 4 ./stencil_large_mpi