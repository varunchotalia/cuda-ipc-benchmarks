#!/bin/bash
#SBATCH --job-name=stencil_bench
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --time=00:30:00
#SBATCH --output=stencil_bench_%j.out

# Benchmark IPC vs MPI at different grid sizes
# Modify --ntasks and --gpus for 2 or 8 GPU tests

module load cuda12.8/toolkit/12.8.0 openmpi/gcc14.3/4.1.8 gcc/14.3.0

cd ~/mpiwrap

NUM_GPUS=$SLURM_NTASKS

echo "=========================================="
echo "Stencil Benchmark - $NUM_GPUS GPUs"
echo "=========================================="

# Grid sizes to test: 1K, 2K, 4K, 8K, 16K, 32K
SIZES="1024 2048 4096 8192 16384 32768"

for SIZE in $SIZES; do
    echo ""
    echo "--- Grid size: ${SIZE} x ${SIZE} ---"
    
    # Compile with this grid size
    sed "s/const int N = [0-9]*/const int N = $SIZE/" stencil_large_contiguous-3.cu | \
    sed "s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" > stencil_bench_ipc.cu
    
    sed "s/const int N = [0-9]*/const int N = $SIZE/" stencil_large_mpi-3.cu | \
    sed "s/const int TOTAL_W = [0-9]*/const int TOTAL_W = $SIZE/" > stencil_bench_mpi.cu
    
    nvcc -o bench_ipc stencil_bench_ipc.cu -lmpi -Wno-deprecated-gpu-targets 2>/dev/null
    nvcc -o bench_mpi stencil_bench_mpi.cu -lmpi -Wno-deprecated-gpu-targets 2>/dev/null
    
    echo "IPC:"
    mpirun -np $NUM_GPUS ./bench_ipc 2>&1 | grep -E "(Total time|Throughput|Global L2)"
    
    echo "MPI:"
    mpirun -np $NUM_GPUS ./bench_mpi 2>&1 | grep -E "(Total time|Throughput|Global L2)"
done

# Cleanup temp files
rm -f stencil_bench_ipc.cu stencil_bench_mpi.cu bench_ipc bench_mpi

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
