#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

extern "C" int MPIX_CUDA_IPC_bench(MPI_Win win, int iters, size_t bytes);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with -np 2\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Select GPU
    int ndev;
    cudaGetDeviceCount(&ndev);
    cudaSetDevice(rank % ndev);
    
    // Allocate GPU buffer (try 1GB, fallback to 256MB)
    size_t bytes = 1ULL << 30;
    void* d_buf;
    if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
        bytes = 256ULL << 20;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            if (rank == 0) fprintf(stderr, "cudaMalloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    if (rank == 0) fprintf(stderr, "Buffer: %.0f MB\n", bytes / 1e6);
    
    // Create MPI window (intercepted by mpiwrap_ipc)
    MPI_Win win;
    MPI_Win_create(d_buf, bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    
    // Run benchmarks
    MPIX_CUDA_IPC_bench(win, 1000, bytes);
    
    // Cleanup
    MPI_Win_free(&win);
    cudaFree(d_buf);
    
    if (rank == 0) fprintf(stderr, "Done.\n");
    MPI_Finalize();
    return 0;
}
