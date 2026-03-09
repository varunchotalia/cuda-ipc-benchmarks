// transpose_gpu_mpi.cu
//
// Multi-GPU transpose benchmark using MPI for exchange.
// Each rank owns a vertical strip of the global matrix A (N x TOTAL_W).
// Per iteration: exchange full local block with one neighbor (ring), then
// transpose the received block locally using a tiled shared-memory kernel.
//
// By default uses host-staging (safe for non-CUDA-aware MPI).
// To use CUDA-aware MPI with device buffers, compile with -DUSE_CUDA_AWARE_MPI.
//
// Compile: nvcc -O3 -lineinfo -o transpose_gpu_mpi transpose_gpu_mpi.cu -lmpi
// Run:     mpirun -np 4 ./transpose_gpu_mpi <iters> <N> <TOTAL_W>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE 32

static void die_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void init_kernel(double* a, int N, int W, int rank) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= N) return;
    a[y * W + x] = (double)(rank) + 1e-6 * (double)(y * W + x);
}

__global__ void transpose_tiled(const double* __restrict__ in,
                                double* __restrict__ out,
                                int N, int W) {
    __shared__ double tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x; // col in input
    int y = blockIdx.y * TILE + threadIdx.y; // row in input

    if (x < W && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    }
    __syncthreads();

    int ox = blockIdx.y * TILE + threadIdx.x; // col in output
    int oy = blockIdx.x * TILE + threadIdx.y; // row in output

    if (ox < N && oy < W) {
        out[oy * N + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: %s <iters> <N> <TOTAL_W>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int iters = atoi(argv[1]);
    int N = atoi(argv[2]);
    int TOTAL_W = atoi(argv[3]);

    if (TOTAL_W % size != 0) {
        if (rank == 0) {
            printf("ERROR: TOTAL_W must be divisible by #ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    int num_devices = 0;
    die_cuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
    int dev = rank % num_devices;
    die_cuda(cudaSetDevice(dev), "cudaSetDevice");
    printf("[Rank %d] Using GPU %d\n", rank, dev);

    int W = TOTAL_W / size;
    size_t bytes = (size_t)N * (size_t)W * sizeof(double);

    if (rank == 0) {
        printf("GPUs: %d\n", size);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * (double)TOTAL_W / 1e6);
        printf("Per GPU: %d x %d\n", N, W);
        printf("Block bytes per GPU: %.2f MB\n", bytes / 1e6);
        printf("Iterations: %d\n", iters);
#ifdef USE_CUDA_AWARE_MPI
        printf("MPI mode: CUDA-aware device buffers\n");
#else
        printf("MPI mode: host staging (safe)\n");
#endif
    }

    // Ring neighbors
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    // Allocate device buffers
    double* d_A = nullptr;
    double* d_recv = nullptr;
    double* d_B = nullptr; // transposed output (W x N)
    die_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    die_cuda(cudaMalloc(&d_recv, bytes), "cudaMalloc d_recv");
    die_cuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");

    // Host staging buffers (only used when not CUDA-aware)
#ifndef USE_CUDA_AWARE_MPI
    double* h_send = (double*)malloc(bytes);
    double* h_recv = (double*)malloc(bytes);
    if (!h_send || !h_recv) {
        fprintf(stderr, "Host allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif

    // Initialize
    dim3 t(16, 16);
    dim3 b((W + t.x - 1) / t.x, (N + t.y - 1) / t.y);
    init_kernel<<<b, t>>>(d_A, N, W, rank);
    die_cuda(cudaGetLastError(), "init_kernel");
    die_cuda(cudaDeviceSynchronize(), "init sync");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    dim3 bt((W + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    for (int it = 0; it < iters; ++it) {
#ifdef USE_CUDA_AWARE_MPI
        MPI_Sendrecv(d_A, (int)(bytes / sizeof(double)), MPI_DOUBLE, send_to, 0,
                     d_recv, (int)(bytes / sizeof(double)), MPI_DOUBLE, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
        die_cuda(cudaMemcpy(h_send, d_A, bytes, cudaMemcpyDeviceToHost), "D2H");
        MPI_Sendrecv(h_send, (int)(bytes / sizeof(double)), MPI_DOUBLE, send_to, 0,
                     h_recv, (int)(bytes / sizeof(double)), MPI_DOUBLE, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        die_cuda(cudaMemcpy(d_recv, h_recv, bytes, cudaMemcpyHostToDevice), "H2D");
#endif

        // Transpose received block into d_B
        transpose_tiled<<<bt, dim3(TILE, TILE)>>>(d_recv, d_B, N, W);
        die_cuda(cudaGetLastError(), "transpose_tiled");
        die_cuda(cudaDeviceSynchronize(), "transpose sync");

        MPI_Barrier(MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double bytes_total = (double)bytes * (double)size * (double)iters;
        double cells_total = (double)N * (double)TOTAL_W * (double)iters;
        printf("\n=== Results (MPI, %d GPUs) ===\n", size);
        printf("Total time: %.6f s (slowest rank)\n", max_time);
        printf("Per iteration: %.6f ms\n", (max_time / iters) * 1e3);
        printf("Throughput: %.2f GB/s (exchange only, global)\n",
               (bytes_total / max_time) / 1e9);
        printf("Transpose throughput: %.2f billion cells/s (global)\n",
               (cells_total / max_time) / 1e9);
    }

#ifndef USE_CUDA_AWARE_MPI
    free(h_send);
    free(h_recv);
#endif
    cudaFree(d_A);
    cudaFree(d_recv);
    cudaFree(d_B);

    MPI_Finalize();
    return 0;
}
