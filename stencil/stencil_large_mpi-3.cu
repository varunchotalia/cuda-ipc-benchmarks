// stencil_large_mpi.cu
//
// Multi-GPU stencil with MPI Send/Recv ghost exchange (baseline)
// Supports 1 to 16 GPUs (any count)
//
// Same computation as IPC version, but ghost data travels:
//   GPU → CPU → MPI → CPU → GPU
//
// Compile: nvcc -o stencil_large_mpi stencil_large_mpi.cu -lmpi
// Run:     mpirun -np 4 ./stencil_large_mpi

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// STENCIL KERNEL
// ============================================================================

__global__
void stencil_kernel(const double* __restrict__ old_grid,
                    double* __restrict__ new_grid,
                    int N, int W, double weight)
{
    int pitch = W + 2;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j > W) return;

    if (i == 0 || i == N - 1) {
        new_grid[i * pitch + j] = 0.0;
        return;
    }

    int idx = i * pitch + j;
    new_grid[idx] = weight * (
        old_grid[idx - pitch] +
        old_grid[idx + pitch] +
        old_grid[idx - 1] +
        old_grid[idx + 1]
    );
}

// ============================================================================
// PACK / UNPACK
// ============================================================================

__global__
void pack_edge(const double* __restrict__ grid, double* __restrict__ buffer,
               int N, int pitch, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    buffer[i] = grid[i * pitch + col];
}

__global__
void unpack_ghost(const double* __restrict__ buffer, double* __restrict__ grid,
                  int N, int pitch, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    grid[i * pitch + col] = buffer[i];
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    printf("[Rank %d] Using GPU %d\n", rank, rank % num_devices);

    // LARGE GRID
    const int N = 16384;
    const int TOTAL_W = 16384;
    const int W = TOTAL_W / size;
    const int pitch = W + 2;

    size_t grid_size  = (size_t)N * pitch * sizeof(double);
    size_t ghost_size = (size_t)N * sizeof(double);

    if (rank == 0) {
        printf("GPUs: %d\n", size);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d + 2 ghost columns\n", N, W);
        printf("Using MPI Send/Recv (not IPC)\n");
    }

    // Allocate grids
    double *d_old, *d_new;
    cudaMalloc(&d_old, grid_size);
    cudaMalloc(&d_new, grid_size);

    // GPU buffers for pack/unpack (one per direction)
    double *d_send_buf_L, *d_send_buf_R;
    double *d_recv_buf_L, *d_recv_buf_R;
    cudaMalloc(&d_send_buf_L, ghost_size);
    cudaMalloc(&d_send_buf_R, ghost_size);
    cudaMalloc(&d_recv_buf_L, ghost_size);
    cudaMalloc(&d_recv_buf_R, ghost_size);

    // HOST buffers for MPI (one per direction)
    double *h_send_buf_L = (double*)malloc(ghost_size);
    double *h_send_buf_R = (double*)malloc(ghost_size);
    double *h_recv_buf_L = (double*)malloc(ghost_size);
    double *h_recv_buf_R = (double*)malloc(ghost_size);

    cudaMemset(d_old, 0, grid_size);
    cudaMemset(d_new, 0, grid_size);

    // Heat source in center of global grid
    int global_center_col = TOTAL_W / 2;
    int my_first_global_col = rank * W;
    int my_last_global_col  = my_first_global_col + W - 1;

    if (global_center_col >= my_first_global_col &&
        global_center_col <= my_last_global_col) {
        int local_col = global_center_col - my_first_global_col + 1;
        int center_i = N / 2;
        double init_val = 100.0;
        cudaMemcpy(&d_old[center_i * pitch + local_col], &init_val,
                   sizeof(double), cudaMemcpyHostToDevice);
        printf("[Rank %d] Heat source at global col %d (local col %d)\n",
               rank, global_center_col, local_col);
    }

    // Neighbors
    int left_rank  = (rank > 0)        ? rank - 1 : -1;
    int right_rank = (rank < size - 1) ? rank + 1 : -1;

    printf("[Rank %d] Neighbors: left=%d, right=%d\n", rank, left_rank, right_rank);

    // Kernel configs
    dim3 stencil_threads(16, 16);
    dim3 stencil_blocks((W + stencil_threads.x - 1) / stencil_threads.x,
                        (N + stencil_threads.y - 1) / stencil_threads.y);

    int copy_threads = 256;
    int copy_blocks = (N + copy_threads - 1) / copy_threads;

    double weight = 0.25;
    int iterations = 100;

    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    for (int iter = 0; iter < iterations; iter++) {

        // === PACK EDGES ===

        // Pack left edge (col 1)
        if (left_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_send_buf_L, N, pitch, 1
            );
        }
        // Pack right edge (col W)
        if (right_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_send_buf_R, N, pitch, W
            );
        }
        cudaDeviceSynchronize();

        // === GPU → CPU ===
        if (left_rank >= 0)
            cudaMemcpy(h_send_buf_L, d_send_buf_L, ghost_size, cudaMemcpyDeviceToHost);
        if (right_rank >= 0)
            cudaMemcpy(h_send_buf_R, d_send_buf_R, ghost_size, cudaMemcpyDeviceToHost);

        // === MPI EXCHANGE ===
        // Send left edge to left neighbor, receive from left neighbor
        if (left_rank >= 0) {
            MPI_Sendrecv(h_send_buf_L, N, MPI_DOUBLE, left_rank, 0,
                         h_recv_buf_L, N, MPI_DOUBLE, left_rank, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Send right edge to right neighbor, receive from right neighbor
        if (right_rank >= 0) {
            MPI_Sendrecv(h_send_buf_R, N, MPI_DOUBLE, right_rank, 1,
                         h_recv_buf_R, N, MPI_DOUBLE, right_rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // === CPU → GPU ===
        if (left_rank >= 0)
            cudaMemcpy(d_recv_buf_L, h_recv_buf_L, ghost_size, cudaMemcpyHostToDevice);
        if (right_rank >= 0)
            cudaMemcpy(d_recv_buf_R, h_recv_buf_R, ghost_size, cudaMemcpyHostToDevice);

        // === UNPACK INTO GRID ===
        if (left_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                d_recv_buf_L, d_old, N, pitch, 0
            );
        }
        if (right_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                d_recv_buf_R, d_old, N, pitch, W + 1
            );
        }

        // === COMPUTE STENCIL ===
        stencil_kernel<<<stencil_blocks, stencil_threads>>>(
            d_old, d_new, N, W, weight
        );

        // Swap
        double* tmp = d_old;
        d_old = d_new;
        d_new = tmp;
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    // Use slowest GPU time
    float max_ms;
    MPI_Reduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Results (MPI Send/Recv, %d GPUs) ===\n", size);
        printf("Grid: %d x %d = %.0f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d\n", N, W);
        printf("Iterations: %d\n", iterations);
        printf("Total time: %.2f ms (slowest GPU)\n", max_ms);
        printf("Per iteration: %.3f ms\n", max_ms / iterations);

        double cells_per_iter = (double)N * TOTAL_W;
        double cells_per_sec = cells_per_iter * iterations / (max_ms / 1000.0);
        printf("Throughput: %.2f billion cells/sec\n", cells_per_sec / 1e9);
    }

    // ------------------------------------------------------------------------
    // Verification: compute L2 norm of local grid
    // ------------------------------------------------------------------------
    double* h_grid = (double*)malloc(grid_size);
    cudaMemcpy(h_grid, d_old, grid_size, cudaMemcpyDeviceToHost);

    double local_norm_sq = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 1; j <= W; j++) {  // skip ghost columns
            double val = h_grid[i * pitch + j];
            local_norm_sq += val * val;
        }
    }
    free(h_grid);

    printf("[Rank %d] Local L2 norm: %.10f\n", rank, sqrt(local_norm_sq));

    double global_norm_sq;
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global L2 norm: %.10f\n", sqrt(global_norm_sq));
    }

    // Cleanup
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_send_buf_L);
    cudaFree(d_send_buf_R);
    cudaFree(d_recv_buf_L);
    cudaFree(d_recv_buf_R);
    free(h_send_buf_L);
    free(h_send_buf_R);
    free(h_recv_buf_L);
    free(h_recv_buf_R);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    MPI_Finalize();

    if (rank == 0) printf("Done.\n");
    return 0;
}
