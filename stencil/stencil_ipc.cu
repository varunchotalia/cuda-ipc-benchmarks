// stencil_large_contiguous.cu
//
// Multi-GPU stencil with CUDA IPC ghost exchange
// Supports 1 to 16 GPUs (any count)
//
// Each GPU owns a vertical strip of the global grid.
// Ghost columns are exchanged with left/right neighbors via IPC.
//
// Memory layout per GPU:
//   [ghost_L] [data col 1 ... data col W] [ghost_R]
//   ghost_L = copy of left neighbor's right edge  (or boundary=0)
//   ghost_R = copy of right neighbor's left edge   (or boundary=0)
//
// Compile: nvcc -o stencil_large_ipc stencil_large_contiguous.cu -lmpi
// Run:     mpirun -np 4 ./stencil_large_ipc

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
                    int N,
                    int W,
                    double weight)
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
void pack_edge(const double* __restrict__ grid,
               double* __restrict__ buffer,
               int N, int pitch, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    buffer[i] = grid[i * pitch + col];
}

__global__
void unpack_ghost(const double* __restrict__ buffer,
                  double* __restrict__ grid,
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

    // GPU selection
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    printf("[Rank %d] Using GPU %d\n", rank, rank % num_devices);

    // ------------------------------------------------------------------------
    // LARGE GRID
    // ------------------------------------------------------------------------
    const int N = 16384;                    // rows
    const int TOTAL_W = 16384;              // total columns
    const int W = TOTAL_W / size;           // columns per GPU
    const int pitch = W + 2;                // +2 for ghost columns

    size_t grid_size  = (size_t)N * pitch * sizeof(double);
    size_t ghost_size = (size_t)N * sizeof(double);

    if (rank == 0) {
        printf("GPUs: %d\n", size);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d + 2 ghost columns\n", N, W);
        printf("Grid memory: %.2f MB per GPU\n", grid_size / 1e6);
        printf("Ghost window: %.2f KB (contiguous)\n", ghost_size / 1e3);
    }

    // ------------------------------------------------------------------------
    // Allocate grids
    // ------------------------------------------------------------------------
    double *d_old, *d_new;
    cudaMalloc(&d_old, grid_size);
    cudaMalloc(&d_new, grid_size);

    // ------------------------------------------------------------------------
    // Ghost buffers: one for LEFT neighbor, one for RIGHT neighbor
    // Each neighbor writes into our receive buffer via IPC
    // ------------------------------------------------------------------------
    double *d_ghost_recv_L, *d_ghost_recv_R;   // peers write here
    double *d_ghost_send_L, *d_ghost_send_R;   // I pack my edges here
    cudaMalloc(&d_ghost_recv_L, ghost_size);
    cudaMalloc(&d_ghost_recv_R, ghost_size);
    cudaMalloc(&d_ghost_send_L, ghost_size);
    cudaMalloc(&d_ghost_send_R, ghost_size);

    // Initialize
    cudaMemset(d_old, 0, grid_size);
    cudaMemset(d_new, 0, grid_size);
    cudaMemset(d_ghost_recv_L, 0, ghost_size);
    cudaMemset(d_ghost_recv_R, 0, ghost_size);

    // Heat source in center of global grid
    int global_center_col = TOTAL_W / 2;
    int my_first_global_col = rank * W;
    int my_last_global_col  = my_first_global_col + W - 1;

    if (global_center_col >= my_first_global_col &&
        global_center_col <= my_last_global_col) {
        int local_col = global_center_col - my_first_global_col + 1; // +1 for ghost
        int center_i = N / 2;
        double init_val = 100.0;
        cudaMemcpy(&d_old[center_i * pitch + local_col], &init_val,
                   sizeof(double), cudaMemcpyHostToDevice);
        printf("[Rank %d] Heat source at global col %d (local col %d)\n",
               rank, global_center_col, local_col);
    }

    // ------------------------------------------------------------------------
    // Neighbors
    // ------------------------------------------------------------------------
    int left_rank  = (rank > 0)        ? rank - 1 : -1;  // -1 = no neighbor
    int right_rank = (rank < size - 1) ? rank + 1 : -1;

    printf("[Rank %d] Neighbors: left=%d, right=%d\n", rank, left_rank, right_rank);

    // ------------------------------------------------------------------------
    // Setup IPC handles with ALL ranks at once (avoids deadlock)
    //
    // Each GPU exposes its LEFT and RIGHT receive buffers.
    // We gather all handles, then each rank opens only its neighbors'.
    // ------------------------------------------------------------------------

    double *peer_recv_L = NULL;  // left neighbor's RIGHT recv buffer
    double *peer_recv_R = NULL;  // right neighbor's LEFT recv buffer

    // Create MPI windows over each recv buffer (intercepted by mpiwrap_ipc)
    size_t ghost_bytes = (size_t)N * sizeof(double);
    MPI_Win win_recv_L, win_recv_R;
    MPI_Win_create(d_ghost_recv_L, ghost_bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_recv_L);
    MPI_Win_create(d_ghost_recv_R, ghost_bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_recv_R);

    // Query neighbors: I write my LEFT edge → left neighbor's RIGHT recv buffer
    if (left_rank >= 0) {
        MPI_Aint sz; int disp;
        MPI_Win_shared_query(win_recv_R, left_rank, &sz, &disp, &peer_recv_L);
    }
    // I write my RIGHT edge → right neighbor's LEFT recv buffer
    if (right_rank >= 0) {
        MPI_Aint sz; int disp;
        MPI_Win_shared_query(win_recv_L, right_rank, &sz, &disp, &peer_recv_R);
    }

    printf("[Rank %d] IPC setup complete\n", rank);

    // ------------------------------------------------------------------------
    // Kernel configs
    // ------------------------------------------------------------------------
    dim3 stencil_threads(16, 16);
    dim3 stencil_blocks((W + stencil_threads.x - 1) / stencil_threads.x,
                        (N + stencil_threads.y - 1) / stencil_threads.y);

    int copy_threads = 256;
    int copy_blocks = (N + copy_threads - 1) / copy_threads;

    double weight = 0.25;
    int iterations = 100;

    // ------------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    for (int iter = 0; iter < iterations; iter++) {

        // === GHOST EXCHANGE ===

        // Pack and send LEFT edge (col 1) to left neighbor
        if (left_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_L, N, pitch, 1
            );
            cudaMemcpyAsync(peer_recv_L, d_ghost_send_L, ghost_size,
                            cudaMemcpyDeviceToDevice);
        }

        // Pack and send RIGHT edge (col W) to right neighbor
        if (right_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_R, N, pitch, W
            );
            cudaMemcpyAsync(peer_recv_R, d_ghost_send_R, ghost_size,
                            cudaMemcpyDeviceToDevice);
        }

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        // Unpack received ghosts into my grid
        if (left_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                d_ghost_recv_L, d_old, N, pitch, 0       // ghost col 0
            );
        }
        if (right_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                d_ghost_recv_R, d_old, N, pitch, W + 1   // ghost col W+1
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

    // ------------------------------------------------------------------------
    // Results - use slowest GPU time
    // ------------------------------------------------------------------------
    float max_ms;
    MPI_Reduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Results (IPC, %d GPUs) ===\n", size);
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

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    MPI_Win_free(&win_recv_L);
    MPI_Win_free(&win_recv_R);
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_ghost_send_L);
    cudaFree(d_ghost_send_R);
    cudaFree(d_ghost_recv_L);
    cudaFree(d_ghost_recv_R);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    MPI_Finalize();

    if (rank == 0) printf("Done.\n");
    return 0;
}
