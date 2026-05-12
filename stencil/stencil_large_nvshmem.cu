// stencil_large_nvshmem.cu
//
// Multi-GPU stencil with NVSHMEM ghost exchange
// Drop-in replacement for the IPC version: same kernels, same layout,
// same per-iteration semantics. Only the ghost-exchange machinery changes.
//
// Key differences vs. the IPC version:
//   - Recv buffers live in the NVSHMEM symmetric heap (nvshmem_malloc).
//   - Pack -> nvshmemx_double_put_on_stream -> barrier -> unpack (all on a stream).
//   - No MPI windows, no manual IPC handle exchange.
//
// Compile (adjust paths to your NVSHMEM install):
//   nvcc -ccbin mpicxx -rdc=true -O3 -arch=sm_90 \
//        -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
//        -o stencil_large_nvshmem stencil_large_nvshmem.cu \
//        -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda -lmpi
//
// Run:
//   mpirun -np 4 ./stencil_large_nvshmem

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// STENCIL KERNEL  (unchanged)
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
// PACK / UNPACK  (unchanged)
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
    // ------------------------------------------------------------------------
    // Bootstrap MPI, then init NVSHMEM from the MPI communicator
    // ------------------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // Pin one GPU per PE on this node
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(mype % num_devices);
    printf("[PE %d] Using GPU %d\n", mype, mype % num_devices);

    // ------------------------------------------------------------------------
    // LARGE GRID  (unchanged)
    // ------------------------------------------------------------------------
    const int N = 16384;
    const int TOTAL_W = 16384;
    const int W = TOTAL_W / npes;
    const int pitch = W + 2;

    size_t grid_size  = (size_t)N * pitch * sizeof(double);
    size_t ghost_size = (size_t)N * sizeof(double);

    if (mype == 0) {
        printf("PEs: %d\n", npes);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d + 2 ghost columns\n", N, W);
        printf("Grid memory: %.2f MB per GPU\n", grid_size / 1e6);
        printf("Ghost window: %.2f KB (contiguous)\n", ghost_size / 1e3);
    }

    // ------------------------------------------------------------------------
    // Allocate grids (private device memory — they don't need to be symmetric)
    // ------------------------------------------------------------------------
    double *d_old, *d_new;
    cudaMalloc(&d_old, grid_size);
    cudaMalloc(&d_new, grid_size);

    // ------------------------------------------------------------------------
    // Ghost buffers
    //   recv buffers MUST be in the symmetric heap (peers put into them)
    //   send buffers can be private — they are only put-source
    // ------------------------------------------------------------------------
    double *d_ghost_recv_L = (double*)nvshmem_malloc(ghost_size);
    double *d_ghost_recv_R = (double*)nvshmem_malloc(ghost_size);
    if (!d_ghost_recv_L || !d_ghost_recv_R) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *d_ghost_send_L, *d_ghost_send_R;
    cudaMalloc(&d_ghost_send_L, ghost_size);
    cudaMalloc(&d_ghost_send_R, ghost_size);

    // Initialize
    cudaMemset(d_old, 0, grid_size);
    cudaMemset(d_new, 0, grid_size);
    cudaMemset(d_ghost_recv_L, 0, ghost_size);
    cudaMemset(d_ghost_recv_R, 0, ghost_size);

    // Heat source in center of global grid
    int global_center_col = TOTAL_W / 2;
    int my_first_global_col = mype * W;
    int my_last_global_col  = my_first_global_col + W - 1;

    if (global_center_col >= my_first_global_col &&
        global_center_col <= my_last_global_col) {
        int local_col = global_center_col - my_first_global_col + 1;
        int center_i = N / 2;
        double init_val = 100.0;
        cudaMemcpy(&d_old[center_i * pitch + local_col], &init_val,
                   sizeof(double), cudaMemcpyHostToDevice);
        printf("[PE %d] Heat source at global col %d (local col %d)\n",
               mype, global_center_col, local_col);
    }

    // ------------------------------------------------------------------------
    // Neighbors
    // ------------------------------------------------------------------------
    int left_pe  = (mype > 0)        ? mype - 1 : -1;
    int right_pe = (mype < npes - 1) ? mype + 1 : -1;
    printf("[PE %d] Neighbors: left=%d, right=%d\n", mype, left_pe, right_pe);

    // ------------------------------------------------------------------------
    // Stream for on-stream NVSHMEM ops
    // ------------------------------------------------------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ------------------------------------------------------------------------
    // Kernel configs  (unchanged)
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
    nvshmem_barrier_all();   // make sure all PEs are ready

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);

    for (int iter = 0; iter < iterations; iter++) {

        // === GHOST EXCHANGE ===

        // Pack my LEFT edge (col 1) and put it into left neighbor's R recv buffer
        if (left_pe >= 0) {
            pack_edge<<<copy_blocks, copy_threads, 0, stream>>>(
                d_old, d_ghost_send_L, N, pitch, 1
            );
            nvshmemx_double_put_on_stream(
                d_ghost_recv_R,   // symmetric: written into left_pe's R buffer
                d_ghost_send_L,
                (size_t)N,
                left_pe,
                stream
            );
        }

        // Pack my RIGHT edge (col W) and put it into right neighbor's L recv buffer
        if (right_pe >= 0) {
            pack_edge<<<copy_blocks, copy_threads, 0, stream>>>(
                d_old, d_ghost_send_R, N, pitch, W
            );
            nvshmemx_double_put_on_stream(
                d_ghost_recv_L,   // symmetric: written into right_pe's L buffer
                d_ghost_send_R,
                (size_t)N,
                right_pe,
                stream
            );
        }

        // Global completion + ordering on the stream.
        // (Equivalent to the cudaDeviceSynchronize + MPI_Barrier pair in the IPC version.)
        nvshmemx_barrier_all_on_stream(stream);

        // Unpack received ghosts into my grid
        if (left_pe >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads, 0, stream>>>(
                d_ghost_recv_L, d_old, N, pitch, 0
            );
        }
        if (right_pe >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads, 0, stream>>>(
                d_ghost_recv_R, d_old, N, pitch, W + 1
            );
        }

        // === COMPUTE STENCIL ===
        stencil_kernel<<<stencil_blocks, stencil_threads, 0, stream>>>(
            d_old, d_new, N, W, weight
        );

        // Swap
        double* tmp = d_old;
        d_old = d_new;
        d_new = tmp;
    }

    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);

    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    // ------------------------------------------------------------------------
    // Results — slowest GPU time
    // ------------------------------------------------------------------------
    float max_ms;
    MPI_Reduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mype == 0) {
        printf("\n=== Results (NVSHMEM, %d GPUs) ===\n", npes);
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
    // Verification: L2 norm
    // ------------------------------------------------------------------------
    double* h_grid = (double*)malloc(grid_size);
    cudaMemcpy(h_grid, d_old, grid_size, cudaMemcpyDeviceToHost);

    double local_norm_sq = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 1; j <= W; j++) {
            double val = h_grid[i * pitch + j];
            local_norm_sq += val * val;
        }
    }
    free(h_grid);

    printf("[PE %d] Local L2 norm: %.10f\n", mype, sqrt(local_norm_sq));

    double global_norm_sq;
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mype == 0) {
        printf("Global L2 norm: %.10f\n", sqrt(global_norm_sq));
    }

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    nvshmem_barrier_all();   // ensure no PE is still touching peer memory

    nvshmem_free(d_ghost_recv_L);
    nvshmem_free(d_ghost_recv_R);
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_ghost_send_L);
    cudaFree(d_ghost_send_R);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);

    nvshmem_finalize();
    MPI_Finalize();

    if (mype == 0) printf("Done.\n");
    return 0;
}
