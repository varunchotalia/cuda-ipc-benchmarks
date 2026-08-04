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

static int local_rank_for_gpu(MPI_Comm comm, int world_rank)
{
    const char* env_names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID"
    };
    for (size_t i = 0; i < sizeof(env_names) / sizeof(env_names[0]); ++i) {
        const char* val = getenv(env_names[i]);
        if (val && *val) {
            char* end = NULL;
            long parsed = strtol(val, &end, 10);
            if (end != val && parsed >= 0) return (int)parsed;
        }
    }

    MPI_Comm local_comm;
    int local_rank = world_rank;
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                            &local_comm) == MPI_SUCCESS) {
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }
    return local_rank;
}

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
    // Lifecycle phase timing (always on; costs one MPI_Wtime per boundary).
    // Reported as PHASE_* lines so a harness can parse them. See
    // results/lulesh_results.csv-style provenance in results/.
    const double t_app_start = MPI_Wtime();
    double t_win_alloc = 0.0, t_win_query = 0.0, t_win_free = 0.0;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // GPU selection
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int local_rank = local_rank_for_gpu(MPI_COMM_WORLD, rank);
    int dev = local_rank % num_devices;
    cudaSetDevice(dev);
    printf("[Rank %d] Using GPU %d\n", rank, dev);

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
    // recv buffers are window allocations: the interposer owns them, which
    // lets it use CUDA IPC on-node or fabric handles across NVLink nodes
    MPI_Win win_recv_L, win_recv_R;
    {
        MPI_Info ipc_info;
        MPI_Info_create(&ipc_info);
        MPI_Info_set(ipc_info, "cuda_ipc", "1");
        MPI_Barrier(MPI_COMM_WORLD);          // align ranks before timing
        const double _t0_alloc = MPI_Wtime();
        MPI_Win_allocate(ghost_size, 1, ipc_info, MPI_COMM_WORLD,
                         &d_ghost_recv_L, &win_recv_L);
        MPI_Win_allocate(ghost_size, 1, ipc_info, MPI_COMM_WORLD,
                         &d_ghost_recv_R, &win_recv_R);
        t_win_alloc = MPI_Wtime() - _t0_alloc;
        MPI_Info_free(&ipc_info);
    }
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
    bool ipc_ok_L = false, ipc_ok_R = false;  // does the IPC put actually work?
    MPI_Barrier(MPI_COMM_WORLD);
    const double _t0_query = MPI_Wtime();

    // Query neighbors: I write my LEFT edge → left neighbor's RIGHT recv buffer.
    // shared_query can fail -- another node, or a GPU-island boundary with no
    // P2P even on the same node -- never assume it succeeded: an unchecked
    // NULL peer pointer written to next iteration is a silent illegal-memory
    // access that can hang or fault the GPU.
    if (left_rank >= 0) {
        MPI_Aint sz; int disp;
        ipc_ok_L = (MPI_Win_shared_query(win_recv_R, left_rank, &sz, &disp,
                                         &peer_recv_L) == MPI_SUCCESS)
                   && peer_recv_L != NULL;
        if (!ipc_ok_L) peer_recv_L = NULL;
    }
    // I write my RIGHT edge → right neighbor's LEFT recv buffer
    if (right_rank >= 0) {
        MPI_Aint sz; int disp;
        ipc_ok_R = (MPI_Win_shared_query(win_recv_L, right_rank, &sz, &disp,
                                         &peer_recv_R) == MPI_SUCCESS)
                   && peer_recv_R != NULL;
        if (!ipc_ok_R) peer_recv_R = NULL;
    }
    if ((left_rank >= 0 && !ipc_ok_L) || (right_rank >= 0 && !ipc_ok_R)) {
        printf("[Rank %d] not IPC-reachable: left=%s right=%s -- using MPI "
               "send/recv fallback\n", rank,
               (left_rank >= 0 && !ipc_ok_L) ? "fallback" : "n/a",
               (right_rank >= 0 && !ipc_ok_R) ? "fallback" : "n/a");
    }

    t_win_query = MPI_Wtime() - _t0_query;
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

    // Untimed warmup iterations (STENCIL_WARMUP, default 0). Kept identical to
    // the staged and GPU-aware variants so all three can be timed on equal
    // terms; default 0 preserves the original timing behavior. Note this
    // variant's IPC handle exchange already happens before the loop, so a
    // warmup changes little here -- that asymmetry is the point of measuring it.
    int warmup = 0;
    {
        const char* w = getenv("STENCIL_WARMUP");
        if (w && *w) {
            char* end = NULL;
            long v = strtol(w, &end, 10);
            if (end != w && v > 0) warmup = (int)v;
        }
    }
    if (rank == 0) printf("Warmup iterations (untimed): %d\n", warmup);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int iter = 0; iter < warmup + iterations; iter++) {

        // Start the clock only once the warmup iterations are done.
        if (iter == warmup) {
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            cudaEventRecord(ev_start);
        }

        // === GHOST EXCHANGE ===
        // tag 0 = data flowing rightward (X's right edge -> X+1's left ghost)
        // tag 1 = data flowing leftward  (X's left edge  -> X-1's right ghost)
        MPI_Request greqs[4]; int ngreq = 0;

        // Post fallback receives before anyone might send
        if (left_rank >= 0 && !ipc_ok_L) {
            MPI_Irecv(d_ghost_recv_L, N, MPI_DOUBLE, left_rank, 0,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (right_rank >= 0 && !ipc_ok_R) {
            MPI_Irecv(d_ghost_recv_R, N, MPI_DOUBLE, right_rank, 1,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }

        // Pack and send LEFT edge (col 1) to left neighbor
        if (left_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_L, N, pitch, 1
            );
            if (ipc_ok_L) {
                cudaMemcpyAsync(peer_recv_L, d_ghost_send_L, ghost_size,
                                cudaMemcpyDeviceToDevice);
            }
        }

        // Pack and send RIGHT edge (col W) to right neighbor
        if (right_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_R, N, pitch, W
            );
            if (ipc_ok_R) {
                cudaMemcpyAsync(peer_recv_R, d_ghost_send_R, ghost_size,
                                cudaMemcpyDeviceToDevice);
            }
        }

        cudaDeviceSynchronize();  // pack kernels + any IPC D2D copies done

        if (left_rank >= 0 && !ipc_ok_L) {
            MPI_Isend(d_ghost_send_L, N, MPI_DOUBLE, left_rank, 1,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (right_rank >= 0 && !ipc_ok_R) {
            MPI_Isend(d_ghost_send_R, N, MPI_DOUBLE, right_rank, 0,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (ngreq > 0) MPI_Waitall(ngreq, greqs, MPI_STATUSES_IGNORE);

        // Completion handshake. A rank writes straight into its neighbour's
        // receive buffer, so the neighbour must not unpack until that write has
        // landed -- and unlike MPI_Sendrecv there is no message to signal it.
        // This is a genuine cost of the one-sided path that the GPU-aware and
        // staged variants do not pay (their blocking Sendrecv is
        // self-synchronising). It must therefore be disclosed as a
        // synchronisation asymmetry when comparing against them.
        //
        // DO NOT replace this with a neighbour-only handshake. That was tried
        // (job 60200) and it is RACY: exchanging a token with each neighbour
        // proves the neighbour finished WRITING, but not that it finished
        // UNPACKING. Without a global sync, ranks drift cumulatively, and a
        // rank can begin iteration i+1's write into a neighbour's receive
        // buffer while that neighbour is still reading iteration i out of it.
        // It failed exactly where drift is largest and compute slack smallest:
        // at 8 ranks, 1024^2 gave L2 3.7550293292 and 4096^2 gave 5.0563202266
        // against the correct 5.1449605829, while every barrier run matched.
        //
        // It is also not faster. Same job, IPC time, barrier vs neighbour-token:
        // 4 ranks 16384^2 36.89 vs 37.04 ms; 32768^2 135.29 vs 135.45 ms. The
        // barrier is marginally CHEAPER -- OpenMPI's intra-node barrier beats
        // four Isend/Irecv plus a Waitall. Measured cost of this barrier at the
        // sizes the paper cites is <=0.4%.
        //
        // A correct neighbour-only scheme needs double-buffered receive buffers
        // or a second post-unpack handshake: more complexity for no gain.
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
    MPI_Barrier(MPI_COMM_WORLD);
    const double _t0_free = MPI_Wtime();
    MPI_Win_free(&win_recv_L);
    MPI_Win_free(&win_recv_R);
    t_win_free = MPI_Wtime() - _t0_free;
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_ghost_send_L);
    cudaFree(d_ghost_send_R);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    {
        const double t_app = MPI_Wtime() - t_app_start;
        double loc[4] = { t_win_alloc, t_win_query, t_win_free, t_app };
        double mx[4];
        MPI_Reduce(loc, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("PHASE_WIN_ALLOCATE_ms %.4f\n", mx[0] * 1000.0);
            printf("PHASE_PEER_QUERY_ms   %.4f\n", mx[1] * 1000.0);
            printf("PHASE_WIN_FREE_ms     %.4f\n", mx[2] * 1000.0);
            printf("PHASE_APP_TOTAL_ms    %.4f\n", mx[3] * 1000.0);
        }
    }

    MPI_Finalize();

    if (rank == 0) printf("Done.\n");
    return 0;
}
