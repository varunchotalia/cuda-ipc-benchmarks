// stencil_large_nvshmem.cu
//
// Multi-GPU stencil with NVSHMEM ghost exchange
// Drop-in replacement for the IPC version: same kernels, same layout,
// same per-iteration semantics. Only the ghost-exchange machinery changes.
//
// Key differences vs. the IPC version:
//   - Recv buffers live in the NVSHMEM symmetric heap (nvshmem_malloc).
//   - Pack -> put(+signal) -> wait/barrier -> unpack (all on a stream).
//   - No MPI windows, no manual IPC handle exchange.
//
// STENCIL_SYNC selects the completion handshake, mirroring stencil_ipc.cu so the
// two one-sided variants stay synchronisation-matched:
//   neighbor (default) -- put_signal to each neighbour, signal_wait_until on the
//                         two incoming signals. Recv buffers are double-buffered
//                         (slot iter%2); see the handshake in the main loop.
//   barrier            -- plain put plus nvshmemx_barrier_all_on_stream.
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
#include <cstring>
#include <cstdint>
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
    int local_rank = local_rank_for_gpu(MPI_COMM_WORLD, mype);
    int dev = local_rank % num_devices;
    cudaSetDevice(dev);
    printf("[PE %d] Using GPU %d\n", mype, dev);

    // ------------------------------------------------------------------------
    // LARGE GRID  (unchanged)
    // ------------------------------------------------------------------------
    const int N = 16384;
    const int TOTAL_W = 16384;
    const int W = TOTAL_W / npes;
    const int pitch = W + 2;

    size_t grid_size  = (size_t)N * pitch * sizeof(double);
    size_t ghost_size = (size_t)N * sizeof(double);

    // Completion handshake, same env var and same default as stencil_ipc.cu:
    // barrier by default, because the put_signal/wait path below has never been
    // run. See the long comment at the corresponding line in stencil_ipc.cu for
    // why -- the two variants must stay synchronisation-matched or the IPC-vs-
    // NVSHMEM comparison stops being apples-to-apples.
    bool neighbor_sync = false;
    {
        const char* s = getenv("STENCIL_SYNC");
        if (s && *s) {
            if (strcmp(s, "barrier") == 0)       neighbor_sync = false;
            else if (strcmp(s, "neighbor") == 0 ||
                     strcmp(s, "neighbour") == 0) neighbor_sync = true;
            else if (mype == 0)
                printf("WARNING: unknown STENCIL_SYNC=\"%s\", using neighbor\n", s);
        }
    }

    // Receive slots, as in stencil_ipc.cu: a PE puts into slot iter%2 of its
    // neighbour's buffer. Needed for correctness under the neighbour handshake;
    // allocated in barrier mode too (which only touches slot 0) so the symmetric
    // heap footprint is identical and an A/B measures only the handshake.
    const int RECV_SLOTS = 2;
    const size_t recv_buf_size = ghost_size * RECV_SLOTS;

    if (mype == 0) {
        printf("PEs: %d\n", npes);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d + 2 ghost columns\n", N, W);
        printf("Grid memory: %.2f MB per GPU\n", grid_size / 1e6);
        printf("Ghost window: %.2f KB (contiguous, %d slots of %.2f KB)\n",
               recv_buf_size / 1e3, RECV_SLOTS, ghost_size / 1e3);
        printf("Completion handshake: %s\n",
               neighbor_sync ? "neighbour-only put_signal/wait"
                             : "global nvshmem barrier");
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
    double *d_ghost_recv_L = (double*)nvshmem_malloc(recv_buf_size);
    double *d_ghost_recv_R = (double*)nvshmem_malloc(recv_buf_size);

    // Signal words for the neighbour handshake. These must also live in the
    // symmetric heap -- a put_signal updates the signal on the TARGET PE, so the
    // address has to be symmetric like the data buffer. Two per PE:
    //   d_sig[SIG_L] is raised by my LEFT neighbour when it has filled my L slot
    //   d_sig[SIG_R] is raised by my RIGHT neighbour when it has filled my R slot
    // Always allocated (16 bytes) so barrier mode has the same heap layout.
    enum { SIG_L = 0, SIG_R = 1, NUM_SIG = 2 };
    uint64_t *d_sig = (uint64_t*)nvshmem_malloc(NUM_SIG * sizeof(uint64_t));

    if (!d_ghost_recv_L || !d_ghost_recv_R || !d_sig) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *d_ghost_send_L, *d_ghost_send_R;
    cudaMalloc(&d_ghost_send_L, ghost_size);
    cudaMalloc(&d_ghost_send_R, ghost_size);

    // Initialize
    cudaMemset(d_old, 0, grid_size);
    cudaMemset(d_new, 0, grid_size);
    cudaMemset(d_ghost_recv_L, 0, recv_buf_size);
    cudaMemset(d_ghost_recv_R, 0, recv_buf_size);
    // Signals start at 0 and only ever increase; iteration i waits for i+1, so
    // they are never reset inside the loop. The nvshmem_barrier_all before the
    // loop is what makes this zeroing visible to every peer before the first put.
    cudaMemset(d_sig, 0, NUM_SIG * sizeof(uint64_t));

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

    // Untimed warmup iterations (STENCIL_WARMUP, default 0), matching
    // stencil_ipc.cu / stencil_mpi.cu / stencil_gpu_mpi.cu exactly: same env
    // var, same default, same total iteration count (warmup + iterations), so
    // the four variants stay comparable and their L2 norms stay identical at a
    // given warmup. Without this the NVSHMEM column could not be run under the
    // same warmup as the others; see results/stencil_results.txt.
    int warmup = 0;
    {
        const char* w = getenv("STENCIL_WARMUP");
        if (w && *w) {
            char* end = NULL;
            long v = strtol(w, &end, 10);
            if (end != w && v > 0) warmup = (int)v;
        }
    }
    if (mype == 0) printf("Warmup iterations (untimed): %d\n", warmup);

    // ------------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------------
    nvshmem_barrier_all();   // make sure all PEs are ready

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int iter = 0; iter < warmup + iterations; iter++) {

        // Start the clock only once the warmup iterations are done.
        if (iter == warmup) {
            cudaStreamSynchronize(stream);
            nvshmem_barrier_all();
            cudaEventRecord(ev_start, stream);
        }

        // === GHOST EXCHANGE ===

        // Receive slot for this iteration. Every PE runs the same
        // warmup+iterations count, so neighbours always agree on the parity.
        // Barrier mode pins slot 0 and behaves exactly as it did before.
        // The symmetric heap is laid out identically on every PE, so offsetting
        // my own pointer names the same slot inside the peer's buffer.
        const int slot = neighbor_sync ? (iter & 1) : 0;
        double* slot_L = d_ghost_recv_L + (size_t)slot * N;
        double* slot_R = d_ghost_recv_R + (size_t)slot * N;

        // Signal value for this iteration: monotonically increasing, so a waiter
        // uses CMP_GE and a neighbour that has run ahead still satisfies it.
        const uint64_t sigval = (uint64_t)iter + 1;

        // Pack my LEFT edge (col 1) and put it into left neighbor's R recv slot
        if (left_pe >= 0) {
            pack_edge<<<copy_blocks, copy_threads, 0, stream>>>(
                d_old, d_ghost_send_L, N, pitch, 1
            );
            if (neighbor_sync) {
                // put_signal raises left_pe's SIG_R only after this payload has
                // been delivered to it -- that ordering guarantee is the whole
                // reason to use put_signal rather than put followed by a
                // separate signal op.
                nvshmemx_double_put_signal_on_stream(
                    slot_R,           // symmetric: left_pe's R slot
                    d_ghost_send_L,
                    (size_t)N,
                    &d_sig[SIG_R],    // symmetric: left_pe's SIG_R
                    sigval,
                    NVSHMEM_SIGNAL_SET,
                    left_pe,
                    stream
                );
            } else {
                nvshmemx_double_put_on_stream(
                    slot_R, d_ghost_send_L, (size_t)N, left_pe, stream
                );
            }
        }

        // Pack my RIGHT edge (col W) and put it into right neighbor's L recv slot
        if (right_pe >= 0) {
            pack_edge<<<copy_blocks, copy_threads, 0, stream>>>(
                d_old, d_ghost_send_R, N, pitch, W
            );
            if (neighbor_sync) {
                nvshmemx_double_put_signal_on_stream(
                    slot_L,           // symmetric: right_pe's L slot
                    d_ghost_send_R,
                    (size_t)N,
                    &d_sig[SIG_L],    // symmetric: right_pe's SIG_L
                    sigval,
                    NVSHMEM_SIGNAL_SET,
                    right_pe,
                    stream
                );
            } else {
                nvshmemx_double_put_on_stream(
                    slot_L, d_ghost_send_R, (size_t)N, right_pe, stream
                );
            }
        }

        // Completion handshake, on the stream either way.
        //
        // A five-point stencil depends only on its two neighbours, so waiting on
        // the two incoming signals is the minimum sufficient sync; a global
        // barrier synchronises the whole PE set and its cost grows with PE count.
        // This mirrors the zero-byte token exchange in stencil_ipc.cu, and the
        // two variants must stay matched: if only one of them dropped its
        // barrier, an IPC-vs-NVSHMEM comparison would be measuring which variant
        // got the optimisation rather than which transport is faster.
        //
        // THE SIGNAL ALONE IS NOT ENOUGH -- it is why the slots alternate. A
        // signal proves the neighbour finished WRITING my slot; it says nothing
        // about whether the neighbour finished UNPACKING the slot I am about to
        // write. The single-buffered form of this scheme was already shown to be
        // racy on the IPC side (commit 8d913ba, job 60200: np=8 1024^2 gave L2
        // 3.7550293292 against the correct 5.1449605829).
        //
        // Two slots are sufficient, with stream order supplying here what the
        // device-wide cudaDeviceSynchronize supplies in the IPC version. My
        // iteration i+2 put reuses slot i%2; it is enqueued after my iteration
        // i+1 wait cleared, which required the neighbour's signal value i+2,
        // which the neighbour raised from its own iteration i+1 put -- and on the
        // neighbour's single in-order stream that put executes only after its
        // iteration i unpack of slot i%2 has completed. So the slot I reuse has
        // always already been read.
        if (neighbor_sync) {
            if (left_pe >= 0) {
                nvshmemx_signal_wait_until_on_stream(
                    &d_sig[SIG_L], NVSHMEM_CMP_GE, sigval, stream
                );
            }
            if (right_pe >= 0) {
                nvshmemx_signal_wait_until_on_stream(
                    &d_sig[SIG_R], NVSHMEM_CMP_GE, sigval, stream
                );
            }
        } else {
            // (Equivalent to the cudaDeviceSynchronize + MPI_Barrier pair in the
            // IPC version's barrier mode.)
            nvshmemx_barrier_all_on_stream(stream);
        }

        // Unpack received ghosts into my grid
        if (left_pe >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads, 0, stream>>>(
                slot_L, d_old, N, pitch, 0
            );
        }
        if (right_pe >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads, 0, stream>>>(
                slot_R, d_old, N, pitch, W + 1
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
    nvshmem_free(d_sig);
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
