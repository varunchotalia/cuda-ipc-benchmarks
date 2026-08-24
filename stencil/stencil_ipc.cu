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
// The receive buffers a peer writes into are double-buffered (slot iter%2).
// STENCIL_SYNC selects the completion handshake: "barrier" (default) uses
// MPI_Barrier, "neighbor" trades a zero-byte MPI_Isend token with each
// neighbour, "ssend" trades the same token with MPI_Ssend so the send does not
// return until the neighbour posts its receive. See the long comment at the
// handshake in the main loop.
//
// Compile: nvcc -o stencil_large_ipc stencil_large_contiguous.cu -lmpi
// Run:     mpirun -np 4 ./stencil_large_ipc

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    // Completion handshake for the one-sided path. Default is the global
    // MPI_Barrier; STENCIL_SYNC=neighbor selects the neighbour-only token
    // exchange.
    //
    // The barrier is the default because every stencil number in the paper
    // (job 59853) is a barrier run; keep it until the paper's numbers are
    // re-measured, not because the token path is unsafe.
    //
    // HISTORY. The FIRST token attempt was reverted in 445c39a as RACY: at 8
    // ranks it produced L2 3.7550293292 (1024^2) and 5.0563202266 (4096^2)
    // against the correct 5.1449605829, while every barrier run matched. A
    // token proves the neighbour finished writing, not unpacking. The
    // double-buffered slots below (slot = iter & 1) are what this second
    // attempt adds to close that race -- iteration i+1 writes the other slot,
    // so it cannot clobber iteration i while a neighbour is still reading it.
    //
    // VALIDATED 2026-08-13, job 63329 (h200x8-03), which is the "matching L2 at
    // 8 ranks" log this comment used to demand before trusting the token path:
    //   - All 144 RESULT rows -- np 2/4/8 x 6 sizes x both handshakes -- report
    //     l2=5.1449605829. Zero mismatches between the barrier and neighbour
    //     arms at any np/size. The race is closed.
    //   - The "NOT faster" note was measured at 4 ranks, where it still holds
    //     (16384^2: 36.86 barrier vs 36.90 token). It does NOT generalise: at 8
    //     ranks the token wins by 5.2% at 2048^2 and 4.6% at 4096^2, where
    //     barrier latency is a real fraction of the step. Elsewhere it is a
    //     wash. Two slots suffice: a neighbour cannot reach iteration i+2's
    //     write without first receiving this rank's iteration-i+1 token, which
    //     is only sent after the cudaDeviceSynchronize that retires iteration
    //     i's unpack.
    // ssend_sync is the advisor's original proposal, added 2026-08-23 so it can
    // be measured rather than argued about: swap the zero-byte MPI_Isend token
    // for a synchronous MPI_Ssend. Ssend does not return until the receiver has
    // posted the matching receive, so it proves the neighbour has REACHED this
    // iteration, which the Isend token does not. Everything else -- the double
    // buffering, the IPC writes, the tags -- is unchanged, so a neighbor-vs-
    // ssend A/B isolates the handshake alone.
    bool neighbor_sync = false, ssend_sync = false;
    {
        const char* s = getenv("STENCIL_SYNC");
        if (s && *s) {
            if (strcmp(s, "barrier") == 0)       neighbor_sync = false;
            else if (strcmp(s, "neighbor") == 0 ||
                     strcmp(s, "neighbour") == 0) neighbor_sync = true;
            else if (strcmp(s, "ssend") == 0)  { neighbor_sync = true;
                                                 ssend_sync    = true; }
            // Do not guess: a typo'd STENCIL_SYNC used to print "using neighbor"
            // while leaving the barrier default in place, so an A/B arm could
            // silently measure the handshake it was not asking for.
            else {
                if (rank == 0)
                    fprintf(stderr, "FATAL: unknown STENCIL_SYNC=\"%s\" "
                            "(expected \"barrier\", \"neighbor\" or "
                            "\"ssend\")\n", s);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }

    // Receive buffers are double-buffered; a rank writes into slot iter%2 of its
    // neighbour's buffer. The neighbour-only handshake NEEDS this for
    // correctness (see the main loop). Both slots are allocated in barrier mode
    // too -- barrier mode only ever touches slot 0, but keeping the allocation
    // byte-identical means a barrier-vs-neighbour A/B isolates the handshake and
    // does not also measure a different setup phase.
    const int RECV_SLOTS = 2;
    const size_t recv_win_size = ghost_size * RECV_SLOTS;

    if (rank == 0) {
        printf("GPUs: %d\n", size);
        printf("Global grid: %d x %d = %.2f million cells\n",
               N, TOTAL_W, (double)N * TOTAL_W / 1e6);
        printf("Per GPU: %d x %d + 2 ghost columns\n", N, W);
        printf("Grid memory: %.2f MB per GPU\n", grid_size / 1e6);
        printf("Ghost window: %.2f KB (contiguous, %d slots of %.2f KB)\n",
               recv_win_size / 1e3, RECV_SLOTS, ghost_size / 1e3);
        printf("Completion handshake: %s\n",
               ssend_sync    ? "neighbour-only Ssend handshake"
             : neighbor_sync ? "neighbour-only token exchange"
                             : "global MPI_Barrier");
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
        MPI_Win_allocate(recv_win_size, 1, ipc_info, MPI_COMM_WORLD,
                         &d_ghost_recv_L, &win_recv_L);
        MPI_Win_allocate(recv_win_size, 1, ipc_info, MPI_COMM_WORLD,
                         &d_ghost_recv_R, &win_recv_R);
        // Send-side buffers belong to this phase too: the comparators time all
        // four of their halo buffers, so timing only the two windows here would
        // understate the wrapper's halo setup and bias the comparison in its
        // favour. The grids (d_old/d_new) stay outside, as in the comparators.
        cudaMalloc(&d_ghost_send_L, ghost_size);
        cudaMalloc(&d_ghost_send_R, ghost_size);
        t_win_alloc = MPI_Wtime() - _t0_alloc;
        MPI_Info_free(&ipc_info);
    }

    // Initialize
    cudaMemset(d_old, 0, grid_size);
    cudaMemset(d_new, 0, grid_size);
    cudaMemset(d_ghost_recv_L, 0, recv_win_size);
    cudaMemset(d_ghost_recv_R, 0, recv_win_size);

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

        // Receive-buffer slot for this iteration. Every rank runs the same
        // warmup+iterations count, so neighbours always agree on the parity.
        // Barrier mode pins slot 0 so it behaves exactly as it did before.
        const int slot = neighbor_sync ? (iter & 1) : 0;
        double* my_recv_L  = d_ghost_recv_L + (size_t)slot * N;
        double* my_recv_R  = d_ghost_recv_R + (size_t)slot * N;
        double* peer_dst_L = ipc_ok_L ? peer_recv_L + (size_t)slot * N : NULL;
        double* peer_dst_R = ipc_ok_R ? peer_recv_R + (size_t)slot * N : NULL;

        // === GHOST EXCHANGE ===
        // tag 0 = data flowing rightward (X's right edge -> X+1's left ghost)
        // tag 1 = data flowing leftward  (X's left edge  -> X-1's right ghost)
        // tag 20 = zero-byte token, "your L slot is filled"  (sent rightward)
        // tag 21 = zero-byte token, "your R slot is filled"  (sent leftward)
        // At most two requests per side (data pair on a fallback side, token
        // pair on an IPC side), so four in flight.
        MPI_Request greqs[4]; int ngreq = 0;

        // Post fallback receives before anyone might send
        if (left_rank >= 0 && !ipc_ok_L) {
            MPI_Irecv(my_recv_L, N, MPI_DOUBLE, left_rank, 0,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (right_rank >= 0 && !ipc_ok_R) {
            MPI_Irecv(my_recv_R, N, MPI_DOUBLE, right_rank, 1,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }

        // Post the token receives up front so a neighbour's token can land while
        // we are still packing, instead of only once we go looking for it. Only
        // IPC sides need one: a fallback side's data Isend/Irecv already carries
        // its own completion signal.
        if (neighbor_sync && !ssend_sync && ipc_ok_L) {
            MPI_Irecv(NULL, 0, MPI_BYTE, left_rank, 20,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (neighbor_sync && !ssend_sync && ipc_ok_R) {
            MPI_Irecv(NULL, 0, MPI_BYTE, right_rank, 21,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }

        // Pack and send LEFT edge (col 1) to left neighbor
        if (left_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_L, N, pitch, 1
            );
            if (ipc_ok_L) {
                cudaMemcpyAsync(peer_dst_L, d_ghost_send_L, ghost_size,
                                cudaMemcpyDeviceToDevice);
            }
        }

        // Pack and send RIGHT edge (col W) to right neighbor
        if (right_rank >= 0) {
            pack_edge<<<copy_blocks, copy_threads>>>(
                d_old, d_ghost_send_R, N, pitch, W
            );
            if (ipc_ok_R) {
                cudaMemcpyAsync(peer_dst_R, d_ghost_send_R, ghost_size,
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

        // The cudaDeviceSynchronize above means my IPC writes have landed, so
        // each neighbour may now be told its slot is filled. My left edge went
        // into the left neighbour's R slot (tag 21); my right edge went into the
        // right neighbour's L slot (tag 20).
        if (neighbor_sync && !ssend_sync && ipc_ok_L) {
            MPI_Isend(NULL, 0, MPI_BYTE, left_rank, 21,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }
        if (neighbor_sync && !ssend_sync && ipc_ok_R) {
            MPI_Isend(NULL, 0, MPI_BYTE, right_rank, 20,
                      MPI_COMM_WORLD, &greqs[ngreq++]);
        }

        if (ngreq > 0) MPI_Waitall(ngreq, greqs, MPI_STATUSES_IGNORE);

        // Ssend handshake. MPI_Ssend blocks until the peer posts its matching
        // receive, so a naive "send both sides, then receive both sides" would
        // deadlock with every rank waiting on a neighbour that is also sending.
        // Rank parity breaks the cycle: even ranks send first, odd ranks
        // receive first, which is the standard odd/even ordering.
        if (ssend_sync) {
            const int even = (rank % 2 == 0);
            for (int pass = 0; pass < 2; pass++) {
                const int sending = (pass == 0) ? even : !even;
                if (sending) {
                    if (ipc_ok_L) MPI_Ssend(NULL, 0, MPI_BYTE, left_rank,  21,
                                            MPI_COMM_WORLD);
                    if (ipc_ok_R) MPI_Ssend(NULL, 0, MPI_BYTE, right_rank, 20,
                                            MPI_COMM_WORLD);
                } else {
                    if (ipc_ok_L) MPI_Recv(NULL, 0, MPI_BYTE, left_rank,  20,
                                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (ipc_ok_R) MPI_Recv(NULL, 0, MPI_BYTE, right_rank, 21,
                                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        // Completion handshake. A rank writes straight into its neighbour's
        // receive buffer, so the neighbour must not unpack until that write has
        // landed -- and unlike MPI_Sendrecv there is no message to signal it.
        // This is a genuine cost of the one-sided path that the GPU-aware and
        // staged variants do not pay (their blocking Sendrecv is
        // self-synchronising). It must therefore be disclosed as a
        // synchronisation asymmetry when comparing against them.
        //
        // A five-point stencil only depends on its two neighbours, so the
        // handshake above is two zero-byte messages per side rather than a
        // global barrier, which synchronises far more than the dependency graph
        // requires and whose cost grows with rank count.
        //
        // THE TOKEN ALONE IS NOT ENOUGH -- it is why the double buffering
        // exists. A token proves the neighbour finished WRITING my slot; it says
        // nothing about whether the neighbour finished UNPACKING the slot I am
        // about to write. A single-buffered version of exactly this scheme was
        // tried (commit 8d913ba, job 60200) and is RACY: ranks drift, and a rank
        // starts iteration i+1's write into a neighbour's buffer while that
        // neighbour is still reading iteration i out of it. It broke where drift
        // is largest and compute slack smallest -- at 8 ranks, 1024^2 gave L2
        // 3.7550293292 and 4096^2 gave 5.0563202266 against the correct
        // 5.1449605829, while every barrier run matched.
        //
        // Alternating slots closes that write-after-read hazard, and two slots
        // are provably sufficient:
        //   - Within iteration i, my write to slot i%2 completed before I sent
        //     the token, and the neighbour's unpack of slot i%2 is enqueued only
        //     after it received that token.
        //   - My iteration i+1 write targets slot (i+1)%2, a different buffer
        //     from the slot the neighbour reads at iteration i. No conflict.
        //   - My iteration i+2 write returns to slot i%2. To get there I had to
        //     clear iteration i+1's Waitall, which required the neighbour's
        //     i+1 token, which it sent only after its own cudaDeviceSynchronize
        //     at iteration i+1 -- and that sync is device-wide, so it guarantees
        //     the neighbour's iteration i unpack of slot i%2 had finished.
        // So a rank can run at most one iteration ahead of its neighbour's
        // write-completion point, and the slot it reuses is always already read.
        //
        // Set STENCIL_SYNC=barrier for the old global barrier. Both modes are
        // correct; keep the A/B available because on one node the barrier was
        // measured marginally CHEAPER (job 60200, 4 ranks: 36.89 vs 37.04 ms at
        // 16384^2, 135.29 vs 135.45 at 32768^2) -- OpenMPI's intra-node barrier
        // beat four Isend/Irecv plus a Waitall. Those numbers predate the
        // double buffering and compare the two handshake IMPLEMENTATIONS, not
        // the cost of having a handshake at all: quantifying that needs a run
        // with no handshake, which cannot be correct. Do not cite <=0.4% as the
        // handshake's cost. The expected payoff for the neighbour path is at
        // higher rank counts and across nodes, where barrier cost scales and
        // the neighbour cost does not.
        if (!neighbor_sync) MPI_Barrier(MPI_COMM_WORLD);

        // Unpack received ghosts into my grid
        if (left_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                my_recv_L, d_old, N, pitch, 0       // ghost col 0
            );
        }
        if (right_rank >= 0) {
            unpack_ghost<<<copy_blocks, copy_threads>>>(
                my_recv_R, d_old, N, pitch, W + 1   // ghost col W+1
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
