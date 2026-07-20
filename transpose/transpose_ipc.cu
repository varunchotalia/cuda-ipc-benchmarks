/*****************************************************************************
 * transpose_ipc.cu
 *
 * Matrix transpose: B = A^T
 * One MPI rank per GPU. Matrix distributed by column blocks.
 *
 * Communication modes (-DCOMM_MODE=N):
 *   0 = CUDA IPC direct   — kernel writes directly to peer's B
 *   1 = CUDA IPC buffered — pack → cudaMemcpyAsync (same stream) → unpack
 *   2 = GPU-aware MPI     — pack → MPI_Sendrecv → unpack
 *   3 = Staged MPI        — pack → D2H → MPI_Sendrecv → H2D → unpack
 *
 * Accumulation (-DACCUMULATE=0 or 1, default 1):
 *   1 = B += A^T (PRK style, A incremented each iteration)
 *   0 = B  = A^T (pure overwrite, A unchanged)
 *
 * Single-kernel mode (-DSINGLE_KERNEL=1, only for COMM_MODE=0):
 *   All peers handled in one kernel launch using blockIdx.z as peer index.
 *   Eliminates per-phase barriers. Only one barrier before and after.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#ifndef COMM_MODE
#define COMM_MODE 0
#endif
#ifndef ACCUMULATE
#define ACCUMULATE 1
#endif
#ifndef SINGLE_KERNEL
#define SINGLE_KERNEL 0
#endif

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(_e));                                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                      \
} while(0)

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

#define TILE 32
#define BROWS 8

/* ==========================================================================
 * KERNEL: transpose_kernel
 *
 * Handles local transpose, direct IPC write, and packing into buffers.
 * ACCUMULATE controls += vs =. A is only modified if ACCUMULATE=1.
 * ========================================================================== */
__global__ void transpose_kernel(
    double * __restrict__ dst,
    int dst_ld,
    int dst_row_off,
#if ACCUMULATE
    double * __restrict__ src,    /* non-const: we modify A with += 1.0 */
#else
    const double * __restrict__ src,
#endif
    int src_ld,
    int src_row_off,
    int Bo,
    int accumulate)               /* runtime flag for += vs = */
{
    __shared__ double tile[TILE][TILE + 1];

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = src[idx];
#if ACCUMULATE
            src[idx] += 1.0;
#endif
        }
    }

    __syncthreads();

    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + dst_row_off) + (size_t)dst_ld * (d_col + j);
            if (accumulate)
                dst[idx] += tile[threadIdx.x][threadIdx.y + j];
            else
                dst[idx]  = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/* ==========================================================================
 * KERNEL: transpose_all_peers_kernel
 *
 * Single kernel that handles ALL remote peers at once.
 * blockIdx.z selects which peer to write to.
 * peer_B_ptrs[z] is the IPC pointer to that peer's B matrix.
 * send_to_list[z] tells which sub-block of A to read.
 *
 * Grid: (tiles_x, tiles_y, num_peers)
 * ========================================================================== */
#if COMM_MODE == 0 && SINGLE_KERNEL
__global__ void transpose_all_peers_kernel(
    double ** __restrict__ peer_B_ptrs,   /* array of pointers to each peer's B */
    int dst_ld,                           /* stride of B = order */
    int colstart,                         /* my colstart = dst_row_off for all peers */
#if ACCUMULATE
    double * __restrict__ src,
#else
    const double * __restrict__ src,
#endif
    int src_ld,
    int Bo,
    int accumulate,
    int my_ID,                            /* this rank's ID, to compute send_to inline */
    int P)                                /* total number of ranks */
{
    __shared__ double tile[TILE][TILE + 1];

    /* peer_idx=0 corresponds to phase=1, peer_idx=1 to phase=2, etc. */
    int peer_idx = blockIdx.z;
    int send_to = (my_ID - (peer_idx + 1) + P) % P; /* actual PE number */
    int src_row_off = send_to * Bo;       /* which sub-block of A to read */

    /* Load tile from local A */
    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = src[idx];
#if ACCUMULATE
            /* Each z-block accesses a different, non-overlapping row range of A
               (src_row_off = send_to * Bo, send_to is unique per z). Increment
               unconditionally — there is no aliasing between z-blocks. */
            src[idx] += 1.0;
#endif
        }
    }

    __syncthreads();

    /* Write transposed tile to peer's B */
    double *dst = peer_B_ptrs[send_to];
    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + colstart) + (size_t)dst_ld * (d_col + j);
            if (accumulate)
                dst[idx] += tile[threadIdx.x][threadIdx.y + j];
            else
                dst[idx]  = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
#endif

/* ==========================================================================
 * KERNEL: unpack_kernel
 * ========================================================================== */
__global__ void unpack_kernel(
    double * __restrict__ B,
    int lda,
    int row_off,
    const double * __restrict__ buf,
    int Bo,
    int accumulate)
{
    int col = blockIdx.x * TILE + threadIdx.y;
    int row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (row < Bo && (col + j) < Bo) {
            size_t didx = (size_t)(row + row_off) + (size_t)lda * (col + j);
            size_t sidx = (size_t)row + (size_t)Bo * (col + j);
            if (accumulate)
                B[didx] += buf[sidx];
            else
                B[didx]  = buf[sidx];
        }
    }
}

/* ==========================================================================
 * KERNEL: init_kernel
 * ========================================================================== */
__global__ void init_kernel(double *A, double *B, int Bo, int lda, int colstart)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < Bo && row < lda) {
        size_t idx = (size_t)row + (size_t)lda * col;
        A[idx] = (double)((double)lda * (col + colstart) + row);
        B[idx] = 0.0;
    }
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */
int main(int argc, char **argv)
{
    int my_ID, P;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc != 3) {
        if (my_ID == 0) fprintf(stderr, "Usage: %s <iterations> <matrix_order>\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    int iterations = atoi(argv[1]);
    int order      = atoi(argv[2]);
    if (order % P != 0) {
        if (my_ID == 0) fprintf(stderr, "ERROR: order must be divisible by num GPUs\n");
        MPI_Finalize(); return 1;
    }

    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int local_rank = local_rank_for_gpu(MPI_COMM_WORLD, my_ID);
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));

    int Bo       = order / P;
    int colstart = Bo * my_ID;
    size_t col_elems = (size_t)order * Bo;
    size_t blk_elems = (size_t)Bo * Bo;
    size_t bytes     = 2ULL * sizeof(double) * order * order;
    if (blk_elems > (size_t)INT_MAX) {
        if (my_ID == 0) fprintf(stderr, "ERROR: Bo*Bo=%zu exceeds INT_MAX; MPI count overflow\n", blk_elems);
        MPI_Finalize(); return 1;
    }
    int blk_count = (int)blk_elems;

    const char *comm_str[] = {"IPC direct", "IPC buffered", "GPU-aware MPI", "Staged MPI"};
    if (my_ID == 0) {
        printf("GPU Matrix transpose: B %s A^T\n", ACCUMULATE ? "+=" : "=");
        printf("Communication: %s%s\n", comm_str[COMM_MODE],
               SINGLE_KERNEL ? " (single-kernel)" : "");
        printf("GPUs = %d, order = %d, Bo = %d, iterations = %d\n",
               P, order, Bo, iterations);
    }

    /* --- Allocate A and B --- */
    double *A_d, *B_d = NULL;
    CUDA_CHECK(cudaMalloc(&A_d, col_elems * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* cuda_ipc info key signals MPI_Win_allocate to use cudaMalloc internally */
    MPI_Info ipc_info;
    MPI_Info_create(&ipc_info);
    MPI_Info_set(ipc_info, "cuda_ipc", "1");

    /* --- IPC DIRECT (mode 0) --- */
    double **peer_B = NULL;
    double **peer_B_dev = NULL;     /* device copy of pointer array (indexed by PE) */
    MPI_Win win_B = MPI_WIN_NULL;
    bool *ipc_ok_B = NULL;          /* per-peer: kernel can write via peer_B[p] */
    double *fb_send = NULL, *fb_recv = NULL;   /* MPI fallback scratch (unreachable peers) */

#if COMM_MODE == 0
    if (P > 1) {
        MPI_Win_allocate(col_elems * sizeof(double), 1,
                         ipc_info, MPI_COMM_WORLD, &B_d, &win_B);

        /* MPI_Win_shared_query can fail per peer -- on another node without
           fabric handles, or same-node across a GPU-island boundary with no
           P2P.  Never assume it succeeded: an unchecked NULL peer pointer
           written to by a later kernel is a silent illegal-memory-access
           that can hang or fault the GPU. */
        peer_B  = (double **)calloc(P, sizeof(double *));
        ipc_ok_B = (bool *)calloc(P, sizeof(bool));
        int nFallback = 0;
        for (int p = 0; p < P; p++) {
            if (p == my_ID) continue;   /* never write to own B via peer path */
            MPI_Aint sz; int disp;
            if (MPI_Win_shared_query(win_B, p, &sz, &disp, &peer_B[p]) == MPI_SUCCESS
                && peer_B[p] != NULL) {
                ipc_ok_B[p] = true;
            } else {
                peer_B[p] = NULL;
                ++nFallback;
            }
        }

#if SINGLE_KERNEL
        if (nFallback > 0) {
            fprintf(stderr,
                "transpose_direct_single: %d of %d peer(s) not IPC-reachable; "
                "single-kernel mode has no MPI fallback path. Rerun with "
                "'direct', 'buffered', 'gpumpi', or 'staged' instead.\n",
                nFallback, P - 1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaMalloc(&peer_B_dev, P * sizeof(double *)));
        CUDA_CHECK(cudaMemcpy(peer_B_dev, peer_B,
                              P * sizeof(double *), cudaMemcpyHostToDevice));
#else
        if (nFallback > 0) {
            CUDA_CHECK(cudaMalloc(&fb_send, blk_elems * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&fb_recv, blk_elems * sizeof(double)));
        }
#endif
        if (my_ID == 0) {
            printf("IPC handles exchanged%s\n",
                   SINGLE_KERNEL ? " — single-kernel mode" : "");
            if (nFallback > 0)
                printf("  %d of %d peers not IPC-reachable; using MPI "
                       "send/recv fallback for them\n", nFallback, P - 1);
        }
    } else {
        CUDA_CHECK(cudaMalloc(&B_d, col_elems * sizeof(double)));
    }
#else
    CUDA_CHECK(cudaMalloc(&B_d, col_elems * sizeof(double)));
#endif

    /* --- IPC BUFFERED (mode 1) --- */
    double *send_buf = NULL, *recv_buf = NULL;
    double **peer_recv = NULL;
    MPI_Win win_recv = MPI_WIN_NULL;
    bool *ipc_ok_recv = NULL;
    double *fb_recv1 = NULL;   /* MPI fallback scratch for unreachable recv_from */

#if COMM_MODE == 1
    if (P > 1) {
        CUDA_CHECK(cudaMalloc(&send_buf, blk_elems * sizeof(double)));

        MPI_Win_allocate(blk_elems * sizeof(double), 1,
                         ipc_info, MPI_COMM_WORLD, &recv_buf, &win_recv);

        /* same caveat as mode 0: query can fail per peer, never assume it
           succeeded */
        peer_recv   = (double **)calloc(P, sizeof(double *));
        ipc_ok_recv = (bool *)calloc(P, sizeof(bool));
        int nFallback = 0;
        for (int p = 0; p < P; p++) {
            if (p == my_ID) continue;
            MPI_Aint sz; int disp;
            if (MPI_Win_shared_query(win_recv, p, &sz, &disp, &peer_recv[p]) == MPI_SUCCESS
                && peer_recv[p] != NULL) {
                ipc_ok_recv[p] = true;
            } else {
                peer_recv[p] = NULL;
                ++nFallback;
            }
        }
        if (nFallback > 0) {
            CUDA_CHECK(cudaMalloc(&fb_recv1, blk_elems * sizeof(double)));
            if (my_ID == 0)
                printf("  %d of %d peers not IPC-reachable; using MPI "
                       "send/recv fallback for them\n", nFallback, P - 1);
        }
    }
#endif

    MPI_Info_free(&ipc_info);

    /* --- MPI MODES (2, 3) --- */
#if COMM_MODE >= 2
    if (P > 1) {
        CUDA_CHECK(cudaMalloc(&send_buf, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&recv_buf, blk_elems * sizeof(double)));
    }
#endif
    double *host_send = NULL, *host_recv = NULL;
#if COMM_MODE == 3
    if (P > 1) {
        CUDA_CHECK(cudaMallocHost(&host_send, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&host_recv, blk_elems * sizeof(double)));
    }
#endif

    /* --- Init --- */
    {
        dim3 blk(16, 16);
        dim3 grd((Bo + 15) / 16, (order + 15) / 16);
        init_kernel<<<grd, blk>>>(A_d, B_d, Bo, order, colstart);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    dim3 tblk(TILE, BROWS);
    dim3 tgrd((Bo + TILE - 1) / TILE, (Bo + TILE - 1) / TILE);

#if COMM_MODE == 0 && SINGLE_KERNEL
    /* 3D grid: x,y = tiles, z = peers */
    dim3 tgrd_all(tgrd.x, tgrd.y, P - 1);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    /* ==================================================================
     * MAIN LOOP
     * ================================================================== */
    double t0 = 0.0;

    for (int iter = 0; iter <= iterations; iter++) {

        if (iter == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }

        /* Phase 0: local transpose */
        transpose_kernel<<<tgrd, tblk, 0, stream>>>(
            B_d, order, colstart,
            A_d, order, colstart,
            Bo, ACCUMULATE);

#if COMM_MODE == 0 && SINGLE_KERNEL
        /* === SINGLE KERNEL: all peers at once === */
        if (P > 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

            transpose_all_peers_kernel<<<tgrd_all, tblk, 0, stream>>>(
                peer_B_dev, order, colstart,
                A_d, order, Bo, ACCUMULATE, my_ID, P);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);
        }
#else
        /* === PER-PHASE LOOP === */
        for (int phase = 1; phase < P; phase++) {
            int send_to   = (my_ID - phase + P) % P;
            int recv_from = (my_ID + phase)     % P;

#if COMM_MODE == 0
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

            {
                /* recv_from's send_to at this same phase is my_ID (the ring
                   pairing is symmetric per phase), so !ipc_ok_B[recv_from]
                   means recv_from can't kernel-write into me either -- post
                   the matching receive before anyone might send. */
                MPI_Request reqs[2]; int nreq = 0;
                if (!ipc_ok_B[recv_from]) {
                    MPI_Irecv(fb_recv, blk_count, MPI_DOUBLE, recv_from, phase,
                              MPI_COMM_WORLD, &reqs[nreq++]);
                }

                if (ipc_ok_B[send_to]) {
                    transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                        peer_B[send_to], order, colstart,
                        A_d, order, send_to * Bo,
                        Bo, ACCUMULATE);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } else {
                    transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                        fb_send, Bo, 0,
                        A_d, order, send_to * Bo,
                        Bo, 0);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    MPI_Isend(fb_send, blk_count, MPI_DOUBLE, send_to, phase,
                              MPI_COMM_WORLD, &reqs[nreq++]);
                }

                if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

                if (!ipc_ok_B[recv_from]) {
                    unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                        B_d, order, recv_from * Bo, fb_recv, Bo, ACCUMULATE);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

#elif COMM_MODE == 1
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo, 0);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            {
                MPI_Request reqs[2]; int nreq = 0;
                if (!ipc_ok_recv[recv_from]) {
                    MPI_Irecv(fb_recv1, blk_count, MPI_DOUBLE, recv_from, phase,
                              MPI_COMM_WORLD, &reqs[nreq++]);
                }

                if (ipc_ok_recv[send_to]) {
                    cudaMemcpyAsync(
                        peer_recv[send_to], send_buf,
                        blk_elems * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } else {
                    MPI_Isend(send_buf, blk_count, MPI_DOUBLE, send_to, phase,
                              MPI_COMM_WORLD, &reqs[nreq++]);
                }

                if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo,
                ipc_ok_recv[recv_from] ? recv_buf : fb_recv1,
                Bo, ACCUMULATE);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

#elif COMM_MODE == 2
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo, 0);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            MPI_Sendrecv(
                send_buf, blk_count, MPI_DOUBLE, send_to,   phase,
                recv_buf, blk_count, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo, ACCUMULATE);

#elif COMM_MODE == 3
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo, 0);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemcpy(host_send, send_buf,
                                  blk_elems * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            MPI_Sendrecv(
                host_send, blk_count, MPI_DOUBLE, send_to,   phase,
                host_recv, blk_count, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(recv_buf, host_recv,
                                  blk_elems * sizeof(double),
                                  cudaMemcpyHostToDevice));

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo, ACCUMULATE);
#endif
        }
#endif
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /* --- Timing --- */
    double local_time = MPI_Wtime() - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* --- Verification --- */
    double *B_h = (double *)malloc(col_elems * sizeof(double));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, col_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double abserr = 0.0;
#if ACCUMULATE
    double addit = ((double)(iterations + 1) * (double)iterations) / 2.0;
    for (size_t j = 0; j < (size_t)Bo; j++)
        for (size_t i = 0; i < (size_t)order; i++) {
            double expected = (double)((double)order * i + j + colstart)
                              * (iterations + 1) + addit;
            abserr += fabs(B_h[i + (size_t)order * j] - expected);
        }
#else
    for (size_t j = 0; j < (size_t)Bo; j++)
        for (size_t i = 0; i < (size_t)order; i++) {
            double expected = (double)order * i + (double)(j + colstart);
            abserr += fabs(B_h[i + (size_t)order * j] - expected);
        }
#endif

    double abserr_tot;
    MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_ID == 0) {
        double abserr_per_elem = abserr_tot / ((double)order * order);
        if (abserr_per_elem < 1.0e-8) {
            printf("Solution validates\n");
            double avgtime = max_time / (double)iterations;
            printf("Rate (MB/s): %lf  Avg time (s): %lf\n",
                   1.0e-06 * bytes / avgtime, avgtime);
        } else {
            printf("ERROR: Per-element error %e exceeds threshold\n", abserr_per_elem);
        }
    }

    /* --- Cleanup --- */
    CUDA_CHECK(cudaStreamDestroy(stream));

#if COMM_MODE == 0
    if (P > 1) {
        MPI_Win_free(&win_B);
        free(peer_B);
        free(ipc_ok_B);
#if SINGLE_KERNEL
        CUDA_CHECK(cudaFree(peer_B_dev));
#else
        if (fb_send) CUDA_CHECK(cudaFree(fb_send));
        if (fb_recv) CUDA_CHECK(cudaFree(fb_recv));
#endif
    }
#endif
#if COMM_MODE == 1
    if (P > 1) {
        MPI_Win_free(&win_recv);  /* also frees recv_buf */
        free(peer_recv);
        free(ipc_ok_recv);
        if (fb_recv1) CUDA_CHECK(cudaFree(fb_recv1));
        CUDA_CHECK(cudaFree(send_buf));
    }
#endif
#if COMM_MODE >= 2
    if (P > 1) { CUDA_CHECK(cudaFree(send_buf)); CUDA_CHECK(cudaFree(recv_buf)); }
#endif
#if COMM_MODE == 3
    if (P > 1) { CUDA_CHECK(cudaFreeHost(host_send)); CUDA_CHECK(cudaFreeHost(host_recv)); }
#endif

    CUDA_CHECK(cudaFree(A_d));
#if COMM_MODE == 0
    if (P == 1) CUDA_CHECK(cudaFree(B_d));
    /* else MPI_Win_free already freed B_d */
#else
    CUDA_CHECK(cudaFree(B_d));
#endif
    free(B_h);

    MPI_Finalize();
    return 0;
}
