/*****************************************************************************
 * transpose_cuda_ipc_noaccum.cu
 *
 * Matrix transpose: B = A^T (NO accumulation — pure overwrite each iteration)
 *
 * Purpose: fair comparison with NVSHMEM no-accum version.
 * A is never modified. B is overwritten each iteration.
 *
 * Communication modes (-DCOMM_MODE=N):
 *   0 = CUDA IPC direct   — kernel writes directly to peer's B
 *   1 = CUDA IPC buffered — pack → cudaMemcpyAsync (same stream) → unpack
 *   2 = GPU-aware MPI     — pack → MPI_Sendrecv → unpack
 *   3 = Staged MPI        — pack → D2H → MPI_Sendrecv → H2D → unpack
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#ifndef COMM_MODE
#define COMM_MODE 0
#endif

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(_e));                                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                      \
} while(0)

#define TILE 32
#define BROWS 8

/* ==========================================================================
 * KERNEL: transpose_kernel
 *
 * Always overwrites (=), never accumulates (+=).
 * A is const — never modified.
 * ========================================================================== */
__global__ void transpose_kernel(
    double * __restrict__ dst,
    int dst_ld,
    int dst_row_off,
    const double * __restrict__ src,    /* const — A never changes */
    int src_ld,
    int src_row_off,
    int Bo)
{
    __shared__ double tile[TILE][TILE + 1];

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = src[idx];
        }
    }

    __syncthreads();

    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + dst_row_off) + (size_t)dst_ld * (d_col + j);
            dst[idx] = tile[threadIdx.x][threadIdx.y + j];   /* = not += */
        }
    }
}

/* ==========================================================================
 * KERNEL: unpack_kernel (NO accumulation — overwrites)
 * ========================================================================== */
__global__ void unpack_kernel(
    double * __restrict__ B,
    int lda,
    int row_off,
    const double * __restrict__ buf,
    int Bo)
{
    int col = blockIdx.x * TILE + threadIdx.y;
    int row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (row < Bo && (col + j) < Bo) {
            B[(size_t)(row + row_off) + (size_t)lda * (col + j)] =    /* = not += */
                buf[(size_t)row + (size_t)Bo * (col + j)];
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
    CUDA_CHECK(cudaSetDevice(my_ID % ndev));

    int Bo       = order / P;
    int colstart = Bo * my_ID;
    size_t col_elems = (size_t)order * Bo;
    size_t blk_elems = (size_t)Bo * Bo;
    size_t bytes     = 2ULL * sizeof(double) * order * order;

    const char *mode_str[] = {
        "CUDA IPC direct (no accum)",
        "CUDA IPC buffered+stream (no accum)",
        "GPU-aware MPI (no accum)",
        "Staged MPI (no accum)"
    };
    if (my_ID == 0) {
        printf("GPU Matrix transpose: B = A^T (no accumulation)\n");
        printf("Communication mode: %s\n", mode_str[COMM_MODE]);
        printf("Number of GPUs       = %d\n", P);
        printf("Matrix order         = %d\n", order);
        printf("Block order          = %d\n", Bo);
        printf("Number of iterations = %d\n", iterations);
    }

    /* --- Allocate A and B --- */
    double *A_d, *B_d;
    CUDA_CHECK(cudaMalloc(&A_d, col_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&B_d, col_elems * sizeof(double)));

    /* --- CUDA stream --- */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* --- IPC DIRECT (mode 0): pointer to every peer's full B --- */
    double **peer_B = NULL;

#if COMM_MODE == 0
    if (P > 1) {
        cudaIpcMemHandle_t my_handle;
        CUDA_CHECK(cudaIpcGetMemHandle(&my_handle, B_d));

        cudaIpcMemHandle_t *all_handles =
            (cudaIpcMemHandle_t *)malloc(P * sizeof(cudaIpcMemHandle_t));
        MPI_Allgather(&my_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      all_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      MPI_COMM_WORLD);

        peer_B = (double **)malloc(P * sizeof(double *));
        for (int p = 0; p < P; p++) {
            if (p == my_ID)
                peer_B[p] = B_d;
            else
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    (void **)&peer_B[p], all_handles[p],
                    cudaIpcMemLazyEnablePeerAccess));
        }
        free(all_handles);
        if (my_ID == 0) printf("IPC handles exchanged — direct B access\n");
    }
#endif

    /* --- IPC BUFFERED (mode 1): pointer to every peer's recv_buf --- */
    double *send_buf = NULL, *recv_buf = NULL;
    double **peer_recv = NULL;

#if COMM_MODE == 1
    if (P > 1) {
        CUDA_CHECK(cudaMalloc(&send_buf, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&recv_buf, blk_elems * sizeof(double)));

        cudaIpcMemHandle_t my_handle;
        CUDA_CHECK(cudaIpcGetMemHandle(&my_handle, recv_buf));

        cudaIpcMemHandle_t *all_handles =
            (cudaIpcMemHandle_t *)malloc(P * sizeof(cudaIpcMemHandle_t));
        MPI_Allgather(&my_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      all_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      MPI_COMM_WORLD);

        peer_recv = (double **)malloc(P * sizeof(double *));
        for (int p = 0; p < P; p++) {
            if (p == my_ID)
                peer_recv[p] = recv_buf;
            else
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    (void **)&peer_recv[p], all_handles[p],
                    cudaIpcMemLazyEnablePeerAccess));
        }
        free(all_handles);
        if (my_ID == 0) printf("IPC handles exchanged — buffered+stream\n");
    }
#endif

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

    /* --- Init matrices --- */
    {
        dim3 blk(16, 16);
        dim3 grd((Bo + 15) / 16, (order + 15) / 16);
        init_kernel<<<grd, blk>>>(A_d, B_d, Bo, order, colstart);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    dim3 tblk(TILE, BROWS);
    dim3 tgrd((Bo + TILE - 1) / TILE, (Bo + TILE - 1) / TILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ==================================================================
     * MAIN LOOP — every iteration produces identical B = A^T
     * ================================================================== */
    double t0 = 0.0;

    for (int iter = 0; iter <= iterations; iter++) {

        if (iter == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }

        /* Phase 0: local transpose (overwrite) */
        transpose_kernel<<<tgrd, tblk, 0, stream>>>(
            B_d, order, colstart,
            A_d, order, colstart,
            Bo);

        /* Phases 1..P-1: remote */
        for (int phase = 1; phase < P; phase++) {
            int send_to   = (my_ID - phase + P) % P;
            int recv_from = (my_ID + phase)     % P;

#if COMM_MODE == 0
            /* ---- IPC DIRECT: kernel writes to peer's B (overwrite) ---- */
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                peer_B[send_to], order, colstart,
                A_d, order, send_to * Bo,
                Bo);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

#elif COMM_MODE == 1
            /* ---- IPC BUFFERED + STREAM ---- */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo);

            cudaMemcpyAsync(
                peer_recv[send_to], send_buf,
                blk_elems * sizeof(double),
                cudaMemcpyDeviceToDevice, stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo);

#elif COMM_MODE == 2
            /* ---- GPU-aware MPI ---- */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            MPI_Sendrecv(
                send_buf, (int)blk_elems, MPI_DOUBLE, send_to,   phase,
                recv_buf, (int)blk_elems, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo);

#elif COMM_MODE == 3
            /* ---- Staged MPI ---- */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemcpy(host_send, send_buf,
                                  blk_elems * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            MPI_Sendrecv(
                host_send, (int)blk_elems, MPI_DOUBLE, send_to,   phase,
                host_recv, (int)blk_elems, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(recv_buf, host_recv,
                                  blk_elems * sizeof(double),
                                  cudaMemcpyHostToDevice));

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo);
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /* --- Timing --- */
    double local_time = MPI_Wtime() - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* --- Verification ---
     * Init: A(row, local_col) = order * (local_col + colstart) + row
     * B = A^T: B(i, j_local) = order * i + (j_local + colstart)
     */
    double *B_h = (double *)malloc(col_elems * sizeof(double));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, col_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double abserr = 0.0;
    for (size_t j = 0; j < (size_t)Bo; j++)
        for (size_t i = 0; i < (size_t)order; i++) {
            double expected = (double)order * i + (double)(j + colstart);
            abserr += fabs(B_h[i + (size_t)order * j] - expected);
        }

    double abserr_tot;
    MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_ID == 0) {
        if (abserr_tot < 1.0e-8) {
            printf("Solution validates\n");
            double avgtime = max_time / (double)iterations;
            printf("Rate (MB/s): %lf  Avg time (s): %lf\n",
                   1.0e-06 * bytes / avgtime, avgtime);
        } else {
            printf("ERROR: Aggregate error %e exceeds threshold\n", abserr_tot);
        }
    }

    /* --- Cleanup --- */
    CUDA_CHECK(cudaStreamDestroy(stream));

#if COMM_MODE == 0
    if (P > 1) {
        for (int p = 0; p < P; p++)
            if (p != my_ID) CUDA_CHECK(cudaIpcCloseMemHandle(peer_B[p]));
        free(peer_B);
    }
#endif
#if COMM_MODE == 1
    if (P > 1) {
        for (int p = 0; p < P; p++)
            if (p != my_ID) CUDA_CHECK(cudaIpcCloseMemHandle(peer_recv[p]));
        free(peer_recv);
        CUDA_CHECK(cudaFree(send_buf));
        CUDA_CHECK(cudaFree(recv_buf));
    }
#endif
#if COMM_MODE >= 2
    if (P > 1) { CUDA_CHECK(cudaFree(send_buf)); CUDA_CHECK(cudaFree(recv_buf)); }
#endif
#if COMM_MODE == 3
    if (P > 1) { CUDA_CHECK(cudaFreeHost(host_send)); CUDA_CHECK(cudaFreeHost(host_recv)); }
#endif

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    free(B_h);

    MPI_Finalize();
    return 0;
}
