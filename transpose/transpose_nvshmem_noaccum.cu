/*****************************************************************************
 * transpose_nvshmem_noaccum.cu
 *
 * Matrix transpose: B = A^T (NO accumulation — pure overwrite each iteration)
 * Using NVSHMEM for GPU-to-GPU communication.
 *
 * Purpose: isolate whether the NVSHMEM direct slowdown comes from the
 * get+put round trip (needed for accumulation) or from scattered writes.
 * Since there's no accumulation, direct mode uses a single nvshmem_double_p
 * per element — no nvshmem_double_g needed.
 *
 * Two modes (-DCOMM_MODE=N):
 *   0 = NVSHMEM direct  — kernel uses nvshmem_double_p to write to peer's B
 *   1 = NVSHMEM buffered — pack → nvshmem_putmem on stream → unpack
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

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
 * KERNEL: transpose_local_kernel
 *
 * Shared-memory tiled transpose. Always overwrites (=), never accumulates.
 * No A modification — A stays constant across iterations.
 * ========================================================================== */
__global__ void transpose_local_kernel(
    double * __restrict__ dst,
    int dst_ld,
    int dst_row_off,
    const double * __restrict__ src,   /* const — we never modify A */
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
 * KERNEL: transpose_nvshmem_direct_kernel
 *
 * Reads from local A, transposes via shared memory, writes each element
 * to target PE's B using nvshmem_double_p. Pure put, no get needed.
 * ========================================================================== */
__global__ void transpose_nvshmem_direct_kernel(
    double * __restrict__ B_sym,
    int dst_ld,
    int dst_row_off,
    const double * __restrict__ A_sym,   /* const — never modified */
    int src_ld,
    int src_row_off,
    int Bo,
    int target_pe)
{
    __shared__ double tile[TILE][TILE + 1];

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = A_sym[idx];
        }
    }

    __syncthreads();

    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + dst_row_off) + (size_t)dst_ld * (d_col + j);
            double val = tile[threadIdx.x][threadIdx.y + j];
            nvshmem_double_p(&B_sym[idx], val, target_pe);  /* pure put, no get */
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
    MPI_Init(&argc, &argv);

    int my_PE, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    if (argc != 3) {
        if (my_PE == 0) fprintf(stderr, "Usage: %s <iterations> <matrix_order>\n", argv[0]);
        nvshmem_finalize(); MPI_Finalize(); return 1;
    }
    int iterations = atoi(argv[1]);
    int order      = atoi(argv[2]);
    if (order % P != 0) {
        if (my_PE == 0) fprintf(stderr, "ERROR: order must be divisible by num GPUs\n");
        nvshmem_finalize(); MPI_Finalize(); return 1;
    }

    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(my_PE % ndev));

    int Bo       = order / P;
    int colstart = Bo * my_PE;
    size_t col_elems = (size_t)order * Bo;
    size_t blk_elems = (size_t)Bo * Bo;
    size_t bytes     = 2ULL * sizeof(double) * order * order;

    const char *mode_str[] = {"NVSHMEM direct (no accum)", "NVSHMEM buffered (no accum)"};
    if (my_PE == 0) {
        printf("GPU Matrix transpose: B = A^T (no accumulation)\n");
        printf("Communication mode: %s\n", mode_str[COMM_MODE]);
        printf("Number of GPUs       = %d\n", P);
        printf("Matrix order         = %d\n", order);
        printf("Block order          = %d\n", Bo);
        printf("Number of iterations = %d\n", iterations);
    }

    /* Allocate from symmetric heap */
    double *A_sym = (double *)nvshmem_malloc(col_elems * sizeof(double));
    double *B_sym = (double *)nvshmem_malloc(col_elems * sizeof(double));
    if (!A_sym || !B_sym) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", my_PE);
        nvshmem_finalize(); MPI_Finalize(); return 1;
    }

    double *send_buf = NULL, *recv_buf = NULL;
#if COMM_MODE == 1
    if (P > 1) {
        send_buf = (double *)nvshmem_malloc(blk_elems * sizeof(double));
        recv_buf = (double *)nvshmem_malloc(blk_elems * sizeof(double));
        if (!send_buf || !recv_buf) {
            fprintf(stderr, "PE %d: nvshmem_malloc for buffers failed\n", my_PE);
            nvshmem_finalize(); MPI_Finalize(); return 1;
        }
    }
#endif

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Init A with known values, zero B */
    {
        dim3 blk(16, 16);
        dim3 grd((Bo + 15) / 16, (order + 15) / 16);
        init_kernel<<<grd, blk, 0, stream>>>(A_sym, B_sym, Bo, order, colstart);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    dim3 tblk(TILE, BROWS);
    dim3 tgrd((Bo + TILE - 1) / TILE, (Bo + TILE - 1) / TILE);

    nvshmem_barrier_all();

    /* ==================================================================
     * MAIN LOOP — every iteration produces identical B = A^T
     * A is never modified. B is overwritten each iteration.
     * ================================================================== */
    double t0 = 0.0;

    for (int iter = 0; iter <= iterations; iter++) {

        if (iter == 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();
            t0 = MPI_Wtime();
        }

        /* Phase 0: local transpose — my A → my B (overwrite) */
        transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
            B_sym, order, colstart,
            A_sym, order, colstart,
            Bo);

        /* Phases 1..P-1: remote */
        for (int phase = 1; phase < P; phase++) {
            int send_to   = (my_PE - phase + P) % P;
            int recv_from = (my_PE + phase)     % P;

#if COMM_MODE == 0
            /* ---- NVSHMEM DIRECT: pure put, no get ---- */
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            transpose_nvshmem_direct_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, colstart,
                A_sym, order, send_to * Bo,
                Bo, send_to);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

#elif COMM_MODE == 1
            /* ---- NVSHMEM BUFFERED ---- */
            transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_sym, order, send_to * Bo,
                Bo);

            nvshmemx_putmem_nbi_on_stream(
                recv_buf, send_buf,
                blk_elems * sizeof(double),
                send_to, stream);

            nvshmemx_quiet_on_stream(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            /* Unpack with = (overwrite, not +=) */
            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, recv_from * Bo, recv_buf, Bo);
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
     * So globally: A(row, global_col) = order * global_col + row
     *
     * B = A^T means: B(i, j_global) = A(j_global, i) = order * i + j_global
     *
     * On this PE, local column j has global column = j + colstart.
     * So: B(i, j_local) = order * i + (j_local + colstart)
     */
    double *B_h = (double *)malloc(col_elems * sizeof(double));
    CUDA_CHECK(cudaMemcpy(B_h, B_sym, col_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double abserr = 0.0;
    for (size_t j = 0; j < (size_t)Bo; j++)
        for (size_t i = 0; i < (size_t)order; i++) {
            double expected = (double)order * i + (double)(j + colstart);
            abserr += fabs(B_h[i + (size_t)order * j] - expected);
        }

    double abserr_tot;
    MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_PE == 0) {
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
    nvshmem_free(A_sym);
    nvshmem_free(B_sym);
#if COMM_MODE == 1
    if (P > 1) {
        nvshmem_free(send_buf);
        nvshmem_free(recv_buf);
    }
#endif
    free(B_h);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
