/*****************************************************************************
 * transpose_nvshmem.cu
 *
 * Matrix transpose: B = A^T using NVSHMEM.
 *
 * Modes (-DCOMM_MODE=N):
 *   0 = NVSHMEM direct  — nvshmem_double_p per element
 *   1 = NVSHMEM buffered — pack → nvshmem_putmem → unpack
 *
 * Flags:
 *   -DACCUMULATE=0/1    (default 1)
 *   -DSINGLE_KERNEL=0/1 (default 0, only for COMM_MODE=0)
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

#define TILE 32
#define BROWS 8

/* ==========================================================================
 * KERNEL: transpose_local_kernel
 * ========================================================================== */
__global__ void transpose_local_kernel(
    double * __restrict__ dst,
    int dst_ld,
    int dst_row_off,
#if ACCUMULATE
    double * __restrict__ src,
#else
    const double * __restrict__ src,
#endif
    int src_ld,
    int src_row_off,
    int Bo,
    int accumulate)
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
 * KERNEL: transpose_nvshmem_direct_kernel (per-phase version)
 * ========================================================================== */
__global__ void transpose_nvshmem_direct_kernel(
    double * __restrict__ B_sym,
    int dst_ld,
    int dst_row_off,
#if ACCUMULATE
    double * __restrict__ A_sym,
#else
    const double * __restrict__ A_sym,
#endif
    int src_ld,
    int src_row_off,
    int Bo,
    int target_pe,
    int accumulate)
{
    __shared__ double tile[TILE][TILE + 1];

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = A_sym[idx];
#if ACCUMULATE
            A_sym[idx] += 1.0;
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
            double val = tile[threadIdx.x][threadIdx.y + j];
#if ACCUMULATE
            double old = nvshmem_double_g(&B_sym[idx], target_pe);
            nvshmem_double_p(&B_sym[idx], old + val, target_pe);
#else
            nvshmem_double_p(&B_sym[idx], val, target_pe);
#endif
        }
    }
}

/* ==========================================================================
 * KERNEL: transpose_nvshmem_all_peers_kernel (single-kernel version)
 *
 * blockIdx.z = peer index. All peers handled in one launch.
 * ========================================================================== */
#if COMM_MODE == 0 && SINGLE_KERNEL
__global__ void transpose_nvshmem_all_peers_kernel(
#if ACCUMULATE
    double * __restrict__ A_sym,
#else
    const double * __restrict__ A_sym,
#endif
    double * __restrict__ B_sym,
    int order,
    int colstart,
    int Bo,
    int accumulate,
    const int * __restrict__ send_to_list)
{
    __shared__ double tile[TILE][TILE + 1];

    int peer_idx = blockIdx.z;
    int target_pe = send_to_list[peer_idx];
    int src_row_off = target_pe * Bo;

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)order * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = A_sym[idx];
#if ACCUMULATE
            A_sym[idx] += 1.0;
#endif
        }
    }

    __syncthreads();

    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + colstart) + (size_t)order * (d_col + j);
            double val = tile[threadIdx.x][threadIdx.y + j];
#if ACCUMULATE
            double old = nvshmem_double_g(&B_sym[idx], target_pe);
            nvshmem_double_p(&B_sym[idx], old + val, target_pe);
#else
            nvshmem_double_p(&B_sym[idx], val, target_pe);
#endif
        }
    }
}
#endif

/* ==========================================================================
 * KERNEL: unpack_kernel
 * ========================================================================== */
__global__ void unpack_kernel(
    double * __restrict__ B, int lda, int row_off,
    const double * __restrict__ buf, int Bo, int accumulate)
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

    const char *comm_str[] = {"NVSHMEM direct", "NVSHMEM buffered"};
    if (my_PE == 0) {
        printf("GPU Matrix transpose: B %s A^T\n", ACCUMULATE ? "+=" : "=");
        printf("Communication: %s%s\n", comm_str[COMM_MODE],
               SINGLE_KERNEL ? " (single-kernel)" : "");
        printf("GPUs = %d, order = %d, Bo = %d, iterations = %d\n",
               P, order, Bo, iterations);
    }

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
    }
#endif

    /* Device-side send_to list for single-kernel mode */
    int *send_to_dev = NULL;
#if COMM_MODE == 0 && SINGLE_KERNEL
    if (P > 1) {
        int *send_to_host = (int *)malloc((P - 1) * sizeof(int));
        for (int phase = 1; phase < P; phase++)
            send_to_host[phase - 1] = (my_PE - phase + P) % P;
        CUDA_CHECK(cudaMalloc(&send_to_dev, (P - 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(send_to_dev, send_to_host,
                              (P - 1) * sizeof(int), cudaMemcpyHostToDevice));
        free(send_to_host);
    }
#endif

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    {
        dim3 blk(16, 16);
        dim3 grd((Bo + 15) / 16, (order + 15) / 16);
        init_kernel<<<grd, blk, 0, stream>>>(A_sym, B_sym, Bo, order, colstart);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    dim3 tblk(TILE, BROWS);
    dim3 tgrd((Bo + TILE - 1) / TILE, (Bo + TILE - 1) / TILE);

#if COMM_MODE == 0 && SINGLE_KERNEL
    dim3 tgrd_all(tgrd.x, tgrd.y, P - 1);
#endif

    nvshmem_barrier_all();

    double t0 = 0.0;

    for (int iter = 0; iter <= iterations; iter++) {

        if (iter == 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();
            t0 = MPI_Wtime();
        }

        /* Phase 0: local */
        transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
            B_sym, order, colstart,
            A_sym, order, colstart,
            Bo, ACCUMULATE);

#if COMM_MODE == 0 && SINGLE_KERNEL
        if (P > 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            transpose_nvshmem_all_peers_kernel<<<tgrd_all, tblk, 0, stream>>>(
                A_sym, B_sym, order, colstart, Bo, ACCUMULATE, send_to_dev);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();
        }
#else
        for (int phase = 1; phase < P; phase++) {
            int send_to   = (my_PE - phase + P) % P;
            int recv_from = (my_PE + phase)     % P;

#if COMM_MODE == 0
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            transpose_nvshmem_direct_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, colstart,
                A_sym, order, send_to * Bo,
                Bo, send_to, ACCUMULATE);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

#elif COMM_MODE == 1
            transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_sym, order, send_to * Bo,
                Bo, 0);

            nvshmemx_putmem_nbi_on_stream(
                recv_buf, send_buf,
                blk_elems * sizeof(double),
                send_to, stream);

            nvshmemx_quiet_on_stream(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            unpack_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, recv_from * Bo, recv_buf, Bo, ACCUMULATE);
#endif
        }
#endif
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    double local_time = MPI_Wtime() - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *B_h = (double *)malloc(col_elems * sizeof(double));
    CUDA_CHECK(cudaMemcpy(B_h, B_sym, col_elems * sizeof(double),
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(A_sym);
    nvshmem_free(B_sym);
#if COMM_MODE == 1
    if (P > 1) { nvshmem_free(send_buf); nvshmem_free(recv_buf); }
#endif
#if COMM_MODE == 0 && SINGLE_KERNEL
    if (send_to_dev) CUDA_CHECK(cudaFree(send_to_dev));
#endif
    free(B_h);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
