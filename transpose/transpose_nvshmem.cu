/*****************************************************************************
 * transpose_nvshmem.cu
 *
 * Matrix transpose: B = A^T using NVSHMEM for GPU-to-GPU communication.
 *
 * Two modes (-DCOMM_MODE=N):
 *   0 = NVSHMEM direct  — kernel uses nvshmem_double_p to write to peer's B
 *   1 = NVSHMEM buffered — pack → nvshmem_putmem on stream → unpack
 *
 * NVSHMEM differences from IPC:
 *   - Memory allocated from "symmetric heap" (nvshmem_malloc)
 *   - All PEs (GPUs) get same-sized allocation at symmetric addresses
 *   - No manual handle exchange — NVSHMEM manages pointers internally
 *   - Can call put/get from inside kernels (device-side communication)
 *
 * Compile:
 *   NVSHMEM_HOME=/lustre/nvwulf/software/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nvshmem
 *   nvcc -O3 -DCOMM_MODE=0 -gencode arch=compute_90,code=sm_90 \
 *        -rdc=true transpose_nvshmem.cu \
 *        -I$NVSHMEM_HOME/include -I$MPI_HOME/include \
 *        -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device \
 *        -L$MPI_HOME/lib -lmpi \
 *        -o transpose_nvshmem_direct
 *
 * Run:
 *   nvshmrun -np 4 ./transpose_nvshmem_direct 100 4096
 *   OR
 *   mpirun -np 4 ./transpose_nvshmem_direct 100 4096
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
 * Standard tiled transpose for local data and for packing into buffers.
 * Same as the non-NVSHMEM version — shared memory tile trick.
 * ========================================================================== */
__global__ void transpose_local_kernel(
    double * __restrict__ dst,    /* where to write                         */
    int dst_ld,                   /* stride of dst                          */
    int dst_row_off,              /* row offset in dst                      */
    double * __restrict__ src,    /* where to read                          */
    int src_ld,                   /* stride of src                          */
    int src_row_off,              /* row offset in src                      */
    int Bo,                       /* block order                            */
    int accumulate)               /* 1 = +=, 0 = =                         */
{
    __shared__ double tile[TILE][TILE + 1];

    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = src[idx];
            src[idx] += 1.0;
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
 * KERNEL: transpose_nvshmem_direct_kernel
 *
 * The NVSHMEM direct kernel. Reads from local A, transposes via shared
 * memory, and uses nvshmem_double_p to write each element directly to
 * the target PE's B matrix. No intermediate buffer.
 *
 * nvshmem_double_p(dest, value, target_pe):
 *   Writes 'value' to address 'dest' on GPU 'target_pe'.
 *   NVSHMEM handles the NVLink transfer internally.
 *   'dest' must be in the symmetric heap (allocated via nvshmem_malloc).
 * ========================================================================== */
__global__ void transpose_nvshmem_direct_kernel(
    double * __restrict__ B_sym,  /* symmetric heap pointer to B            */
    int dst_ld,                   /* stride of B = order                    */
    int dst_row_off,              /* row offset in target's B               */
    double * __restrict__ A_sym,  /* symmetric heap pointer to my A         */
    int src_ld,                   /* stride of A = order                    */
    int src_row_off,              /* which sub-block of A to read           */
    int Bo,                       /* block order                            */
    int target_pe)                /* which GPU to write to                  */
{
    __shared__ double tile[TILE][TILE + 1];

    /* Load tile from local A into shared memory */
    int s_col = blockIdx.x * TILE + threadIdx.y;
    int s_row = blockIdx.y * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (s_row < Bo && (s_col + j) < Bo) {
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = A_sym[idx];
            A_sym[idx] += 1.0;
        }
    }

    __syncthreads();

    /* Write transposed tile directly to target PE's B using nvshmem */
    int d_col = blockIdx.y * TILE + threadIdx.y;
    int d_row = blockIdx.x * TILE + threadIdx.x;

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + dst_row_off) + (size_t)dst_ld * (d_col + j);
            double val = tile[threadIdx.x][threadIdx.y + j];
            /* nvshmem_double_p: write one double to remote PE's B_sym[idx]
               NVSHMEM may batch/optimize these writes internally */
            double old = nvshmem_double_g(&B_sym[idx], target_pe);
            nvshmem_double_p(&B_sym[idx], old + val, target_pe);
        }
    }
}

/* ==========================================================================
 * KERNEL: unpack_add_kernel
 * ========================================================================== */
__global__ void unpack_add_kernel(
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
            B[(size_t)(row + row_off) + (size_t)lda * (col + j)] +=
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
    /* --- MPI init first, then NVSHMEM on top of MPI --- */
    MPI_Init(&argc, &argv);

    int my_PE, P;   /* PE = "processing element" = NVSHMEM's term for rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* --- Initialize NVSHMEM using MPI as the bootstrap ---
       This tells NVSHMEM to use MPI for its initial setup
       (discovering GPUs, establishing connections). After init,
       NVSHMEM handles all GPU-to-GPU communication itself. */
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    /* Verify NVSHMEM agrees with MPI on rank/size */
    int nvshmem_pe = nvshmem_my_pe();       /* my NVSHMEM rank               */
    int nvshmem_npes = nvshmem_n_pes();     /* total NVSHMEM ranks           */

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

    /* --- GPU setup --- */
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(my_PE % ndev));

    int Bo       = order / P;
    int colstart = Bo * my_PE;
    size_t col_elems = (size_t)order * Bo;
    size_t blk_elems = (size_t)Bo * Bo;
    size_t bytes     = 2ULL * sizeof(double) * order * order;

    const char *mode_str[] = {"NVSHMEM (direct)", "NVSHMEM (buffered)"};
    if (my_PE == 0) {
        printf("GPU Matrix transpose: B = A^T\n");
        printf("Communication mode: %s\n", mode_str[COMM_MODE]);
        printf("Number of GPUs       = %d\n", P);
        printf("Matrix order         = %d\n", order);
        printf("Block order          = %d\n", Bo);
        printf("Number of iterations = %d\n", iterations);
    }

    /* --- Allocate A and B from NVSHMEM symmetric heap ---
       nvshmem_malloc: every PE calls this with the same size.
       The returned pointer is a "symmetric address" — it's valid
       on all PEs. When PE 0 writes to B_sym[100] on PE 2,
       NVSHMEM knows exactly where that is on PE 2's GPU. */
    double *A_sym = (double *)nvshmem_malloc(col_elems * sizeof(double));
    double *B_sym = (double *)nvshmem_malloc(col_elems * sizeof(double));
    if (!A_sym || !B_sym) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", my_PE);
        nvshmem_finalize(); MPI_Finalize(); return 1;
    }

    /* --- Allocate send/recv buffers for buffered mode ---
       Also from symmetric heap so nvshmem_putmem can target them. */
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

    /* --- CUDA stream --- */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* --- Init matrices --- */
    {
        dim3 blk(16, 16);
        dim3 grd((Bo + 15) / 16, (order + 15) / 16);
        init_kernel<<<grd, blk, 0, stream>>>(A_sym, B_sym, Bo, order, colstart);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    dim3 tblk(TILE, BROWS);
    dim3 tgrd((Bo + TILE - 1) / TILE, (Bo + TILE - 1) / TILE);

    /* Sync all PEs before timing */
    nvshmem_barrier_all();

    /* ==================================================================
     * MAIN LOOP
     * ================================================================== */
    double t0 = 0.0;

    for (int iter = 0; iter <= iterations; iter++) {

        if (iter == 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();
            t0 = MPI_Wtime();
        }

        /* Phase 0: local transpose (same as all other versions) */
        transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
            B_sym, order, colstart,
            A_sym, order, colstart,
            Bo, 1);

        /* Phases 1..P-1: remote */
        for (int phase = 1; phase < P; phase++) {
            int send_to   = (my_PE - phase + P) % P;
            int recv_from = (my_PE + phase)     % P;

#if COMM_MODE == 0
            /* ---- NVSHMEM DIRECT ----
               Kernel reads from local A, transposes in shared memory,
               writes each element to target PE's B using nvshmem_double_p.
               NVSHMEM handles NVLink routing internally.
               nvshmem_barrier_all replaces MPI_Barrier for sync. */
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            transpose_nvshmem_direct_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, colstart,      /* dst: target's B              */
                A_sym, order, send_to * Bo,  /* src: my A sub-block          */
                Bo, send_to);                /* target PE                    */

            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

#elif COMM_MODE == 1
            /* ---- NVSHMEM BUFFERED ----
               Pack into local send_buf, then bulk transfer to peer's
               recv_buf using nvshmemx_putmem_nbi_on_stream.
               
               nbi = non-blocking, returns immediately.
               on_stream = ordered with kernel on same stream.
               nvshmem_quiet = wait for all outstanding puts to complete. */

            /* Pack + transpose into local send_buf */
            transpose_local_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_sym, order, send_to * Bo,
                Bo, 0);

            /* Bulk transfer send_buf → peer's recv_buf.
               This is the NVSHMEM equivalent of cudaMemcpyAsync to peer.
               NVSHMEM uses its internal transport (NVLink/IB) optimally. */
            nvshmemx_putmem_nbi_on_stream(
                recv_buf,                    /* dst on target PE              */
                send_buf,                    /* src on my PE                  */
                blk_elems * sizeof(double),  /* size in bytes                 */
                send_to,                     /* target PE                     */
                stream);                     /* CUDA stream for ordering      */

            /* Ensure put completes, then sync with all PEs */
            nvshmemx_quiet_on_stream(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvshmem_barrier_all();

            /* Unpack recv_buf into B */
            unpack_add_kernel<<<tgrd, tblk, 0, stream>>>(
                B_sym, order, recv_from * Bo, recv_buf, Bo);
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /* --- Timing --- */
    double local_time = MPI_Wtime() - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* --- Verification --- */
    double *B_h = (double *)malloc(col_elems * sizeof(double));
    CUDA_CHECK(cudaMemcpy(B_h, B_sym, col_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double abserr = 0.0;
    double addit = ((double)(iterations + 1) * (double)iterations) / 2.0;
    for (size_t j = 0; j < (size_t)Bo; j++)
        for (size_t i = 0; i < (size_t)order; i++) {
            double expected = (double)((double)order * i + j + colstart)
                              * (iterations + 1) + addit;
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
