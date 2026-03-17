/*****************************************************************************
 * transpose_cuda_ipc.cu
 *
 * Matrix transpose: B = A^T
 * One MPI rank per GPU. Matrix distributed by column blocks.
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
#define COMM_MODE 1
#endif

/* Macro to check every CUDA call for errors and abort if one fails */
#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(_e));                                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                      \
} while(0)

#define TILE 32    /* Width/height of one tile (matches warp size for coalescing) */
#define BROWS 8    /* Rows of threads per block. 32*8=256 threads per block.
                      Each thread handles TILE/BROWS = 4 elements via the loop. */

/* ==========================================================================
 * KERNEL: transpose_kernel
 *
 * This ONE kernel handles all cases:
 *   - Local transpose   (dst = my B,       dst_ld = order)
 *   - IPC direct write  (dst = peer's B,   dst_ld = order)
 *   - Pack into buffer  (dst = send_buf,   dst_ld = Bo)
 *
 * Parameters:
 *   dst         — pointer to destination (my B, peer's B, or send_buf)
 *   dst_ld      — stride of destination (order for B, Bo for send_buf)
 *   dst_row_off — row offset in destination (where to start writing)
 *   src         — pointer to source (always my A)
 *   src_ld      — stride of source (always order)
 *   src_row_off — row offset in source (which sub-block to read)
 *   Bo          — block order: width and height of the sub-block
 *   accumulate  — 1 means dst += (writing into B), 0 means dst = (packing)
 * ========================================================================== */
__global__ void transpose_kernel(
    double * __restrict__ dst,    /* where to write (B or send_buf)            */
    int dst_ld,                   /* stride of dst (distance between columns)  */
    int dst_row_off,              /* row offset: skip this many rows in dst    */
    double * __restrict__ src,    /* where to read (always A)                  */
    int src_ld,                   /* stride of src (distance between columns)  */
    int src_row_off,              /* row offset: skip this many rows in src    */
    int Bo,                       /* block order: sub-block is Bo × Bo         */
    int accumulate)               /* 1 = += into dst, 0 = overwrite dst        */
{
    /* Scratchpad: 32×33 doubles, on-chip, shared by all 256 threads.
       +1 padding prevents bank conflicts when reading transposed. */
    __shared__ double tile[TILE][TILE + 1];

    /* --- Where to read from src ---
       blockIdx picks which 32×32 tile.
       threadIdx picks which element within the tile.
       threadIdx.x maps to rows so adjacent threads read adjacent addresses. */
    int s_col = blockIdx.x * TILE + threadIdx.y;   /* column within sub-block */
    int s_row = blockIdx.y * TILE + threadIdx.x;   /* row within sub-block    */

    /* Load tile from src into scratchpad.
       Loop runs 4 times (TILE/BROWS = 32/8 = 4) because we have 8 rows
       of threads but 32 rows of data. Each thread loads 4 elements. */
    #pragma unroll
    for (int j = 0;               /* j steps through column offsets            */
         j < TILE;                /* until we've covered all 32 columns        */
         j += BROWS) {            /* jump by 8 (number of thread rows)         */
        if (s_row < Bo && (s_col + j) < Bo) {   /* bounds check               */
            /* Column-major index: row + stride * col */
            size_t idx = (size_t)(s_row + src_row_off) + (size_t)src_ld * (s_col + j);
            tile[threadIdx.y + j][threadIdx.x] = src[idx];  /* global → shared */
            src[idx] += 1.0;     /* increment A for verification (ignore this) */
        }
    }

    /* All 256 threads must finish loading before anyone reads the tile */
    __syncthreads();

    /* --- Where to write in dst ---
       blockIdx.x and blockIdx.y are SWAPPED compared to the read.
       This is the transpose: what was column-tile becomes row-tile. */
    int d_col = blockIdx.y * TILE + threadIdx.y;   /* note: blockIdx.y not .x */
    int d_row = blockIdx.x * TILE + threadIdx.x;   /* note: blockIdx.x not .y */

    /* Write tile from scratchpad to dst.
       tile indices are swapped: we wrote tile[col][row], we read tile[row][col].
       That swap IS the transpose. */
    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (d_row < Bo && (d_col + j) < Bo) {
            size_t idx = (size_t)(d_row + dst_row_off) + (size_t)dst_ld * (d_col + j);
            if (accumulate)
                dst[idx] += tile[threadIdx.x][threadIdx.y + j];  /* += into B  */
            else
                dst[idx]  = tile[threadIdx.x][threadIdx.y + j];  /* = into buf */
        }
    }
}

/* ==========================================================================
 * KERNEL: unpack_add_kernel
 *
 * After receiving data, scatter-add from dense recv_buf into strided B.
 * No transpose — the sender already did that during packing.
 *
 * Parameters:
 *   B       — destination matrix (strided, stride = lda)
 *   lda     — stride of B (= order, full matrix height)
 *   row_off — which rows in B this data belongs to (recv_from * Bo)
 *   buf     — source buffer (dense, stride = Bo)
 *   Bo      — block order
 * ========================================================================== */
__global__ void unpack_add_kernel(
    double * __restrict__ B,         /* destination: my B matrix               */
    int lda,                         /* stride of B = order                    */
    int row_off,                     /* row offset: where in B to scatter      */
    const double * __restrict__ buf, /* source: dense recv buffer              */
    int Bo)                          /* block order: buffer is Bo × Bo         */
{
    int col = blockIdx.x * TILE + threadIdx.y;  /* column in sub-block        */
    int row = blockIdx.y * TILE + threadIdx.x;  /* row in sub-block           */

    #pragma unroll
    for (int j = 0; j < TILE; j += BROWS) {
        if (row < Bo && (col + j) < Bo) {
            /* Read from buf (dense, stride Bo), add into B (strided, stride lda) */
            B[(size_t)(row + row_off) + (size_t)lda * (col + j)] +=
                buf[(size_t)row + (size_t)Bo * (col + j)];
        }
    }
}

/* ==========================================================================
 * KERNEL: init_kernel
 *
 * Fill A with known values, zero out B.
 * A(i,j) = order * (j + colstart) + i
 * B(i,j) = 0.0
 * ========================================================================== */
__global__ void init_kernel(
    double *A,      /* matrix to fill with known values                       */
    double *B,      /* matrix to zero out                                     */
    int Bo,         /* block order: number of columns this GPU owns           */
    int lda,        /* leading dimension = order (number of rows)             */
    int colstart)   /* global column index of this GPU's first column         */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* local column        */
    int row = blockIdx.y * blockDim.y + threadIdx.y;   /* row                 */
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
    int my_ID;  /* This GPU's rank (0, 1, 2, or 3)                           */
    int P;      /* Total number of GPUs                                       */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* --- Parse command line: <iterations> <matrix_order> --- */
    if (argc != 3) {
        if (my_ID == 0) fprintf(stderr, "Usage: %s <iterations> <matrix_order>\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    int iterations = atoi(argv[1]);  /* how many times to repeat transpose    */
    int order      = atoi(argv[2]);  /* full matrix is order × order          */
    if (order % P != 0) {
        if (my_ID == 0) fprintf(stderr, "ERROR: order must be divisible by num GPUs\n");
        MPI_Finalize(); return 1;
    }

    /* --- GPU setup: assign one GPU per MPI rank --- */
    int ndev;  /* how many GPUs are on this machine                           */
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(my_ID % ndev));  /* rank 0 → GPU 0, rank 1 → GPU 1, etc */

    int Bo       = order / P;       /* block order: columns per GPU           */
    int colstart = Bo * my_ID;      /* first global column this GPU owns      */
    size_t col_elems = (size_t)order * Bo;  /* elements in A (or B) per GPU   */
    size_t blk_elems = (size_t)Bo * Bo;     /* elements in one Bo×Bo sub-block*/
    size_t bytes = 2ULL * sizeof(double) * order * order; /* total data for rate calc */

    const char *mode_str[] = {
        "CUDA IPC (direct)",
        "CUDA IPC (buffered+stream)",
        "GPU-aware MPI",
        "Staged MPI"
    };
    if (my_ID == 0) {
        printf("GPU Matrix transpose: B = A^T\n");
        printf("Communication mode: %s\n", mode_str[COMM_MODE]);
        printf("Number of GPUs       = %d\n", P);
        printf("Matrix order         = %d\n", order);
        printf("Block order          = %d\n", Bo);
        printf("Number of iterations = %d\n", iterations);
    }

    /* --- Allocate A and B on this GPU --- */
    double *A_d;   /* this GPU's chunk of the original matrix                 */
    double *B_d;   /* this GPU's chunk of the transposed result               */
    CUDA_CHECK(cudaMalloc(&A_d, col_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&B_d, col_elems * sizeof(double)));

    /* --- Create a CUDA stream ---
       A stream is an ordered queue of GPU operations.
       Operations on the same stream execute in order.
       This lets us put pack_kernel then cudaMemcpyAsync on the same
       stream — the memcpy auto-waits for the kernel without a sync. */
    cudaStream_t stream;  /* our one stream for all GPU work                  */
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* ==================================================================
     * IPC DIRECT (mode 0): get pointer to every peer's full B matrix.
     * No buffers needed — kernel writes directly into peer's B.
     * ================================================================== */
    double **peer_B = NULL;  /* array of pointers: peer_B[p] → GPU p's B     */

#if COMM_MODE == 0
    if (P > 1) {
        /* IPC handle: small struct (~64 bytes) that describes how to access
           a GPU memory allocation from another GPU */
        cudaIpcMemHandle_t my_handle;  /* handle describing MY B_d            */
        CUDA_CHECK(cudaIpcGetMemHandle(&my_handle, B_d));

        /* Collect everyone's handle. After this, all_handles[p] describes
           GPU p's B matrix */
        cudaIpcMemHandle_t *all_handles =  /* array of handles from all GPUs  */
            (cudaIpcMemHandle_t *)malloc(P * sizeof(cudaIpcMemHandle_t));
        MPI_Allgather(&my_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      all_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                      MPI_COMM_WORLD);

        /* Open each handle to get a usable device pointer */
        peer_B = (double **)malloc(P * sizeof(double *));
        for (int p = 0; p < P; p++) {
            if (p == my_ID)
                peer_B[p] = B_d;          /* I already have my own pointer    */
            else
                /* Turn the handle into a pointer I can use in kernels.
                   This pointer goes to GPU p's memory over NVLink. */
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    (void **)&peer_B[p],   /* output: usable pointer          */
                    all_handles[p],        /* input: handle from GPU p         */
                    cudaIpcMemLazyEnablePeerAccess));
        }
        free(all_handles);
        if (my_ID == 0) printf("IPC handles exchanged — direct B access\n");
    }
#endif

    /* ==================================================================
     * IPC BUFFERED (mode 1): get pointer to every peer's recv_buf.
     * Pack → cudaMemcpyAsync on same stream → unpack.
     * ================================================================== */
    double *send_buf = NULL;   /* local buffer: kernel packs transposed data here */
    double *recv_buf = NULL;   /* local buffer: peer writes into this via IPC     */
    double **peer_recv = NULL; /* array of pointers: peer_recv[p] → GPU p's recv_buf */

#if COMM_MODE == 1
    if (P > 1) {
        CUDA_CHECK(cudaMalloc(&send_buf, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&recv_buf, blk_elems * sizeof(double)));

        /* Same IPC handle exchange, but for recv_buf instead of B */
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

    /* ==================================================================
     * MPI MODES (2, 3): just need local send/recv buffers, no IPC
     * ================================================================== */
#if COMM_MODE >= 2
    if (P > 1) {
        CUDA_CHECK(cudaMalloc(&send_buf, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&recv_buf, blk_elems * sizeof(double)));
    }
#endif
    double *host_send = NULL;  /* pinned CPU memory for staged D2H copy       */
    double *host_recv = NULL;  /* pinned CPU memory for staged H2D copy       */
#if COMM_MODE == 3
    if (P > 1) {
        /* cudaMallocHost = pinned memory. Faster for CPU↔GPU copies because
           it can't be swapped to disk by the OS. */
        CUDA_CHECK(cudaMallocHost(&host_send, blk_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&host_recv, blk_elems * sizeof(double)));
    }
#endif

    /* --- Init matrices on GPU --- */
    {
        dim3 blk(16, 16);         /* 256 threads per block for init           */
        dim3 grd((Bo + 15) / 16,  /* enough blocks to cover Bo columns        */
                 (order + 15) / 16); /* enough blocks to cover order rows      */
        init_kernel<<<grd, blk>>>(A_d, B_d, Bo, order, colstart);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* --- Kernel launch config for transpose --- */
    dim3 tblk(TILE, BROWS);   /* 32 × 8 = 256 threads per block              */
    dim3 tgrd(
        (Bo + TILE - 1) / TILE,  /* number of tiles across (columns)          */
        (Bo + TILE - 1) / TILE); /* number of tiles down (rows)               */
    /* Example: Bo=4096 → 128 × 128 = 16384 thread blocks                    */

    MPI_Barrier(MPI_COMM_WORLD);  /* everyone ready before timing starts      */

    /* ==================================================================
     * MAIN LOOP
     * ================================================================== */
    double t0 = 0.0;  /* start time (set after warmup iteration)              */

    for (int iter = 0;             /* iteration counter                       */
         iter <= iterations;       /* one extra warmup iteration (iter 0)     */
         iter++) {

        /* Start timer after warmup iteration completes */
        if (iter == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();     /* MPI wall clock time                      */
        }

        /* --- Phase 0: local transpose (my A → my B, no communication) ---
           src_row_off = colstart: read rows colstart..colstart+Bo from A
           dst_row_off = colstart: write to same row range in B
           This is the diagonal block — data is already on this GPU. */
        transpose_kernel<<<tgrd, tblk, 0, stream>>>(
            B_d, order, colstart,       /* dst: my B, stride=order            */
            A_d, order, colstart,       /* src: my A, stride=order            */
            Bo, 1);                     /* accumulate: B += ...               */

        /* --- Phases 1..P-1: exchange with each other GPU --- */
        for (int phase = 1;        /* phase counter (skip 0, that was local)  */
             phase < P;            /* one phase per remote peer               */
             phase++) {

            int send_to = (my_ID - phase + P) % P;  /* who I send data to    */
            int recv_from = (my_ID + phase) % P;     /* who sends data to me  */

#if COMM_MODE == 0
            /* ---- IPC DIRECT ----
               One kernel reads from my A and writes straight to peer's B.
               Two barriers: first ensures peer is ready to be written to,
               second ensures my write is visible before peer reads it. */
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                peer_B[send_to],         /* dst: peer's B matrix via IPC      */
                order,                   /* dst stride: same as any B         */
                colstart,                /* dst row offset: my cols → their rows */
                A_d,                     /* src: my A matrix                  */
                order,                   /* src stride                        */
                send_to * Bo,            /* src row offset: the sub-block     */
                Bo, 1);                  /* accumulate: B += ...              */

            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);

#elif COMM_MODE == 1
            /* ---- IPC BUFFERED + STREAM ----
               Pack and memcpy on the SAME stream. The stream guarantees
               the memcpy starts only after the pack kernel finishes.
               No sync needed between them — just one sync + barrier at the end. */

            /* Pack: transpose from A into dense send_buf */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf,                /* dst: local send buffer            */
                Bo,                      /* dst stride: Bo (dense, no gaps)   */
                0,                       /* dst row offset: 0 (buffer starts at 0) */
                A_d,                     /* src: my A                         */
                order,                   /* src stride                        */
                send_to * Bo,            /* src row offset: sub-block to send */
                Bo, 0);                  /* don't accumulate: buf = ...       */

            /* Bulk transfer to peer's recv_buf.
               Async on same stream → auto-waits for pack kernel.
               Uses DMA engine → full NVLink bandwidth. */
            cudaMemcpyAsync(
                peer_recv[send_to],      /* dst: peer's recv_buf via IPC      */
                send_buf,                /* src: my local send_buf            */
                blk_elems * sizeof(double),
                cudaMemcpyDeviceToDevice,
                stream);                 /* SAME stream as the kernel above   */

            /* One sync point: wait for both kernel and memcpy to finish */
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD); /* ensure peer's recv_buf is ready   */

            /* Unpack: scatter-add from my recv_buf into B */
            unpack_add_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d,                     /* dst: my B                         */
                order,                   /* B's stride                        */
                recv_from * Bo,          /* row offset in B for this data     */
                recv_buf,                /* src: my recv_buf (peer wrote here)*/
                Bo);                     /* buffer stride                     */

#elif COMM_MODE == 2
            /* ---- GPU-AWARE MPI ----
               Same as buffered but MPI handles the transfer. */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo, 0);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            /* MPI takes device pointers directly, handles transfer internally */
            MPI_Sendrecv(
                send_buf, (int)blk_elems, MPI_DOUBLE, send_to,   phase,
                recv_buf, (int)blk_elems, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            unpack_add_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo);

#elif COMM_MODE == 3
            /* ---- STAGED MPI ----
               GPU → pinned host → MPI → pinned host → GPU. Slow. */
            transpose_kernel<<<tgrd, tblk, 0, stream>>>(
                send_buf, Bo, 0,
                A_d, order, send_to * Bo,
                Bo, 0);
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

            unpack_add_kernel<<<tgrd, tblk, 0, stream>>>(
                B_d, order, recv_from * Bo, recv_buf, Bo);
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(stream)); /* finish all phases       */
    }

    /* --- Timing --- */
    double local_time = MPI_Wtime() - t0;  /* elapsed time on this GPU        */
    double max_time;  /* slowest GPU's time (determines overall performance)   */
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* --- Verification: copy B to host and check against expected values --- */
    double *B_h = (double *)malloc(col_elems * sizeof(double)); /* host copy of B */
    CUDA_CHECK(cudaMemcpy(B_h, B_d, col_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double abserr = 0.0;  /* sum of absolute errors on this GPU               */
    /* addit accounts for the += 1.0 that happens each iteration in the kernel */
    double addit = ((double)(iterations + 1) * (double)iterations) / 2.0;
    for (size_t j = 0; j < (size_t)Bo; j++)        /* each local column       */
        for (size_t i = 0; i < (size_t)order; i++) {  /* each row             */
            double expected = (double)((double)order * i + j + colstart)
                              * (iterations + 1) + addit;
            abserr += fabs(B_h[i + (size_t)order * j] - expected);
        }

    double abserr_tot;  /* global error summed across all GPUs                 */
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
