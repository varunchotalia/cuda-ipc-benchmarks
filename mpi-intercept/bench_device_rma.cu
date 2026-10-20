// bench_device_rma.cu - overhead of MPIX_Put/MPIX_Get over raw pointer access
//
//   LD_PRELOAD=$MPIWRAP_LIB mpirun -np 1 ./bench_device_rma
//
// Every rank targets rank+1, so np=1 measures the same device (the target is
// my own window). Each case runs three ways:
//
//   raw  copy through the pointer from host MPI_Win_shared_query: no lookup,
//        no checks. Scope fixed at compile time.
//   ct   MPIX_Put / MPIX_Get with the scope as a compile-time constant
//   rt   the same calls with the scope passed in as a kernel argument
//
// raw and the MPIX calls use the same copy loop (mpix_copy), so ct - raw is
// the cost of the handle lookup and argument checks, and rt - ct is the cost
// of not knowing the scope at compile time.
//
// Timing: thread 0's clock64() around ITERS back-to-back operations, after a
// warmup launch; the median of TRIALS launches. Cycles are the primary unit;
// ns use the SM clock attribute, which may differ from the clock actually run.
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "mpiw_device.cuh"

#define NSLOTS 64     // operations cycle through this many slots of the window
#define MAXB   4096   // largest message, bytes
#define ITERS  1000
#define TRIALS 7

enum { RAW = 0, CT = 1, RT = 2 };
enum { OP_PUT = 0, OP_PUT_FLUSH = 1, OP_GET_FLUSH = 2 };
static const char* OP_NAME[]     = {"put", "put+flush", "get+flush"};
static const char* SCOPE_NAME[]  = {"thread", "warp", "block"};
static const int   SCOPE_THREADS[] = {1, 32, 256};

struct Args {
    char*      raw_peer;   // target window base from host MPI_Win_shared_query
    MPI_Win    dwin;       // device handle
    char*      local;      // origin buffer, MAXB bytes
    int        bytes;
    int        target;
    int        scope_rt;   // only read by the RT kernels
    long long* cycles;     // out: thread 0's cycles for ITERS operations
    int*       rc;         // out: OR of all MPIX return codes
};

template <int METHOD, int OP, int S>
__global__ void bench(Args a)
{
    const int scope = (METHOD == RT) ? a.scope_rt : S;   // RT: opaque to the compiler
    unsigned me, n;
    mpix_scope_rank(S, &me, &n);                          // used by raw only
    const int count = a.bytes / 8;
    int rc = 0;

    __syncthreads();
    long long t0 = clock64();
    for (int it = 0; it < ITERS; it++) {
        MPI_Aint disp = (MPI_Aint)(it % NSLOTS) * (MAXB / 8);   // disp_unit = 8
        if (METHOD == RAW) {
            char* peer = a.raw_peer + disp * 8;
            if (OP == OP_GET_FLUSH) mpix_copy(a.local, peer, a.bytes, me, n);
            else                    mpix_copy(peer, a.local, a.bytes, me, n);
            if (OP != OP_PUT) {     // the same steps MPIX_Win_flush_all takes
                __threadfence_system();
                if (S == MPIX_SCOPE_WARP)  __syncwarp();
                if (S == MPIX_SCOPE_BLOCK) __syncthreads();
            }
        } else {
            if (OP == OP_GET_FLUSH)
                rc |= MPIX_Get(a.local, count, MPIX_ABI_DOUBLE, a.target, disp,
                               count, MPIX_ABI_DOUBLE, a.dwin, scope);
            else
                rc |= MPIX_Put(a.local, count, MPIX_ABI_DOUBLE, a.target, disp,
                               count, MPIX_ABI_DOUBLE, a.dwin, scope);
            if (OP != OP_PUT) rc |= MPIX_Win_flush_all(a.dwin, scope);
        }
    }
    __syncthreads();
    long long t1 = clock64();

    if (rc) atomicOr(a.rc, rc);
    if (threadIdx.x == 0) *a.cycles = t1 - t0;
}

// Pick the instantiation for (method, op, scope) at runtime.
template <int M, int O, int S> static void go(const Args& a) {
    bench<M, O, S><<<1, SCOPE_THREADS[S]>>>(a);
}
template <int M, int O> static void go_s(int s, const Args& a) {
    if (s == 0) go<M, O, 0>(a); else if (s == 1) go<M, O, 1>(a); else go<M, O, 2>(a);
}
template <int M> static void go_o(int o, int s, const Args& a) {
    if (o == 0) go_s<M, 0>(s, a); else if (o == 1) go_s<M, 1>(s, a); else go_s<M, 2>(s, a);
}
static void launch(int m, int o, int s, const Args& a) {
    if (m == RAW) go_o<RAW>(o, s, a); else if (m == CT) go_o<CT>(o, s, a); else go_o<RT>(o, s, a);
}

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "rank %d: %s failed: %s\n", rank, #call, cudaGetErrorString(e_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, lrank, ndev;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &lrank);
    CK(cudaGetDeviceCount(&ndev));
    const int dev = lrank % ndev;
    CK(cudaSetDevice(dev));
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));

    const size_t win_bytes = (size_t)NSLOTS * MAXB;
    char* win_buf;
    CK(cudaMalloc((void**)&win_buf, win_bytes));
    CK(cudaMemset(win_buf, 0, win_bytes));
    MPI_Win win;
    MPI_Win_create(win_buf, win_bytes, 8, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Info info;
    char s[16];
    snprintf(s, sizeof s, "%d", dev);
    MPI_Info_create(&info);
    MPI_Info_set(info, "device_id", s);
    MPI_Win dwin;
    if (MPIX_Win_get_device_handle(win, info, &dwin) != MPI_SUCCESS) {
        fprintf(stderr, "rank %d: MPIX_Win_get_device_handle failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Info_free(&info);

    const int target = (rank + 1) % size;
    void* raw_peer = nullptr;
    MPI_Aint qsize; int qdisp;
    if (MPI_Win_shared_query(win, target, &qsize, &qdisp, &raw_peer) != MPI_SUCCESS) {
        fprintf(stderr, "rank %d: target %d unreachable\n", rank, target);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Args a;
    a.raw_peer = (char*)raw_peer;
    a.dwin = dwin;
    a.target = target;
    CK(cudaMalloc((void**)&a.local, MAXB));
    CK(cudaMemset(a.local, 1, MAXB));
    CK(cudaMalloc((void**)&a.cycles, sizeof(long long)));
    CK(cudaMalloc((void**)&a.rc, sizeof(int)));
    CK(cudaMemset(a.rc, 0, sizeof(int)));

    if (rank == 0)
        printf("# node %s GPU %d, np=%d, target %s, SM clock %.0f MHz, %d iters x %d trials (median)\n"
               "# %-9s %-6s %5s %9s %9s %9s %9s %9s %9s\n",
               getenv("SLURMD_NODENAME") ? getenv("SLURMD_NODENAME") : "?", dev, size,
               target == rank ? "self (same device)" : "rank+1", clock_khz / 1e3,
               ITERS, TRIALS, "op", "scope", "bytes",
               "raw_cyc", "ct_cyc", "rt_cyc", "raw_ns", "ct-raw", "rt-ct");

    MPI_Win_lock_all(0, win);
    const int sizes[] = {8, 64, 512, 4096};
    for (int o = 0; o < 3; o++)
    for (int sc = 0; sc < 3; sc++)
    for (int b : sizes) {
        double cyc[3];
        for (int m = 0; m < 3; m++) {
            a.bytes = b;
            a.scope_rt = sc;
            long long t[TRIALS];
            launch(m, o, sc, a);                      // warmup
            CK(cudaDeviceSynchronize());
            for (int k = 0; k < TRIALS; k++) {
                launch(m, o, sc, a);
                CK(cudaGetLastError());
                CK(cudaMemcpy(&t[k], a.cycles, sizeof t[k], cudaMemcpyDeviceToHost));
            }
            std::sort(t, t + TRIALS);
            cyc[m] = (double)t[TRIALS / 2] / ITERS;
        }
        const double ns_per_cyc = 1e6 / clock_khz;
        if (rank == 0)
            printf("RESULT %-9s %-6s %5d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n",
                   OP_NAME[o], SCOPE_NAME[sc], b, cyc[RAW], cyc[CT], cyc[RT],
                   cyc[RAW] * ns_per_cyc, cyc[CT] - cyc[RAW], cyc[RT] - cyc[CT]);
    }
    MPI_Win_unlock_all(win);

    int h_rc = 0;
    CK(cudaMemcpy(&h_rc, a.rc, sizeof h_rc, cudaMemcpyDeviceToHost));
    if (h_rc) printf("rank %d: ERROR: an MPIX call returned %d; timings are invalid\n", rank, h_rc);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);
    cudaFree(a.local); cudaFree(a.cycles); cudaFree(a.rc); cudaFree(win_buf);
    MPI_Comm_free(&node);
    MPI_Finalize();
    return h_rc ? 1 : 0;
}
