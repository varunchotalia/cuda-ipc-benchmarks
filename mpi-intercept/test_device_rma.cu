// test_device_rma.cu - device-initiated Put/Get through WinIPC's device handle
//
//   LD_PRELOAD=$MPIWRAP_LIB mpirun -np 4 ./test_device_rma
//
// Each rank puts into rank+1 from a kernel, once per scope (thread, warp,
// block), then gets back what it put. A target this GPU cannot reach (e.g.
// across the NVLink islands of h200x8-04) must fail with MPI_ERR_OTHER; that
// is reported as SKIP, not FAIL. Ends with one "RESULT" line from rank 0.
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mpiw_device.cuh"

#define K 1024   // doubles per region

// Window layout: region (test, src) starts at disp (test * size + src) * K.
enum { T_PUT_THREAD, T_PUT_WARP, T_PUT_BLOCK, NPUT };
static const int   SCOPE[NPUT]   = {MPIX_SCOPE_THREAD, MPIX_SCOPE_WARP, MPIX_SCOPE_BLOCK};
static const int   THREADS[NPUT] = {1, 32, 256};
static const char* NAME[NPUT]    = {"thread", "warp", "block"};

static double val(int test, int src, int i) { return src * 100000.0 + test * 10000.0 + i; }

__global__ void put_kernel(const double* src, int target, MPI_Aint disp,
                           MPI_Win dwin, int scope, int* rc) {
    int r = MPIX_Put(src, K, MPIX_ABI_DOUBLE, target, disp, K, MPIX_ABI_DOUBLE, dwin, scope);
    int f = MPIX_Win_flush_all(dwin, scope);
    if (threadIdx.x == 0) { rc[0] = r; rc[1] = f; }
}

__global__ void get_kernel(double* dst, int target, MPI_Aint disp,
                           MPI_Win dwin, int scope, int* rc) {
    int r = MPIX_Get(dst, K, MPIX_ABI_DOUBLE, target, disp, K, MPIX_ABI_DOUBLE, dwin, scope);
    int f = MPIX_Win_flush_all(dwin, scope);
    if (threadIdx.x == 0) { rc[0] = r; rc[1] = f; }
}

// Bad arguments, one call each, from a single thread.
// Targets myself: always reachable, so each call fails only for its own reason.
__global__ void error_kernel(MPI_Win dwin, int t, int size, MPI_Aint win_elems,
                             MPI_Datatype host_double, int* rc) {
    double x = 0;
    rc[0] = MPIX_Put(&x, 1, MPIX_ABI_DOUBLE, size, 0, 1, MPIX_ABI_DOUBLE, dwin, MPIX_SCOPE_THREAD);
    rc[1] = MPIX_Put(&x, 1, host_double, t, 0, 1, host_double, dwin, MPIX_SCOPE_THREAD);
    rc[2] = MPIX_Put(&x, 1, MPIX_ABI_DOUBLE, t, 0, 1, MPIX_ABI_INT64_T, dwin, MPIX_SCOPE_THREAD);
    rc[3] = MPIX_Put(&x, 1, MPIX_ABI_DOUBLE, t, 0, 2, MPIX_ABI_DOUBLE, dwin, MPIX_SCOPE_THREAD);
    rc[4] = MPIX_Put(&x, 1, MPIX_ABI_DOUBLE, t, win_elems, 1, MPIX_ABI_DOUBLE, dwin, MPIX_SCOPE_THREAD);
    rc[5] = MPIX_Put(&x, 1, MPIX_ABI_DOUBLE, t, 0, 1, MPIX_ABI_DOUBLE, dwin, 7);
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
    int dev = lrank % ndev;
    if (size > 64) {   // all_rc below holds 64 ranks
        if (rank == 0) fprintf(stderr, "run with at most 64 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CK(cudaSetDevice(dev));

    int fails = 0, skips = 0;
    auto check = [&](bool ok, const char* what, int got, int want) {
        if (!ok) { fails++; printf("rank %d: FAIL %s: got %d, want %d\n", rank, what, got, want); }
    };

    const size_t elems = (size_t)NPUT * size * K;
    double* win_buf;
    CK(cudaMalloc((void**)&win_buf, elems * sizeof(double)));
    CK(cudaMemset(win_buf, 0, elems * sizeof(double)));
    MPI_Win win;
    MPI_Win_create(win_buf, elems * sizeof(double), sizeof(double), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);

    // ---- host-side checks of MPIX_Win_get_device_handle ----
    MPI_Win dwin = MPI_WIN_NULL, dwin2 = MPI_WIN_NULL;
    int rc = MPIX_Win_get_device_handle(win, MPI_INFO_NULL, &dwin);
    check(rc == MPI_ERR_INFO_NOKEY, "no device_id", rc, MPI_ERR_INFO_NOKEY);

    MPI_Info info;
    char s[16];
    MPI_Info_create(&info);
    snprintf(s, sizeof s, "%d", dev + 1);
    MPI_Info_set(info, "device_id", s);
    rc = MPIX_Win_get_device_handle(win, info, &dwin);
    check(rc == MPI_ERR_INFO_VALUE, "wrong device_id", rc, MPI_ERR_INFO_VALUE);
    MPI_Info_set(info, "device_id", "abc");
    rc = MPIX_Win_get_device_handle(win, info, &dwin);
    check(rc == MPI_ERR_INFO_VALUE, "non-numeric device_id", rc, MPI_ERR_INFO_VALUE);

    char* host_mem = (char*)malloc(64);
    MPI_Win host_win;
    MPI_Win_create(host_mem, 64, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &host_win);
    snprintf(s, sizeof s, "%d", dev);
    MPI_Info_set(info, "device_id", s);
    rc = MPIX_Win_get_device_handle(host_win, info, &dwin);
    check(rc == MPI_ERR_WIN, "host-memory window", rc, MPI_ERR_WIN);
    MPI_Win_free(&host_win);
    free(host_mem);

    rc = MPIX_Win_get_device_handle(win, info, &dwin);
    check(rc == MPI_SUCCESS, "get_device_handle", rc, MPI_SUCCESS);
    rc = MPIX_Win_get_device_handle(win, info, &dwin2);
    check(rc == MPI_SUCCESS && dwin2 == dwin, "second call returns same handle",
          dwin2 == dwin, 1);
    MPI_Info_free(&info);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);

    int* d_rc;
    CK(cudaMalloc((void**)&d_rc, 8 * sizeof(int)));
    int h_rc[8];

    // ---- device-side argument errors ----
    error_kernel<<<1, 1>>>(dwin, rank, size, (MPI_Aint)elems, MPI_DOUBLE, d_rc);
    CK(cudaMemcpy(h_rc, d_rc, 6 * sizeof(int), cudaMemcpyDeviceToHost));
    const int want_err[6] = {MPI_ERR_RANK, MPI_ERR_TYPE, MPI_ERR_TYPE,
                             MPI_ERR_COUNT, MPI_ERR_RMA_RANGE, MPI_ERR_ARG};
    const char* err_name[6] = {"target out of range", "Open MPI host MPI_DOUBLE",
                               "mismatched datatypes", "mismatched counts",
                               "disp past window end", "bad scope"};
    for (int i = 0; i < 6; i++) check(h_rc[i] == want_err[i], err_name[i], h_rc[i], want_err[i]);

    // ---- Put to rank+1, one test per scope ----
    const int target = (rank + 1) % size, source = (rank - 1 + size) % size;
    double* d_src;
    double* h = (double*)malloc(elems * sizeof(double));
    CK(cudaMalloc((void**)&d_src, K * sizeof(double)));
    int put_rc[NPUT];

    MPI_Win_lock_all(0, win);   // host opens the epoch on win, as in the spec
    for (int t = 0; t < NPUT; t++) {
        for (int i = 0; i < K; i++) h[i] = val(t, rank, i);
        CK(cudaMemcpy(d_src, h, K * sizeof(double), cudaMemcpyHostToDevice));
        put_kernel<<<1, THREADS[t]>>>(d_src, target, (MPI_Aint)(t * size + rank) * K,
                                      dwin, SCOPE[t], d_rc);
        CK(cudaMemcpy(h_rc, d_rc, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        put_rc[t] = h_rc[0];
        if (h_rc[0] == MPI_ERR_OTHER) {
            skips++;
            printf("rank %d: SKIP put/%s -> %d: unreachable (MPI_ERR_OTHER)\n",
                   rank, NAME[t], target);
        } else {
            check(h_rc[0] == MPI_SUCCESS, "put rc", h_rc[0], MPI_SUCCESS);
        }
        check(h_rc[1] == MPI_SUCCESS, "flush rc", h_rc[1], MPI_SUCCESS);
    }
    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);

    // Did the data land? Check what my source put into me, unless it could not.
    int all_rc[NPUT * 64];
    MPI_Allgather(put_rc, NPUT, MPI_INT, all_rc, NPUT, MPI_INT, MPI_COMM_WORLD);
    CK(cudaMemcpy(h, win_buf, elems * sizeof(double), cudaMemcpyDeviceToHost));
    for (int t = 0; t < NPUT; t++) {
        if (all_rc[source * NPUT + t] != MPI_SUCCESS) continue;
        const double* reg = h + (size_t)(t * size + source) * K;
        int bad = 0;
        for (int i = 0; i < K; i++) bad += reg[i] != val(t, source, i);
        if (bad) {
            fails++;
            printf("rank %d: FAIL put/%s from %d: %d of %d values wrong (first: %g, want %g)\n",
                   rank, NAME[t], source, bad, K, reg[0], val(t, source, 0));
        }
    }

    // ---- Get back from rank+1 what I put there, thread and block scope ----
    MPI_Win_lock_all(0, win);
    for (int t : {T_PUT_THREAD, T_PUT_BLOCK}) {
        if (put_rc[t] != MPI_SUCCESS) continue;   // nothing to read back
        CK(cudaMemset(d_src, 0, K * sizeof(double)));
        get_kernel<<<1, THREADS[t]>>>(d_src, target, (MPI_Aint)(t * size + rank) * K,
                                      dwin, SCOPE[t], d_rc);
        CK(cudaMemcpy(h_rc, d_rc, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        check(h_rc[0] == MPI_SUCCESS, "get rc", h_rc[0], MPI_SUCCESS);
        CK(cudaMemcpy(h, d_src, K * sizeof(double), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < K; i++) bad += h[i] != val(t, rank, i);
        if (bad) {
            fails++;
            printf("rank %d: FAIL get/%s from %d: %d of %d values wrong\n",
                   rank, NAME[t], target, bad, K);
        }
    }
    MPI_Win_unlock_all(win);

    int tot_fail = 0, tot_skip = 0;
    MPI_Reduce(&fails, &tot_fail, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&skips, &tot_skip, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("RESULT np=%d %s fails=%d skips=%d\n", size,
               tot_fail ? "FAIL" : "PASS", tot_fail, tot_skip);

    MPI_Win_free(&win);   // also frees the device handle
    cudaFree(d_src); cudaFree(d_rc); cudaFree(win_buf); free(h);
    MPI_Comm_free(&node);
    MPI_Finalize();
    return tot_fail ? 1 : 0;
}
