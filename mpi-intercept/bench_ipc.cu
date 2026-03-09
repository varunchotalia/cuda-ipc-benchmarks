#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// Kernels from bench_kernels.cu
extern "C" __global__ void copy_kernel(const double*, double*, size_t, int);
extern "C" __global__ void pingpong_kernel(const double*, double*, double*,
                                           size_t, int, int*, int*, int);
extern "C" __global__ void atomic_exch_bench_kernel(int*, int);

// ============================================================================
// HELPERS
// ============================================================================

static double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

static double stddev(const std::vector<double>& v, double m) {
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return sqrt(s / v.size());
}

// Timed kernel launch - returns seconds
static double timed_launch(void (*kernel_func)(), cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0;
}

// ============================================================================
// BANDWIDTH BENCHMARK
// ============================================================================

static void bench_bandwidth(double* src, double* dst, size_t bytes, int iters,
                           int rank, int size, cudaEvent_t ev0, cudaEvent_t ev1)
{
    size_t n = bytes / sizeof(double);
    int threads = 128;
    int blocks = std::min(4096, (int)((n + threads - 1) / threads));
    
    const int SAMPLES = 10;
    std::vector<double> bw_samples;
    
    for (int s = 0; s < SAMPLES; s++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Everyone launches the kernel
        cudaEventRecord(ev0);
        copy_kernel<<<blocks, threads>>>(src, dst, n, iters);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        
        if (rank == 0) {
            float ms;
            cudaEventElapsedTime(&ms, ev0, ev1);
            double secs = ms / 1000.0;
            double bw = (bytes * iters) / secs / 1e9;
            bw_samples.push_back(bw);
        }
    }
    
    if (rank == 0) {
        double m = mean(bw_samples);
        double sd = stddev(bw_samples, m);
        fprintf(stderr, "[BW] %8zu bytes, %4d iters: %.2f ± %.2f GB/s\n",
                bytes, iters, m, sd);
    }
}

// ============================================================================
// PING-PONG BENCHMARK
// ============================================================================

static void bench_pingpong(double* src, double* my_win, double* peer_win,
                          int* my_flag, int* peer_flag, size_t bytes, int pairs,
                          int rank, cudaEvent_t ev0, cudaEvent_t ev1)
{
    // Get cooperative grid size
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    
    if (!props.cooperativeLaunch) {
        if (rank == 0) fprintf(stderr, "[PINGPONG] Not supported on this GPU\n");
        return;
    }
    
    int blocks = props.multiProcessorCount;
    int threads = 256;
    size_t n = bytes / sizeof(double);
    int initiator = (rank == 0) ? 1 : 0;
    
    cudaMemset(my_flag, 0, sizeof(int));
    cudaMemset(peer_flag, 0, sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    
    void* args[] = {&src, &my_win, &peer_win, &n, &pairs, &my_flag, &peer_flag, &initiator};
    
    cudaEventRecord(ev0);
    cudaLaunchCooperativeKernel((void*)pingpong_kernel, blocks, threads, args, 0, 0);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    
    if (rank == 0) {
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        double secs = ms / 1000.0;
        double rtt = secs / pairs * 1e6;  // microseconds
        double bw = (bytes * 2 * pairs) / secs / 1e9;
        fprintf(stderr, "[PINGPONG] %zu bytes, %d pairs: RTT=%.2f us, BW=%.2f GB/s\n",
                bytes, pairs, rtt, bw);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// ATOMIC BENCHMARK
// ============================================================================

static void bench_atomic(int rank, cudaEvent_t ev0, cudaEvent_t ev1) {
    if (rank != 0) return;
    
    int* flag;
    cudaMalloc(&flag, sizeof(int));
    cudaMemset(flag, 0, sizeof(int));
    
    int iters = 1000000;
    
    cudaEventRecord(ev0);
    atomic_exch_bench_kernel<<<1, 1>>>(flag, iters);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    
    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1);
    double ns_per_op = (ms * 1e6) / iters;
    
    fprintf(stderr, "[ATOMIC] %d ops: %.2f ns/op\n", iters, ns_per_op);
    cudaFree(flag);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

extern "C"
int MPIX_CUDA_IPC_bench(MPI_Win win, int iters, size_t max_bytes)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Need 2+ ranks\n");
        return MPI_ERR_OTHER;
    }
    
    // Check modes
    bool local = getenv("MPIX_LOCAL") != nullptr;
    bool signal = getenv("MPIX_SIGNAL") != nullptr;
    bool atomic = getenv("MPIX_ATOMIC") != nullptr;
    
    // Get peer pointer
    int peer = (rank + 1) % size;
    MPI_Aint qsize;
    int disp;
    void *my_base, *peer_base;
    MPI_Win_shared_query(win, rank, &qsize, &disp, &my_base);
    MPI_Win_shared_query(win, peer, &qsize, &disp, &peer_base);
    
    if (!peer_base) {
        if (rank == 0) fprintf(stderr, "Failed to get peer pointer\n");
        return MPI_ERR_OTHER;
    }
    
    size_t win_bytes = qsize;
    if (max_bytes == 0 || max_bytes > win_bytes) max_bytes = win_bytes;
    
    // Allocate source
    double* src;
    cudaMalloc(&src, max_bytes);
    cudaMemset(src, 0xA5, max_bytes);
    
    // Local destination for baseline
    double* local_dst = nullptr;
    if (local) {
        cudaMalloc(&local_dst, max_bytes);
    }
    
    // Destination: local or peer
    double* dst = local ? local_dst : (double*)peer_base;
    
    // Events
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    
    // Iteration counts to test
    std::vector<int> iter_counts;
    if (iters > 0) {
        iter_counts.push_back(iters);
    } else {
        iter_counts = {1000, 2000, 4000, 8000, 100000};
    }
    
    // --- BANDWIDTH ---
    if (rank == 0) fprintf(stderr, "\n=== BANDWIDTH (%s) ===\n", local ? "LOCAL" : "IPC");
    
    for (size_t bytes = 4096; bytes <= max_bytes; bytes *= 2) {
        for (int it : iter_counts) {
            bench_bandwidth(src, dst, bytes, it, rank, size, ev0, ev1);
        }
    }
    
    // --- PING-PONG ---
    if (signal && !local && size == 2) {
        if (rank == 0) fprintf(stderr, "\n=== PING-PONG ===\n");
        
        size_t flag_offset = win_bytes - sizeof(int);
        int* my_flag = (int*)((char*)my_base + flag_offset);
        int* peer_flag = (int*)((char*)peer_base + flag_offset);
        size_t payload = std::min(max_bytes, flag_offset);
        
        bench_pingpong(src, (double*)my_base, (double*)peer_base,
                      my_flag, peer_flag, payload, iters, rank, ev0, ev1);
    }
    
    // --- ATOMIC ---
    if (atomic) {
        if (rank == 0) fprintf(stderr, "\n=== ATOMIC ===\n");
        bench_atomic(rank, ev0, ev1);
    }
    
    // Cleanup
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(src);
    if (local_dst) cudaFree(local_dst);
    
    return MPI_SUCCESS;
}

