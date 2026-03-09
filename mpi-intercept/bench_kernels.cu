// bench_kernels.cu
// 
// GPU kernels for IPC benchmarking. Four kernels:
//   1. empty_kernel         - measures kernel launch overhead
//   2. copy_kernel          - measures memory bandwidth  
//   3. pingpong_kernel      - measures round-trip latency between GPUs
//   4. atomic_exch_bench    - measures atomic operation cost
//
// All kernels use extern "C" to prevent C++ name mangling,
// making them easier to reference from other files.

// bench_kernels.cu - GPU kernels for IPC benchmarking

#include <cstddef>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Empty kernel - measures launch overhead
extern "C" __global__ void empty_kernel(int iters) {
    int dummy = 0;
    for (int k = 0; k < iters; k++) dummy += k;
    if (dummy == -1) ((volatile int*)0)[0] = dummy;  // prevent optimization
}

// Copy kernel - measures bandwidth
extern "C" __global__
void copy_kernel(const double* src, double* dst, size_t n, int iters) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (int k = 0; k < iters; k++) {
        for (size_t i = tid; i < n; i += stride) {
            dst[i] = src[i];
        }
    }
}

// Ping-pong kernel - measures round-trip latency
// Both GPUs launch this. initiator=1 sends first, initiator=0 waits first.
extern "C" __global__
void pingpong_kernel(const double* src,      // my data to send
                     double* my_win,          // peer writes here (I read)
                     double* peer_win,        // I write here (peer reads)
                     size_t n,                // payload size in doubles
                     int pairs,               // number of ping-pongs
                     int* my_flag,            // I poll this
                     int* peer_flag,          // I signal this
                     int initiator)
{
    cg::grid_group grid = cg::this_grid();
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int p = 0; p < pairs; p++) {
        int ping_token = 2 * p + 1;
        int pong_token = 2 * p + 2;

        if (initiator) {
            // SEND ping
            for (size_t i = tid; i < n; i += stride) peer_win[i] = src[i];
            grid.sync();
            __threadfence_system();
            if (grid.thread_rank() == 0) atomicExch(peer_flag, ping_token);
            
            // WAIT for pong
            if (grid.thread_rank() == 0) {
                while (atomicAdd(my_flag, 0) < pong_token);
            }
            grid.sync();
            
            // READ pong
            for (size_t i = tid; i < n; i += stride) {
                volatile double x = my_win[i]; (void)x;
            }
            grid.sync();
        } else {
            // WAIT for ping
            if (grid.thread_rank() == 0) {
                while (atomicAdd(my_flag, 0) < ping_token);
            }
            grid.sync();
            
            // READ ping
            for (size_t i = tid; i < n; i += stride) {
                volatile double x = my_win[i]; (void)x;
            }
            grid.sync();
            
            // SEND pong
            for (size_t i = tid; i < n; i += stride) peer_win[i] = src[i];
            grid.sync();
            __threadfence_system();
            if (grid.thread_rank() == 0) atomicExch(peer_flag, pong_token);
            grid.sync();
        }
    }
}

// Atomic benchmark - measures signaling cost
extern "C" __global__
void atomic_exch_bench_kernel(int* flag, int iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < iters; i++) atomicExch(flag, i);
    }
}
