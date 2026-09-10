// Does a raw CUDA IPC write actually land across a peer-accessible island
// boundary?  No interposer, no benchmark -- just the mechanism.
//
// WHY: on h200x8-04 (GPUs 0-3 and 4-7 are NVLink islands, PCIe between them)
// the stencil at 8 ranks returns L2 0.6600931148 instead of 5.1449605829,
// deterministically, at full speed, in BOTH sync modes.  Double buffering
// makes no difference, so it is not a synchronisation race -- the data is
// simply not arriving.  Meanwhile every rank reports "IPC setup complete" and
// cudaDeviceCanAccessPeer(3,4) returns TRUE, so nothing in the stack thinks
// anything is wrong and the application never takes its MPI fallback.
//
// This builds the full pair matrix directly.  Every rank fills its buffer with
// a known pattern, publishes an IPC handle, then each rank writes a sentinel
// into every peer it can open.  After a barrier each rank checks which
// sentinels actually arrived.
//
// Reachability and delivery are reported SEPARATELY, because the whole problem
// is that they disagree:
//   OPEN  = cudaIpcOpenMemHandle succeeded
//   CAN   = cudaDeviceCanAccessPeer said yes
//   LAND  = the sentinel was actually there afterwards
// A pair with OPEN=1 CAN=1 LAND=0 is the silent-loss case, and if it shows up
// here then this is CUDA IPC behaviour on this topology, not a WinIPC bug --
// which means the fix is to detect it, and cudaDeviceCanAccessPeer is not the
// test that can.

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"[rank %d] %s:%d %s -> %s\n",rank,__FILE__,__LINE__,#call, \
            cudaGetErrorString(e)); } } while(0)

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ndev = 0; cudaGetDeviceCount(&ndev);
    const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int mydev = (lr ? atoi(lr) : rank) % ndev;
    CK(cudaSetDevice(mydev));

    const int N = 256;                      // ints per buffer
    int* buf = nullptr;
    CK(cudaMalloc(&buf, N * sizeof(int)));
    // slot p of my buffer is where rank p is supposed to write 1000+p
    CK(cudaMemset(buf, 0, N * sizeof(int)));

    // publish handles and device ordinals
    cudaIpcMemHandle_t mine;
    CK(cudaIpcGetMemHandle(&mine, buf));
    std::vector<cudaIpcMemHandle_t> all(size);
    MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, all.data(), sizeof(mine),
                  MPI_BYTE, MPI_COMM_WORLD);
    std::vector<int> devs(size);
    MPI_Allgather(&mydev, 1, MPI_INT, devs.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> opened(size, 0), can(size, 0);
    std::vector<int*> peer(size, nullptr);

    for (int p = 0; p < size; p++) {
        if (p == rank) { peer[p] = buf; opened[p] = 1; can[p] = 1; continue; }
        int c = 0;
        if (cudaDeviceCanAccessPeer(&c, mydev, devs[p]) == cudaSuccess) can[p] = c;
        void* ptr = nullptr;
        if (cudaIpcOpenMemHandle(&ptr, all[p], cudaIpcMemLazyEnablePeerAccess)
            == cudaSuccess) { peer[p] = (int*)ptr; opened[p] = 1; }
        else cudaGetLastError();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // write my sentinel into slot `rank` of every peer I could open
    const int sentinel = 1000 + rank;
    for (int p = 0; p < size; p++) {
        if (p == rank || !opened[p]) continue;
        CK(cudaMemcpy(peer[p] + rank, &sentinel, sizeof(int),
                      cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // check what actually arrived in MY buffer
    std::vector<int> got(size);
    CK(cudaMemcpy(got.data(), buf, size * sizeof(int), cudaMemcpyDeviceToHost));

    // report one line per (writer -> me) pair
    for (int p = 0; p < size; p++) {
        if (p == rank) continue;
        int land = (got[p] == 1000 + p) ? 1 : 0;
        printf("PAIR writer=%d(dev%d) -> reader=%d(dev%d) OPEN=%d CAN=%d "
               "LAND=%d expected=%d got=%d\n",
               p, devs[p], rank, mydev, opened[p], can[p], land,
               1000 + p, got[p]);
    }
    fflush(stdout);

    for (int p = 0; p < size; p++)
        if (p != rank && peer[p]) cudaIpcCloseMemHandle(peer[p]);
    CK(cudaFree(buf));
    MPI_Finalize();
    return 0;
}
