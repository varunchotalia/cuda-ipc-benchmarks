// mpiwrap_ipc.cc - MPI interception for CUDA IPC
//
// Intercepts MPI_Win_create/shared_query/free to add CUDA IPC support.
// Use with: LD_PRELOAD=./libmpiwrap_ipc.so mpirun ...

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mpi.h>
#include <cuda_runtime_api.h>

#define LOG(fmt, ...) fprintf(stderr, "[mpiwrap] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// WINDOW METADATA
// ============================================================================

struct WinMeta {
    MPI_Comm comm;
    int rank, size;
    size_t win_size;
    cudaIpcMemHandle_t* handles;  // IPC handle from each rank
    void** opened;                // cached opened pointers
    void* self_base;              // my own base pointer
};

static std::map<MPI_Win, WinMeta*> g_wins;

// ============================================================================
// HELPERS
// ============================================================================

static bool is_device_ptr(const void* p) {
    if (!p) return false;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) return false;
    return attr.type == cudaMemoryTypeDevice;
}

// ============================================================================
// INTERCEPTED FUNCTIONS
// ============================================================================

extern "C" {

int MPI_Win_create(void* base, MPI_Aint size, int disp,
                   MPI_Info info, MPI_Comm comm, MPI_Win* win)
{
    // Not a GPU pointer? Use standard MPI
    if (!is_device_ptr(base)) {
        return PMPI_Win_create(base, size, disp, info, comm, win);
    }
    
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    LOG("[rank %d] GPU window, setting up IPC", rank);
    
    // Export my handle
    cudaIpcMemHandle_t my_handle;
    if (cudaIpcGetMemHandle(&my_handle, base) != cudaSuccess) {
        LOG("FATAL: cudaIpcGetMemHandle failed");
        MPI_Abort(comm, 1);
    }
    
    // Exchange handles
    auto* handles = (cudaIpcMemHandle_t*)malloc(nprocs * sizeof(cudaIpcMemHandle_t));
    MPI_Allgather(&my_handle, sizeof(my_handle), MPI_BYTE,
                  handles, sizeof(my_handle), MPI_BYTE, comm);
    
    // Create real window
    int rc = PMPI_Win_create(base, size, disp, info, comm, win);
    if (rc != MPI_SUCCESS) MPI_Abort(comm, rc);
    
    // Store metadata
    auto* m = new WinMeta();
    m->comm = comm;
    m->rank = rank;
    m->size = nprocs;
    m->win_size = size;
    m->handles = handles;
    m->opened = (void**)calloc(nprocs, sizeof(void*));
    m->self_base = base;
    m->opened[rank] = base;  // self is already "opened"
    
    g_wins[*win] = m;
    return rc;
}

int MPI_Win_shared_query(MPI_Win win, int target,
                         MPI_Aint* size, int* disp, void* baseptr)
{
    auto it = g_wins.find(win);
    if (it == g_wins.end()) {
        return PMPI_Win_shared_query(win, target, size, disp, baseptr);
    }
    
    WinMeta* m = it->second;
    
    if (target < 0 || target >= m->size) return MPI_ERR_RANK;
    
    // Lazy open
    if (!m->opened[target]) {
        void* ptr;
        if (cudaIpcOpenMemHandle(&ptr, m->handles[target],
                                 cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
            LOG("Failed to open IPC handle for rank %d", target);
            return MPI_ERR_OTHER;
        }
        m->opened[target] = ptr;
    }
    
    if (baseptr) *(void**)baseptr = m->opened[target];
    if (size) *size = m->win_size;
    if (disp) *disp = 1;
    return MPI_SUCCESS;
}

int MPI_Win_free(MPI_Win* win)
{
    auto it = g_wins.find(*win);
    if (it != g_wins.end()) {
        WinMeta* m = it->second;
        for (int i = 0; i < m->size; i++) {
            if (i != m->rank && m->opened[i]) {
                cudaIpcCloseMemHandle(m->opened[i]);
            }
        }
        free(m->opened);
        free(m->handles);
        delete m;
        g_wins.erase(it);
    }
    return PMPI_Win_free(win);
}

} // extern "C"
