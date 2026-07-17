// mpiwrap_ipc.cc - MPI interception for CUDA IPC
//
// Intercepts MPI_Win_create/allocate/shared_query/free to add CUDA IPC support.
// Use with: LD_PRELOAD=./libmpiwrap.so mpirun ...

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

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
    bool self_base_owned;         // true if we cudaMalloc'd it (MPI_Win_allocate)
    // fabric mode: multi-node NVLink (GB200/GH200 NVL-class systems)
    bool fabric = false;
    unsigned long long* fab_sizes = nullptr;              // padded bytes per rank
    CUmemGenericAllocationHandle* fab_handles = nullptr;  // own + imported
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

static bool info_has_cuda(MPI_Info info) {
    if (info == MPI_INFO_NULL) return false;
    char val[8]; int flag;
    MPI_Info_get(info, "cuda_ipc", 7, val, &flag);
    return flag && val[0] == '1';
}

static bool ranks_span_nodes(MPI_Comm comm) {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int nsize, csize;
    MPI_Comm_size(node, &nsize);
    MPI_Comm_size(comm, &csize);
    MPI_Comm_free(&node);
    return nsize != csize;
}

#if CUDA_VERSION >= 12040
// Fabric handles (CU_MEM_HANDLE_TYPE_FABRIC + IMEX) let peers on OTHER nodes
// of a multi-node NVLink domain map this allocation. Requires VMM-allocated
// memory, so this path only exists for MPI_Win_allocate, where we own the
// allocation -- not MPI_Win_create over an app cudaMalloc pointer.
static bool fabric_supported() {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int curdev = 0;
    if (cudaGetDevice(&curdev) != cudaSuccess) return false;
    CUdevice dev;
    if (cuDeviceGet(&dev, curdev) != CUDA_SUCCESS) return false;
    int v = 0;
    if (cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                             dev) != CUDA_SUCCESS) return false;
    return v != 0;
}

static int fabric_win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                               MPI_Comm comm, void* baseptr, MPI_Win* win)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    int curdev = 0;
    cudaGetDevice(&curdev);

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = curdev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    unsigned long long padded = ((size + gran - 1) / gran) * gran;
    if (padded == 0) padded = gran;

    CUmemGenericAllocationHandle mine;
    if (cuMemCreate(&mine, padded, &prop, 0) != CUDA_SUCCESS) {
        LOG("FATAL: cuMemCreate(FABRIC, %llu bytes) failed -- is the IMEX "
            "daemon running?", padded);
        MPI_Abort(comm, 1);
    }
    CUmemFabricHandle myfh;
    memset(&myfh, 0, sizeof(myfh));
    if (cuMemExportToShareableHandle(&myfh, mine, CU_MEM_HANDLE_TYPE_FABRIC, 0)
        != CUDA_SUCCESS) {
        LOG("FATAL: cuMemExportToShareableHandle(FABRIC) failed");
        MPI_Abort(comm, 1);
    }

    auto* allfh = (CUmemFabricHandle*)malloc(nprocs * sizeof(CUmemFabricHandle));
    MPI_Allgather(&myfh, sizeof(myfh), MPI_BYTE,
                  allfh, sizeof(myfh), MPI_BYTE, comm);
    auto* sizes = (unsigned long long*)malloc(nprocs * sizeof(unsigned long long));
    MPI_Allgather(&padded, 1, MPI_UNSIGNED_LONG_LONG,
                  sizes, 1, MPI_UNSIGNED_LONG_LONG, comm);

    auto* handles = (CUmemGenericAllocationHandle*)
        malloc(nprocs * sizeof(CUmemGenericAllocationHandle));
    void** opened = (void**)calloc(nprocs, sizeof(void*));
    CUmemAccessDesc acc;
    memset(&acc, 0, sizeof(acc));
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = curdev;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (int r = 0; r < nprocs; r++) {
        if (r == rank) {
            handles[r] = mine;
        }
        else if (cuMemImportFromShareableHandle(&handles[r], &allfh[r],
                                                CU_MEM_HANDLE_TYPE_FABRIC)
                 != CUDA_SUCCESS) {
            LOG("FATAL: fabric import from rank %d failed", r);
            MPI_Abort(comm, 1);
        }
        CUdeviceptr va = 0;
        if (cuMemAddressReserve(&va, sizes[r], gran, 0, 0) != CUDA_SUCCESS ||
            cuMemMap(va, sizes[r], 0, handles[r], 0) != CUDA_SUCCESS ||
            cuMemSetAccess(va, sizes[r], &acc, 1) != CUDA_SUCCESS) {
            LOG("FATAL: fabric map for rank %d failed", r);
            MPI_Abort(comm, 1);
        }
        opened[r] = (void*)va;
    }
    free(allfh);

    // zero-size window: purely the app-visible key for shared_query/free
    int rc = PMPI_Win_create(NULL, 0, disp_unit, info, comm, win);
    if (rc != MPI_SUCCESS) MPI_Abort(comm, rc);

    auto* m = new WinMeta();
    m->comm = comm;
    m->rank = rank;
    m->size = nprocs;
    m->win_size = padded;
    m->handles = nullptr;
    m->opened = opened;
    m->self_base = opened[rank];
    m->self_base_owned = false;
    m->fabric = true;
    m->fab_sizes = sizes;
    m->fab_handles = handles;
    g_wins[*win] = m;

    *(void**)baseptr = opened[rank];
    LOG("fabric window: %d ranks, %llu bytes/rank (multi-node NVLink)",
        nprocs, padded);
    return MPI_SUCCESS;
}
#endif /* CUDA_VERSION >= 12040 */

// ============================================================================
// INTERCEPTED FUNCTIONS
// ============================================================================

extern "C" {

int MPI_Win_create(void* base, MPI_Aint size, int disp,
                   MPI_Info info, MPI_Comm comm, MPI_Win* win)
{
    if (!is_device_ptr(base))
        return PMPI_Win_create(base, size, disp, info, comm, win);

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    cudaIpcMemHandle_t my_handle;
    if (cudaIpcGetMemHandle(&my_handle, base) != cudaSuccess) {
        LOG("FATAL: cudaIpcGetMemHandle failed");
        MPI_Abort(comm, 1);
    }

    auto* handles = (cudaIpcMemHandle_t*)malloc(nprocs * sizeof(cudaIpcMemHandle_t));
    MPI_Allgather(&my_handle, sizeof(my_handle), MPI_BYTE,
                  handles, sizeof(my_handle), MPI_BYTE, comm);

    // spanning communicators get a zero-size key window: the app only uses
    // the handle for shared_query/free, and registering device memory with
    // the MPI library across nodes is exactly what we are bypassing
    int rc = ranks_span_nodes(comm)
                 ? PMPI_Win_create(NULL, 0, disp, info, comm, win)
                 : PMPI_Win_create(base, size, disp, info, comm, win);
    if (rc != MPI_SUCCESS) MPI_Abort(comm, rc);

    auto* m = new WinMeta();
    m->comm = comm;
    m->rank = rank;
    m->size = nprocs;
    m->win_size = size;
    m->handles = handles;
    m->opened = (void**)calloc(nprocs, sizeof(void*));
    m->self_base = base;
    m->self_base_owned = false;
    m->opened[rank] = base;

    g_wins[*win] = m;
    return rc;
}

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                     MPI_Comm comm, void* baseptr, MPI_Win* win)
{
    if (!info_has_cuda(info))
        return PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);

    if (ranks_span_nodes(comm)) {
#if CUDA_VERSION >= 12040
        int ok = fabric_supported() ? 1 : 0, allok = 0;
        MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, comm);
        if (allok)
            return fabric_win_allocate(size, disp_unit, info, comm, baseptr, win);
#endif
        LOG("window spans nodes without fabric support: same-node peers map "
            "via CUDA IPC; shared_query fails for remote peers so the app "
            "can fall back to MPI for them");
        // fall through: legacy per-peer IPC with lazy opens
    }

    void* d_ptr;
    if (cudaMalloc(&d_ptr, size) != cudaSuccess) {
        LOG("cudaMalloc failed for size %zu", (size_t)size);
        return MPI_ERR_NO_MEM;
    }

    int rc = MPI_Win_create(d_ptr, size, disp_unit, info, comm, win);
    if (rc != MPI_SUCCESS) { cudaFree(d_ptr); return rc; }

    g_wins[*win]->self_base_owned = true;

    *(void**)baseptr = d_ptr;
    return MPI_SUCCESS;
}

int MPI_Win_shared_query(MPI_Win win, int target,
                         MPI_Aint* size, int* disp, void* baseptr)
{
    auto it = g_wins.find(win);
    if (it == g_wins.end())
        return PMPI_Win_shared_query(win, target, size, disp, baseptr);

    WinMeta* m = it->second;

    if (target < 0 || target >= m->size) return MPI_ERR_RANK;

    if (m->fabric) {
        if (baseptr) *(void**)baseptr = m->opened[target];
        if (size)    *size = (MPI_Aint)m->fab_sizes[target];
        if (disp)    *disp = 1;
        return MPI_SUCCESS;
    }

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
    if (size)    *size = m->win_size;
    if (disp)    *disp = 1;
    return MPI_SUCCESS;
}

int MPI_Win_free(MPI_Win* win)
{
    auto it = g_wins.find(*win);
    if (it != g_wins.end()) {
        WinMeta* m = it->second;
        if (m->fabric) {
            for (int i = 0; i < m->size; i++) {
                if (m->opened[i]) {
                    cuMemUnmap((CUdeviceptr)m->opened[i], m->fab_sizes[i]);
                    cuMemAddressFree((CUdeviceptr)m->opened[i], m->fab_sizes[i]);
                }
                cuMemRelease(m->fab_handles[i]);
            }
            free(m->fab_sizes);
            free(m->fab_handles);
            free(m->opened);
            delete m;
            g_wins.erase(it);
            return PMPI_Win_free(win);
        }
        for (int i = 0; i < m->size; i++) {
            if (i != m->rank && m->opened[i])
                cudaIpcCloseMemHandle(m->opened[i]);
        }
        if (m->self_base_owned)
            cudaFree(m->self_base);
        free(m->opened);
        free(m->handles);
        delete m;
        g_wins.erase(it);
    }
    return PMPI_Win_free(win);
}

} // extern "C"
