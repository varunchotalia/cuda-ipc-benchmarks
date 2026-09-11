// mpiwrap_ipc.cc - WinIPC: MPI interception for CUDA IPC
//
// (Formerly "mpiwrap". This file name, the built .so and the CMake target keep
// the old spelling for now -- queued Slurm jobs reference them by name.)
//
// Intercepts MPI_Win_create/allocate/shared_query/free to add CUDA IPC support.
// Use with: LD_PRELOAD=./libmpiwrap.so mpirun ...
//
// WINIPC_DISABLE_FABRIC=1 forces cross-node windows onto the per-peer
// hybrid-MPI fallback even on fabric-capable (NVL72-class) hardware --
// useful for an apples-to-apples fabric-vs-no-fabric comparison.
// MPIWRAP_DISABLE_FABRIC is still honoured as a deprecated alias, so an
// already-submitted job or a collaborator's script does not silently lose the
// setting and run WITH fabric when it meant to run without.
//
// WINIPC_VERIFY_PEERS=0 skips the peer delivery probe (see
// verify_peer_delivery below) and restores the pre-2026-09-10 behaviour, in
// which a peer that opens but never delivers is reported as reachable. Use it
// only to reproduce measurements taken before the probe existed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define LOG(fmt, ...) fprintf(stderr, "[winipc] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// WINDOW METADATA
// ============================================================================

struct WinMeta {
    MPI_Comm comm;
    int rank, size;
    size_t win_size;                        // MY size (peer_sizes has the rest)
    int disp_unit = 1;                      // as passed at window construction
    cudaIpcMemHandle_t* handles = nullptr;  // IPC handle from each rank
    size_t* peer_sizes = nullptr;           // size of EACH rank, gathered at create
    int* peer_devs = nullptr;               // CUDA device ordinal of EACH rank
    void** opened = nullptr;                // cached opened pointers
    int* reachable = nullptr;               // probe verdict, NULL if not probed
    void* self_base = nullptr;              // my own base pointer
    bool self_base_owned = false;           // true if we cudaMalloc'd it
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

// All MPI work performed BY the interposer goes through PMPI_*, not MPI_*.
// Calling MPI_* here would re-enter the profiling layer: our own calls would be
// intercepted again by this library, and by any other PMPI tool layered with it
// (Score-P and friends), which both distorts what such a tool reports and makes
// the composition order significant. An interposer should be invisible to the
// profiling interface it is built on.
static bool info_has_cuda(MPI_Info info) {
    if (info == MPI_INFO_NULL) return false;
    char val[8]; int flag;
    PMPI_Info_get(info, "cuda_ipc", 7, val, &flag);
    return flag && val[0] == '1';
}

static bool ranks_span_nodes(MPI_Comm comm) {
    MPI_Comm node;
    PMPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int nsize, csize;
    PMPI_Comm_size(node, &nsize);
    PMPI_Comm_size(comm, &csize);
    PMPI_Comm_free(&node);
    return nsize != csize;
}

#if CUDA_VERSION >= 12040
// Fabric handles (CU_MEM_HANDLE_TYPE_FABRIC + IMEX) let peers on OTHER nodes
// of a multi-node NVLink domain map this allocation. Requires VMM-allocated
// memory, so this path only exists for MPI_Win_allocate, where we own the
// allocation -- not MPI_Win_create over an app cudaMalloc pointer.
static bool fabric_supported() {
    // Lets a benchmark force the per-peer hybrid-MPI path even on hardware
    // that supports fabric handles, e.g. to compare against the fabric path.
    // MPIWRAP_DISABLE_FABRIC is the pre-rebrand name, still honoured: dropping
    // it would turn a stale script into a silent fabric-ON run.
    const char* disable = getenv("WINIPC_DISABLE_FABRIC");
    if (!disable) disable = getenv("MPIWRAP_DISABLE_FABRIC");
    if (disable && disable[0] == '1') return false;
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
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &nprocs);
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
        PMPI_Abort(comm, 1);
    }
    CUmemFabricHandle myfh;
    memset(&myfh, 0, sizeof(myfh));
    if (cuMemExportToShareableHandle(&myfh, mine, CU_MEM_HANDLE_TYPE_FABRIC, 0)
        != CUDA_SUCCESS) {
        LOG("FATAL: cuMemExportToShareableHandle(FABRIC) failed");
        PMPI_Abort(comm, 1);
    }

    auto* allfh = (CUmemFabricHandle*)malloc(nprocs * sizeof(CUmemFabricHandle));
    PMPI_Allgather(&myfh, sizeof(myfh), MPI_BYTE,
                  allfh, sizeof(myfh), MPI_BYTE, comm);
    auto* sizes = (unsigned long long*)malloc(nprocs * sizeof(unsigned long long));
    PMPI_Allgather(&padded, 1, MPI_UNSIGNED_LONG_LONG,
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
            PMPI_Abort(comm, 1);
        }
        CUdeviceptr va = 0;
        if (cuMemAddressReserve(&va, sizes[r], gran, 0, 0) != CUDA_SUCCESS ||
            cuMemMap(va, sizes[r], 0, handles[r], 0) != CUDA_SUCCESS ||
            cuMemSetAccess(va, sizes[r], &acc, 1) != CUDA_SUCCESS) {
            LOG("FATAL: fabric map for rank %d failed", r);
            PMPI_Abort(comm, 1);
        }
        opened[r] = (void*)va;
    }
    free(allfh);

    // zero-size window: purely the app-visible key for shared_query/free
    int rc = PMPI_Win_create(NULL, 0, disp_unit, info, comm, win);
    if (rc != MPI_SUCCESS) PMPI_Abort(comm, rc);

    auto* m = new WinMeta();
    m->comm = comm;
    m->rank = rank;
    m->size = nprocs;
    m->win_size = padded;
    m->disp_unit = disp_unit;
    m->handles = nullptr;
    m->peer_sizes = nullptr;   // fabric path reports sizes from fab_sizes
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
// PEER DELIVERY PROBE
// ============================================================================

// Both cudaDeviceCanAccessPeer and a successful cudaIpcOpenMemHandle report
// REACHABILITY, and on a node whose GPUs form bridged NVLink islands the two
// can agree that a peer is fine while its writes never arrive. On h200x8-04
// (GPUs 0-3 and 4-7 are NVLink islands, PCIe between them)
// cudaDeviceCanAccessPeer(3,4) returns TRUE and the open succeeds, yet job
// 90162 -- built at 95a4607, i.e. WITH the canAccessPeer gate in
// MPI_Win_shared_query -- returned stencil L2 0.6600931148 instead of
// 5.1449605829 at np=8, with no rank reporting a failure and no rank taking
// its MPI fallback. np=4, inside a single island, was correct.
//
// Only an actual write, checked by the rank that owns the memory, separates
// "the mapping opened" from "the store landed". Each rank writes a sentinel
// into slot[myrank] of every peer; the Alltoall then turns "whose writes
// reached me" into "which of my writes reached their target", which is the
// direction MPI_Win_shared_query has to gate on. A peer that does not deliver
// is reported unreachable, which is what makes the documented MPI fallback
// actually happen on this topology.
//
// Costs one extra pass of opens, two barriers and one Alltoall per window.
// Because it opens every peer eagerly it also moves the per-peer
// cudaIpcOpenMemHandle cost out of MPI_Win_shared_query and into window
// construction -- set WINIPC_VERIFY_PEERS=0 to reproduce setup-cost numbers
// measured before this existed.
static const unsigned long long PROBE_MAGIC = 0x57494E4950430000ULL; // "WINIPC"

static void verify_peer_delivery(WinMeta* m)
{
    const char* off = getenv("WINIPC_VERIFY_PEERS");
    if (off && off[0] == '0') return;
    if (m->size < 2 || !m->handles || !m->opened) return;

    const size_t slot_bytes = (size_t)m->size * sizeof(unsigned long long);

    unsigned long long* saved = (unsigned long long*)malloc(slot_bytes);
    unsigned long long* got   = (unsigned long long*)malloc(slot_bytes);
    int* seen   = (int*)calloc(m->size, sizeof(int));
    int* landed = (int*)calloc(m->size, sizeof(int));
    // Device-side source for the sentinel: the applications reach a peer with
    // a device-to-device copy, so the probe has to use one too. A host-to-
    // device write can take a different route and would not prove the path the
    // app is about to use.
    unsigned long long* d_sentinel = nullptr;
    if (cudaMalloc((void**)&d_sentinel, sizeof(unsigned long long))
        != cudaSuccess) {
        cudaGetLastError();
        d_sentinel = nullptr;
    }

    // EVERY reason to skip has to be unanimous: a rank that bails out here
    // after the others have entered the barriers below would hang the job.
    int ok = (m->win_size >= slot_bytes && saved && got && seen && landed
              && d_sentinel) ? 1 : 0;
    int all_ok = 0;
    PMPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, m->comm);
    if (!all_ok) {
        if (m->rank == 0)
            LOG("peer delivery probe skipped (window < %zu bytes, or setup "
                "allocation failed) -- a peer that opens but never delivers "
                "will NOT be detected on this window", slot_bytes);
        free(saved); free(got); free(seen); free(landed);
        if (d_sentinel) cudaFree(d_sentinel);
        return;
    }

    // MPI_Win_create is handed the application's own buffer, which may already
    // hold live data, so the probe region is saved and put back afterwards.
    cudaMemcpy(saved, m->self_base, slot_bytes, cudaMemcpyDeviceToHost);
    cudaMemset(m->self_base, 0, slot_bytes);
    cudaDeviceSynchronize();
    PMPI_Barrier(m->comm);

    const unsigned long long sentinel = PROBE_MAGIC + (unsigned long long)m->rank;
    cudaMemcpy(d_sentinel, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice);
    for (int p = 0; p < m->size; p++) {
        if (p == m->rank) continue;
        if (!m->opened[p]) {
            void* ptr = nullptr;
            if (cudaIpcOpenMemHandle(&ptr, m->handles[p],
                                     cudaIpcMemLazyEnablePeerAccess)
                != cudaSuccess) {
                cudaGetLastError();   // cannot even open: stays unreachable
                continue;
            }
            m->opened[p] = ptr;
        }
        unsigned long long* slot = (unsigned long long*)m->opened[p] + m->rank;
        if (cudaMemcpy(slot, d_sentinel, sizeof(sentinel),
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
            cudaGetLastError();
    }
    cudaDeviceSynchronize();
    PMPI_Barrier(m->comm);

    cudaMemcpy(got, m->self_base, slot_bytes, cudaMemcpyDeviceToHost);
    for (int p = 0; p < m->size; p++)
        seen[p] = (p == m->rank) ||
                  (got[p] == PROBE_MAGIC + (unsigned long long)p);

    // seen[p]   = "p's write reached me"
    // landed[p] = "my write reached p"   (the transpose, which is what the
    // caller of shared_query is about to rely on)
    PMPI_Alltoall(seen, 1, MPI_INT, landed, 1, MPI_INT, m->comm);

    cudaMemcpy(m->self_base, saved, slot_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int bad = 0;
    for (int p = 0; p < m->size; p++) {
        if (p == m->rank || landed[p]) continue;
        bad++;
        if (m->opened[p]) {   // opened cleanly, delivered nothing: drop it so
            cudaIpcCloseMemHandle(m->opened[p]);  // nobody can write through it
            m->opened[p] = nullptr;
        }
    }
    m->reachable = landed;
    if (bad)
        LOG("rank %d: %d of %d peers opened but did not deliver; reporting "
            "them unreachable so the caller falls back to MPI",
            m->rank, bad, m->size - 1);

    free(saved); free(got); free(seen);
    cudaFree(d_sentinel);
}

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
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &nprocs);

    cudaIpcMemHandle_t my_handle;
    if (cudaIpcGetMemHandle(&my_handle, base) != cudaSuccess) {
        LOG("FATAL: cudaIpcGetMemHandle failed");
        PMPI_Abort(comm, 1);
    }

    auto* handles = (cudaIpcMemHandle_t*)malloc(nprocs * sizeof(cudaIpcMemHandle_t));
    PMPI_Allgather(&my_handle, sizeof(my_handle), MPI_BYTE,
                  handles, sizeof(my_handle), MPI_BYTE, comm);

    // Gather every rank's window size too. MPI_Win_shared_query is specified to
    // report the TARGET's size, so returning our own would be wrong for any
    // window whose ranks allocate different amounts -- the collective is already
    // here, so this costs one extra Allgather at construction and nothing later.
    auto* peer_sizes = (size_t*)malloc(nprocs * sizeof(size_t));
    size_t my_size = (size_t)size;
    PMPI_Allgather(&my_size, sizeof(size_t), MPI_BYTE,
                  peer_sizes, sizeof(size_t), MPI_BYTE, comm);

    // Gather each rank's CUDA device ordinal. shared_query needs it to ask
    // cudaDeviceCanAccessPeer before trusting an opened handle -- see the long
    // comment there. One more Allgather at construction, nothing later.
    auto* peer_devs = (int*)malloc(nprocs * sizeof(int));
    int my_dev = -1;
    cudaGetDevice(&my_dev);
    PMPI_Allgather(&my_dev, 1, MPI_INT, peer_devs, 1, MPI_INT, comm);

    // spanning communicators get a zero-size key window: the app only uses
    // the handle for shared_query/free, and registering device memory with
    // the MPI library across nodes is exactly what we are bypassing
    int rc = ranks_span_nodes(comm)
                 ? PMPI_Win_create(NULL, 0, disp, info, comm, win)
                 : PMPI_Win_create(base, size, disp, info, comm, win);
    if (rc != MPI_SUCCESS) PMPI_Abort(comm, rc);

    auto* m = new WinMeta();
    m->comm = comm;
    m->rank = rank;
    m->size = nprocs;
    m->win_size = size;
    m->disp_unit = disp;
    m->handles = handles;
    m->peer_sizes = peer_sizes;
    m->peer_devs = peer_devs;
    m->opened = (void**)calloc(nprocs, sizeof(void*));
    m->self_base = base;
    m->self_base_owned = false;
    m->opened[rank] = base;

    g_wins[*win] = m;
    verify_peer_delivery(m);
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
        PMPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, comm);
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
        // Same rule as shared_query: we report the failure through the MPI
        // return code, so clear it rather than leaving it for the app's next
        // kernel launch to trip over.
        cudaGetLastError();
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
        if (disp)    *disp = m->disp_unit;
        return MPI_SUCCESS;
    }

    // Probe verdict, when one was taken, outranks every other signal here: it
    // is the only one that observed an actual store arriving.
    if (m->reachable && !m->reachable[target]) {
        LOG("Rank %d: peer %d opened but did not deliver during the window "
            "probe; reporting unreachable so the caller falls back to MPI",
            m->rank, target);
        return MPI_ERR_OTHER;
    }

    if (!m->opened[target]) {
        // Peer-access gate. cudaIpcOpenMemHandle is asked for LAZY peer
        // enablement, so on a node whose GPUs form separate peer-accessible
        // islands it can RETURN SUCCESS for a target in the other island. The
        // caller then writes through a pointer that never lands, gets no error,
        // and silently computes on stale ghost data.
        //
        // Observed on h200x8-04 (GPUs 0-3 and 4-7 are NVLink islands, PCIe
        // between them). Job 83189: at 8 ranks the stencil returned L2
        // 0.6600931148 instead of 5.1449605829, deterministically and at full
        // speed, while ranks 3 and 4 -- the pair straddling the boundary --
        // both reported "IPC setup complete" and never took the MPI fallback.
        // np=4 fits inside one island and was correct.
        //
        // Section III-C of the paper states that a failed peer query is
        // reported so the application can fall back to MPI. That was true only
        // for peers on other nodes, where the open itself fails. Ask the
        // hardware directly instead of inferring reachability from the open.
        //
        // Only meaningful when both ranks are on this node; if they are not,
        // the open fails on its own and the check below never runs.
        if (m->peer_devs) {
            int mydev = -1, can = 0;
            cudaGetDevice(&mydev);
            int peerdev = m->peer_devs[target];
            if (mydev >= 0 && peerdev >= 0 && mydev != peerdev &&
                cudaDeviceCanAccessPeer(&can, mydev, peerdev) == cudaSuccess &&
                !can) {
                cudaGetLastError();
                LOG("Rank %d (dev %d) cannot peer-access rank %d (dev %d); "
                    "reporting unreachable so the caller falls back to MPI",
                    m->rank, mydev, target, peerdev);
                return MPI_ERR_OTHER;
            }
        }

        void* ptr;
        cudaError_t cerr = cudaIpcOpenMemHandle(&ptr, m->handles[target],
                                                cudaIpcMemLazyEnablePeerAccess);
        if (cerr != cudaSuccess) {
            // Expected whenever the window spans nodes without fabric support:
            // the remote peer's handle is unopenable here and the app falls
            // back to MPI for it. Clear the error before returning -- CUDA
            // keeps it as the thread's last error, and the app's next kernel
            // launch would otherwise report this failure as its own (thrust
            // throws "parallel_for failed: cudaErrorInvalidDevice").
            cudaGetLastError();
            LOG("Failed to open IPC handle for rank %d: %s", target,
                cudaGetErrorString(cerr));
            return MPI_ERR_OTHER;
        }
        m->opened[target] = ptr;
    }

    if (baseptr) *(void**)baseptr = m->opened[target];
    if (size)    *size = (MPI_Aint)(m->peer_sizes ? m->peer_sizes[target]
                                                  : m->win_size);
    if (disp)    *disp = m->disp_unit;
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
        free(m->peer_devs);
        free(m->reachable);
        if (m->self_base_owned)
            cudaFree(m->self_base);
        free(m->opened);
        free(m->handles);
        free(m->peer_sizes);
        delete m;
        g_wins.erase(it);
    }
    return PMPI_Win_free(win);
}

} // extern "C"
