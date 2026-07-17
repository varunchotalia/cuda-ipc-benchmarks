// comm_direct.h -- Mode B: no pack, no unpack.
// Sender kernels write halo contributions straight into the RECEIVER's
// field arrays through CUDA-IPC mappings of the peer fields: atomicAdd for
// the force-summation phase (multiple neighbors legitimately contribute to
// shared edge/corner nodes), plain stores for position/velocity sync.
// MonoQ is the exception: its target arrays (delv_xi/eta/zeta) are per-step
// pool allocations whose addresses cannot be premapped, so MonoQ remote-
// packs into the peer's packed recv buffer and unpacks locally.
// The GPU comm functions live in lulesh-comms-direct.cu, which replaces
// lulesh-comms-gpu.cu in this build.
#ifndef LULESH_COMM_DIRECT_H
#define LULESH_COMM_DIRECT_H

#include "comm_ipc_common.h"

// Mode B keeps global-barrier synchronization: senders write into the
// receiver's FIELD arrays, whose readiness is bounded by the receiver's
// local compute, not by its unpack -- the per-neighbor token protocol of
// the packed modes does not cover that dependency.  Single node only, so
// the barrier spans at most 8 ranks.
#undef COMM_RECV_SKIP
static inline int commDirectRecvBarrier(Domain& domain)
{
   if (!g_commActive) return 0 ;
   for (Index_t i = 0; i < 26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }
   cudaDeviceSynchronize() ;
   MPI_Barrier(MPI_COMM_WORLD) ;
   return 1 ;
}
#define COMM_RECV_SKIP(domain, msgType, xferFields, dx, dy, dz, doRecv, planeOnly) \
   commDirectRecvBarrier(domain)

// implemented in lulesh-comms-direct.cu
void commDirectMapFields(Domain* d) ;
void commDirectUnmapFields(Domain* d, int myRank) ;

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   // packed buffer + peer mapping still needed for the MonoQ path
   commIpcAllocAndMapPacked(d, comBufSize) ;
   commDirectMapFields(d) ;
}

static inline void commTeardown(Domain* d, int myRank)
{
   commDirectUnmapFields(d, myRank) ;
   commIpcUnmapAndFreePacked(d, myRank) ;
}

#endif
