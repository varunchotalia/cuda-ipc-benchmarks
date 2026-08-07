// comm_ipc_packed.h -- handwritten CUDA-IPC allocation and teardown for the
// packed recv buffer.  Split out of comm_ipc_common.h so that it is included
// ONLY by the backends that actually perform their own handle exchange:
// comm_ipc.h (Mode A/C) and comm_direct.h (Mode B, for the MonoQ path).
//
// WHY THE SPLIT EXISTS -- do not merge this back.
// WinIPC (comm_mpiwrap.h) establishes the same d_peerRecv[] mapping through
// MPI_Win_allocate + MPI_Win_shared_query and never calls these functions.
// While they lived in comm_ipc_common.h, which comm_mpiwrap.h includes for
// the shared token/steady-state machinery, they were still textually present
// in the wrapper build's translation units.  The paper's mechanism-
// localization claim is that a wrapper-mediated build contains NO cudaIpc*
// call sites, and scripts/check_no_ipc_calls.sh verifies exactly that against
// the preprocessed TUs -- so their mere presence made the check fail (3 sites
// per TU) even though they were never invoked.  Keeping them here makes the
// claim true as stated rather than true-with-an-asterisk.
#ifndef LULESH_COMM_IPC_PACKED_H
#define LULESH_COMM_IPC_PACKED_H

#include "comm_ipc_common.h"

// Allocate the standard recv buffers and map every IPC-reachable peer's
// d_commDataRecv through an explicit CUDA-IPC handle exchange.  Peers on
// other nodes, and same-node peers CUDA can't actually open a handle to
// (e.g. GPU islands without full P2P), get d_peerRecv[r] = NULL (MPI
// fallback) -- same-node is attempted, not assumed reachable.
static inline void commIpcAllocAndMapPacked(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   COMM_CUDA_OK(cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0)) ;
   COMM_CUDA_OK(cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t))) ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   // node identity: world rank of each rank's node-local leader
   MPI_Comm node ;
   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                       MPI_INFO_NULL, &node) ;
   int leader = myRank ;
   MPI_Bcast(&leader, 1, MPI_INT, 0, node) ;
   MPI_Comm_free(&node) ;
   int *leaders = new int[d->m_numRanks] ;
   MPI_Allgather(&leader, 1, MPI_INT, leaders, 1, MPI_INT, MPI_COMM_WORLD) ;

   cudaIpcMemHandle_t myHandle ;
   if (cudaIpcGetMemHandle(&myHandle, d->d_commDataRecv) != cudaSuccess) {
      fprintf(stderr, "rank %d: cudaIpcGetMemHandle failed\n", myRank) ;
      MPI_Abort(MPI_COMM_WORLD, 1) ;
   }
   cudaIpcMemHandle_t *allHandles = new cudaIpcMemHandle_t[d->m_numRanks] ;
   MPI_Allgather(&myHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                 allHandles, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                 MPI_COMM_WORLD) ;
   d->d_peerRecv = new Real_t*[d->m_numRanks] ;
   int nFallback = 0 ;
   for (int r = 0; r < d->m_numRanks; ++r) {
      if (r == myRank) {
         d->d_peerRecv[r] = d->d_commDataRecv ;
      }
      else if (leaders[r] != leaders[myRank]) {
         d->d_peerRecv[r] = NULL ;   // other node: MPI fallback
         ++nFallback ;
      }
      else if (cudaIpcOpenMemHandle((void **)&d->d_peerRecv[r],
                                    allHandles[r],
                                    cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
         // same node, but CUDA couldn't open a handle to this peer (e.g. a
         // GPU-island boundary with no P2P): fall back, don't abort
         d->d_peerRecv[r] = NULL ;
         ++nFallback ;
      }
   }
   /* Printed unconditionally -- see the note in comm_mpiwrap.h.  On a
      multi-node machine this is the number that decides how to label the
      column: cudaIpcGetMemHandle handles never cross a node, so at 27 or 64
      ranks (4 GPUs/node) only about 3 of a domain's 26 neighbours are
      same-node and the rest run over GPU-aware MPI.  A run that reports a
      large fallback count is not measuring CUDA IPC. */
   if (myRank == 0) {
      printf("comm: %d of %d peers not IPC-reachable, using MPI send/recv "
             "fallback for them\n", nFallback, (int)d->m_numRanks - 1) ;
   }
   delete [] allHandles ;
   delete [] leaders ;
}

static inline void commIpcUnmapAndFreePacked(Domain* d, int myRank)
{
   int nRanks ;
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks) ;
   for (int r = 0; r < nRanks; ++r) {
      if (r != myRank && d->d_peerRecv[r]) {
         cudaIpcCloseMemHandle(d->d_peerRecv[r]) ;
      }
   }
   delete [] d->d_peerRecv ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
   cudaFree(d->d_commDataRecv) ;
}

#endif
