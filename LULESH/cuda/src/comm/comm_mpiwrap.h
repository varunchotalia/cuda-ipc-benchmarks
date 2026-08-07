// comm_mpiwrap.h -- CUDA IPC through the WinIPC interceptor (was: mpiwrap;
// the file name, the .so and the IPC_VIA_MPIWRAP macro keep the old spelling
// because queued Slurm jobs reference them).
// The application only speaks portable MPI window code: MPI_Win_allocate
// (info key cuda_ipc=1) for the device recv buffer + MPI_Win_shared_query
// per peer.  Vanilla MPI cannot shared_query such a window; libmpiwrap.so
// (LD_PRELOAD) intercepts both calls and backs them with CUDA IPC on a
// single node, yielding the same d_peerRecv mapping as comm_ipc.h -- or,
// on multi-node NVLink systems (GB200/GH200 NVL-class), with CUDA fabric
// handles.  Win_allocate (rather than Win_create over a cudaMalloc
// pointer) is what makes the fabric path possible: fabric handles can
// only be exported from allocations the interposer owns.
#ifndef LULESH_COMM_MPIWRAP_H
#define LULESH_COMM_MPIWRAP_H

#include "comm_ipc_common.h"

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   COMM_CUDA_OK(cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0)) ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   MPI_Info info ;
   MPI_Info_create(&info) ;
   MPI_Info_set(info, "cuda_ipc", "1") ;
   // E2a breakdown: window construction. The interposer's cudaMalloc and its
   // collective handle Allgather both happen inside MPI_Win_allocate, so they
   // are not separable from the application side and are reported together.
   const double _t0win = MPI_Wtime() ;
   MPI_Win_allocate((MPI_Aint)(comBufSize*sizeof(Real_t)), sizeof(Real_t),
                    info, MPI_COMM_WORLD, (void *)&d->d_commDataRecv,
                    &d->ipcWin) ;
   g_phaseWinAllocMs = (MPI_Wtime() - _t0win) * 1000.0 ;
   MPI_Info_free(&info) ;
   d->d_peerRecv = new Real_t*[d->m_numRanks] ;
   int nFallback = 0 ;
   // E2a breakdown: lazy peer mapping. Each shared_query triggers at most one
   // cudaIpcOpenMemHandle inside the interposer.
   const double _t0q = MPI_Wtime() ;
   for (int r = 0; r < d->m_numRanks; ++r) {
      MPI_Aint sz ;
      int disp ;
      if (MPI_Win_shared_query(d->ipcWin, r, &sz, &disp,
                               (void **)&d->d_peerRecv[r]) != MPI_SUCCESS) {
         d->d_peerRecv[r] = NULL ;
      }
      if (r == myRank && d->d_peerRecv[r] == NULL) {
         // self-query always succeeds under the interposer
         fprintf(stderr, "rank %d: MPI_Win_shared_query failed for self "
                         "-- run with LD_PRELOAD=libmpiwrap.so\n", myRank) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
      if (r != myRank && d->d_peerRecv[r] == NULL) ++nFallback ;
   }
   g_phasePeerQueryMs = (MPI_Wtime() - _t0q) * 1000.0 ;
   /* Printed unconditionally: the transport mix is what the numbers mean.
      "0 of N not IPC-reachable" is the positive statement that every peer,
      including cross-node ones, is served by the window -- without it a
      clean run leaves no record of which transport was measured. */
   if (myRank == 0) {
      printf("winipc: %d of %d peers not IPC-reachable, using MPI "
             "send/recv fallback for them\n", nFallback,
             (int)d->m_numRanks - 1) ;
   }
}

static inline void commTeardown(Domain* d, int myRank)
{
   (void)myRank ;
   // The interceptor closes its mappings and frees the window allocation
   // (it owns d_commDataRecv) inside MPI_Win_free.
   MPI_Win_free(&d->ipcWin) ;
   delete [] d->d_peerRecv ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
}

#endif
