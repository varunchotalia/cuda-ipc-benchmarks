// comm_mpiwrap.h -- CUDA IPC through the mpiwrap interceptor.
// The application only speaks portable MPI window code: MPI_Win_create
// over the device recv buffer + MPI_Win_shared_query per peer.  Vanilla
// MPI cannot shared_query a created window; libmpiwrap.so (LD_PRELOAD)
// intercepts both calls and backs them with CUDA IPC, yielding the same
// d_peerRecv mapping as comm_ipc.h.
#ifndef LULESH_COMM_MPIWRAP_H
#define LULESH_COMM_MPIWRAP_H

#include "comm_ipc_common.h"

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;
   cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t)) ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   MPI_Win_create(d->d_commDataRecv, (MPI_Aint)(comBufSize*sizeof(Real_t)),
                  sizeof(Real_t), MPI_INFO_NULL, MPI_COMM_WORLD,
                  &d->ipcWin) ;
   d->d_peerRecv = new Real_t*[d->m_numRanks] ;
   for (int r = 0; r < d->m_numRanks; ++r) {
      MPI_Aint sz ;
      int disp ;
      if (MPI_Win_shared_query(d->ipcWin, r, &sz, &disp,
                               (void **)&d->d_peerRecv[r]) != MPI_SUCCESS
          || d->d_peerRecv[r] == NULL) {
         fprintf(stderr, "rank %d: MPI_Win_shared_query failed for rank %d "
                         "-- run with LD_PRELOAD=libmpiwrap.so\n", myRank, r) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
   }
}

static inline void commTeardown(Domain* d, int myRank)
{
   (void)myRank ;
   // The interceptor closes its CUDA IPC mappings inside MPI_Win_free.
   MPI_Win_free(&d->ipcWin) ;
   delete [] d->d_peerRecv ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
   cudaFree(d->d_commDataRecv) ;
}

#endif
