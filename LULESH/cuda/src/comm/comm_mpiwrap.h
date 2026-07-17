// comm_mpiwrap.h -- CUDA IPC through the mpiwrap interceptor.
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
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   MPI_Info info ;
   MPI_Info_create(&info) ;
   MPI_Info_set(info, "cuda_ipc", "1") ;
   MPI_Win_allocate((MPI_Aint)(comBufSize*sizeof(Real_t)), sizeof(Real_t),
                    info, MPI_COMM_WORLD, (void *)&d->d_commDataRecv,
                    &d->ipcWin) ;
   MPI_Info_free(&info) ;
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
   // The interceptor closes its mappings and frees the window allocation
   // (it owns d_commDataRecv) inside MPI_Win_free.
   MPI_Win_free(&d->ipcWin) ;
   delete [] d->d_peerRecv ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
}

#endif
