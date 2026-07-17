// comm_shmwin.h -- MPI shared-memory window.
// commDataRecv is this rank's slice of one MPI_Win_allocate_shared segment;
// MPI_Win_shared_query gives a direct pointer into each peer's slice, so a
// send is a single D2H cudaMemcpy into the neighbor's unpack buffer.
// Synchronization is Win_sync + barriers instead of send/recv completion.
// Single node only.  Unlike the other one-sided backends, this one is also
// active for the init-time host CommSend (COMM_HOST_* overrides below).
#ifndef LULESH_COMM_SHMWIN_H
#define LULESH_COMM_SHMWIN_H

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   int myRank, nodeRank, nodeSize ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                       MPI_INFO_NULL, &d->shmComm) ;
   MPI_Comm_rank(d->shmComm, &nodeRank) ;
   MPI_Comm_size(d->shmComm, &nodeSize) ;
   if (nodeSize != d->m_numRanks || nodeRank != myRank) {
      fprintf(stderr, "COMM_SHMWIN requires all ranks on a single node "
                      "(node holds %d of %d ranks)\n",
              nodeSize, (int)d->m_numRanks) ;
      MPI_Abort(MPI_COMM_WORLD, 1) ;
   }
   MPI_Win_allocate_shared((MPI_Aint)(comBufSize*sizeof(Real_t)),
                           sizeof(Real_t), MPI_INFO_NULL, d->shmComm,
                           &d->commDataRecv, &d->shmWin) ;
   d->peerRecv = new Real_t*[d->m_numRanks] ;
   for (int r = 0; r < d->m_numRanks; ++r) {
      MPI_Aint sz ;
      int disp ;
      MPI_Win_shared_query(d->shmWin, r, &sz, &disp,
                           (void **)&d->peerRecv[r]) ;
   }
   // Pin the whole node segment (slices are contiguous from rank 0's base)
   // so device-to-peer copies are true DMA.
   MPI_Aint lastSz ;
   int lastDisp ;
   Real_t *lastBase ;
   MPI_Win_shared_query(d->shmWin, d->m_numRanks-1, &lastSz, &lastDisp,
                        (void **)&lastBase) ;
   size_t segBytes = (char *)lastBase + lastSz - (char *)d->peerRecv[0] ;
   cudaHostRegister(d->peerRecv[0], segBytes, 0) ;
   MPI_Win_lock_all(MPI_MODE_NOCHECK, d->shmWin) ;

   cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t)) ;
}

static inline void commTeardown(Domain* d, int myRank)
{
   (void)myRank ;
   MPI_Win_unlock_all(d->shmWin) ;
   cudaHostUnregister(d->peerRecv[0]) ;
   MPI_Win_free(&d->shmWin) ;   // frees the segment backing commDataRecv
   delete [] d->peerRecv ;
   MPI_Comm_free(&d->shmComm) ;
   cudaFree(d->d_commDataRecv) ;
}

#define COMM_RUNTIME_INIT()     ((void)0)
#define COMM_RUNTIME_SHUTDOWN() cudaDeviceReset()

// Nobody posts receives -- neighbors write into our window slice.  The
// device sync + barrier keep them from overwriting data the previous comm
// phase is still unpacking.
static inline int commRecvSkip(Domain& domain)
{
   for (Index_t i=0; i<26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }
   cudaDeviceSynchronize() ;
   MPI_Barrier(domain.shmComm) ;
   return 1 ;
}
#define COMM_RECV_SKIP(domain, msgType, xferFields, dx, dy, dz, doRecv, planeOnly) \
   commRecvSkip(domain)
#define COMM_RECV_POST_BUF(domain) ((domain).commDataRecv)
#define COMM_RECV_BASE(domain)     ((domain).commDataRecv)

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(reqPtr) ; \
      cudaMemcpyAsync((domain).peerRecv[toRank] + \
                      shmRecvOffset(domain, toRank, dc, dr, dp, \
                                    xferFields, doSend, planeOnly), \
                      d_src, (count)*sizeof(Real_t), \
                      cudaMemcpyDeviceToHost, stream); \
      cudaStreamSynchronize(stream); \
   } while (0)

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(devStage) ; (void)(reqPtr) ; \
      Real_t *dst_ = (domain).peerRecv[toRank] + \
                     shmRecvOffset(domain, toRank, dc, dr, dp, \
                                   xferFields, doSend, planeOnly) ; \
      for (Index_t fi=0; fi<xferFields; ++fi) { \
         cudaMemcpyAsync(&dst_[fi], &((domain).*fieldData[fi])(idx), \
                         sizeof(Real_t), cudaMemcpyDeviceToHost, stream); \
      } \
      cudaStreamSynchronize(stream); \
   } while (0)

// Publish our writes and wait for every rank's halo data to land; the
// MPI_Waits in the unpack routines are then no-ops on MPI_REQUEST_NULL.
#define COMM_SEND_FINISH(domain, stream, status) \
   do { \
      (void)(status) ; \
      MPI_Win_sync((domain).shmWin) ; \
      MPI_Barrier((domain).shmComm) ; \
      MPI_Win_sync((domain).shmWin) ; \
   } while (0)

#define COMM_UNPACK_H2D(d_dst, src, bytes, stream) \
   cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream)

#define COMM_ADD_CORNER(stream, destPtr, comBuf, fi) \
   AddCorner<<<1,1,0,stream>>>(destPtr, (comBuf)[fi])
#define COMM_COPY_CORNER(stream, destPtr, comBuf, fi) \
   CopyCorner<<<1,1,0,stream>>>(destPtr, (comBuf)[fi])

#define COMM_MONOQ_COPY(dst, src, bytes, stream) \
   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)

// Host CommSend (init exchange): the CPU pack loops write straight into the
// peer's window slice -- a zero-copy put -- and nothing is Isent.
#define COMM_HOST_SEND_DEST(domain, toRank, dc, dr, dp, defaultPtr, xf, doSend, planeOnly) \
   ((domain).peerRecv[toRank] + \
    shmRecvOffset(domain, toRank, dc, dr, dp, xf, doSend, planeOnly))
#define COMM_HOST_ISEND(buf, count, baseType, toRank, msgType, reqPtr) \
   do { (void)(buf) ; (void)(baseType) ; (void)(reqPtr) ; } while (0)
#define COMM_HOST_SEND_FINISH(domain, status) \
   do { \
      (void)(status) ; \
      MPI_Win_sync((domain).shmWin) ; \
      MPI_Barrier((domain).shmComm) ; \
      MPI_Win_sync((domain).shmWin) ; \
   } while (0)

#endif
