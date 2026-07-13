// comm_ipc_common.h -- transfer/unpack machinery shared by the two CUDA-IPC
// backends (comm_ipc.h and comm_mpiwrap.h).  Both end up with d_peerRecv[r]
// = rank r's d_commDataRecv mapped into this process; they differ only in
// how the mapping is established.  A send is one device-to-device
// cudaMemcpy over NVLink into the peer GPU's unpack buffer.
#ifndef LULESH_COMM_IPC_COMMON_H
#define LULESH_COMM_IPC_COMMON_H

#define COMM_RUNTIME_INIT()     ((void)0)
#define COMM_RUNTIME_SHUTDOWN() cudaDeviceReset()

// Allocate the standard recv buffers and map every peer's d_commDataRecv
// through an explicit CUDA-IPC handle exchange.
static inline void commIpcAllocAndMapPacked(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;
   cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t)) ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
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
   for (int r = 0; r < d->m_numRanks; ++r) {
      if (r == myRank) {
         d->d_peerRecv[r] = d->d_commDataRecv ;
      }
      else if (cudaIpcOpenMemHandle((void **)&d->d_peerRecv[r],
                                    allHandles[r],
                                    cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
         fprintf(stderr, "rank %d: cudaIpcOpenMemHandle for rank %d failed "
                         "(CUDA IPC requires a single node)\n", myRank, r) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
   }
   delete [] allHandles ;
}

static inline void commIpcUnmapAndFreePacked(Domain* d, int myRank)
{
   int nRanks ;
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks) ;
   for (int r = 0; r < nRanks; ++r) {
      if (r != myRank) cudaIpcCloseMemHandle(d->d_peerRecv[r]) ;
   }
   delete [] d->d_peerRecv ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
   cudaFree(d->d_commDataRecv) ;
}

// One-sided after init: no receives posted; neighbors write our device
// buffer through the IPC mapping.
static inline int commRecvSkip(Domain& domain)
{
   if (!g_commActive) return 0 ;
   for (Index_t i=0; i<26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }
   cudaDeviceSynchronize() ;
   MPI_Barrier(MPI_COMM_WORLD) ;
   return 1 ;
}
#define COMM_RECV_SKIP(domain)     commRecvSkip(domain)
#define COMM_RECV_POST_BUF(domain) ((domain).commDataRecv)
#define COMM_RECV_BASE(domain)     ((domain).d_commDataRecv)

#ifdef IPC_REMOTE_PACK
// Mode C: the pack kernels write straight into the peer's packed recv
// buffer (no local staging, no separate copy), so the "send" is already
// done once packing finishes.
#define COMM_PACK_DEST(domain, toRank, dc, dr, dp, localPtr, xferFields, doSend, planeOnly) \
   ((domain).d_peerRecv[toRank] + \
    shmRecvOffset(domain, toRank, dc, dr, dp, xferFields, doSend, planeOnly))
#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { (void)(hostStage) ; (void)(reqPtr) ; (void)(d_src) ; } while (0)
#else
// Mode A: pack locally, then one D2D copy into the peer's recv buffer.
#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(reqPtr) ; \
      cudaMemcpyAsync((domain).d_peerRecv[toRank] + \
                      shmRecvOffset(domain, toRank, dc, dr, dp, \
                                    xferFields, doSend, planeOnly), \
                      d_src, (count)*sizeof(Real_t), \
                      cudaMemcpyDeviceToDevice, stream); \
   } while (0)
#endif

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(devStage) ; (void)(reqPtr) ; \
      Real_t *dst_ = (domain).d_peerRecv[toRank] + \
                     shmRecvOffset(domain, toRank, dc, dr, dp, \
                                   xferFields, doSend, planeOnly) ; \
      for (Index_t fi=0; fi<xferFields; ++fi) { \
         cudaMemcpyAsync(&dst_[fi], &((domain).*fieldData[fi])(idx), \
                         sizeof(Real_t), cudaMemcpyDeviceToDevice, stream); \
      } \
   } while (0)

// Drain our copies into the peers' recv buffers, then barrier so every
// rank's halo data has landed before anyone unpacks.
#define COMM_SEND_FINISH(domain, stream, status) \
   do { \
      (void)(status) ; \
      cudaStreamSynchronize(stream) ; \
      MPI_Barrier(MPI_COMM_WORLD) ; \
   } while (0)

// Halo data is already device-resident; no staging copy.
#define COMM_UNPACK_H2D(d_dst, src, bytes, stream) \
   do { (void)(d_dst) ; (void)(src) ; (void)(bytes) ; } while (0)

#define COMM_ADD_CORNER(stream, destPtr, comBuf, fi) \
   AddCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])
#define COMM_COPY_CORNER(stream, destPtr, comBuf, fi) \
   CopyCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])

#define COMM_MONOQ_COPY(dst, src, bytes, stream) \
   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)

#endif
