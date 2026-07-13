// comm_staged.h -- baseline two-sided MPI with host staging.
// GPU packs -> D2H into commDataSend -> MPI_Isend; MPI_Irecv into
// commDataRecv -> H2D into d_commDataRecv -> GPU unpacks.
#ifndef LULESH_COMM_STAGED_H
#define LULESH_COMM_STAGED_H

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;
   cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t)) ;
}

static inline void commTeardown(Domain* d, int myRank)
{
   (void)myRank ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
   cudaFree(d->d_commDataRecv) ;
}

#define COMM_RUNTIME_INIT()     ((void)0)
#define COMM_RUNTIME_SHUTDOWN() cudaDeviceReset()

#define COMM_RECV_SKIP(domain)     (0)
#define COMM_RECV_POST_BUF(domain) ((domain).commDataRecv)
#define COMM_RECV_BASE(domain)     ((domain).commDataRecv)

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      cudaMemcpyAsync(hostStage, d_src, (count)*sizeof(Real_t), \
                      cudaMemcpyDeviceToHost, stream); \
      cudaStreamSynchronize(stream); \
      MPI_Isend(hostStage, count, baseType, toRank, msgType, \
                MPI_COMM_WORLD, reqPtr) ; \
   } while (0)

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      Real_t *comBuf_ = (hostStage) ; \
      for (Index_t fi=0; fi<xferFields; ++fi) { \
         cudaMemcpyAsync(&comBuf_[fi], &((domain).*fieldData[fi])(idx), \
                         sizeof(Real_t), cudaMemcpyDeviceToHost, stream); \
      } \
      cudaStreamSynchronize(stream); \
      MPI_Isend(comBuf_, xferFields, baseType, toRank, msgType, \
                MPI_COMM_WORLD, reqPtr) ; \
   } while (0)

#define COMM_SEND_FINISH(domain, stream, status) \
   MPI_Waitall(26, (domain).sendRequest, status)

#define COMM_UNPACK_H2D(d_dst, src, bytes, stream) \
   cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream)

#define COMM_ADD_CORNER(stream, destPtr, comBuf, fi) \
   AddCorner<<<1,1,0,stream>>>(destPtr, (comBuf)[fi])
#define COMM_COPY_CORNER(stream, destPtr, comBuf, fi) \
   CopyCorner<<<1,1,0,stream>>>(destPtr, (comBuf)[fi])

#define COMM_MONOQ_COPY(dst, src, bytes, stream) \
   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)

#endif
