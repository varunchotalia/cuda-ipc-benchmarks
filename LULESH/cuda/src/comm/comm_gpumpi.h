// comm_gpumpi.h -- GPU-aware two-sided MPI.
// Device pointers are passed straight to MPI_Isend/Irecv; no staging
// copies.  Requires a CUDA-aware MPI.  The init exchange still posts
// receives into the host buffer (g_commActive false).
#ifndef LULESH_COMM_GPUMPI_H
#define LULESH_COMM_GPUMPI_H

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

#define COMM_RECV_SKIP(domain) (0)
// After init, receives land straight in the device buffer.  The device
// sync keeps MPI from overwriting regions the previous phase's async
// unpack copies are still reading.
#define COMM_RECV_POST_BUF(domain) \
   (g_commActive ? (cudaDeviceSynchronize(), (domain).d_commDataRecv) \
                 : (domain).commDataRecv)
#define COMM_RECV_BASE(domain) ((domain).d_commDataRecv)

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; \
      cudaStreamSynchronize(stream); /* pack must finish before MPI reads */ \
      MPI_Isend(d_src, count, baseType, toRank, msgType, \
                MPI_COMM_WORLD, reqPtr) ; \
   } while (0)

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; \
      Real_t *d_comBuf_ = (devStage) ; \
      for (Index_t fi=0; fi<xferFields; ++fi) { \
         cudaMemcpyAsync(&d_comBuf_[fi], &((domain).*fieldData[fi])(idx), \
                         sizeof(Real_t), cudaMemcpyDeviceToDevice, stream); \
      } \
      cudaStreamSynchronize(stream); \
      MPI_Isend(d_comBuf_, xferFields, baseType, toRank, msgType, \
                MPI_COMM_WORLD, reqPtr) ; \
   } while (0)

#define COMM_SEND_FINISH(domain, stream, status) \
   MPI_Waitall(26, (domain).sendRequest, status)

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
