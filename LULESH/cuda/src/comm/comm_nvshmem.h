// comm_nvshmem.h -- NVSHMEM symmetric heap.
// d_commDataRecv lives on the symmetric heap, so a send is one
// nvshmemx_putmem_on_stream from the local pack buffer straight into the
// peer GPU's unpack buffer.  One PE per GPU required.
#ifndef LULESH_COMM_NVSHMEM_H
#define LULESH_COMM_NVSHMEM_H

#include <nvshmem.h>
#include <nvshmemx.h>

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;

   // nvshmem_malloc is collective and needs the same size on every PE;
   // boundary ranks have smaller comBufSize, so use the global max.
   int myBuf = (int)comBufSize, maxBuf = 0 ;
   MPI_Allreduce(&myBuf, &maxBuf, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
   d->d_commDataRecv = (Real_t *)nvshmem_malloc((size_t)maxBuf*sizeof(Real_t)) ;
   if (d->d_commDataRecv == NULL) {
      fprintf(stderr, "nvshmem_malloc of %d Real_t failed\n", maxBuf) ;
      MPI_Abort(MPI_COMM_WORLD, 1) ;
   }
}

static inline void commTeardown(Domain* d, int myRank)
{
   (void)myRank ;
   cudaHostUnregister(d->commDataRecv) ;
   delete [] d->commDataRecv ;
   // must precede nvshmem_finalize() in COMM_RUNTIME_SHUTDOWN
   nvshmem_free(d->d_commDataRecv) ;
}

#define COMM_RUNTIME_INIT() \
   do { \
      nvshmemx_init_attr_t attr_ ; \
      MPI_Comm comm_ = MPI_COMM_WORLD ; \
      attr_.mpi_comm = &comm_ ; \
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr_) ; \
   } while (0)
#define COMM_RUNTIME_SHUTDOWN() nvshmem_finalize()

// One-sided after init: no receives posted; neighbors put into our
// symmetric recv buffer.
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

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(reqPtr) ; \
      nvshmemx_putmem_on_stream((domain).d_commDataRecv + \
                                shmRecvOffset(domain, toRank, dc, dr, dp, \
                                              xferFields, doSend, planeOnly), \
                                d_src, (count)*sizeof(Real_t), toRank, stream); \
   } while (0)

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; (void)(devStage) ; (void)(reqPtr) ; \
      Real_t *dst_ = (domain).d_commDataRecv + \
                     shmRecvOffset(domain, toRank, dc, dr, dp, \
                                   xferFields, doSend, planeOnly) ; \
      for (Index_t fi=0; fi<xferFields; ++fi) { \
         nvshmemx_putmem_on_stream(&dst_[fi], &((domain).*fieldData[fi])(idx), \
                                   sizeof(Real_t), toRank, stream); \
      } \
   } while (0)

// Flush all puts, then barrier so every rank's halo data has landed.
#define COMM_SEND_FINISH(domain, stream, status) \
   do { \
      (void)(status) ; \
      nvshmemx_quiet_on_stream(stream) ; \
      cudaStreamSynchronize(stream) ; \
      MPI_Barrier(MPI_COMM_WORLD) ; \
   } while (0)

#define COMM_UNPACK_H2D(d_dst, src, bytes, stream) \
   do { (void)(d_dst) ; (void)(src) ; (void)(bytes) ; } while (0)

#define COMM_ADD_CORNER(stream, destPtr, comBuf, fi) \
   AddCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])
#define COMM_COPY_CORNER(stream, destPtr, comBuf, fi) \
   CopyCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])

#define COMM_MONOQ_COPY(dst, src, bytes, stream) \
   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)

#endif
