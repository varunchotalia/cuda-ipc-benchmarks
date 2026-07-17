// comm_ipc_common.h -- transfer/unpack machinery shared by the two CUDA-IPC
// backends (comm_ipc.h and comm_mpiwrap.h).  Both end up with d_peerRecv[r]
// = rank r's d_commDataRecv mapped into this process; they differ only in
// how the mapping is established.
//
// Hybrid transport: d_peerRecv[r] == NULL means rank r is not IPC-reachable
// (e.g. it lives on another node).  IPC peers get one-sided puts; NULL
// peers fall back to real MPI send/recv of the same packed messages, with
// receives posted into the device recv buffer (requires CUDA-aware MPI).
//
// Synchronization is per-neighbor, not global: instead of world barriers,
// zero-byte token messages mirror the original send/recv matching.
//   tag msgType   : "data delivered" -- receiver's unpack MPI_Waits on it
//                   through the same recvRequest slots as real messages
//   tag msgType+1 : "my recv buffer is free" -- sender waits for it before
//                   putting into that peer
// This costs O(neighbors) messages per phase and scales with node count,
// where a barrier would cost O(all ranks).
#ifndef LULESH_COMM_IPC_COMMON_H
#define LULESH_COMM_IPC_COMMON_H

#define COMM_RUNTIME_INIT()     ((void)0)
#define COMM_RUNTIME_SHUTDOWN() cudaDeviceReset()

// Allocate the standard recv buffers and map every IPC-reachable peer's
// d_commDataRecv through an explicit CUDA-IPC handle exchange.  Peers on
// other nodes get d_peerRecv[r] = NULL (MPI fallback).
static inline void commIpcAllocAndMapPacked(Domain* d, Index_t comBufSize)
{
   d->commDataRecv = new Real_t[comBufSize] ;
   cudaHostRegister(d->commDataRecv, comBufSize*sizeof(Real_t), 0) ;
   cudaMalloc(&d->d_commDataRecv, comBufSize*sizeof(Real_t)) ;

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
         fprintf(stderr, "rank %d: cudaIpcOpenMemHandle for same-node "
                         "rank %d failed\n", myRank, r) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
   }
   if (myRank == 0 && nFallback > 0) {
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

/******************************************/
/* per-neighbor synchronization tokens */

static char s_commTokDummy ;

// sender: block until the peer's recv buffer is free for this phase
static inline void commReadyWait(int toRank, int msgType)
{
   MPI_Recv(&s_commTokDummy, 0, MPI_BYTE, toRank, msgType + 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE) ;
}

// deferred "data delivered" tokens: recorded per put, sent after the
// stream sync in COMM_SEND_FINISH (the puts are asynchronous until then)
static int          s_commTokPeer[26] ;
static int          s_commTokTag[26] ;
static MPI_Request *s_commTokReq[26] ;
static int          s_commTokN = 0 ;

static inline void commTokenRecord(int toRank, int msgType, MPI_Request *req)
{
   s_commTokPeer[s_commTokN] = toRank ;
   s_commTokTag[s_commTokN]  = msgType ;
   s_commTokReq[s_commTokN]  = req ;
   ++s_commTokN ;
}

static inline void commTokenFlush(cudaStream_t stream)
{
   cudaStreamSynchronize(stream) ;   // all puts have landed
   for (int i = 0; i < s_commTokN; ++i) {
      MPI_Isend(&s_commTokDummy, 0, MPI_BYTE, s_commTokPeer[i],
                s_commTokTag[i], MPI_COMM_WORLD, s_commTokReq[i]) ;
   }
   s_commTokN = 0 ;
}

/******************************************/
/* receive side: replaces CommRecv's posting loop after init.
   Mirrors CommRecv's guarded slot order exactly.  IPC in-neighbors get a
   zero-byte token Irecv (arrival sync through the same recvRequest slot
   the unpack routines already MPI_Wait on) plus a "buffer free" ready
   send; MPI-fallback in-neighbors get the original real Irecv, posted
   into the DEVICE recv buffer. */

static inline int commRecvBegin(Domain& domain, int msgType,
                                Index_t xferFields,
                                Index_t dx, Index_t dy, Index_t dz,
                                bool doRecv, bool planeOnly)
{
   if (!g_commActive) return 0 ;   // init nodalMass exchange: plain MPI

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   int tp = (int)domain.tp() ;
   int tp2 = tp*tp ;
   bool rowMin   = (domain.rowLoc() != 0),   rowMax   = (domain.rowLoc() != tp-1) ;
   bool colMin   = (domain.colLoc() != 0),   colMax   = (domain.colLoc() != tp-1) ;
   bool planeMin = (domain.planeLoc() != 0), planeMax = (domain.planeLoc() != tp-1) ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0, emsg = 0, cmsg = 0 ;

   for (Index_t i = 0; i < 26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }
   /* the previous phase's unpack (its H2D copies and kernels) must have
      consumed the recv buffer before we hand it back to our neighbors */
   cudaDeviceSynchronize() ;

#define COMM_RB_POST(from, count, offset, slot)                             \
   do {                                                                     \
      if (domain.d_peerRecv[from]) {                                        \
         MPI_Irecv(&s_commTokDummy, 0, MPI_BYTE, from, msgType,             \
                   MPI_COMM_WORLD, &domain.recvRequest[slot]) ;             \
         MPI_Send(&s_commTokDummy, 0, MPI_BYTE, from, msgType + 1,          \
                  MPI_COMM_WORLD) ;                                         \
      } else {                                                              \
         MPI_Irecv(&domain.d_commDataRecv[offset], count, baseType,         \
                   from, msgType, MPI_COMM_WORLD,                           \
                   &domain.recvRequest[slot]) ;                             \
      }                                                                     \
   } while (0)

   {
      const bool    g[6] = { planeMin && doRecv, planeMax,
                             rowMin && doRecv,   rowMax,
                             colMin && doRecv,   colMax } ;
      const int     d[6] = { -tp2, tp2, -tp, tp, -1, 1 } ;
      const Index_t c[6] = { dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz } ;
      for (int f = 0; f < 6; ++f) {
         if (!g[f]) continue ;
         COMM_RB_POST(myRank + d[f], c[f]*xferFields,
                      pmsg * maxPlaneComm, pmsg) ;
         ++pmsg ;
      }
   }
   if (!planeOnly) {
      const bool ge[12] = {
         rowMin && colMin && doRecv,   rowMin && planeMin && doRecv,
         colMin && planeMin && doRecv, rowMax && colMax,
         rowMax && planeMax,           colMax && planeMax,
         rowMax && colMin,             rowMin && planeMax,
         colMin && planeMax,           rowMin && colMax && doRecv,
         rowMax && planeMin && doRecv, colMax && planeMin && doRecv } ;
      const int de[12] = { -tp-1, -tp2-tp, -tp2-1,
                            tp+1,  tp2+tp,  tp2+1,
                            tp-1,  tp2-tp,  tp2-1,
                           -tp+1, -tp2+tp, -tp2+1 } ;
      const Index_t ce[12] = { dz, dx, dy, dz, dx, dy,
                               dz, dx, dy, dz, dx, dy } ;
      for (int e = 0; e < 12; ++e) {
         if (!ge[e]) continue ;
         COMM_RB_POST(myRank + de[e], ce[e]*xferFields,
                      pmsg * maxPlaneComm + emsg * maxEdgeComm,
                      pmsg + emsg) ;
         ++emsg ;
      }
      const bool gc[8] = {
         rowMin && colMin && planeMin && doRecv,
         rowMin && colMin && planeMax,
         rowMin && colMax && planeMin && doRecv,
         rowMin && colMax && planeMax,
         rowMax && colMin && planeMin && doRecv,
         rowMax && colMin && planeMax,
         rowMax && colMax && planeMin && doRecv,
         rowMax && colMax && planeMax } ;
      const int dc[8] = { -tp2-tp-1,  tp2-tp-1, -tp2-tp+1,  tp2-tp+1,
                          -tp2+tp-1,  tp2+tp-1, -tp2+tp+1,  tp2+tp+1 } ;
      for (int cn = 0; cn < 8; ++cn) {
         if (!gc[cn]) continue ;
         COMM_RB_POST(myRank + dc[cn], xferFields,
                      pmsg * maxPlaneComm + emsg * maxEdgeComm +
                      cmsg * CACHE_COHERENCE_PAD_REAL,
                      pmsg + emsg + cmsg) ;
         ++cmsg ;
      }
   }
#undef COMM_RB_POST
   return 1 ;
}

#define COMM_RECV_SKIP(domain, msgType, xferFields, dx, dy, dz, doRecv, planeOnly) \
   commRecvBegin(domain, msgType, xferFields, dx, dy, dz, doRecv, planeOnly)
#define COMM_RECV_POST_BUF(domain) ((domain).commDataRecv)
#define COMM_RECV_BASE(domain)     ((domain).d_commDataRecv)

/******************************************/
/* send side */

#ifdef IPC_REMOTE_PACK
// Mode C: for IPC peers the pack kernels themselves write into the peer's
// packed recv buffer, so the "put" is done when packing finishes; the
// ready-wait must precede the pack-kernel ENQUEUE.  MPI-fallback peers
// pack locally (mode-A style) and get a real Isend in COMM_SEND_MSG.
#define COMM_PACK_DEST(domain, toRank, dc, dr, dp, localPtr, xferFields, doSend, planeOnly, msgType) \
   ((domain).d_peerRecv[toRank]                                              \
    ? (commReadyWait(toRank, msgType),                                       \
       (domain).d_peerRecv[toRank] +                                         \
       shmRecvOffset(domain, toRank, dc, dr, dp, xferFields, doSend, planeOnly)) \
    : (localPtr))

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; \
      if ((domain).d_peerRecv[toRank]) { \
         commTokenRecord(toRank, msgType, reqPtr) ; \
      } else { \
         cudaStreamSynchronize(stream); /* local pack must finish */ \
         MPI_Isend(d_src, count, baseType, toRank, msgType, \
                   MPI_COMM_WORLD, reqPtr) ; \
      } \
   } while (0)
#else
// Mode A: pack locally, then one D2D copy into the IPC peer's recv buffer
// (or a real Isend of the packed message for MPI-fallback peers).
#define COMM_PACK_DEST(domain, toRank, dc, dr, dp, localPtr, xferFields, doSend, planeOnly, msgType) \
   (localPtr)

#define COMM_SEND_MSG(domain, toRank, dc, dr, dp, d_src, hostStage, count, \
                      baseType, msgType, reqPtr, stream, xferFields, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; \
      if ((domain).d_peerRecv[toRank]) { \
         commReadyWait(toRank, msgType) ; \
         cudaMemcpyAsync((domain).d_peerRecv[toRank] + \
                         shmRecvOffset(domain, toRank, dc, dr, dp, \
                                       xferFields, doSend, planeOnly), \
                         d_src, (count)*sizeof(Real_t), \
                         cudaMemcpyDeviceToDevice, stream); \
         commTokenRecord(toRank, msgType, reqPtr) ; \
      } else { \
         cudaStreamSynchronize(stream); \
         MPI_Isend(d_src, count, baseType, toRank, msgType, \
                   MPI_COMM_WORLD, reqPtr) ; \
      } \
   } while (0)
#endif

#define COMM_SEND_CORNER(domain, toRank, dc, dr, dp, fieldData, idx, \
                         hostStage, devStage, xferFields, baseType, msgType, \
                         reqPtr, stream, doSend, planeOnly) \
   do { \
      (void)(hostStage) ; \
      if ((domain).d_peerRecv[toRank]) { \
         commReadyWait(toRank, msgType) ; \
         Real_t *dst_ = (domain).d_peerRecv[toRank] + \
                        shmRecvOffset(domain, toRank, dc, dr, dp, \
                                      xferFields, doSend, planeOnly) ; \
         for (Index_t fi=0; fi<xferFields; ++fi) { \
            cudaMemcpyAsync(&dst_[fi], &((domain).*fieldData[fi])(idx), \
                            sizeof(Real_t), cudaMemcpyDeviceToDevice, stream); \
         } \
         commTokenRecord(toRank, msgType, reqPtr) ; \
      } else { \
         Real_t *d_stage_ = (devStage) ; \
         for (Index_t fi=0; fi<xferFields; ++fi) { \
            cudaMemcpyAsync(&d_stage_[fi], &((domain).*fieldData[fi])(idx), \
                            sizeof(Real_t), cudaMemcpyDeviceToDevice, stream); \
         } \
         cudaStreamSynchronize(stream); \
         MPI_Isend(d_stage_, xferFields, baseType, toRank, msgType, \
                   MPI_COMM_WORLD, reqPtr) ; \
      } \
   } while (0)

// Drain the puts, then release per-neighbor "delivered" tokens and
// complete the fallback Isends.  No global barrier: each receiver's
// unpack MPI_Waits provide exactly the arrival sync it needs.
#define COMM_SEND_FINISH(domain, stream, status) \
   do { \
      commTokenFlush(stream) ; \
      MPI_Waitall(26, (domain).sendRequest, status) ; \
   } while (0)

// Halo data is device-resident either way (IPC put or CUDA-aware recv);
// no host staging copy.
#define COMM_UNPACK_H2D(d_dst, src, bytes, stream) \
   do { (void)(d_dst) ; (void)(src) ; (void)(bytes) ; } while (0)

#define COMM_ADD_CORNER(stream, destPtr, comBuf, fi) \
   AddCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])
#define COMM_COPY_CORNER(stream, destPtr, comBuf, fi) \
   CopyCornerPtr<<<1,1,0,stream>>>(destPtr, &(comBuf)[fi])

#define COMM_MONOQ_COPY(dst, src, bytes, stream) \
   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)

#endif
