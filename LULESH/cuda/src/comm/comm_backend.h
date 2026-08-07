// comm/comm_backend.h
//
// Dispatches to exactly one halo-exchange backend and supplies the pieces
// they share.  A backend defines HOW bytes move; the pack/unpack logic in
// lulesh-comms*.cu is identical for all of them.
//
// Backend API (macros or static inline functions):
//   commAllocRecv(Domain*, Index_t)      recv buffers + peer state
//   commTeardown(Domain*, int myRank)    cleanup before MPI_Finalize
//   COMM_RUNTIME_INIT()                  process-level init (after cuda_init)
//   COMM_RUNTIME_SHUTDOWN()              process-level shutdown
//   COMM_RECV_SKIP(domain)               one-sided: sync + skip posting recvs
//   COMM_RECV_POST_BUF(domain)           where MPI_Irecv should land
//   COMM_SEND_MSG(...)                   one face/edge message
//   COMM_SEND_CORNER(...)                one corner message
//   COMM_SEND_FINISH(domain,stream,st)   completion / publication
//   COMM_RECV_BASE(domain)               buffer the unpack routines read
//   COMM_UNPACK_H2D(dst,src,bytes,strm)  staging copy into device, or no-op
//   COMM_ADD_CORNER / COMM_COPY_CORNER   corner unpack kernel launch
//   COMM_MONOQ_COPY(dst,src,bytes,strm)  MonoQ unpack copy (H2D or D2D)
// Host-side CommSend hooks (COMM_HOST_*) default below; only the
// shared-memory-window backend overrides them.

#ifndef LULESH_COMM_BACKEND_H
#define LULESH_COMM_BACKEND_H

#if USE_MPI

// The init-time nodalMass exchange is host-packed plain MPI in every
// backend; the one-sided backends switch on after it completes.
extern bool g_commActive ;

/* Lifecycle phase accounting (E2a). Defined in lulesh.cu, reported as PHASE_*
   lines at the end of the run.

   g_phaseCommSetupMs is measured in lulesh.cu around the single
   commAllocRecv() call, so it is uniform across ALL backends: whatever a
   backend needs to become ready to exchange halos is charged to it.

   The two fields below are an OPTIONAL finer breakdown that only a
   window-based backend can provide. A backend that sets them must ensure
   they sum to no more than its share of setup; a backend that leaves them
   at zero simply reports no breakdown, and only the uniform total is
   comparable across backends. Currently only WinIPC (comm_mpiwrap.h) does. */
extern double g_phaseCommSetupMs ;   /* all backends: whole commAllocRecv */
extern double g_phaseCommFreeMs ;    /* all backends: whole commTeardown  */
extern double g_phaseWinAllocMs ;    /* window construction only, or 0    */
extern double g_phasePeerQueryMs ;   /* peer query / mapping only, or 0   */

/* Setup-path error checking.  Every backend allocates and maps its recv
   buffers before any solver work runs, and these calls used to go unchecked:
   a failure surfaced much later as a segfault or a bare MPI abort with
   nothing naming the cause.  That is exactly what the GB200 NVL72 27- and
   64-rank runs reported ("FAILED (crash)", no attribution).  Check at the
   call site so the log names the rank, the call and the CUDA error. */
#define COMM_CUDA_OK(call)                                                   \
   do {                                                                      \
      cudaError_t _e_ = (call) ;                                             \
      if (_e_ != cudaSuccess) {                                              \
         int _r_ = -1 ; MPI_Comm_rank(MPI_COMM_WORLD, &_r_) ;                \
         fprintf(stderr, "rank %d: %s:%d: %s failed: %s\n", _r_,             \
                 __FILE__, __LINE__, #call, cudaGetErrorString(_e_)) ;       \
         fflush(stderr) ;                                                    \
         MPI_Abort(MPI_COMM_WORLD, 1) ;                                      \
      }                                                                      \
   } while (0)

/* Mirrors the posting order of CommRecv() to compute the offset within rank
   recvRank's commDataRecv where the message arriving from direction
   (dCol,dRow,dPlane) -- the sender's position relative to the receiver --
   is expected.  Lets a sender write halo data directly into the receiver's
   recv buffer instead of MPI_Isend.  Valid because every rank's subdomain
   has identical dimensions, so maxPlaneSize/maxEdgeSize agree, and doSend
   at the send site always equals doRecv at the matching CommRecv site. */
static inline Index_t shmRecvOffset(Domain& domain, int recvRank,
                                    int dCol, int dRow, int dPlane,
                                    Index_t xferFields, bool doRecv,
                                    bool planeOnly)
{
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0, emsg = 0, cmsg = 0 ;
   int tp = domain.tp() ;
   int rPlane = recvRank / (tp*tp) ;
   int rRow   = (recvRank % (tp*tp)) / tp ;
   int rCol   = recvRank % tp ;
   bool rowMin   = (rRow   != 0), rowMax   = (rRow   != tp-1) ;
   bool colMin   = (rCol   != 0), colMax   = (rCol   != tp-1) ;
   bool planeMin = (rPlane != 0), planeMax = (rPlane != tp-1) ;

#define SHM_FACE(cond, c, r, p) \
   if (cond) { if (dCol==(c) && dRow==(r) && dPlane==(p)) \
      return pmsg*maxPlaneComm ; ++pmsg ; }
#define SHM_EDGE(cond, c, r, p) \
   if (cond) { if (dCol==(c) && dRow==(r) && dPlane==(p)) \
      return pmsg*maxPlaneComm + emsg*maxEdgeComm ; ++emsg ; }
#define SHM_CORNER(cond, c, r, p) \
   if (cond) { if (dCol==(c) && dRow==(r) && dPlane==(p)) \
      return pmsg*maxPlaneComm + emsg*maxEdgeComm + \
             cmsg*CACHE_COHERENCE_PAD_REAL ; ++cmsg ; }

   SHM_FACE(planeMin && doRecv,  0,  0, -1)
   SHM_FACE(planeMax,            0,  0,  1)
   SHM_FACE(rowMin && doRecv,    0, -1,  0)
   SHM_FACE(rowMax,              0,  1,  0)
   SHM_FACE(colMin && doRecv,   -1,  0,  0)
   SHM_FACE(colMax,              1,  0,  0)
   if (!planeOnly) {
      SHM_EDGE(rowMin && colMin && doRecv,   -1, -1,  0)
      SHM_EDGE(rowMin && planeMin && doRecv,  0, -1, -1)
      SHM_EDGE(colMin && planeMin && doRecv, -1,  0, -1)
      SHM_EDGE(rowMax && colMax,              1,  1,  0)
      SHM_EDGE(rowMax && planeMax,            0,  1,  1)
      SHM_EDGE(colMax && planeMax,            1,  0,  1)
      SHM_EDGE(rowMax && colMin,             -1,  1,  0)
      SHM_EDGE(rowMin && planeMax,            0, -1,  1)
      SHM_EDGE(colMin && planeMax,           -1,  0,  1)
      SHM_EDGE(rowMin && colMax && doRecv,    1, -1,  0)
      SHM_EDGE(rowMax && planeMin && doRecv,  0,  1, -1)
      SHM_EDGE(colMax && planeMin && doRecv,  1,  0, -1)
      SHM_CORNER(rowMin && colMin && planeMin && doRecv, -1, -1, -1)
      SHM_CORNER(rowMin && colMin && planeMax,           -1, -1,  1)
      SHM_CORNER(rowMin && colMax && planeMin && doRecv,  1, -1, -1)
      SHM_CORNER(rowMin && colMax && planeMax,            1, -1,  1)
      SHM_CORNER(rowMax && colMin && planeMin && doRecv, -1,  1, -1)
      SHM_CORNER(rowMax && colMin && planeMax,           -1,  1,  1)
      SHM_CORNER(rowMax && colMax && planeMin && doRecv,  1,  1, -1)
      SHM_CORNER(rowMax && colMax && planeMax,            1,  1,  1)
   }
#undef SHM_FACE
#undef SHM_EDGE
#undef SHM_CORNER

   fprintf(stderr, "shmRecvOffset: rank %d expects no message from "
                   "direction (%d,%d,%d)\n", recvRank, dCol, dRow, dPlane) ;
   MPI_Abort(MPI_COMM_WORLD, 1) ;
   return 0 ;
}

#if defined(COMM_DIRECT)
#include "comm_direct.h"
#elif defined(COMM_NVSHMEM)
#include "comm_nvshmem.h"
#elif defined(COMM_IPC) && defined(IPC_VIA_MPIWRAP)
#include "comm_mpiwrap.h"
#elif defined(COMM_IPC)
#include "comm_ipc.h"
#elif defined(COMM_SHMWIN)
#include "comm_shmwin.h"
#elif defined(COMM_GPUMPI)
#include "comm_gpumpi.h"
#else
#include "comm_staged.h"
#endif

// Where the GPU pack kernels write.  Default: the local device staging
// buffer.  Remote-pack backends override this to point straight into the
// peer's mapped recv buffer.
#ifndef COMM_PACK_DEST
#define COMM_PACK_DEST(domain, toRank, dc, dr, dp, localPtr, xferFields, doSend, planeOnly, msgType) \
   (localPtr)
#endif

// Host-side CommSend hooks: by default the init exchange packs into
// commDataSend and Isends it (matching the Irecvs CommRecv posts).
#ifndef COMM_HOST_SEND_DEST
#define COMM_HOST_SEND_DEST(domain, toRank, dc, dr, dp, defaultPtr, xf, doSend, planeOnly) \
   (defaultPtr)
#define COMM_HOST_ISEND(buf, count, baseType, toRank, msgType, reqPtr) \
   MPI_Isend(buf, count, baseType, toRank, msgType, MPI_COMM_WORLD, reqPtr)
#define COMM_HOST_SEND_FINISH(domain, status) \
   MPI_Waitall(26, (domain).sendRequest, status)
#endif

#endif /* USE_MPI */
#endif /* LULESH_COMM_BACKEND_H */
