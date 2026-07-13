// lulesh-comms-direct.cu -- Mode B ("direct") halo exchange.
//
// Replaces lulesh-comms-gpu.cu in the COMM_DIRECT build.  There is no pack
// and no unpack: for each halo message the sender launches one kernel that
// reads its own boundary values and writes them straight into the
// RECEIVER's field arrays through CUDA-IPC mappings established at setup.
//
//   - SBN (force summation): atomicAdd, because up to seven neighbors
//     legitimately contribute to the same shared edge/corner node and
//     non-atomic cross-GPU += would lose updates.
//   - SyncPosVel: plain stores (overlapping writers carry the value of the
//     same physical node).
//   - MonoQ: its destinations (delv_xi/eta/zeta) are per-step pool
//     allocations whose addresses cannot be premapped, so MonoQ remote-
//     packs into the peer's packed recv buffer and unpacks locally.
//
// Synchronization: a device-sync + barrier on entry to CommSendGpu
// guarantees every rank finished writing its own fields before any remote
// write lands; a stream-sync + barrier on exit guarantees all remote
// writes have landed before anyone reads.
//
// Only the structured (-s) path is supported: it allocates the nodal
// fields before SetupCommBuffers, so the peer-field mappings taken there
// are valid.  The unstructured (-u) path does not call SetupCommBuffers
// and has not been tested with any comm backend, this one included.

#if USE_MPI
#include <mpi.h>
#endif

#include "lulesh.h"

#if USE_MPI

/******************************************/
/* peer mappings for the persistent nodal fields */

#define NDIRECT_FIELDS 9

static Real_t **s_peerField[NDIRECT_FIELDS] ;

void commDirectMapFields(Domain* d)
{
   Real_t* mine[NDIRECT_FIELDS] = {
      d->x.raw(),  d->y.raw(),  d->z.raw(),
      d->xd.raw(), d->yd.raw(), d->zd.raw(),
      d->fx.raw(), d->fy.raw(), d->fz.raw() } ;

   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   int n = (int)d->m_numRanks ;
   cudaIpcMemHandle_t *all = new cudaIpcMemHandle_t[n] ;

   for (int f = 0; f < NDIRECT_FIELDS; ++f) {
      cudaIpcMemHandle_t h ;
      if (cudaIpcGetMemHandle(&h, mine[f]) != cudaSuccess) {
         fprintf(stderr, "rank %d: cudaIpcGetMemHandle(field %d) failed\n",
                 myRank, f) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
      MPI_Allgather(&h, sizeof(h), MPI_BYTE, all, sizeof(h), MPI_BYTE,
                    MPI_COMM_WORLD) ;
      s_peerField[f] = new Real_t*[n] ;
      for (int r = 0; r < n; ++r) {
         if (r == myRank) {
            s_peerField[f][r] = mine[f] ;
         }
         else if (cudaIpcOpenMemHandle((void **)&s_peerField[f][r], all[r],
                                       cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
            fprintf(stderr, "rank %d: cudaIpcOpenMemHandle(field %d, "
                            "rank %d) failed\n", myRank, f, r) ;
            MPI_Abort(MPI_COMM_WORLD, 1) ;
         }
      }
   }
   delete [] all ;
}

void commDirectUnmapFields(Domain* d, int myRank)
{
   (void)d ;
   int n ;
   MPI_Comm_size(MPI_COMM_WORLD, &n) ;
   for (int f = 0; f < NDIRECT_FIELDS; ++f) {
      for (int r = 0; r < n; ++r) {
         if (r != myRank) cudaIpcCloseMemHandle(s_peerField[f][r]) ;
      }
      delete [] s_peerField[f] ;
   }
}

static int fieldIndex(Domain_member f)
{
   if (f == &Domain::get_x)  return 0 ;
   if (f == &Domain::get_y)  return 1 ;
   if (f == &Domain::get_z)  return 2 ;
   if (f == &Domain::get_xd) return 3 ;
   if (f == &Domain::get_yd) return 4 ;
   if (f == &Domain::get_zd) return 5 ;
   if (f == &Domain::get_fx) return 6 ;
   if (f == &Domain::get_fy) return 7 ;
   if (f == &Domain::get_fz) return 8 ;
   return -1 ;
}

/******************************************/
/* index maps -- identical formulas to SendPlane/AddPlane and
   SendEdge/AddEdge in the packed variants */

__device__ __forceinline__
Index_t faceIdx(int T, int tid, Index_t dx, Index_t dy, Index_t dz)
{
   switch (T) {
   case 0:  return tid ;
   case 1:  return dx*dy*(dz-1) + tid ;
   case 2:  return (tid/dx)*dx*dy + tid%dx ;
   case 3:  return dx*(dy-1) + (tid/dx)*dx*dy + tid%dx ;
   case 4:  return (tid/dy)*dx*dy + (tid%dy)*dx ;
   default: return dx-1 + (tid/dy)*dx*dy + (tid%dy)*dx ;
   }
}

__device__ __forceinline__
Index_t edgeIdx(int T, int i, Index_t dx, Index_t dy, Index_t dz)
{
   switch (T) {
   case 0:  return i*dx*dy ;
   case 1:  return i ;
   case 2:  return i*dx ;
   case 3:  return dx*dy - 1 + i*dx*dy ;
   case 4:  return dx*(dy-1) + dx*dy*(dz-1) + i ;
   case 5:  return dx*dy*(dz-1) + dx - 1 + i*dx ;
   case 6:  return dx*(dy-1) + i*dx*dy ;
   case 7:  return dx*dy*(dz-1) + i ;
   case 8:  return dx*dy*(dz-1) + i*dx ;
   case 9:  return dx - 1 + i*dx*dy ;
   case 10: return dx*(dy-1) + i ;
   default: return dx - 1 + i*dx ;
   }
}

/******************************************/
/* remote-write kernels */

__global__ void DirectFace(Real_t *dst, const Real_t *src, Index_t n,
                           int sT, int rT, int doAdd,
                           Index_t dx, Index_t dy, Index_t dz)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x ;
   if (tid >= n) return ;
   Real_t v = src[faceIdx(sT, tid, dx, dy, dz)] ;
   if (doAdd) atomicAdd(&dst[faceIdx(rT, tid, dx, dy, dz)], v) ;
   else       dst[faceIdx(rT, tid, dx, dy, dz)] = v ;
}

__global__ void DirectEdge(Real_t *dst, const Real_t *src, Index_t n,
                           int sT, int rT, int doAdd,
                           Index_t dx, Index_t dy, Index_t dz)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x ;
   if (i >= n) return ;
   Real_t v = src[edgeIdx(sT, i, dx, dy, dz)] ;
   if (doAdd) atomicAdd(&dst[edgeIdx(rT, i, dx, dy, dz)], v) ;
   else       dst[edgeIdx(rT, i, dx, dy, dz)] = v ;
}

__global__ void DirectCorner(Real_t *dst, const Real_t *src, int doAdd)
{
   if (doAdd) atomicAdd(dst, src[0]) ;
   else       dst[0] = src[0] ;
}

/* MonoQ remote-pack: same gather as SendPlane, but the packed destination
   lives in the peer GPU's recv buffer */
__global__ void PackFace(Real_t *destAddr, const Real_t *srcAddr, Index_t n,
                         int T, Index_t dx, Index_t dy, Index_t dz)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x ;
   if (tid >= n) return ;
   destAddr[tid] = srcAddr[faceIdx(T, tid, dx, dy, dz)] ;
}

/******************************************/
/* MonoQ: remote-pack into the peer's packed recv buffer */

static void MonoQSendDirect(Domain& domain, Index_t xferFields,
                            Domain_member *fieldData,
                            Index_t dx, Index_t dy, Index_t dz,
                            bool doSend, cudaStream_t stream)
{
   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   int tp  = (int)domain.tp() ;
   int tp2 = tp*tp ;
   bool rowMin   = (domain.rowLoc() != 0),   rowMax   = (domain.rowLoc() != tp-1) ;
   bool colMin   = (domain.colLoc() != 0),   colMax   = (domain.colLoc() != tp-1) ;
   bool planeMin = (domain.planeLoc() != 0), planeMax = (domain.planeLoc() != tp-1) ;
   const int block = 128 ;

   static const int  ST[6]     = { 0, 1, 2, 3, 4, 5 } ;
   static const int  DC[6]     = { 0, 0, 0, 0, -1, 1 } ;
   static const int  DR[6]     = { 0, 0, -1, 1, 0, 0 } ;
   static const int  DP[6]     = { -1, 1, 0, 0, 0, 0 } ;
   const int  delta[6]  = { -tp2, tp2, -tp, tp, -1, 1 } ;
   const bool guard[6]  = { planeMin, planeMax && doSend,
                            rowMin,   rowMax   && doSend,
                            colMin,   colMax   && doSend } ;
   const Index_t count[6] = { dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz } ;

   for (int f = 0; f < 6; ++f) {
      if (!guard[f]) continue ;
      int toRank = myRank + delta[f] ;
      /* source direction as seen by the receiver is the opposite of ours */
      Real_t *dst = domain.d_peerRecv[toRank] +
                    shmRecvOffset(domain, toRank, -DC[f], -DR[f], -DP[f],
                                  xferFields, doSend, true) ;
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         PackFace<<<(count[f]+block-1)/block, block, 0, stream>>>(
            dst + fi*count[f], &(domain.*fieldData[fi])(0), count[f],
            ST[f], dx, dy, dz) ;
      }
   }
}

/******************************************/

void CommSendGpu(Domain& domain, Int_t msgType,
                 Index_t xferFields, Domain_member *fieldData,
                 Index_t dx, Index_t dy, Index_t dz,
                 bool doSend, bool planeOnly, cudaStream_t stream)
{
   if (domain.numRanks() == 1)
      return ;

   if (planeOnly) {
      /* MonoQ: remote-pack (destination fields are per-step temporaries).
         Same entry-sync rule as the direct path: the device sync orders the
         pack kernels after the delv_* producers regardless of which stream
         they ran on, and the barrier keeps the entry/exit protocol uniform
         across all direct-mode sends. */
      cudaDeviceSynchronize() ;
      MPI_Barrier(MPI_COMM_WORLD) ;
      MonoQSendDirect(domain, xferFields, fieldData, dx, dy, dz, doSend,
                      stream) ;
      cudaStreamSynchronize(stream) ;
      MPI_Barrier(MPI_COMM_WORLD) ;
      return ;
   }

   int doAdd = (msgType == MSG_COMM_SBN) ;
   int myRank ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   int tp  = (int)domain.tp() ;
   int tp2 = tp*tp ;
   bool rowMin   = (domain.rowLoc() != 0),   rowMax   = (domain.rowLoc() != tp-1) ;
   bool colMin   = (domain.colLoc() != 0),   colMax   = (domain.colLoc() != tp-1) ;
   bool planeMin = (domain.planeLoc() != 0), planeMax = (domain.planeLoc() != tp-1) ;
   const int block = 128 ;

   /* resolve source pointers and the peer-field tables */
   const Real_t *src[6] ;
   Real_t **peer[6] ;
   for (Index_t fi = 0; fi < xferFields; ++fi) {
      int f = fieldIndex(fieldData[fi]) ;
      if (f < 0) {
         fprintf(stderr, "COMM_DIRECT: unmapped field in message type %d\n",
                 (int)msgType) ;
         MPI_Abort(MPI_COMM_WORLD, 1) ;
      }
      src[fi]  = &(domain.*fieldData[fi])(0) ;
      peer[fi] = s_peerField[f] ;
   }

   /* every rank must finish writing its own fields before any remote
      write can land in them */
   cudaDeviceSynchronize() ;
   MPI_Barrier(MPI_COMM_WORLD) ;

   /* faces: sender pack-type -> receiver unpack-type */
   {
      static const int ST[6] = { 0, 1, 2, 3, 4, 5 } ;
      static const int RT[6] = { 1, 0, 3, 2, 5, 4 } ;
      const int  delta[6] = { -tp2, tp2, -tp, tp, -1, 1 } ;
      const bool guard[6] = { planeMin, planeMax && doSend,
                              rowMin,   rowMax   && doSend,
                              colMin,   colMax   && doSend } ;
      const Index_t count[6] = { dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz } ;
      for (int f = 0; f < 6; ++f) {
         if (!guard[f]) continue ;
         int toRank = myRank + delta[f] ;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            DirectFace<<<(count[f]+block-1)/block, block, 0, stream>>>(
               peer[fi][toRank], src[fi], count[f], ST[f], RT[f], doAdd,
               dx, dy, dz) ;
         }
      }
   }

   /* edges */
   {
      static const int ST[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } ;
      static const int RT[12] = { 3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8 } ;
      const int delta[12] = { -tp-1, -tp2-tp, -tp2-1,
                               tp+1,  tp2+tp,  tp2+1,
                               tp-1,  tp2-tp,  tp2-1,
                              -tp+1, -tp2+tp, -tp2+1 } ;
      const bool guard[12] = {
         rowMin && colMin,             rowMin && planeMin,
         colMin && planeMin,           rowMax && colMax   && doSend,
         rowMax && planeMax && doSend, colMax && planeMax && doSend,
         rowMax && colMin   && doSend, rowMin && planeMax && doSend,
         colMin && planeMax && doSend, rowMin && colMax,
         rowMax && planeMin,           colMax && planeMin } ;
      const Index_t count[12] = { dz, dx, dy, dz, dx, dy,
                                  dz, dx, dy, dz, dx, dy } ;
      for (int e = 0; e < 12; ++e) {
         if (!guard[e]) continue ;
         int toRank = myRank + delta[e] ;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            DirectEdge<<<(count[e]+block-1)/block, block, 0, stream>>>(
               peer[fi][toRank], src[fi], count[e], ST[e], RT[e], doAdd,
               dx, dy, dz) ;
         }
      }
   }

   /* corners: source corner -> mirrored corner on the receiver */
   {
      const Index_t srcIdx[8] = {
         0,                         dx*dy*(dz-1),
         dx-1,                      dx*dy*(dz-1) + dx-1,
         dx*(dy-1),                 dx*dy*(dz-1) + dx*(dy-1),
         dx*dy - 1,                 dx*dy*dz - 1 } ;
      const Index_t dstIdx[8] = {
         dx*dy*dz - 1,              dx*dy - 1,
         dx*(dy-1) + dx*dy*(dz-1),  dx*(dy-1),
         dx-1 + dx*dy*(dz-1),       dx-1,
         dx*dy*(dz-1),              0 } ;
      const int delta[8] = { -tp2-tp-1,  tp2-tp-1, -tp2-tp+1,  tp2-tp+1,
                             -tp2+tp-1,  tp2+tp-1, -tp2+tp+1,  tp2+tp+1 } ;
      const bool guard[8] = {
         rowMin && colMin && planeMin,
         rowMin && colMin && planeMax && doSend,
         rowMin && colMax && planeMin,
         rowMin && colMax && planeMax && doSend,
         rowMax && colMin && planeMin,
         rowMax && colMin && planeMax && doSend,
         rowMax && colMax && planeMin,
         rowMax && colMax && planeMax && doSend } ;
      for (int c = 0; c < 8; ++c) {
         if (!guard[c]) continue ;
         int toRank = myRank + delta[c] ;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            DirectCorner<<<1, 1, 0, stream>>>(
               peer[fi][toRank] + dstIdx[c], src[fi] + srcIdx[c], doAdd) ;
         }
      }
   }

   /* all remote writes must land before anyone reads their halos */
   cudaStreamSynchronize(stream) ;
   MPI_Barrier(MPI_COMM_WORLD) ;
}

/******************************************/
/* Unpack routines: nothing to do -- the senders already wrote our fields */

void CommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData,
                cudaStream_t *streams)
{
   (void)xferFields ; (void)fieldData ; (void)streams ;
   if (domain.numRanks() == 1)
      return ;
   /* neighbor contributions were atomically added in CommSendGpu */
}

void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams)
{
   (void)streams ;
   if (domain.numRanks() == 1)
      return ;
   /* neighbor values were stored directly in CommSendGpu */
}

/* MonoQ still unpacks: copy each face message from our packed recv buffer
   into the ghost regions of the (per-step) delv arrays */
void CommMonoQGpu(Domain& domain, cudaStream_t stream)
{
   if (domain.numRanks() == 1)
      return ;

   Index_t xferFields = 3 ;
   Domain_member fieldData[3] ;
   Index_t fieldOffset[3] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t pmsg = 0 ;
   Index_t dx = domain.sizeX ;
   Index_t dy = domain.sizeY ;
   Index_t dz = domain.sizeZ ;
   int tp = (int)domain.tp() ;
   bool rowMin   = (domain.rowLoc() != 0),   rowMax   = (domain.rowLoc() != tp-1) ;
   bool colMin   = (domain.colLoc() != 0),   colMax   = (domain.colLoc() != tp-1) ;
   bool planeMin = (domain.planeLoc() != 0), planeMax = (domain.planeLoc() != tp-1) ;

   fieldData[0] = &Domain::get_delv_xi ;
   fieldData[1] = &Domain::get_delv_eta ;
   fieldData[2] = &Domain::get_delv_zeta ;
   fieldOffset[0] = domain.numElem ;
   fieldOffset[1] = domain.numElem ;
   fieldOffset[2] = domain.numElem ;

   const bool guard[6] = { planeMin, planeMax, rowMin, rowMax,
                           colMin, colMax } ;
   const Index_t count[6] = { dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz } ;

   for (int f = 0; f < 6; ++f) {
      if (!guard[f]) continue ;
      Real_t *srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr,
                         count[f]*sizeof(Real_t),
                         cudaMemcpyDeviceToDevice, stream);
         srcAddr += count[f] ;
         fieldOffset[fi] += count[f] ;
      }
      ++pmsg ;
   }
}

#endif /* USE_MPI */
