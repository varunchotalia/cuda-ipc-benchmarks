// mpiw_device.cuh - device-initiated RMA for WinIPC windows (first cut)
#pragma once
#include <mpi.h>
#include <stdint.h>
#include "mpiw_dev_win.h"

// ----------------------------------------------------------------------------
// Predefined datatypes, as the MPI standard ABI encodes them
// (mpi-standard chap-abi/abi.tex, "Handle Constants").
//
// Open MPI 4.1.8 does not use the standard ABI: its MPI_DOUBLE is the address
// of a host global (ompi_mpi_double), which device code cannot reference. ABI
// handles are small integers, so they work in device code. They cannot reuse
// the MPI_ names, which mpi.h already defines, hence MPIX_ABI_.
//
// Datatype handle bits: 11:8 = 0010, 7:3 = category, 2:0 = instance.
// Fixed-size types have bit 6 set and keep log2(size in bytes) in bits 5:3,
// so their size is read from the handle itself, with no table.
// ----------------------------------------------------------------------------
#define MPIX_ABI_DT(bits) ((MPI_Datatype)(uintptr_t)(bits))

// fixed-size: size is encoded in the handle
#define MPIX_ABI_INT8_T         MPIX_ABI_DT(0x240)  // 0010 01000 000
#define MPIX_ABI_UINT8_T        MPIX_ABI_DT(0x241)  // 0010 01000 001
#define MPIX_ABI_CHAR           MPIX_ABI_DT(0x243)  // 0010 01000 011
#define MPIX_ABI_SIGNED_CHAR    MPIX_ABI_DT(0x244)  // 0010 01000 100
#define MPIX_ABI_UNSIGNED_CHAR  MPIX_ABI_DT(0x245)  // 0010 01000 101
#define MPIX_ABI_BYTE           MPIX_ABI_DT(0x247)  // 0010 01000 111
#define MPIX_ABI_INT16_T        MPIX_ABI_DT(0x248)  // 0010 01001 000
#define MPIX_ABI_UINT16_T       MPIX_ABI_DT(0x249)  // 0010 01001 001
#define MPIX_ABI_INT32_T        MPIX_ABI_DT(0x250)  // 0010 01010 000
#define MPIX_ABI_UINT32_T       MPIX_ABI_DT(0x251)  // 0010 01010 001
#define MPIX_ABI_INT64_T        MPIX_ABI_DT(0x258)  // 0010 01011 000
#define MPIX_ABI_UINT64_T       MPIX_ABI_DT(0x259)  // 0010 01011 001

// not fixed-size in the ABI (C leaves their width to the platform)
#define MPIX_ABI_SHORT              MPIX_ABI_DT(0x208)  // 0010 00001 000
#define MPIX_ABI_INT                MPIX_ABI_DT(0x209)  // 0010 00001 001
#define MPIX_ABI_LONG               MPIX_ABI_DT(0x20A)  // 0010 00001 010
#define MPIX_ABI_LONG_LONG          MPIX_ABI_DT(0x20B)  // 0010 00001 011
#define MPIX_ABI_UNSIGNED_SHORT     MPIX_ABI_DT(0x20C)  // 0010 00001 100
#define MPIX_ABI_UNSIGNED           MPIX_ABI_DT(0x20D)  // 0010 00001 101
#define MPIX_ABI_UNSIGNED_LONG      MPIX_ABI_DT(0x20E)  // 0010 00001 110
#define MPIX_ABI_UNSIGNED_LONG_LONG MPIX_ABI_DT(0x20F)  // 0010 00001 111
#define MPIX_ABI_FLOAT              MPIX_ABI_DT(0x210)  // 0010 00010 000
#define MPIX_ABI_C_FLOAT_COMPLEX    MPIX_ABI_DT(0x211)  // 0010 00010 001
#define MPIX_ABI_DOUBLE             MPIX_ABI_DT(0x214)  // 0010 00010 100
#define MPIX_ABI_C_DOUBLE_COMPLEX   MPIX_ABI_DT(0x215)  // 0010 00010 101

// Element size in bytes, or -1 if the handle is not a supported ABI datatype
// (that includes Open MPI's own MPI_DOUBLE, a host address far above 4095).
__host__ __device__ __forceinline__ int MPIX_ABI_Type_size(MPI_Datatype type) {
  uintptr_t h = (uintptr_t)type;
  if ((h & ~(uintptr_t)0xFF) != 0x200) return -1;    // not in the datatype range
  if (h & 0x40) return 1 << ((h >> 3) & 0x7);        // fixed-size: decode bits 5:3
  switch (h) {                                       // the rest need the platform
    case 0x208: case 0x20C: return sizeof(short);
    case 0x209: case 0x20D: return sizeof(int);
    case 0x20A: case 0x20E: return sizeof(long);
    case 0x20B: case 0x20F: return sizeof(long long);
    case 0x210:             return sizeof(float);
    case 0x211:             return 2 * sizeof(float);
    case 0x214:             return sizeof(double);
    case 0x215:             return 2 * sizeof(double);
  }
  return -1;
}

// ----------------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------------

// This thread's position inside the scope (me) and the scope's size (n).
__device__ __forceinline__ void
mpix_scope_rank(int scope, unsigned *me, unsigned *n) {
  unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  if (scope == MPIX_SCOPE_BLOCK) { *me = tid;             *n = blockDim.x * blockDim.y * blockDim.z; }
  else if (scope == MPIX_SCOPE_WARP) { *me = tid % warpSize; *n = warpSize; }
  else                           { *me = 0;               *n = 1; }
}

// The threads of a scope split one copy: thread `me` of `n` takes every n-th
// piece. 8-byte pieces when both ends and the length allow it, else bytes.
__device__ __forceinline__ void
mpix_copy(char *dst, const char *src, size_t bytes, unsigned me, unsigned n) {
  if ((((uintptr_t)dst | (uintptr_t)src | bytes) & 7) == 0) {
    for (size_t i = me; i < bytes / 8; i += n)
      ((uint64_t *)dst)[i] = ((const uint64_t *)src)[i];
  } else {
    for (size_t i = me; i < bytes; i += n)
      dst[i] = src[i];
  }
}

// Checks shared by Put and Get. On success, *addr is the target address in
// this process's address space and *bytes the transfer size.
__device__ __forceinline__ int
mpix_resolve(int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp,
             int target_count, MPI_Datatype target_datatype,
             const mpiw_dev_win *w, int scope, char **addr, size_t *bytes) {
  if (scope < MPIX_SCOPE_THREAD || scope > MPIX_SCOPE_BLOCK) return MPI_ERR_ARG;
  int esize = MPIX_ABI_Type_size(origin_datatype);
  if (esize < 0 || target_datatype != origin_datatype) return MPI_ERR_TYPE;
  if (origin_count < 0 || target_count != origin_count) return MPI_ERR_COUNT;
  if (target_rank < 0 || target_rank >= w->comm_size) return MPI_ERR_RANK;
  if (w->transport[target_rank] != MPIX_TRANSPORT_IPC)  // unreachable, as in
    return MPI_ERR_OTHER;                               // host shared_query
  MPI_Aint off = target_disp * w->disp_unit[target_rank];
  *bytes = (size_t)origin_count * esize;
  if (target_disp < 0 || off + (MPI_Aint)*bytes > w->peer_size[target_rank])
    return MPI_ERR_RMA_RANGE;
  *addr = w->peer_base[target_rank] + off;
  return MPI_SUCCESS;
}

// ----------------------------------------------------------------------------
// Device bindings. Same arguments as MPI_Put / MPI_Get / MPI_Win_flush_all,
// plus `scope`. `win` must be the device_win from MPIX_Win_get_device_handle.
// ----------------------------------------------------------------------------

__device__ __forceinline__ int
MPIX_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
         int target_rank, MPI_Aint target_disp, int target_count,
         MPI_Datatype target_datatype, MPI_Win win, int scope) {
  char *dst; size_t bytes; unsigned me, n;
  int rc = mpix_resolve(origin_count, origin_datatype, target_rank, target_disp,
                        target_count, target_datatype,
                        (const mpiw_dev_win *)win, scope, &dst, &bytes);
  if (rc != MPI_SUCCESS) return rc;
  mpix_scope_rank(scope, &me, &n);
  mpix_copy(dst, (const char *)origin_addr, bytes, me, n);  // stores over NVLink
  return MPI_SUCCESS;
}

__device__ __forceinline__ int
MPIX_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
         int target_rank, MPI_Aint target_disp, int target_count,
         MPI_Datatype target_datatype, MPI_Win win, int scope) {
  char *src; size_t bytes; unsigned me, n;
  int rc = mpix_resolve(origin_count, origin_datatype, target_rank, target_disp,
                        target_count, target_datatype,
                        (const mpiw_dev_win *)win, scope, &src, &bytes);
  if (rc != MPI_SUCCESS) return rc;
  mpix_scope_rank(scope, &me, &n);
  mpix_copy((char *)origin_addr, src, bytes, me, n);        // loads over NVLink
  return MPI_SUCCESS;
}

// Completes the operations issued by the threads of the calling scope, and
// no others: a thread scope covers that thread, a block scope covers the
// whole block. Nothing wider than a block. Every thread of the scope calls it.
// After it returns, Put data is visible at the target and Get data is in the
// origin buffer for every thread of the scope.
__device__ __forceinline__ int MPIX_Win_flush_all(MPI_Win win, int scope) {
  (void)win;
  if (scope < MPIX_SCOPE_THREAD || scope > MPIX_SCOPE_BLOCK) return MPI_ERR_ARG;
  __threadfence_system();              // my own loads/stores are done, system-wide
  if (scope == MPIX_SCOPE_WARP)  __syncwarp();      // ...and so are my warp's
  if (scope == MPIX_SCOPE_BLOCK) __syncthreads();   // ...and so are my block's
  return MPI_SUCCESS;
}
