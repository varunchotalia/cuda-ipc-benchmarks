// comm_ipc.h -- CUDA IPC with an explicit handle exchange.
// Every rank cudaIpcGetMemHandle's its device recv buffer, handles are
// allgathered, and cudaIpcOpenMemHandle maps each peer's buffer locally.
// Single node only.
#ifndef LULESH_COMM_IPC_H
#define LULESH_COMM_IPC_H

#include "comm_ipc_common.h"
// handwritten handle exchange + teardown (deliberately NOT in comm_ipc_common.h
// -- see the header comment there for why the wrapper build must not see it)
#include "comm_ipc_packed.h"

static inline void commAllocRecv(Domain* d, Index_t comBufSize)
{
   commIpcAllocAndMapPacked(d, comBufSize) ;
}

static inline void commTeardown(Domain* d, int myRank)
{
   commIpcUnmapAndFreePacked(d, myRank) ;
}

#endif
