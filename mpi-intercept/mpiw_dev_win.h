// mpiw_dev_win.h - device-side window metadata shared by host and device code
//
// Plain C++ on purpose: the interposer (mpiwrap_ipc.cc) is compiled by mpicxx,
// never nvcc, so this header cannot contain __device__ code. The device
// functions live in mpiw_device.cuh, which includes this file.
#pragma once
#include <mpi.h>

// How a device thread reaches each target rank.
enum {
  MPIX_TRANSPORT_NONE = 0,   // unreachable from this GPU: operations fail
  MPIX_TRANSPORT_IPC  = 1,   // peer_base[r] is a mapped pointer: plain loads/stores
  // MPIX_TRANSPORT_NIC = 2, // later: GDAKI (the GPU posts work to the NIC)
};

// Which threads take part in one call. Every thread of the scope must make
// the same call with the same arguments (like nvshmemx_*_warp / *_block).
enum {
  MPIX_SCOPE_THREAD = 0,
  MPIX_SCOPE_WARP   = 1,
  MPIX_SCOPE_BLOCK  = 2,
};

// Filled by MPIX_Win_get_device_handle and copied to device memory as ONE
// allocation: this struct, then its arrays. MPI_Win_free releases it.
struct mpiw_dev_win {
  int        my_rank;
  int        comm_size;
  int        device_id;
  char     **peer_base;   // [comm_size] IPC-mapped bases, in MY address space
  MPI_Aint  *peer_size;   // [comm_size] for bounds checks
  int       *disp_unit;   // [comm_size]
  int       *transport;   // [comm_size] MPIX_TRANSPORT_*
};

#ifdef __cplusplus
extern "C" {
#endif
// Proposed MPI_Win_get_device_handle (mpiwg-rma PR #13), as an extension.
// info must carry "device_id", the CUDA device the window was created on.
// *device_win is a DEVICE pointer to an mpiw_dev_win, typed as MPI_Win: pass
// it only to device functions. Host calls (lock_all, unlock_all, free) still
// take the original win.
int MPIX_Win_get_device_handle(MPI_Win win, MPI_Info info, MPI_Win *device_win);
#ifdef __cplusplus
}
#endif
