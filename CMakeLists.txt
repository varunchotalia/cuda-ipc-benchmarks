cmake_minimum_required(VERSION 3.18)

# must precede project(): otherwise CMake caches the compiler-detected arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)   # H200; use 100 for GB200/NVL72
endif()

project(cuda-ipc-benchmarks LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# One-command build of everything: the mpiwrap interposer, the transpose and
# stencil benchmarks, and all LULESH halo-exchange variants.
#
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100    # GB200/NVL72 (sm_100)
#   cmake --build build -j                           # H200 = 90 (the default)
#
# NVSHMEM targets build when NVSHMEM_HOME is set (env var or -DNVSHMEM_HOME).
# See scripts/run_nvl72.sh for a scheduler-agnostic runner.
# ---------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
# nvcc-compiled targets don't always inherit MPI::MPI_CXX's include dirs
include_directories(${MPI_CXX_INCLUDE_DIRS})

if(NOT NVSHMEM_HOME AND DEFINED ENV{NVSHMEM_HOME})
  set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE PATH "NVSHMEM install root")
endif()
if(NVSHMEM_HOME)
  message(STATUS "NVSHMEM: ${NVSHMEM_HOME}")
else()
  message(STATUS "NVSHMEM_HOME not set: skipping NVSHMEM targets")
endif()

# ----- mpiwrap interposer ---------------------------------------------------
# Plain C++ (never nvcc: it drags in libnvomp, which breaks LD_PRELOAD).
# Driver API linked for the multi-node NVLink fabric-handle path.
add_library(mpiwrap SHARED mpi-intercept/mpiwrap_ipc.cc)
target_link_libraries(mpiwrap PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver)
target_include_directories(mpiwrap PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# ----- transpose ------------------------------------------------------------
# COMM_MODE: 0 = IPC direct (kernel writes through window pointers, needs the
# interposer preloaded), 1 = IPC buffered, 2 = GPU-aware MPI, 3 = staged.
function(add_transpose name mode)
  add_executable(${name} transpose/transpose_ipc.cu)
  target_compile_definitions(${name} PRIVATE COMM_MODE=${mode} ACCUMULATE=1 ${ARGN})
  target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart)
endfunction()
add_transpose(transpose_direct        0)
add_transpose(transpose_direct_single 0 SINGLE_KERNEL=1)
add_transpose(transpose_buffered      1)
add_transpose(transpose_gpumpi        2)
add_transpose(transpose_staged        3)

# ----- stencil ---------------------------------------------------------------
add_executable(stencil_ipc stencil/stencil_ipc.cu)
target_link_libraries(stencil_ipc PRIVATE MPI::MPI_CXX CUDA::cudart)
add_executable(stencil_mpi stencil/stencil_mpi.cu)
target_link_libraries(stencil_mpi PRIVATE MPI::MPI_CXX CUDA::cudart)

# ----- LULESH ---------------------------------------------------------------
set(LULESH_DIR ${CMAKE_SOURCE_DIR}/LULESH/cuda/src)
set(LULESH_BASE_SRCS
    ${LULESH_DIR}/allocator.cu
    ${LULESH_DIR}/lulesh.cu
    ${LULESH_DIR}/lulesh-comms.cu)

function(add_lulesh name gpu_comms)
  add_executable(${name} ${LULESH_BASE_SRCS} ${LULESH_DIR}/${gpu_comms})
  target_compile_definitions(${name} PRIVATE USE_MPI=1 ${ARGN})
  target_include_directories(${name} PRIVATE ${LULESH_DIR})
  target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart)
endfunction()

add_lulesh(lulesh_staged     lulesh-comms-gpu.cu)
add_lulesh(lulesh_gpumpi     lulesh-comms-gpu.cu COMM_GPUMPI)
add_lulesh(lulesh_shmwin     lulesh-comms-gpu.cu COMM_SHMWIN)
add_lulesh(lulesh_ipc        lulesh-comms-gpu.cu COMM_IPC)
add_lulesh(lulesh_ipc_rp     lulesh-comms-gpu.cu COMM_IPC IPC_REMOTE_PACK)
add_lulesh(lulesh_mpiwrap    lulesh-comms-gpu.cu COMM_IPC IPC_VIA_MPIWRAP)
add_lulesh(lulesh_mpiwrap_rp lulesh-comms-gpu.cu COMM_IPC IPC_VIA_MPIWRAP IPC_REMOTE_PACK)
add_lulesh(lulesh_direct     lulesh-comms-direct.cu COMM_IPC COMM_DIRECT)

# ----- NVSHMEM targets (device-initiated puts need separable compilation) ---
if(NVSHMEM_HOME)
  function(add_nvshmem_props name)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${name} PRIVATE ${NVSHMEM_HOME}/include)
    target_link_directories(${name} PRIVATE ${NVSHMEM_HOME}/lib)
    target_link_libraries(${name} PRIVATE
        ${NVSHMEM_HOME}/lib/libnvshmem_device.a nvshmem_host)
  endfunction()

  add_lulesh(lulesh_nvshmem lulesh-comms-gpu.cu COMM_NVSHMEM)
  add_nvshmem_props(lulesh_nvshmem)

  add_executable(transpose_nvshmem_direct transpose/transpose_nvshmem.cu)
  target_compile_definitions(transpose_nvshmem_direct PRIVATE COMM_MODE=0)
  target_link_libraries(transpose_nvshmem_direct PRIVATE MPI::MPI_CXX CUDA::cudart)
  add_nvshmem_props(transpose_nvshmem_direct)

  add_executable(transpose_nvshmem_buffered transpose/transpose_nvshmem.cu)
  target_compile_definitions(transpose_nvshmem_buffered PRIVATE COMM_MODE=1)
  target_link_libraries(transpose_nvshmem_buffered PRIVATE MPI::MPI_CXX CUDA::cudart)
  add_nvshmem_props(transpose_nvshmem_buffered)

  add_executable(stencil_nvshmem stencil/stencil_large_nvshmem.cu)
  target_link_libraries(stencil_nvshmem PRIVATE MPI::MPI_CXX CUDA::cudart)
  add_nvshmem_props(stencil_nvshmem)
endif()
