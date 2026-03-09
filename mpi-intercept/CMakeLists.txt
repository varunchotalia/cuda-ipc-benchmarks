# cmake_minimum_required(VERSION 3.16)
# project(MPIWrap LANGUAGES C CXX CUDA)

# cmake_policy(SET CMP0104 NEW)
# set(CMAKE_CUDA_ARCHITECTURES 80)   # A100 / sm_80; tweak if needed

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(MPI REQUIRED)
# find_package(CUDAToolkit REQUIRED)

# # ----- Interposer library -----
# add_library(mpiwrap SHARED mpiwrap_ipc.cc bench_kernels.cu)
# set_target_properties(mpiwrap PROPERTIES
#   CXX_STANDARD 17
#   CUDA_STANDARD 17
#   CXX_STANDARD_REQUIRED YES
#   CUDA_STANDARD_REQUIRED YES
# )
# target_link_libraries(mpiwrap PRIVATE MPI::MPI_CXX CUDA::cudart)
# target_include_directories(mpiwrap PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# # ----- Test executable -----
# add_executable(test_ipc_win test_ipc_win.cu bench_ipc.cu)
# set_target_properties(test_ipc_win PROPERTIES
#   CUDA_STANDARD 17
#   CUDA_STANDARD_REQUIRED YES
# )
# target_link_libraries(test_ipc_win PRIVATE mpiwrap MPI::MPI_CXX CUDA::cudart)
# target_include_directories(test_ipc_win PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

cmake_minimum_required(VERSION 3.16)
project(MPIWrap LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 80)   # A100 / sm_80

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ----- Interposer + benchmark library -----
add_library(mpiwrap SHARED
    mpiwrap_ipc.cc
    bench_kernels.cu
    bench_ipc.cu
)

set_target_properties(mpiwrap PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD_REQUIRED YES
)

target_link_libraries(mpiwrap PRIVATE MPI::MPI_CXX CUDA::cudart)
target_include_directories(mpiwrap PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# ----- Test executable -----
add_executable(test_ipc_win
    test_ipc_win.cu
)

set_target_properties(test_ipc_win PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
)

target_link_libraries(test_ipc_win PRIVATE mpiwrap MPI::MPI_CXX CUDA::cudart)
target_include_directories(test_ipc_win PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
