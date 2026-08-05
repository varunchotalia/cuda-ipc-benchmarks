#!/bin/bash
# Build the interposer from source and print its provenance.
#
# Source this (or run it) as the FIRST step of any job that LD_PRELOADs the
# interposer, then use $MPIWRAP_LIB. Two reasons:
#
#  1. Provenance. The .so is gitignored, so a checked-out tree can carry an
#     artifact built from anything. Two hand-built copies had drifted apart --
#     ./libmpiwrap.so (2026-05-04) and mpi-intercept/libmpiwrap.so
#     (2026-07-17) -- and different harnesses loaded different ones. Their
#     interposed symbols matched and the July deltas are multi-node/fabric
#     only (dead code on nvwulf), so results were unaffected; but "which
#     binary produced this number" was unanswerable from the log. Now the
#     build and its md5 are in every job's output.
#  2. The login node has no nvcc and no GPU, so building in-job is the only
#     place a compile error can surface at all -- and putting it first means
#     it surfaces in the first minute instead of after the queue wait.
#
# NEVER build this with nvcc. Per CMakeLists.txt, nvcc drags in libnvomp,
# which breaks LD_PRELOAD. Plain C++ only: the .cc has no kernels, just the
# CUDA runtime + driver API (-lcuda is needed for the cuMem* fabric path).
#
# Sets: MPIWRAP_LIB -- absolute path to the freshly built .so

REPO="${REPO:-/lustre/nvwulf/home/vchotalia/mpiwrap}"
mkdir -p "$REPO/build"   # the mpicxx fallback writes here too, and cmake may
                         # never run to create it

echo "--- building libmpiwrap.so ($(git -C "$REPO" log -1 --format='%h %s' -- mpi-intercept/mpiwrap_ipc.cc)) ---"

if command -v cmake >/dev/null 2>&1 && cmake -S "$REPO" -B "$REPO/build" >/dev/null 2>&1 \
   && cmake --build "$REPO/build" --target mpiwrap -j 8 >/dev/null 2>&1 \
   && [ -f "$REPO/build/libmpiwrap.so" ]; then
    MPIWRAP_LIB="$REPO/build/libmpiwrap.so"
    echo "built via CMake (CMakeLists.txt target: mpiwrap)"
else
    # Fallback: the exact link line CMakeLists.txt:42-46 expresses, in case
    # cmake is unavailable on the compute node. Still plain C++, never nvcc.
    echo "cmake unavailable or failed -- falling back to direct mpicxx"
    mpicxx -O3 -shared -fPIC "$REPO/mpi-intercept/mpiwrap_ipc.cc" \
           -o "$REPO/build/libmpiwrap.so" \
           -I"${CUDA_HOME:-/usr/local/cuda}/include" \
           -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart -lcuda || {
        echo "FATAL: interposer build failed -- aborting before wasting the slot"
        exit 1
    }
    MPIWRAP_LIB="$REPO/build/libmpiwrap.so"
fi

export MPIWRAP_LIB
ls -la "$MPIWRAP_LIB"
md5sum "$MPIWRAP_LIB"
# Confirm the four interposed entry points are actually exported. A silently
# short symbol list means the LD_PRELOAD will no-op and every "wrapper" run
# will really be plain MPI.
echo "--- interposed symbols ---"
nm -D --defined-only "$MPIWRAP_LIB" | grep -E 'MPI_Win_(create|allocate|shared_query|free)' \
  || echo "WARNING: expected MPI_Win_* symbols not found"
