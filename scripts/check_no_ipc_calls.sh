#!/bin/bash
# E7 -- assert that the wrapper-mediated build targets contain ZERO CUDA IPC
# call sites, and report the count for every target.
#
# The paper's mechanism-localization claim is that an application using the
# interposer speaks only portable MPI window code: no cudaIpc* anywhere in the
# application's own translation units. This script is what makes that claim
# defensible rather than assertable.
#
# WHY PREPROCESSED TRANSLATION UNITS, NOT A SOURCE GREP
#
#   1. False positive. LULESH/cuda/src/lulesh.h mentions cudaIpc in a *comment*
#      inside #ifdef COMM_IPC, and the wrapper build compiles through that
#      block. A naive `grep cudaIpc *.h *.cu` therefore reports a hit for the
#      mpiwrap target, and a reviewer will find it. Preprocessing strips
#      comments, so the claim survives contact.
#   2. False negative. A source grep cannot see which branches of the #ifdef
#      maze a given -D combination actually selects. Only the preprocessor
#      knows what is really in the build.
#
# WHY THE SYSTEM-HEADER FILTER
#
#   `nvcc -E` inlines cuda_runtime_api.h, which *declares* cudaIpcOpenMemHandle,
#   cudaIpcGetMemHandle, etc. Grepping the raw preprocessed output for
#   'cudaIpc[A-Za-z]*(' matches those declarations and reports a nonzero count
#   for every target including staged, which is meaningless. We therefore track
#   the `# <line> "<file>"` line markers the preprocessor emits and count a hit
#   only when the current file lives inside this repository.
#
# Usage:  scripts/check_no_ipc_calls.sh
# Exit:   0 if every must-be-zero target is clean, 1 otherwise.

set -uo pipefail
REPO="${REPO:-/lustre/nvwulf/home/vchotalia/mpiwrap}"
SRC="$REPO/LULESH/cuda/src"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0 2>/dev/null

command -v nvcc >/dev/null || { echo "FATAL: nvcc not on PATH. Run:"; \
  echo "  module load cuda12.8/toolkit/12.8.1 openmpi/gcc14.3/4.1.8 gcc/14.3.0"; exit 1; }

MPI_INC=$(mpicc --showme:incdirs 2>/dev/null | xargs -I{} echo -I{})
BASE="-arch=sm_90 -O3 -DNDEBUG -DUSE_MPI -I$SRC $MPI_INC"

# Count cudaIpc* call sites in a preprocessed TU, ignoring anything that came
# from outside the repo (i.e. CUDA/MPI/system headers).
# NOTE ON THE PATH TEST: nvcc records repo headers with the *relative* path it
# was given (e.g. comm/comm_ipc_common.h under -I.), and only system headers
# with absolute paths. So "in repo" means: relative path, or absolute path
# under $REPO. Testing only the absolute prefix silently counts zero repo hits
# while counting CUDA's own declarations -- which is exactly backwards.
count_tu() {
    awk -v repo="$REPO" '
        # Preprocessor line marker: # <lineno> "<filename>" [flags]
        /^#[ \t]+[0-9]+[ \t]+"/ {
            f = $0; sub(/^#[ \t]+[0-9]+[ \t]+"/, "", f); sub(/".*$/, "", f)
            inrepo = (substr(f, 1, 1) != "/") || (index(f, repo) == 1)
            next
        }
        inrepo { total += gsub(/cudaIpc[A-Za-z]*[ \t]*\(/, "&") }
        END { print total + 0 }
    ' "$1"
}

# Same filter, but reports file -> count so a failure is actionable.
where_tu() {
    awk -v repo="$REPO" '
        /^#[ \t]+[0-9]+[ \t]+"/ {
            f = $0; sub(/^#[ \t]+[0-9]+[ \t]+"/, "", f); sub(/".*$/, "", f)
            inrepo = (substr(f, 1, 1) != "/") || (index(f, repo) == 1); cur = f
            next
        }
        inrepo { n = gsub(/cudaIpc[A-Za-z]*[ \t]*\(/, "&"); if (n > 0) w[cur] += n }
        END { for (k in w) printf "        %3d  %s\n", w[k], k }
    ' "$1"
}

# target : defines : sources
TARGETS=(
  "staged      ::allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "gpumpi      :-DCOMM_GPUMPI:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "shmwin      :-DCOMM_SHMWIN:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "ipc         :-DCOMM_IPC:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "ipc_rp      :-DCOMM_IPC -DIPC_REMOTE_PACK:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "direct      :-DCOMM_IPC -DCOMM_DIRECT:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-direct.cu"
  "mpiwrap     :-DCOMM_IPC -DIPC_VIA_MPIWRAP:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
  "mpiwrap_rp  :-DCOMM_IPC -DIPC_VIA_MPIWRAP -DIPC_REMOTE_PACK:allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu"
)
# Targets that MUST report zero -- the mechanism-localization claim.
MUST_BE_ZERO="mpiwrap mpiwrap_rp"

echo "==================================================================="
echo "E7: cudaIpc* call sites per build target (preprocessed TUs)"
echo "==================================================================="
printf "%-14s %8s   %s\n" "target" "sites" "verdict"

FAIL=0
for entry in "${TARGETS[@]}"; do
    NAME="${entry%%:*}";      NAME="${NAME// /}"
    REST="${entry#*:}"
    DEFS="${REST%%:*}"
    SRCS="${REST#*:}"
    TOTAL=0
    for s in $SRCS; do
        if ! nvcc $BASE $DEFS -E "$SRC/$s" -o "$TMP/$NAME.$s.i" 2>"$TMP/$NAME.$s.err"; then
            echo "  PREPROCESS FAILED: $NAME / $s"; head -3 "$TMP/$NAME.$s.err"; FAIL=1; continue
        fi
        TOTAL=$(( TOTAL + $(count_tu "$TMP/$NAME.$s.i") ))
    done
    VERDICT=""
    if [[ " $MUST_BE_ZERO " == *" $NAME "* ]]; then
        if [ "$TOTAL" -eq 0 ]; then VERDICT="PASS (must be 0)"
        else VERDICT="*** FAIL -- must be 0 ***"; FAIL=1; SHOW_WHERE="$NAME"; fi
    else
        VERDICT="(handwritten IPC expected)"
    fi
    printf "%-14s %8s   %s\n" "$NAME" "$TOTAL" "$VERDICT"
    if [ "${SHOW_WHERE:-}" = "$NAME" ]; then
        for s in $SRCS; do [ -f "$TMP/$NAME.$s.i" ] && where_tu "$TMP/$NAME.$s.i"; done | sort -u
        SHOW_WHERE=""
    fi
done

# ---- stencil + transpose IPC builds -------------------------------------
echo ""
echo "--- stencil / transpose (wrapper builds must also be 0) ---"
printf "%-24s %8s   %s\n" "source" "sites" "verdict"
for f in "$REPO/stencil/stencil_ipc.cu" "$REPO/stencil/stencil_mpi.cu" \
         "$REPO/stencil/stencil_gpu_mpi.cu"; do
    [ -f "$f" ] || continue
    B="$(basename "$f")"
    if nvcc -arch=sm_90 -O3 $MPI_INC -E "$f" -o "$TMP/$B.i" 2>/dev/null; then
        N=$(count_tu "$TMP/$B.i")
        if [ "$B" = "stencil_ipc.cu" ]; then
            [ "$N" -eq 0 ] && V="PASS (must be 0)" || { V="*** FAIL -- must be 0 ***"; FAIL=1; }
        else
            V="(no IPC expected)"
        fi
        printf "%-24s %8s   %s\n" "$B" "$N" "$V"
    else
        printf "%-24s %8s   %s\n" "$B" "-" "preprocess failed"; FAIL=1
    fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: PASS -- every wrapper-mediated target has zero cudaIpc* call sites."
else
    echo "RESULT: FAIL -- see above."
fi
exit $FAIL
