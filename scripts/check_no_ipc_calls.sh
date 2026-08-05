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
# TWO COUNTS, AND THEY ARE NOT INTERCHANGEABLE
#
#   DISTINCT  -- unique source call sites, deduped by file:line across the
#                target's translation units. A call site in a header that three
#                TUs include is ONE site. This is the number to cite in prose
#                ("the handwritten path performs N cudaIpc calls"), and it is
#                what a human reading the source counts.
#   TU-SUMMED -- occurrences summed over every TU compiled into the target, so
#                that same header site counts three times. This is what the
#                compiler actually sees, useful for spotting accidental
#                inclusion, but it is NOT a count of calls in the program.
#
# Mixing them is an easy way to put an inconsistent pair of numbers in a paper.
# `direct` is the worked example:
#     8 distinct  = 5 in lulesh-comms-direct.cu (the field-write path)
#                 + 3 in comm_ipc_packed.h (Mode B still uses the packed path
#                   for MonoQ, whose transient pool allocations cannot be
#                   premapped)
#    14 TU-summed = 5 (one TU) + 9 (those same 3 header sites, seen in 3 TUs)
# Note 5 is the field-write subtotal, NOT direct's distinct total -- quoting it
# as the total is precisely the error these two columns exist to prevent.
# Both are printed so the distinction is impossible to miss.
#
# NOTE ON THE PATH TEST: nvcc records repo headers with the *relative* path it
# was given (e.g. comm/comm_ipc_common.h under -I.), and only system headers
# with absolute paths. So "in repo" means: relative path, or absolute path
# under $REPO. Testing only the absolute prefix silently counts zero repo hits
# while counting CUDA's own declarations -- which is exactly backwards.
#
# Dedup keys use the basename: every source file in this tree has a unique
# name, and the same header can appear with a relative path in one TU and an
# absolute one in another, which would otherwise split into two keys.

# Emit "basename:line<TAB>count" for each in-repo cudaIpc call site in a TU.
records_tu() {
    awk -v repo="$REPO" '
        # Preprocessor line marker: # <lineno> "<filename>" [flags]
        /^#[ \t]+[0-9]+[ \t]+"/ {
            ln = $2 + 0
            f = $0; sub(/^#[ \t]+[0-9]+[ \t]+"/, "", f); sub(/".*$/, "", f)
            inrepo = (substr(f, 1, 1) != "/") || (index(f, repo) == 1)
            nparts = split(f, p, "/"); cur = p[nparts]
            curln = ln - 1        # the NEXT physical line is line `ln`
            next
        }
        {
            curln++
            if (inrepo) {
                n = gsub(/cudaIpc[A-Za-z]*[ \t]*\(/, "&")
                if (n > 0) printf "%s:%d\t%d\n", cur, curln, n
            }
        }
    ' "$1"
}

# distinct total: max count per unique file:line, summed
total_distinct() { awk -F'\t' '{ if ($2 > m[$1]) m[$1] = $2 }
                               END { for (k in m) s += m[k]; print s + 0 }' "$1"; }
# TU-summed total: every occurrence in every TU
total_summed()   { awk -F'\t' '{ s += $2 } END { print s + 0 }' "$1"; }
# per-file distinct breakdown, for actionable failures
where_distinct() {
    awk -F'\t' '{ split($1, a, ":"); if ($2 > m[$1]) m[$1] = $2; file[$1] = a[1] }
                END { for (k in m) w[file[k]] += m[k]
                      for (f in w) printf "        %3d  %s\n", w[f], f }' "$1"
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
printf "%-14s %9s %10s   %s\n" "target" "distinct" "TU-summed" "verdict"

FAIL=0
for entry in "${TARGETS[@]}"; do
    NAME="${entry%%:*}";      NAME="${NAME// /}"
    REST="${entry#*:}"
    DEFS="${REST%%:*}"
    SRCS="${REST#*:}"
    : > "$TMP/$NAME.rec"
    for s in $SRCS; do
        if ! nvcc $BASE $DEFS -E "$SRC/$s" -o "$TMP/$NAME.$s.i" 2>"$TMP/$NAME.$s.err"; then
            echo "  PREPROCESS FAILED: $NAME / $s"; head -3 "$TMP/$NAME.$s.err"; FAIL=1; continue
        fi
        records_tu "$TMP/$NAME.$s.i" >> "$TMP/$NAME.rec"
    done
    TOTAL=$(total_distinct "$TMP/$NAME.rec")
    SUMMED=$(total_summed  "$TMP/$NAME.rec")
    VERDICT=""
    if [[ " $MUST_BE_ZERO " == *" $NAME "* ]]; then
        if [ "$TOTAL" -eq 0 ]; then VERDICT="PASS (must be 0)"
        else VERDICT="*** FAIL -- must be 0 ***"; FAIL=1; SHOW_WHERE="$NAME"; fi
    else
        VERDICT="(handwritten IPC expected)"
    fi
    printf "%-14s %9s %10s   %s\n" "$NAME" "$TOTAL" "$SUMMED" "$VERDICT"
    if [ "${SHOW_WHERE:-}" = "$NAME" ]; then
        where_distinct "$TMP/$NAME.rec"
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
        records_tu "$TMP/$B.i" > "$TMP/$B.rec"
        N=$(total_distinct "$TMP/$B.rec")   # single TU, so distinct == summed
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
