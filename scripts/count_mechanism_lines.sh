#!/bin/bash
# E7 -- mechanism-localization line counts for the before/after figure.
#
# Counts EFFECTIVE lines only: blank lines and comment-only lines are excluded
# (a comment-heavy file should not look like more mechanism). Counting is done
# by a small awk state machine that also tracks /* */ blocks.
#
# Three numbers the figure needs:
#   handwritten setup+teardown   comm_ipc_packed.h : commIpcAllocAndMapPacked
#                                                  + commIpcUnmapAndFreePacked
#   interposed  setup+teardown   comm_mpiwrap.h    : commAllocRecv + commTeardown
#   shared steady state          comm_ipc_common.h : token protocol + transfer
#                                                    machinery, identical for both
#
# The interposed count is reported twice: as-is, and with the E2a lifecycle
# instrumentation subtracted. The instrumentation is measurement scaffolding,
# not mechanism, and it inflates the interposed side -- i.e. leaving it in
# understates the paper's own claim. Both numbers are printed so the choice is
# explicit rather than silent.
#
# Usage: scripts/count_mechanism_lines.sh

set -uo pipefail
REPO="${REPO:-/lustre/nvwulf/home/vchotalia/mpiwrap}"
CD="$REPO/LULESH/cuda/src/comm"

# effective_lines <file> <start> <end>
effective_lines() {
    awk -v s="$2" -v e="$3" '
        NR < s || NR > e { next }
        {
            line = $0
            # strip /* ... */ spans (single-line and multi-line)
            if (inblock) {
                if (match(line, /\*\//)) { line = substr(line, RSTART+2); inblock = 0 }
                else next
            }
            while (match(line, /\/\*/)) {
                pre = substr(line, 1, RSTART-1)
                rest = substr(line, RSTART+2)
                if (match(rest, /\*\//)) { line = pre substr(rest, RSTART+2) }
                else { line = pre; inblock = 1; break }
            }
            sub(/\/\/.*$/, "", line)          # strip // comments
            gsub(/[ \t]+/, "", line)          # strip whitespace
            if (length(line) > 0) n++
        }
        END { print n + 0 }
    ' "$1"
}

# Count E2a instrumentation lines inside a range (timing scaffolding only).
instr_lines() {
    awk -v s="$2" -v e="$3" '
        NR < s || NR > e { next }
        /g_phase(WinAlloc|PeerQuery|CommSetup|CommFree)Ms|_t0win|_t0q|MPI_Wtime/ { n++ }
        END { print n + 0 }
    ' "$1"
}

fn_range() { # fn_range <file> <function-name> -> "start end"
    awk -v fn="$2" '
        $0 ~ ("^static inline .*" fn "[ \t]*\\(") { s = NR }
        s && /^\}/ && NR >= s { print s, NR; exit }
    ' "$1"
}

echo "==================================================================="
echo "E7: mechanism-localization line counts (effective lines)"
echo "==================================================================="

HW=0
for fn in commIpcAllocAndMapPacked commIpcUnmapAndFreePacked; do
    R=$(fn_range "$CD/comm_ipc_packed.h" "$fn"); set -- $R
    N=$(effective_lines "$CD/comm_ipc_packed.h" "$1" "$2")
    printf "  handwritten  %-28s lines %3s-%-3s  %3s\n" "$fn" "$1" "$2" "$N"
    HW=$(( HW + N ))
done

IW=0; INSTR=0
for fn in commAllocRecv commTeardown; do
    R=$(fn_range "$CD/comm_mpiwrap.h" "$fn"); set -- $R
    N=$(effective_lines "$CD/comm_mpiwrap.h" "$1" "$2")
    I=$(instr_lines    "$CD/comm_mpiwrap.h" "$1" "$2")
    printf "  interposed   %-28s lines %3s-%-3s  %3s  (incl %s instr)\n" "$fn" "$1" "$2" "$N" "$I"
    IW=$(( IW + N )); INSTR=$(( INSTR + I ))
done

SHARED=$(effective_lines "$CD/comm_ipc_common.h" 1 100000)

echo "-------------------------------------------------------------------"
printf "  handwritten setup+teardown          %4s\n" "$HW"
printf "  interposed  setup+teardown          %4s  (as committed)\n" "$IW"
printf "  interposed  setup+teardown          %4s  (mechanism only, -%s E2a instr)\n" \
       "$(( IW - INSTR ))" "$INSTR"
printf "  shared steady state                 %4s  (comm_ipc_common.h, identical both paths)\n" "$SHARED"
echo "-------------------------------------------------------------------"
printf "  reduction in mechanism code         %4s lines  (%s%%)\n" \
       "$(( HW - (IW - INSTR) ))" \
       "$(awk -v a="$HW" -v b="$(( IW - INSTR ))" 'BEGIN{printf "%.0f", (a-b)*100.0/a}')"
echo ""
echo "cudaIpc* call sites per target: run scripts/check_no_ipc_calls.sh"
