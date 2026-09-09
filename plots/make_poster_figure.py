#!/usr/bin/env python3
"""Single combined results figure for the poster.

    plots/poster_summary.pdf   (vector, for print)
    plots/poster_summary.png   (300 dpi, for slides/preview)

One figure, four panels in a 2x2, one story: an interposed MPI-window
abstraction (WinIPC) reaches hand-written CUDA IPC performance on every
benchmark, beats host-staged MPI everywhere, and keeps doing so when the
window spans nodes over multi-node NVLink.

    (a) Stencil   -- speedup over host-staged MPI, 4 GPUs, 6 grid sizes
    (b) Transpose -- achieved bandwidth vs matrix order, 4 GPUs, B += A^T
    (c) LULESH    -- figure of merit by backend, 8 GPUs, 5 reps, min/max bars
    (d) Scale-out -- speedup over host-staged MPI on a GB200 NVL scale-out
                     system, 16 and 32 GPUs, transpose and stencil

WHY 2x2 AND NOT THE 1x3 THIS FIGURE USED TO BE. Panels (a)-(c) are all single
node: three benchmarks, one machine, one claim. Panel (d) is the second claim
-- that the same interposed window still wins when it crosses nodes -- and
without it a reader has to take the abstraction's scale-out behaviour on
trust. The 2x2 also gives every panel more width than the old 16.2 x 5.0 strip
did, which is what let panel (c)'s labels stop fighting panel (b).

WHY PANEL (d) IS NOT "LULESH ON GB200". LULESH is the obvious fourth panel and
it cannot be drawn. On that system LULESH crashes for gpumpi/ipc/ipc_rp/nvshmem
at 27 and 64 ranks, and every WinIPC run that did complete at 27/64 disagrees
with the staged reference on Final Origin Energy. A correctness-suspect bar has
no place on a poster. Panel (d) therefore carries the two benchmarks whose
scale-out numbers were cleared, and LULESH stays single-node in panel (c).

WHY PANEL (d) IS A RATIO AND NOT ITS OWN UNITS. Transpose is measured in GB/s
and stencil in Gcells/s; there is no honest shared axis. Normalising both to
their own host-staged MPI baseline puts four groups on one scale and matches
panel (a)'s idiom, so "1.0 = host-staged MPI" means the same thing in both
halves of the figure. The absolute GB200 numbers live in the separate
gb200_transpose.pdf / gb200_stencil.pdf figures, which is where a reader who
wants rates should be sent.

PROVENANCE -- each of (a)-(c) is ONE job, so no panel stitches measurements
taken under different builds or UCX settings:

    (a) job 59853, h200x8-04, 2026-08-03. UCX defaults (UCX_TLS unset),
        STENCIL_WARMUP=20, 100 timed iterations, ranks 2/4/8 same node.
        Source: results/stencil_results.txt, "RE-BASELINE (authoritative)".
    (b) job 63328, h200x4-04, 2026-08-13. 100 timed iterations, warmup 20,
        UCX defaults, 10 modes x 5 orders x 2 accumulate settings in one
        allocation. Source: results/transpose_single_job_63328.md.
        NOT results/transpose_results.md -- that file's IPC/MPI half is
        2026-05-05 data stitched to 2026-08-07 NVSHMEM data, and its
        buffered-IPC column is ~10-12% low at the small orders.
    (c) jobs 60796-60800, h200x8-03, 2026-08-06. -s 45, 3145 iterations,
        8 ranks on 8 H200 SXM, 5 single-rep jobs x 9 variants. Source:
        results/lulesh_variance.csv -- the SAME file the paper's Figure 5 is
        generated from, so the two figures cannot disagree. All nine medians
        match that figure to three decimals.
    (d) plots/local_nvl/gb200_nvl.csv -- the same cleared subset that
        gb200_transpose.pdf and gb200_stencil.pdf are drawn from. Read from
        that CSV rather than transcribed here, so the poster cannot drift away
        from the paper figures.

DISCLOSURE LIMITS ON PANEL (d) -- these are conditions, not preferences:

  * The machine is identified ONLY as a "GB200 NVL scale-out system", at
    "16 GPUs (4 nodes)" and "32 GPUs (8 nodes)". No NVL36/NVL72 wiring, no
    host names, no partition, no toolchain, no job IDs. That is the condition
    the 16/32-GPU figures were cleared under on 2026-08-08.
  * 64 GPUs is excluded. It exists in the underlying data and is NOT covered
    by the clearance. Do not add it, and do not call panel (d) a scaling
    study -- it is two points, not a curve.
  * Panel (d) reads plots/local_nvl/gb200_nvl.csv, which is in
    .git/info/exclude. THIS GENERATOR MUST THEREFORE STAY LOCAL TOO, and it
    already is (untracked). Do not hardcode the CSV's values into this file
    to make it self-contained -- that would move the unreleased per-variant
    record into a file that could be committed.

  * Stencil in panel (d) deliberately has NO GPU-aware MPI bar. Every stencil
    run on that system used zero warmup iterations, so lazy CUDA-aware
    connection setup was charged entirely to that one variant (121.85 ms at
    16 GPUs against 22.89 for staged). We retracted the identical artifact on
    our own cluster, where GPU-aware MPI ties WinIPC once warmed up -- which
    is exactly what panel (a) shows. The slot is marked "not measured", not
    left as an ambiguous gap that reads as zero.

ERROR BARS. Only panel (c) has repetitions -- five independent runs per
variant -- so only panel (c) draws them (min/max whiskers, bar = median).
Panels (a), (b) and (d) are single runs per cell and are drawn without
whiskers rather than with invented ones. Within-job spread on LULESH is
<=0.26% CoV, which is why the whiskers are barely visible; the honest caveat,
recorded in results/lulesh_lifecycle_61540.md, is that CROSS-session drift
reaches 1.6%, so these whiskers understate true run-to-run spread. Do not
present them as a confidence interval.

WHAT IS DELIBERATELY NOT ON THIS FIGURE:
  * IPC direct (single-kernel), on either machine. Fastest column in every
    transpose table, and flagged in RESULTS_NVL16_NVL32.md as likely timing
    only the fused kernel -- on GB200 its kernel time FALLS with rank count,
    so it never pays for the remote exchange. It is a measurement bug, not a
    transport. (Panel (d)'s "WinIPC" is the buffered variant, matching panel
    (b); the separate non-fused direct variant is quoted in the caption only.)
  * LULESH at 27 and 64 ranks on GB200 -- see above.
  * The neighbour-only sync variant. Validated (job 63329) but every stencil
    number in the paper is a barrier run; mixing arms in one panel would be
    a provenance error.

Palette: dataviz reference categorical slots in fixed order -- blue (slot 1),
orange (slot 2), aqua (slot 3), violet (slot 7) -- plus a neutral grey for the
host-staged baseline, which is a reference line rather than a competing
mechanism. Colour means the same mechanism in all four panels: blue is always
the interposed window, orange always GPU-aware MPI, aqua always NVSHMEM,
violet always hand-written CUDA IPC, grey always host-staged MPI. Panel (d)
therefore does NOT use the light/dark buffered-vs-direct palette that
gb200_transpose.pdf uses -- consistency across this figure's own panels wins,
and panel (d) plots one variant per mechanism so it does not need the second
channel. The validator (scripts/validate_palette.js) could not be run on this
host -- no node binary -- so the slots are used unmodified from
references/palette.md rather than re-stepped.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import SURFACE, INK, INK2, GRID  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# ---- mechanism -> colour, fixed for all four panels -------------------------
WINIPC  = "#2a78d6"   # slot 1, blue    -- interposed MPI window (this work)
GPUMPI  = "#eb6834"   # slot 2, orange  -- GPU-aware MPI
NVSHMEM = "#1baf7a"   # slot 3, aqua    -- NVSHMEM
HANDIPC = "#4a3aa7"   # slot 7, violet  -- hand-written CUDA IPC
STAGED  = "#8a8a86"   # neutral grey    -- host-staged MPI (the baseline)

# Poster type sizes: this figure is authored at final size and must stay
# readable at ~2 m, so the paper's 7 pt body is replaced with 11 pt here.
# Do not render this at 14 in and let the poster template shrink it.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def recede(ax, axis="y"):
    """Recessive grid behind the marks, no top/right spines."""
    ax.grid(axis=axis, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# =============================================================================
# (a) Stencil -- job 59853, 4 GPUs, times in ms, 100 timed iterations
# =============================================================================
SIZES = ["1024²", "2048²", "4096²", "8192²", "16384²", "32768²"]
STENCIL_MS = {
    "winipc": [3.95, 4.02, 6.07, 12.92, 39.00, 143.04],
    "gpumpi": [4.71, 4.93, 7.05, 13.53, 39.75, 143.73],
    "staged": [7.80, 9.48, 13.20, 23.42, 57.50, 161.44],
}

# =============================================================================
# (b) Transpose -- job 63328, 4 GPUs, B += A^T, GB/s (median over reps)
#
# WHY ACCUMULATE AND NOT OVERWRITE. Job 63328 measured both. B += A^T is the
# Parallel Research Kernels operation the benchmark is defined by; B = A^T is
# an extension we added. It is also the setting every number the paper quotes
# comes from -- the abstract's "up to 3.0x higher throughput than GPU-aware
# MPI" is single-kernel at 1024^2 under accumulate (591 vs 202 GB/s here).
# Overwrite is the friendlier picture (722 vs 207, and 1349 GB/s at 4096^2
# where accumulate manages 883, because accumulating into peer memory has to
# read the destination before writing the sum). Showing the friendlier one on
# a poster while the paper reports the other is how the two end up disagreeing
# in front of a reviewer, so the poster follows the paper.
# =============================================================================
ORDERS = [1024, 2048, 4096, 8192, 16384]
TRANSPOSE_GBS = {
    # Two WinIPC variants, because they tell opposite halves of the story and
    # showing only the buffered one made the panel say WinIPC loses to
    # GPU-aware MPI at every order. The direct variant writes each peer's block
    # from one fused kernel with no packing and no per-phase barrier, which is
    # where the abstraction's advantage actually lives (2.9x at 1024^2). It
    # gives that lead back at the large orders, where the fixed launch and
    # sync costs it removes stop mattering and accumulate's read-modify-write
    # traffic dominates. Both curves, or neither.
    "winipc_d": [591.0, 1167.8, 882.5,  889.2,  884.7],  # direct, single-kernel
    "winipc":  [188.1, 526.6, 836.7, 1014.2, 1062.4],   # buffered
    "gpumpi":  [202.2, 547.7, 853.2, 1022.8, 1077.2],
    "nvshmem": [111.1, 355.9, 723.4,  979.9, 1073.8],   # buffered
    "staged":  [ 78.0, 113.4, 128.7,  115.7,  120.6],
}

# =============================================================================
# (c) LULESH -- job 61540, 8 GPUs, 5 reps, FOM in zones/s
# =============================================================================
# variant key -> (label drawn, colour, mechanism note). The CSV keys are still
# `mpiwrap*` because renaming the build identifiers would break queued Slurm
# jobs whose script text Slurm froze at submit time; the rename to WinIPC lives
# here, at the display edge, exactly as make_lulesh_plots.py does it.
# Panel (c) is the paper's LULESH figure, not a poster-specific redraw. Same
# source CSV, same nine variants, same labels, same mode colouring, so a reader
# who has seen the paper sees the same bars and a reviewer cannot find two
# different numbers for one experiment.
#
# It previously read results/lulesh_poster_fom.csv (job 61540) while the paper
# figure reads results/lulesh_variance.csv (jobs 60796-60800). Both are valid
# 5-rep samples of the same configuration and they agree to ~1.8%, but they are
# not the same numbers -- direct was 1.80 here against 1.832 in the paper. One
# of those had to go, and the paper's is the published one.
#
# Colours are by MECHANISM MODE, matching make_lulesh_plots.py: mode B direct
# field writes, mode C remote-pack, mode A pack+copy, two-sided MPI, host
# shared window. That grouping is what makes the handwritten/interposed pairs
# read as pairs.
BLUE_P, GREEN_P, MAGENTA_P, YELLOW_P = "#2a78d6", "#008300", "#e87ba4", "#eda100"
LULESH_CATEGORY = {
    "direct": "B", "ipc_rp": "C", "mpiwrap_rp": "C",
    "ipc": "A", "mpiwrap": "A", "nvshmem": "A",
    "gpumpi": "T", "staged": "T", "shmwin": "W",
}
LULESH_MODE_COLOR = {"B": BLUE_P, "C": GREEN_P, "A": MAGENTA_P,
                     "T": YELLOW_P, "W": STAGED}
LULESH_MODE_LABEL = {
    "B": "mode B - direct field writes",
    "C": "mode C - remote-pack",
    "A": "mode A - pack + copy",
    "T": "two-sided MPI",
    "W": "host shared window",
}
LULESH_MODE_ORDER = ["B", "C", "A", "T", "W"]
# Same display rename the paper figure applies: the CSV keys stay `mpiwrap*`
# because the build identifiers and queued job scripts still use them.
LULESH_DISPLAY = {"mpiwrap": "winipc", "mpiwrap_rp": "winipc_rp"}

LULESH_CSV = os.path.join(ROOT, "results", "lulesh_variance.csv")


def load_lulesh():
    """variant -> (median, min, max) FOM in Mzone/s, from the 5-rep CSV."""
    reps = {}
    with open(LULESH_CSV) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#")):
            reps.setdefault(row["variant"], []).append(float(row["fom_z_per_s"]))
    out = {}
    for variant, vals in reps.items():
        vals = sorted(v / 1e6 for v in vals)
        out[variant] = (float(np.median(vals)), vals[0], vals[-1])
    return out


# =============================================================================
# (d) GB200 NVL scale-out -- speedup over host-staged MPI at 16 and 32 GPUs
# =============================================================================
# Read, not transcribed: this is the same cleared subset that draws
# gb200_transpose.pdf and gb200_stencil.pdf, so the poster and the paper
# figures cannot drift apart. The file is local-only by clearance -- see the
# DISCLOSURE LIMITS note at the top before touching any of this.
GB200_CSV = os.path.join(HERE, "local_nvl", "gb200_nvl.csv")

# Series drawn per benchmark, and which CSV row feeds each. The staged row is
# the denominator, never a bar. WinIPC is the BUFFERED variant in both
# benchmarks, matching panel (b); the faster non-fused direct variant is
# mentioned in the caption instead of drawn, so one colour keeps one meaning.
GB200_PLAN = {
    "transpose": {
        "baseline": "staged MPI",
        "series": [("WinIPC buffered", WINIPC),
                   ("GPU-aware MPI",   GPUMPI),
                   ("NVSHMEM buffered", NVSHMEM)],
    },
    "stencil": {
        "baseline": "host-staged MPI",
        # GPU-aware MPI is absent on purpose (warmup=0 artifact, see the top
        # of this file). It is marked "not measured" in the panel rather than
        # dropped silently, because a missing bar in a group reads as a zero.
        "series": [("WinIPC",  WINIPC),
                   (None,      GPUMPI),
                   ("NVSHMEM", NVSHMEM)],
    },
}
GB200_GPUS = [16, 32]


def load_gb200():
    """{benchmark: {series: {gpus: value}}} straight from the cleared CSV."""
    if not os.path.exists(GB200_CSV):
        sys.exit(
            f"missing {GB200_CSV}\n"
            "Panel (d) is drawn from the local-only GB200 CSV (see\n"
            ".git/info/exclude). This generator cannot run without it, and the\n"
            "numbers must not be hardcoded here -- read the DISCLOSURE LIMITS\n"
            "note at the top of this file.")
    out = {}
    with open(GB200_CSV) as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["benchmark"], {}) \
               .setdefault(row["series"], {})[int(row["gpus"])] = float(row["value"])
    return out


# =============================================================================
# figure
# =============================================================================
fig, ((axa, axb), (axc, axd)) = plt.subplots(
    2, 2, figsize=(14.0, 10.4),
    gridspec_kw={"wspace": 0.24, "hspace": 0.42},
)

# ---- panel (a): speedup over host-staged MPI --------------------------------
# Ratio rather than milliseconds: the raw times span 4-161 ms, so on a linear
# axis the 32768² group sets the scale and the small grids -- where the gap is
# widest -- collapse to a few pixels. Normalising puts all six sizes in one
# readable band. Same measurement, different presentation.
x = np.arange(len(SIZES))
w = 0.36
sp_winipc = [s / v for s, v in zip(STENCIL_MS["staged"], STENCIL_MS["winipc"])]
sp_gpumpi = [s / v for s, v in zip(STENCIL_MS["staged"], STENCIL_MS["gpumpi"])]

recede(axa)
axa.bar(x - w / 2, sp_winipc, w, color=WINIPC, label="WinIPC (interposed)", zorder=3)
axa.bar(x + w / 2, sp_gpumpi, w, color=GPUMPI, label="GPU-aware MPI", zorder=3)
axa.axhline(1.0, color=STAGED, linewidth=2.0, zorder=4)
axa.text(-0.45, 1.05, "host-staged MPI = 1.0",
         ha="left", va="bottom", fontsize=10, color=INK2)

# Every bar carries its ratio, as in panel (d). Labelling only the extremes
# made a reader interpolate the middle four groups by eye off a 0-2.75 axis.
for i in range(len(SIZES)):
    # At the two largest grids the bars are within 0.02x of each other, so two
    # labels sitting at the same height run together ("1.13x1.12x"). Lift the
    # GPU-aware label clear whenever the pair is too close to separate
    # horizontally; elsewhere the bar heights already do the separating.
    lift = 0.13 if abs(sp_winipc[i] - sp_gpumpi[i]) < 0.15 else 0.03
    axa.text(x[i] - w / 2, sp_winipc[i] + 0.03, f"{sp_winipc[i]:.2f}×",
             ha="center", va="bottom", fontsize=8, color=INK)
    axa.text(x[i] + w / 2, sp_gpumpi[i] + lift, f"{sp_gpumpi[i]:.2f}×",
             ha="center", va="bottom", fontsize=8, color=INK2)

axa.set_xticks(x)
axa.set_xticklabels(SIZES, rotation=20, ha="right")
axa.set_ylim(0, 2.75)
axa.set_ylabel("Speedup over host-staged MPI")
axa.set_xlabel("Grid size")
axa.set_title("(a)  Stencil: 4 GPUs, one node", loc="left", color=INK)
axa.legend(frameon=False, loc="upper right", ncol=1)

# ---- panel (b): transpose bandwidth -----------------------------------------
recede(axb)
for key, label, colour, marker in [
    ("winipc_d", "WinIPC direct",      HANDIPC, "^"),
    ("winipc",  "WinIPC buffered",     WINIPC,  "o"),
    ("gpumpi",  "GPU-aware MPI",       GPUMPI,  "s"),
    ("nvshmem", "NVSHMEM",             NVSHMEM, "^"),
    ("staged",  "Host-staged MPI",     STAGED,  "D"),
]:
    axb.plot(ORDERS, TRANSPOSE_GBS[key], marker=marker, color=colour,
             label=label, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.6)

axb.set_xscale("log", base=2)
axb.set_xticks(ORDERS)
axb.set_xticklabels([f"{o}²" for o in ORDERS], rotation=20, ha="right")
axb.set_xlabel("Matrix order")
axb.set_ylabel("Achieved bandwidth (GB/s)")
axb.set_title("(b)  Transpose  B += Aᵀ: 4 GPUs, one node", loc="left", color=INK)
# Lower right: the five curves all climb left-to-right and converge at the top
# right, so the upper left -- where this legend used to sit -- is exactly where
# the direct variant's rise is. Down here it covers only the flat host-staged
# line and empty space below the crossover.
axb.legend(frameon=False, loc="lower right", ncol=1, fontsize=9.5,
           borderaxespad=1.8)

# The one number a reader should take away from this panel. Named in the text
# rather than pointed at with a leader line -- any arrow to the 16384² point
# has to cross the NVSHMEM and staged curves to get there.
# The 16384^2 summary moved to the figure caption: it is prose, it was sitting
# in the only clear space on the panel, and the legend needs that space more
# than a sentence does.

# ---- panel (c): LULESH figure of merit --------------------------------------
lulesh = load_lulesh()
# Descending by FOM, exactly as the paper figure orders it.
order = sorted(lulesh, key=lambda v: -lulesh[v][0])
labels = [LULESH_DISPLAY.get(v, v) for v in order]
colours = [LULESH_MODE_COLOR[LULESH_CATEGORY[v]] for v in order]
med = np.array([lulesh[v][0] for v in order])
lo = np.array([lulesh[v][1] for v in order])
hi = np.array([lulesh[v][2] for v in order])

xc = np.arange(len(order))
recede(axc)
axc.bar(xc, med, 0.72, color=colours, zorder=3,
        yerr=[med - lo, hi - med],
        error_kw=dict(ecolor=INK2, elinewidth=1.4, capsize=3, zorder=4))
axc.set_xticks(xc)
axc.set_xticklabels(labels, rotation=25, ha="right")
axc.tick_params(axis="x", labelsize=9.5)
axc.set_ylabel("Figure of merit (Mzone/s)")
axc.set_title("(c)  LULESH: 8 GPUs, one node, 5-run median", loc="left", color=INK)
axc.set_ylim(0, max(med) * 1.38)

for xi, m in zip(xc, med):
    axc.text(xi, m + max(med) * 0.035, f"{m:.2f}",
             va="bottom", ha="center", fontsize=9.5, color=INK)

from matplotlib.patches import Patch  # noqa: E402
axc.legend(handles=[Patch(facecolor=LULESH_MODE_COLOR[c],
                          label=LULESH_MODE_LABEL[c])
                    for c in LULESH_MODE_ORDER],
           frameon=False, loc="upper right", ncol=2, columnspacing=0.7,
           handlelength=1.0, handletextpad=0.35, borderaxespad=0.1,
           fontsize=9)

# ---- panel (d): GB200 NVL scale-out, normalised ------------------------------
gb = load_gb200()
recede(axd)

# Four groups: {transpose, stencil} x {16, 32} GPUs. Each group draws only the
# series it has -- three on the transpose side, two on the stencil side, each
# set centred in its own group. Holding a fixed three-slot grid left a gap on
# the stencil side that had to be captioned away; a group that is simply two
# bars wide needs no explaining, and the caption still says why GPU-aware MPI
# is absent for anyone who counts.
groups = [("transpose", 16), ("transpose", 32), ("stencil", 16), ("stencil", 32)]
xd = np.arange(len(groups), dtype=float)
GROUP_W = 0.76

# Values first, drawing second: the ghost slot standing in for the absent
# stencil GPU-aware series is sized from the panel's own peak, so the peak has
# to be known before anything is drawn rather than accumulated as bars appear.
cells = []      # (x, colour, value, width)
for gi, (bench, gpus) in enumerate(groups):
    plan = GB200_PLAN[bench]
    # Resolve this group's series first so the absent ones never take a slot.
    present = []
    for slot in range(len(plan["series"])):
        name, colour = plan["series"][slot]
        if name is None:
            continue                    # not measured: takes no slot at all
        base = gb[bench][plan["baseline"]][gpus]
        present.append((colour, gb[bench][name][gpus] / base))

    n = len(present)
    wd = GROUP_W / n
    for k, (colour, val) in enumerate(present):
        offs = -GROUP_W / 2 + wd * (k + 0.5)
        cells.append((xd[gi] + offs, colour, val, wd * 0.88))

peak = max(v for _, _, v, _ in cells)
top = peak * 1.30

for xpos, colour, val, bw in cells:
    axd.bar(xpos, val, bw, color=colour, zorder=3)
    axd.text(xpos, val + top * 0.015, f"{val:.1f}×", ha="center",
             va="bottom", fontsize=9.5, color=INK, zorder=4)

axd.axhline(1.0, color=STAGED, linewidth=2.0, zorder=4)
axd.text(-0.48, 1.06, "host-staged MPI = 1.0",
         ha="left", va="bottom", fontsize=10, color=INK2)

axd.set_xticks(xd)
axd.set_xticklabels([f"Transpose\n16 GPUs\n(4 nodes)",
                     f"Transpose\n32 GPUs\n(8 nodes)",
                     f"Stencil\n16 GPUs\n(4 nodes)",
                     f"Stencil\n32 GPUs\n(8 nodes)"], fontsize=9.5)
axd.set_ylim(0, top)
axd.set_ylabel("Speedup over host-staged MPI")
axd.set_title("(d)  GB200 NVL scale-out: multi-node NVLink window",
              loc="left", color=INK)
axd.legend(handles=[
    Patch(facecolor=WINIPC, label="WinIPC (interposed)"),
    Patch(facecolor=GPUMPI, label="GPU-aware MPI"),
    Patch(facecolor=NVSHMEM, label="NVSHMEM"),
], frameon=False, loc="upper right", ncol=1, handlelength=1.0,
           handletextpad=0.35, borderaxespad=0.1)

# Caption, hard-wrapped by hand. Matplotlib does not wrap fig.text, so a
# single long string silently runs off the right edge of the canvas -- which
# it did, truncating the stencil GPU-aware caveat mid-word.
fig.text(0.008, 0.074,
         "(a)-(c) H200 SXM (NVSwitch), single node.  (c) is the paper's LULESH "
         "figure: same data, bars are medians of 5 runs, whiskers min/max.",
         fontsize=9, color=INK2, ha="left")
fig.text(0.008, 0.052,
         "(b) B += Aᵀ, the Parallel Research Kernels operation the paper "
         "reports.  At 16384²: WinIPC buffered is 8.8× host-staged MPI and "
         "1.4% behind GPU-aware MPI.",
         fontsize=9, color=INK2, ha="left")
fig.text(0.008, 0.030,
         "(d) GB200 NVL scale-out system, 4 and 8 nodes, cross-node CUDA "
         "fabric-handle window.  WinIPC = buffered variant, as in (b) — its "
         "direct variant reaches 5.3× (16 GPUs) and 3.2× (32 GPUs) on transpose.",
         fontsize=9, color=INK2, ha="left")
fig.text(0.008, 0.008,
         "Stencil GPU-aware MPI is omitted from (d): those runs used zero warmup, "
         "so one-time lazy connection setup was charged to that variant alone.  "
         "(d) is two configurations, not a scaling curve.",
         fontsize=9, color=INK2, ha="left")

fig.subplots_adjust(left=0.062, right=0.982, top=0.945, bottom=0.150)

for path, kwargs in [("poster_summary.pdf", dict(metadata={"CreationDate": None})),
                     ("poster_summary.png", dict(dpi=300))]:
    out = os.path.join(HERE, path)
    fig.savefig(out, **kwargs)
    print(f"wrote {out}")
plt.close(fig)
