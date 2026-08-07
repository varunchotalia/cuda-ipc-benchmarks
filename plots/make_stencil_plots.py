#!/usr/bin/env python3
"""Stencil figures for the paper. Writes three single-column PDFs:

    plots/stencil_time.pdf      time, 4 GPUs, three modes
    plots/stencil_speedup.pdf   speedup over host-staged MPI, 4 GPUs
    plots/stencil_overhead.pdf  GPU-aware MPI overhead vs interposed, 2/4/8 ranks

The throughput panel of the previous three-panel composite is not emitted: it
is the same measurement as the time panel (throughput is cells/time), so the
two panels could not disagree and printing both spends a column restating one
result. Time is kept because it is the quantity measured.

The speedup panel exists because the time panel cannot show the small grids.
Time spans 4 ms to 161 ms across the six sizes, so on a linear axis the
32768² group sets the scale and the 1024²-4096² groups collapse into a few
pixels -- exactly where the interposed-vs-GPU-aware gap is widest (2.4x vs
1.9x over staged at 2048²). Normalising to host-staged MPI removes the
size-dependent magnitude and puts every point in a 1.1-2.4 band, so all six
sizes are readable at one scale. It is a re-plot of the same job-59853
measurements as the time panel, not a new experiment.

The overhead panel is a separate file rather than a composite member because
it is descriptive -- the 2- and 8-rank points are not a controlled scaling
study -- so the paper must be able to place or drop it independently.

Data source: job 59853 (h200x8-04, 2026-08-03) -- the consolidated re-baseline.
UCX defaults (UCX_TLS deliberately unset), STENCIL_WARMUP=20, uniform
-O3 -gencode arch=compute_90,code=sm_90, 100 timed iterations, rank counts
2/4/8 all from the same node. See results/stencil_results.txt, section
"RE-BASELINE (authoritative)".

Three explicitly-named series per review feedback -- "Interposed",
"GPU-aware MPI", "Host-staged MPI" -- replacing the earlier two-series
"IPC"/"MPI" version, which omitted the parity result that is now the central
stencil finding. GPU-aware MPI is drawn in the same purple used for that series
in the transpose figure, so the two figures read consistently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import COL_W, use_paper_style, save, GRID  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

use_paper_style()

SIZES = ["1024²", "2048²", "4096²", "8192²", "16384²", "32768²"]
x = np.arange(len(SIZES))

# ---- job 59853: time (ms) and throughput (Gcell/s), 100 timed iterations ----
T = {  # ranks -> mode -> times
    2: {"ipc": [2.26, 3.10, 6.54, 19.66, 71.52, 278.78],
        "gpu": [3.95, 3.17, 6.88, 19.94, 71.80, 279.14],
        "stg": [3.61, 5.08, 9.50, 23.72, 77.74, 287.43]},
    4: {"ipc": [3.95, 4.02, 6.07, 12.92, 39.00, 143.04],
        "gpu": [4.71, 4.93, 7.05, 13.53, 39.75, 143.73],
        "stg": [7.80, 9.48, 13.20, 23.42, 57.50, 161.44]},
    8: {"ipc": [4.10, 4.42, 5.29, 8.97, 22.13, 74.59],
        "gpu": [4.88, 5.12, 6.02, 9.80, 22.99, 75.69],
        "stg": [8.03, 9.47, 12.12, 19.72, 41.33, 95.08]},
}
G4 = {"ipc": [26.57, 104.23, 276.36, 519.26, 688.26, 750.68],
      "gpu": [22.29, 85.07, 237.95, 495.88, 675.25, 747.08],
      "stg": [13.45, 44.26, 127.12, 286.58, 466.86, 665.12]}

IPC_C, GPU_C, STG_C = "#1f77b4", "#9467bd", "#ff7f0e"
LBL = {"ipc": "Interposed", "gpu": "GPU-aware MPI", "stg": "Host-staged MPI"}
RANK_C = {2: "#dcd0ec", 4: "#a97fc9", 8: "#6a3d9a"}

# =============== figure 1: time, 4 GPUs ===============
fig, ax = plt.subplots(figsize=(COL_W, 2.35))
w = 0.27
ax.bar(x - w, T[4]["ipc"], w, color=IPC_C, label=LBL["ipc"])
ax.bar(x,     T[4]["gpu"], w, color=GPU_C, label=LBL["gpu"])
ax.bar(x + w, T[4]["stg"], w, color=STG_C, label=LBL["stg"])
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Grid size")
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.legend(loc="upper left", frameon=False, handlelength=1.4,
          borderaxespad=0.2, labelspacing=0.25)
ax.grid(True, axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
save(fig, "plots/stencil_time.pdf")

# =============== figure 2: speedup over host-staged MPI, 4 GPUs ===============
# Same job-59853 numbers as figure 1, divided by the host-staged time at the
# same grid size. Bars start at 0 so the bar length is proportional to the
# ratio it encodes; the dashed line at 1.0 is host-staged itself, drawn in the
# same orange that series wears in figure 1 so the baseline is identifiable
# without a third bar. Every bar is labelled -- twelve bars is few enough to
# label them all, and reading the small grids is the entire reason this panel
# exists.
fig, ax = plt.subplots(figsize=(COL_W, 2.35))
# bw, not w: figure 3 below reuses the w = 0.27 set for the three-series
# figure 1, and rebinding it here silently widened those bars into each other.
bw = 0.36
SPEEDUP = {m: [s / v for s, v in zip(T[4]["stg"], T[4][m])] for m in ("ipc", "gpu")}
for off, mode, colour in ((-bw / 2, "ipc", IPC_C), (bw / 2, "gpu", GPU_C)):
    ax.bar(x + off, SPEEDUP[mode], bw * 0.94, color=colour, label=LBL[mode])
    for xi, v in zip(x + off, SPEEDUP[mode]):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=5,
                color="#3a3a3a")
ax.axhline(1.0, color=STG_C, ls="--", lw=0.9, zorder=3,
           label="Baseline: host-staged MPI")
ax.set_ylabel("Speedup over host-staged MPI (×)")
ax.set_xlabel("Grid size")
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.set_ylim(0, max(max(v) for v in SPEEDUP.values()) * 1.16)
ax.legend(loc="upper right", frameon=False, handlelength=1.4,
          borderaxespad=0.2, labelspacing=0.25)
ax.grid(True, axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
save(fig, "plots/stencil_speedup.pdf")

# =============== figure 3: parity with GPU-aware MPI ===============
# How much slower GPU-aware MPI is than the interposed path. Log y because the
# values span 0.1%-75%: small grids are latency-bound and noisy, large grids
# converge to parity. All values are plotted -- nothing is clipped.
fig, ax = plt.subplots(figsize=(COL_W, 2.35))
for i, np_ in enumerate((2, 4, 8)):
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    ax.bar(x + (i - 1) * w, d, w, color=RANK_C[np_], label=f"{np_} GPUs")
ax.axhline(1.0, color="#8a8a8a", ls=":", lw=0.9)
ax.text(-0.48, 1.06, "1%", fontsize=5.5, color="#6a6a6a", ha="left")
ax.set_yscale("log")
ax.set_ylabel("% slower than interposed (log scale)")
ax.set_xlabel("Grid size")
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.legend(loc="upper right", frameon=False, handlelength=1.4,
          borderaxespad=0.2, labelspacing=0.25)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, which="both")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for i, np_ in enumerate((2, 4, 8)):     # label the two largest grids
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    for k in (4, 5):
        ax.text(k + (i - 1) * w, d[k] * 1.12, f"{d[k]:.1f}",
                ha="center", fontsize=5, color="#3a3a3a")
fig.tight_layout()
save(fig, "plots/stencil_overhead.pdf")

for np_ in (2, 4, 8):
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    print(f"np={np_} GPU-aware vs interposed: " +
          ", ".join(f"{s}={v:+.1f}%" for s, v in zip(SIZES, d)))

# Speedup over host-staged at every rank count, for the text. Only np=4 is
# plotted, but the 2- and 8-rank columns are what justify calling the range
# "1.1x to 1.9x" in the results file, so print them too.
for np_ in (2, 4, 8):
    for mode in ("ipc", "gpu"):
        sp = [s / v for s, v in zip(T[np_]["stg"], T[np_][mode])]
        print(f"np={np_} {LBL[mode]:>14s} vs host-staged: " +
              ", ".join(f"{s}={v:.2f}x" for s, v in zip(SIZES, sp)))
