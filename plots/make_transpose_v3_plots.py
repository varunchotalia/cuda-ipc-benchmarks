#!/usr/bin/env python3
"""Transpose order sweep from job 90161.

    plots/transpose_v3_sweep.pdf / .png

Two panels, laid out like the paper's Figure 3a and 3b so this reads as that
figure with a denser x-axis rather than as a new one:

    (a) accumulate, B += A^T   -- WinIPC variants vs MPI
    (b) overwrite,  B = A^T    -- same series

NVSHMEM IS DELIBERATELY NOT PLOTTED, even though this job measured it. Its
numbers here are wrong by an order of magnitude against job 63328 on the same
node: nvsingle reads 7.8 GB/s at order 1024 where 63328 measured 355.7, and
nvbuffered 12.2 against 111.1. Every affected run still reports "Solution
validates" and the log carries no NVSHMEM warning, so this is a transport or
placement problem rather than a correctness one -- the same source md5, the
same build flags, and the same launch as v2, which makes it unexplained rather
than understood. Plotting it would put a 45x-wrong curve on a figure next to
correct ones. The WinIPC and MPI series in the same job reproduce 63328 to
within a few percent and converge to ~0% at the large orders, so they are
sound; only the nv* modes are suspect. Restore panel (c) once NVSHMEM here is
explained.

WHAT THIS ADDS OVER THE PAPER'S FIGURE 3. That figure has five orders, all
powers of two, one run per point. This job measured THIRTEEN orders including
non-powers-of-two (1536, 3072, 6144, 12288, 24576, 49152) with three
repetitions per point, and reaches 65536 rather than stopping at 16384. The
non-power-of-two points matter because the single-kernel curve turns over
somewhere between 2048 and 4096, and with only powers of two that turn is two
segments of a straight line -- you cannot see where it happens or how sharp it
is.

Job 90161, h200x4-04, 2026-09-09. 100 timed iterations after 20 untimed, three
reps, UCX defaults. An earlier attempt at this sweep (87249) was OOM-killed at
order 65536: the verifier mirrors each rank's whole B block to the host, 8 GB
per rank at that order, against a 44 GB partition default. Points are medians
of the three reps.

Rates use the paper's Eq. 1 convention, 2N^2 * sizeof(double) / t, which counts
the same bytes for accumulate and overwrite even though accumulate must read
the destination before writing the sum. So (a) and (b) are not directly
comparable as bandwidths -- the gap between them is partly that convention.
"""
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LOG = os.path.join(ROOT, "transpose_fig_v3_90161.out")

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINIPC, GPUMPI, NVSHMEM = "#2a78d6", "#eb6834", "#1baf7a"
HANDIPC, STAGED = "#4a3aa7", "#8a8a86"
PERPHASE, NVALT1, NVALT2 = "#9a86e0", "#0f8f63", "#7fd4b0"

# mode key -> (label, colour, marker)
STYLE = {
    "single":     ("WinIPC direct, single-kernel", HANDIPC,  "^"),
    "direct":     ("WinIPC direct, per-phase",     PERPHASE, "v"),
    "buffered":   ("WinIPC buffered",              WINIPC,   "o"),
    "gpumpi":     ("GPU-aware MPI",                GPUMPI,   "s"),
    "staged":     ("Host-staged MPI",              STAGED,   "D"),
    "nvsingle":   ("NVSHMEM single-kernel",        NVALT1,   "^"),
    "nvdirect":   ("NVSHMEM direct",               NVSHMEM,  "v"),
    "nvbuffered": ("NVSHMEM buffered",             NVALT2,   "o"),
}
PANELS = [
    (1, ["single", "direct", "buffered", "gpumpi", "staged"],
     "(a)  Accumulate,  B += Aᵀ"),
    (0, ["single", "direct", "buffered", "gpumpi", "staged"],
     "(b)  Overwrite,  B = Aᵀ"),
]


def load(path=LOG):
    """(accum, mode, order) -> median GB/s over reps; fails loudly on a gap."""
    reps, bad = {}, 0
    with open(path) as fh:
        for line in fh:
            if "RESULT transpose" not in line:
                continue
            d = dict(re.findall(r"(\w+)=([\w.+-]+)", line))
            if d.get("validates") != "1":
                bad += 1
                continue
            key = (int(d["accum"]), d["mode"], int(d["order"]))
            reps.setdefault(key, []).append(float(d["mbps"]) / 1000.0)
    if not reps:
        raise SystemExit(f"no RESULT lines parsed from {path}")
    if bad:
        raise SystemExit(f"{bad} rows failed validation in {path}; not plotting")
    return {k: st.median(v) for k, v in reps.items()}, reps


DATA, REPS = load()
ORDERS = sorted({o for _, _, o in DATA})

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE)
for ax, (accum, modes, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for mode in modes:
        label, colour, marker = STYLE[mode]
        xs = [o for o in ORDERS if (accum, mode, o) in DATA]
        ys = [DATA[(accum, mode, o)] for o in xs]
        ax.plot(xs, ys, marker=marker, markersize=4.5, linewidth=1.8,
                color=colour, label=label, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ORDERS)
    ax.set_xticklabels([str(o) for o in ORDERS], rotation=55, ha="right",
                       fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Matrix order")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)

axes[0].set_ylabel("Aggregate effective rate (GB/s)")
# Scale to what is plotted, not to every mode measured.
plotted = [DATA[(a, m, o)] for a, modes, _ in PANELS for m in modes
           for o in ORDERS if (a, m, o) in DATA]
top = max(plotted) * 1.10
for ax in axes:
    ax.set_ylim(0, top)

# Hard-wrapped by hand: matplotlib does not wrap fig.text, it just runs off
# the right edge and truncates. Narrowing the figure to two panels made the
# previous three-panel wording overflow.
for y, line in [
    (0.075, "Job 90161, 4 GPUs on one H200 node, 13 matrix orders, medians of "
            "3 reps, 100 timed iterations after 20 untimed."),
    (0.052, "Non-power-of-two orders are included so the single-kernel "
            "turnover is resolved rather than interpolated between powers of two."),
    (0.029, "NVSHMEM was measured but is not plotted: its rates in this job "
            "disagree with job 63328 by up to 45x and are unexplained."),
    (0.006, "Rates follow the paper's Eq. 1 (2N²·sizeof(double)/t), which counts "
            "the same bytes for both operations although accumulate must read "
            "the destination first, so (a) and (b) are not comparable as "
            "bandwidths."),
]:
    fig.text(0.006, y, line, fontsize=8.5, color=INK2, ha="left")

fig.subplots_adjust(left=0.072, right=0.99, top=0.93, bottom=0.285, wspace=0.14)
for name, kw in [("transpose_v3_sweep.pdf", dict(metadata={"CreationDate": None})),
                 ("transpose_v3_sweep.png", dict(dpi=200))]:
    out = os.path.join(HERE, name)
    fig.savefig(out, **kw)
    print(f"wrote {out}")
plt.close(fig)
