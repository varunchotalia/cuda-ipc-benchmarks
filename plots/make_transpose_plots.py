#!/usr/bin/env python3
"""Transpose figures for the paper (Fig. 3). Writes two single-column PDFs:

    plots/transpose_ipc_accum.pdf      IPC -- With Accumulation (B += A^T)
    plots/transpose_ipc_noaccum.pdf    IPC -- No Accumulation  (B = A^T)

One file per panel, rather than a composite, so the LaTeX side can place them
one-per-column without regenerating anything.

DATA: job 90161, h200x4-04, 2026-09-09. Four H200 GPUs in one peer-accessible
domain, 100 timed iterations after 20 untimed, UCX defaults (UCX_TLS unset),
three reps per point; each point is the MEDIAN of the three reps. Read from
results/transpose_90161.csv, which scripts/build_transpose_90161_csv.py builds
from the (gitignored) raw log and which refuses to build if the log header
disagrees with those settings or any run failed validation. Update that CSV and
rerun; do not edit figures by hand.

Until 2026-09-22 these panels came from results/transpose_results.md (job
28917, 2026-05-05, one run per point, one untimed iteration). That source is
superseded for Fig. 3 and Table III.

The IPC-vs-NVSHMEM panel this script used to write is no longer generated: the
paper dropped the intra-node NVSHMEM comparison because it does not reproduce
(results/nvshmem_not_reproducible.md). plots/transpose_ipc_vs_nvshmem.pdf is a
leftover from the May/August data and is not a paper figure.
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import COL_W, use_paper_style, save, GRID  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

use_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "transpose_90161.csv")
OUT = HERE

# The published figure plots 1024^2..16384^2; keep that range and format. Job
# 90161 also has non-power-of-two orders and orders up to 65536; those are in
# plots/transpose_v3_sweep.pdf, not here.
ORDERS = [1024, 2048, 4096, 8192, 16384]
SIZES = [f"{o}\u00b2" for o in ORDERS]
X = range(len(SIZES))
BUCKET = {"accum": "1", "noaccum": "0"}


def load(path=SRC):
    reps = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r["warmup"], r["iters"]) != ("20", "100"):
                raise SystemExit(f"{path}: row with warmup={r['warmup']} "
                                 f"iters={r['iters']}")
            key = (r["accum"], r["mode"], int(r["order"]))
            reps.setdefault(key, []).append(float(r["gbps"]))
    return {k: statistics.median(v) for k, v in reps.items()}


DATA = load()


def series(bucket, mode):
    """(x, y) for one mode; every point must be present."""
    ys = []
    for o in ORDERS:
        key = (BUCKET[bucket], mode, o)
        if key not in DATA:
            raise SystemExit(f"no data for {key} in {SRC}")
        ys.append(DATA[key])
    return list(X), ys


# style: (source mode name, legend label, colour, marker, linestyle)
IPC_SERIES = [
    ("single",   "IPC direct (single-kernel)", "#2ca02c", "o", "-"),
    ("direct",   "IPC direct (per-phase)",     "#1f77b4", "s", "-"),
    ("buffered", "IPC buffered",               "#d62728", "^", "--"),
    ("gpumpi",   "GPU-aware MPI",              "#9467bd", "D", "-."),
    ("staged",   "Staged MPI",                 "#8a8a86", "v", ":"),
]

def panel(bucket, spec, title, path, ncol=1, legend_only=None):
    """legend_only: if given, only these labels get a legend entry.
    The other series are still drawn, just not keyed."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for mode, label, colour, marker, ls in spec:
        xs, ys = series(bucket, mode)
        keyed = legend_only is None or label in legend_only
        ax.plot(xs, ys, ls, color=colour, marker=marker,
                label=label if keyed else "_nolegend_")
    ax.set_title(title)
    ax.set_ylabel("GB/s")
    ax.set_xlabel("Matrix size")
    ax.set_xticks(list(X))
    ax.set_xticklabels(SIZES)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Bottom-right, in an opaque white box. The box matters: the curves rise
    # left-to-right, so the corner is the emptiest region, but the flat
    # Staged MPI line runs along the bottom of the IPC panels and a
    # transparent legend would sit ambiguously on top of it.
    # Lifted off the axis floor: "lower right" alone puts the box on top of
    # the flat Staged MPI series, which runs along the bottom of the IPC
    # panels. bbox_to_anchor is in axes fractions, so this clears it at any
    # y-limit rather than at one hard-coded data value.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16), ncol=ncol,
                    frameon=True, facecolor="white", edgecolor=GRID,
                    framealpha=1.0, handlelength=1.6, borderaxespad=0.4,
                    labelspacing=0.22, borderpad=0.4, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(5)
    fig.tight_layout()
    save(fig, path)


panel("accum", IPC_SERIES, "IPC — With Accumulation (B += A\u1d40)",
      os.path.join(OUT, "transpose_ipc_accum.pdf"))
panel("noaccum", IPC_SERIES, "IPC — No Accumulation (B = A\u1d40)",
      os.path.join(OUT, "transpose_ipc_noaccum.pdf"))
