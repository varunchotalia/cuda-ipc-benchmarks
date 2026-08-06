#!/usr/bin/env python3
"""Transpose figures for the paper. Writes three single-column PDFs:

    plots/transpose_ipc_accum.pdf      IPC -- With Accumulation (B += A^T)
    plots/transpose_ipc_noaccum.pdf    IPC -- No Accumulation  (B = A^T)
    plots/transpose_ipc_vs_nvshmem.pdf IPC vs NVSHMEM -- No Accumulation

Three panels of the previous six-panel composite are deliberately NOT emitted:
"IPC -- Accum vs No-Accum", "NVSHMEM -- With Accumulation" and
"NVSHMEM -- No Accumulation". Their content is either restated by the two IPC
panels or subsumed by the IPC-vs-NVSHMEM comparison.

One file per panel, rather than a composite, so the LaTeX side can place them
one-per-column or three-across without regenerating anything.

PROVENANCE. This generator did not exist before 2026-08-04: the previous
figure, plots/transpose_benchmark_ipc_nvshmem.png, was committed as a finished
image with no script, so it could not be regenerated and its numbers could not
be checked against the results file. Everything here is parsed from
results/transpose_results.md, which is therefore the single source of truth --
update that file and rerun, do not edit figures by hand.

Data: 2026-05-05, four H200 GPUs in one peer-accessible domain, mean of 100
timed iterations after one untimed, UCX defaults (UCX_TLS deliberately unset --
see the appendix in the results file for why pinning it is harmful).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import COL_W, use_paper_style, save, GRID  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

use_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "transpose_results.md")
OUT = HERE

# ---------------------------------------------------------------------------
# Parse results/transpose_results.md
#
# Only the "## 4 GPUs" sections are read; the 2-GPU section is a separate
# experiment and must not be mixed in. A cell may hold a non-numeric placeholder
# such as "(rerun pending)", which becomes None and is skipped when plotting
# rather than being silently treated as zero.
# ---------------------------------------------------------------------------
ROW = re.compile(r"^\s*(\d+²)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$")


def parse(path=SRC):
    data = {"accum": {}, "noaccum": {}}
    sizes = []
    bucket = None
    with open(path) as f:
        for line in f:
            if line.startswith("## "):
                if "4 GPUs" not in line:
                    bucket = None                    # e.g. the 2-GPU section
                elif "Without Accumulation" in line:
                    bucket = "noaccum"
                elif "With Accumulation" in line:
                    bucket = "accum"
                else:
                    bucket = None
                continue
            if bucket is None or "|" not in line or line.startswith("---"):
                continue
            m = ROW.match(line)
            if not m:
                continue
            size, mode, raw = m.group(1), m.group(2), m.group(3)
            if mode.lower().startswith("mode"):       # header row
                continue
            try:
                val = float(raw)
            except ValueError:
                val = None                            # "(rerun pending)" etc.
            data[bucket].setdefault(mode, {})[size] = val
            if size not in sizes:
                sizes.append(size)
    return data, sizes


DATA, SIZES = parse()
if not DATA["accum"] or not DATA["noaccum"]:
    raise SystemExit(f"parsed no 4-GPU data from {SRC}")

# The composite figure plotted 1024^2..16384^2; keep that range.
SIZES = [s for s in SIZES if s in
         {"1024²", "2048²", "4096²", "8192²", "16384²"}]
X = range(len(SIZES))


def series(bucket, mode):
    """(x, y) for one mode, dropping sizes with no measurement."""
    row = DATA[bucket].get(mode, {})
    xs, ys = [], []
    for i, s in enumerate(SIZES):
        v = row.get(s)
        if v is not None:
            xs.append(i)
            ys.append(v)
    if not xs:
        raise SystemExit(f"no data for mode {mode!r} in {bucket}")
    return xs, ys


# style: (source mode name, legend label, colour, marker, linestyle)
IPC_SERIES = [
    ("IPC direct (single-K)",  "Direct (single-kernel)", "#2ca02c", "o", "-"),
    ("IPC direct (per-phase)", "Direct (per-phase)",     "#1f77b4", "s", "-"),
    ("IPC buffered",           "Buffered",               "#d62728", "^", "--"),
    ("GPU-aware MPI",          "GPU-aware MPI",          "#9467bd", "D", "-."),
    ("Staged MPI",             "Staged MPI",             "#8a8a86", "v", ":"),
]

VS_SERIES = [
    ("IPC direct (single-K)",  "IPC single-kernel", "#2ca02c", "o", "-"),
    ("IPC direct (per-phase)", "IPC per-phase",     "#1f77b4", "s", "-"),
    ("IPC buffered",           "IPC buffered",      "#d62728", "^", "-."),
    ("GPU-aware MPI",          "GPU-aware MPI",     "#9467bd", "D", "-."),
    ("NVSHMEM direct",         "NVSHMEM direct",    "#ff7f0e", "o", "--"),
    ("NVSHMEM single-K",       "NVSHMEM single-K",  "#eda100", "D", "--"),
    ("NVSHMEM buffered",       "NVSHMEM buffered",  "#9a7fc9", "^", "--"),
]


def panel(bucket, spec, title, path, ncol=1):
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for mode, label, colour, marker, ls in spec:
        xs, ys = series(bucket, mode)
        ax.plot(xs, ys, ls, color=colour, marker=marker, label=label)
    ax.set_title(title)
    ax.set_ylabel("GB/s")
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
    leg = ax.legend(loc="lower right", ncol=ncol, frameon=True,
                    facecolor="white", edgecolor=GRID, framealpha=1.0,
                    handlelength=1.6, borderaxespad=0.4, labelspacing=0.22,
                    borderpad=0.4, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(5)
    fig.tight_layout()
    save(fig, path)


panel("accum", IPC_SERIES, "IPC — With Accumulation (B += A\u1d40)",
      os.path.join(OUT, "transpose_ipc_accum.pdf"))
panel("noaccum", IPC_SERIES, "IPC — No Accumulation (B = A\u1d40)",
      os.path.join(OUT, "transpose_ipc_noaccum.pdf"))
# seven series: two legend columns keep the box off the data
panel("noaccum", VS_SERIES, "IPC vs NVSHMEM — No Accumulation",
      os.path.join(OUT, "transpose_ipc_vs_nvshmem.pdf"), ncol=2)
