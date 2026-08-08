"""Shared figure style for the paper's plots.

Every figure is emitted as PDF, not PNG. Vector output is what fixes the
"fonts too small" review comment: a raster figure scaled down to a column
shrinks its text with it, whereas a PDF sized for its final placement renders
type at the size set here regardless of how the LaTeX side scales the box.

COL_W is one IEEE two-column-format column. Figures are authored at final
size, so font sizes below are the sizes that actually appear in the PDF -- do
not size a figure at 8 inches and let \\includegraphics shrink it.

pdf.fonttype 42 embeds TrueType rather than Type 3, which keeps text
selectable and searchable and avoids the Type 3 warnings some camera-ready
checkers emit.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W = 3.4          # inches: one IEEE column
FULL_W = 7.16        # inches: both columns, for anything spanning the page

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
GRID    = "#e4e3df"


def use_paper_style():
    """Apply the shared rcParams. Call once at the top of a generator."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.2,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, path):
    # CreationDate=None suppresses the wall-clock timestamp matplotlib embeds in
    # the PDF. Without it every regeneration rewrites every figure byte-for-byte
    # differently, so `git status` cannot distinguish "this figure's data
    # changed" from "the generator ran again" -- which is precisely the
    # confusion that let stale figures sit beside fresh tables (see the
    # provenance note in results/transpose_results.md). With it, a figure shows
    # up as modified only when its numbers actually moved.
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")
