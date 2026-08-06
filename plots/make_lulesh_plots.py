#!/usr/bin/env python3
"""LULESH charts for the paper. Writes two single-column PDFs:

    plots/lulesh_variants_sxm.pdf  all nine variants, ranked by elapsed
    plots/lulesh_modes_sxm.pdf     the three send modes, handwritten vs interposed

VARIANCE (E1). If results/lulesh_variance.csv exists it is used to draw
min/max whiskers on the matched pairs in chart 2, and the bar becomes the
median of the repetitions rather than a single run. Expected columns:
    variant,rep,elapsed_s_hi
one row per (variant, repetition) with no pre-aggregation. Absent the file the
charts fall back to the single-run values in lulesh_results.csv and print a
notice, so a missing variance file is visible rather than silent.

All numbers are read from results/lulesh_results.csv -- nothing is hard-coded
here, so the figures and the committed results file cannot drift apart.

Data: job 60150 (h200x8-03, SXM/NVSwitch all-to-all), 8 ranks, -s 45, full
sedov run to t=0.01, 3145 iterations, **UCX defaults** (UCX_TLS unset).

Supersedes job 46979, which exported UCX_TLS=self,sm,cuda_copy,cuda_ipc. That
pinning inflated the staged baseline from 1.96 s to 3.56 s while moving every
other variant <=10%, so all speedup-vs-staged figures shifted: direct went from
2.78x to +56.8% z/s (1.57x), and shmwin flipped sign -- previously reported as
1.70x faster than staged, it is actually ~5% slower, which is unsurprising
since it stages through a HOST window. gpumpi is now included: under defaults
it beats staged, so relegating it to an appendix is no longer justified.

Palette: dataviz reference categorical slots, fixed order (validated set).
Marks: thin bars, 2px surface gaps, recessive grid, text in ink tokens.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import COL_W, use_paper_style, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
GRID    = "#e4e3df"
# categorical slots 1-4, fixed order; grey is a neutral for the host-window case
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
GREY = "#8a8a86"

use_paper_style()

# ---- data: read from results/lulesh_results.csv (single source of truth) ----
# Do not hard-code values here. The CSV carries node, command, environment,
# correctness value and both precisions; see its header block.
import csv, os

CSV = os.path.join(os.path.dirname(__file__), "..", "results", "lulesh_results.csv")
CATEGORY = {  # variant -> plot category
    "direct": "B", "ipc_rp": "C", "mpiwrap_rp": "C",
    "ipc": "A", "mpiwrap": "A", "nvshmem": "A",
    "gpumpi": "T", "staged": "T", "shmwin": "W",
}

def load(path=CSV):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    rd = csv.DictReader(rows)
    out = {}
    for r in rd:
        out[r["variant"]] = {
            "elapsed": float(r["elapsed_s_hi"]),
            "gain": float(r["gain_pct"]) / 100.0,
            "mode": r["mode"],
        }
    return out

DATA = load()
missing = set(CATEGORY) - set(DATA)
if missing:
    raise SystemExit(f"CSV is missing variants: {sorted(missing)}")

# ascending elapsed == descending performance
variants = [(k, CATEGORY[k], DATA[k]["elapsed"], DATA[k]["gain"])
            for k in sorted(DATA, key=lambda k: DATA[k]["elapsed"])]

MODE_COLOR = {"B": BLUE, "C": GREEN, "A": MAGENTA, "T": YELLOW, "W": GREY}
MODE_LABEL = {
    "B": "mode B - direct field writes",
    "C": "mode C - remote-pack",
    "A": "mode A - pack + copy",
    "T": "two-sided MPI",
    "W": "host shared window",
}
CAT_ORDER = ["B", "C", "A", "T", "W"]

# ---- optional variance data from E1 (jobs 60796-60800) --------------------
VAR_CSV = os.path.join(os.path.dirname(__file__), "..", "results",
                       "lulesh_variance.csv")


def load_variance(path=VAR_CSV):
    """variant -> (median, min, max) over repetitions, or {} if unavailable."""
    if not os.path.exists(path):
        return {}
    reps = {}
    with open(path) as f:
        rows = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    for r in csv.DictReader(rows):
        reps.setdefault(r["variant"], []).append(float(r["elapsed_s_hi"]))
    out = {}
    for v, xs in reps.items():
        xs.sort()
        n = len(xs)
        med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        out[v] = (med, xs[0], xs[-1])
    return out


VAR = load_variance()
if VAR:
    print(f"variance: {len(VAR)} variants with repetitions -- "
          "bars are medians, whiskers are min/max")
else:
    print("variance: results/lulesh_variance.csv absent -- single-run values, "
          "no whiskers (rerun once E1 jobs 60796-60800 land)")

# =============== chart 1: all nine variants ===============
fig, ax = plt.subplots(figsize=(COL_W, 3.3))
names   = [v[0] for v in variants][::-1]
times   = [v[2] for v in variants][::-1]
gains   = [v[3] for v in variants][::-1]
colors  = [MODE_COLOR[v[1]] for v in variants][::-1]

bars = ax.barh(names, times, height=0.62, color=colors,
               edgecolor=SURFACE, linewidth=1.0)
for bar, t, g in zip(bars, times, gains):
    lab = f"{t:.3f} s" if g == 0.0 else f"{t:.3f} s ({g*100:+.1f}%)"
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height()/2, lab,
            va="center", ha="left", fontsize=5.5, color=INK)

ax.set_xlabel("elapsed (s) - lower is better")
ax.set_xlim(0, 3.15)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(left=False)
ax.set_title("LULESH halo exchange, 8 ranks / 8x H200 SXM\n"
             "(-s 45, 3145 iterations, default UCX)",
             color=INK, loc="left", pad=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[c]) for c in CAT_ORDER]
ax.legend(handles, [MODE_LABEL[c] for c in CAT_ORDER],
          loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          frameon=False, labelcolor=INK2, handlelength=1.2,
          columnspacing=1.0, labelspacing=0.3)
fig.tight_layout()
# PDF is the paper artifact; the PNG exists only because LULESH/cuda/README.md
# embeds these two inline and GitHub cannot render a PDF. Emitting both from
# the same run keeps the README image from drifting away from the data.
fig.savefig("plots/lulesh_variants_sxm.png", dpi=200)
save(fig, "plots/lulesh_variants_sxm.pdf")

# =============== chart 2: the three send modes, ipc vs mpiwrap ===============
fig, ax = plt.subplots(figsize=(COL_W, 2.7))
modes      = ["A\npack + copy", "C\nremote-pack", "B\ndirect writes"]
IPC_V  = ["ipc", "ipc_rp", "direct"]
WRAP_V = ["mpiwrap", "mpiwrap_rp", None]


def bar_value(v):
    """Median over reps when variance data exists, else the single run."""
    if v in VAR:
        return VAR[v][0]
    return DATA[v]["elapsed"]


def err(v):
    """[[lo],[hi]] whisker offsets from the plotted value, or None."""
    if v not in VAR:
        return None
    med, lo, hi = VAR[v]
    return [[med - lo], [hi - med]]


ipc_times  = [bar_value(v) for v in IPC_V]
wrap_times = [bar_value(v) for v in WRAP_V[:2]]

x = range(3)
w = 0.32
EBAR = dict(ecolor=INK2, capsize=1.8, elinewidth=0.7, capthick=0.7)
b1 = ax.bar([i - w/2 for i in x], ipc_times, width=w, color=BLUE,
            edgecolor=SURFACE, linewidth=1.0, label="hand-written CUDA IPC")
for i, v in enumerate(IPC_V):
    e = err(v)
    if e:
        ax.errorbar(i - w/2, ipc_times[i], yerr=e, fmt="none", **EBAR)
b2 = ax.bar([i + w/2 for i in x[:2]], wrap_times, width=w, color=GREEN,
            edgecolor=SURFACE, linewidth=1.0,
            label="interposed MPI windows")
for i, v in enumerate(WRAP_V[:2]):
    e = err(v)
    if e:
        ax.errorbar(i + w/2, wrap_times[i], yerr=e, fmt="none", **EBAR)

for bar, t in zip(b1, ipc_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.3f}",
            ha="center", fontsize=5.5, color=INK)
for bar, t in zip(b2, wrap_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.3f}",
            ha="center", fontsize=5.5, color=INK)
ax.text(2 + w/2, 0.55, "hand-written\nIPC only", ha="center",
        va="center", fontsize=5.5, color=INK2, style="italic")
ax.set_xlim(-0.55, 2.62)

ax.set_xticks(list(x))
ax.set_xticklabels(modes)
ax.set_ylabel("elapsed (s) - lower is better")
ax.set_ylim(0, 2.05)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(bottom=False)
ax.set_title("Matched CUDA IPC and interposed MPI-window paths\n"
             "(8 ranks / 8x H200 SXM, -s 45, default UCX)",
             color=INK, loc="left", pad=8)
ax.legend(loc="upper right", frameon=False, labelcolor=INK2,
          handlelength=1.2, labelspacing=0.3)
fig.tight_layout()
fig.savefig("plots/lulesh_modes_sxm.png", dpi=200)   # README embed, see above
save(fig, "plots/lulesh_modes_sxm.pdf")
