#!/usr/bin/env python3
"""LULESH charts for the paper. Writes two single-column PDFs:

    plots/lulesh_variants_sxm.pdf  all nine variants, ranked by FOM (Gzone/s)
    plots/lulesh_modes_sxm.pdf     the three send modes, handwritten vs interposed

THE TWO CHARTS PLOT OPPOSITE POLARITIES, DELIBERATELY. Chart 1 is the figure
of merit -- the established LULESH metric, and time-neutral -- so higher is
better. Chart 2 stays on elapsed seconds because the matched-pair residual
between ipc and winipc reads better in time than as a ratio of rates, so lower
is better. Both axis labels state their direction explicitly; do not drop that
text, it is the only thing stopping two adjacent figures from being read the
same way round.

VARIANCE (E1). If results/lulesh_variance.csv exists it is used to draw
min/max whiskers on BOTH charts, and each bar becomes the median of the
repetitions rather than a single run -- on the FOM axis for chart 1 and the
elapsed axis for chart 2. Build it with scripts/build_lulesh_variance_csv.py,
which emits one row per (variant, job) with no pre-aggregation; `variant`,
`elapsed_s_hi` and `fom_z_per_s` are read here, the rest are carried for
provenance and for the paper's paired analysis.

CAUTION: the bars are then medians over jobs 60796-60800, while the CSV's
`gain_pct` column is job 60150's single run. Chart 1 therefore recomputes its
percentages from the plotted values instead of reading gain_pct, so the labels
always agree with the bars. The two differ by 0.3-1.0 percentage points.

`rep` indexes the JOB SUBMISSION, not an in-process loop: run_lulesh_verify
runs each variant once per job, so E1's five submissions give five independent
runs per variant. `jobid` is carried alongside it so the paper's matched-pair
residual (WinIPC - ipc within a job) can be formed explicitly rather than by
trusting that rep maps 1:1 onto a job -- the builder's --check-pairing asserts
that it does.

Absent the file the charts fall back to the single-run values in
lulesh_results.csv and print a notice, so a missing variance file is visible
rather than silent.

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

# variant key -> label drawn on the figure. The interposed-window backend was
# renamed mpiwrap -> WinIPC, but the CSV keys, the make targets and the binary
# names are still `mpiwrap*`: renaming those would break the queued Slurm jobs,
# whose script text Slurm froze at submit time. So the rename lives here, at the
# display edge, and the recorded data keeps its original keys. Drop this map
# once the build-side identifiers are renamed too.
DISPLAY = {"mpiwrap": "winipc", "mpiwrap_rp": "winipc_rp"}


def label(v):
    return DISPLAY.get(v, v)


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
            "fom": float(r["fom_z_per_s"]) / FOM_PER_GZ,
            "gain": float(r["gain_pct"]) / 100.0,
            "mode": r["mode"],
        }
    return out


# The CSV column is NAMED fom_z_per_s but is not in zones/s -- it is off by a
# factor of 1000. Cross-checked against the header's own zone_cycles:
#   zone_cycles 2292705000 / elapsed 1.25024 s = 1.8338e9 zone-cycles/s,
#   while the column reads 1833813.8. Ratio exactly 1000.
# So Gzone/s = column / 1e6, NOT column / 1e9 -- dividing by 1e9 would print
# 0.0018 Gzone/s and look like a unit error rather than a value. Do not
# "simplify" this constant without re-deriving it from zone_cycles.
FOM_PER_GZ = 1.0e6

DATA = load()
missing = set(CATEGORY) - set(DATA)
if missing:
    raise SystemExit(f"CSV is missing variants: {sorted(missing)}")

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
    """variant -> {"elapsed": (med,min,max), "fom": (med,min,max)}, or {}.

    Both metrics are carried because the two charts now plot different axes:
    chart 1 is FOM (higher is better), chart 2 stays on elapsed seconds (lower
    is better). Whiskers must be computed on whatever quantity is drawn --
    min/max of elapsed are NOT the endpoints of min/max of FOM once rounded,
    and reusing one for the other would draw whiskers that do not bracket
    their own bar.
    """
    if not os.path.exists(path):
        return {}
    reps = {}
    with open(path) as f:
        rows = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    for r in csv.DictReader(rows):
        d = reps.setdefault(r["variant"], {"elapsed": [], "fom": []})
        d["elapsed"].append(float(r["elapsed_s_hi"]))
        d["fom"].append(float(r["fom_z_per_s"]) / FOM_PER_GZ)

    def summarise(xs):
        xs = sorted(xs)
        n = len(xs)
        med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        return (med, xs[0], xs[-1])

    return {v: {k: summarise(xs) for k, xs in d.items()}
            for v, d in reps.items()}


VAR = load_variance()
if VAR:
    print(f"variance: {len(VAR)} variants with repetitions -- "
          "bars are medians, whiskers are min/max")
else:
    print("variance: results/lulesh_variance.csv absent -- single-run values, "
          "no whiskers (rerun once E1 jobs 60796-60800 land)")

# =============== chart 1: all nine variants, figure of merit ===============
# FOM rather than elapsed seconds: it is the established LULESH metric and it is
# time-neutral. Same runs, same data -- only the quantity plotted changes.
# Ordering is descending FOM so "best" stays at the top as it was under
# ascending elapsed; FOM inverts the sense, so the sort key has to invert too.


def fom_value(v):
    """Median FOM over reps when variance data exists, else the single run."""
    if v in VAR:
        return VAR[v]["fom"][0]
    return DATA[v]["fom"]


def fom_err(v):
    """[[lo],[hi]] whisker offsets from the plotted FOM, or None."""
    if v not in VAR:
        return None
    med, lo, hi = VAR[v]["fom"]
    return [[med - lo], [hi - med]]


# Gains are recomputed from the PLOTTED values against the plotted staged bar,
# not taken from the CSV's gain_pct. gain_pct is job 60150's single run, but the
# bars are medians over jobs 60796-60800 whenever the variance CSV is present.
# Reading one off the other would print a percentage the bars do not support.
fom_order = sorted(DATA, key=lambda k: -fom_value(k))
staged_fom = fom_value("staged")

names  = [label(v) for v in fom_order][::-1]
foms   = [fom_value(v) for v in fom_order][::-1]
gains  = [fom_value(v) / staged_fom - 1.0 for v in fom_order][::-1]
colors = [MODE_COLOR[CATEGORY[v]] for v in fom_order][::-1]
errs   = [fom_err(v) for v in fom_order][::-1]

fig, ax = plt.subplots(figsize=(COL_W, 3.0))
bars = ax.barh(names, foms, height=0.62, color=colors,
               edgecolor=SURFACE, linewidth=1.0)
EBAR1 = dict(ecolor=INK2, capsize=1.8, elinewidth=0.7, capthick=0.7)
for i, e in enumerate(errs):
    if e:
        ax.errorbar(foms[i], i, xerr=e, fmt="none", **EBAR1)

# Value label on every bar. Table VII is dropping its numeric columns, so this
# figure becomes the only place the nine-variant numbers appear -- a reader
# checking the percentages quoted in the text has nowhere else to look.
for i, (bar, f, g) in enumerate(zip(bars, foms, gains)):
    lab = f"{f:.3f} Gz/s" if abs(g) < 1e-12 else f"{f:.3f} Gz/s ({g*100:+.1f}%)"
    hi = errs[i][1][0] if errs[i] else 0.0
    ax.text(bar.get_width() + hi + 0.035, bar.get_y() + bar.get_height()/2,
            lab, va="center", ha="left", fontsize=5.5, color=INK)

ax.set_xlabel("FOM (Gzone/s) - higher is better")
ax.set_xlim(0, 2.80)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(left=False)

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

# =============== chart 2: the three send modes, ipc vs WinIPC ===============
fig, ax = plt.subplots(figsize=(COL_W, 2.45))
modes      = ["A\npack + copy", "C\nremote-pack", "B\ndirect writes"]
IPC_V  = ["ipc", "ipc_rp", "direct"]
WIN_V = ["mpiwrap", "mpiwrap_rp", None]


def bar_value(v):
    """Median elapsed over reps when variance data exists, else the single run.

    Chart 2 deliberately stays on elapsed seconds: the matched-pair residual
    between ipc/winipc reads better in time than as a ratio of rates. That
    leaves the two adjacent LULESH figures with OPPOSITE polarity, which is why
    both axis labels carry an explicit "lower is better" / "higher is better".
    """
    if v in VAR:
        return VAR[v]["elapsed"][0]
    return DATA[v]["elapsed"]


def err(v):
    """[[lo],[hi]] whisker offsets from the plotted elapsed value, or None."""
    if v not in VAR:
        return None
    med, lo, hi = VAR[v]["elapsed"]
    return [[med - lo], [hi - med]]


ipc_times  = [bar_value(v) for v in IPC_V]
wrap_times = [bar_value(v) for v in WIN_V[:2]]

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
            label="WinIPC (interposed MPI windows)")
for i, v in enumerate(WIN_V[:2]):
    e = err(v)
    if e:
        ax.errorbar(i + w/2, wrap_times[i], yerr=e, fmt="none", **EBAR)

for bar, t in zip(b1, ipc_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.3f}",
            ha="center", fontsize=5.5, color=INK)
for bar, t in zip(b2, wrap_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.3f}",
            ha="center", fontsize=5.5, color=INK)
# Mode B has no interposed counterpart (Section IV-F: Thrust owns the field
# allocations, and a window must own its storage to be exportable). Draw the
# absent bar as a dashed outline and label it, rather than leaving a floating
# caption in white space -- otherwise the single bar reads as lost data.
ax.bar(2 + w/2, DATA["direct"]["elapsed"], width=w, facecolor="none",
       edgecolor=GREY, linewidth=0.7, linestyle=(0, (2, 1.6)), zorder=1)
ax.text(2 + w/2, DATA["direct"]["elapsed"] / 2, "no interposed counterpart",
        ha="center", va="center", rotation=90, fontsize=5, color=INK2)
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
ax.legend(loc="upper right", frameon=False, labelcolor=INK2,
          handlelength=1.2, labelspacing=0.3)
fig.tight_layout()
fig.savefig("plots/lulesh_modes_sxm.png", dpi=200)   # README embed, see above
save(fig, "plots/lulesh_modes_sxm.pdf")
