#!/usr/bin/env python3
"""LULESH README charts. Writes plots/lulesh_variants_sxm.png and
plots/lulesh_modes_sxm.png.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
GRID    = "#e4e3df"
# categorical slots 1-4, fixed order; grey is a neutral for the host-window case
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
GREY = "#8a8a86"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

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

# =============== chart 1: all nine variants ===============
fig, ax = plt.subplots(figsize=(8.6, 5.5), dpi=160)
names   = [v[0] for v in variants][::-1]
times   = [v[2] for v in variants][::-1]
gains   = [v[3] for v in variants][::-1]
colors  = [MODE_COLOR[v[1]] for v in variants][::-1]

bars = ax.barh(names, times, height=0.62, color=colors,
               edgecolor=SURFACE, linewidth=2)
for bar, t, g in zip(bars, times, gains):
    lab = (f"{t:.3f} s   (baseline)" if g == 0.0
           else f"{t:.3f} s   ({g*100:+.1f}% z/s)")
    ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2, lab,
            va="center", ha="left", fontsize=9.5, color=INK)

ax.set_xlabel("elapsed (s) - lower is better", fontsize=10)
ax.set_xlim(0, 2.85)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(left=False, labelsize=10.5)
ax.set_title("LULESH halo exchange - full sedov run, 8 ranks / 8x H200 SXM\n"
             "(-s 45, 3145 iterations, default UCX transport selection)",
             fontsize=11.5, color=INK, loc="left", pad=14)

handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[c]) for c in CAT_ORDER]
ax.legend(handles, [MODE_LABEL[c] for c in CAT_ORDER],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
          frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plots/lulesh_variants_sxm.png", bbox_inches="tight")
plt.close(fig)

# =============== chart 2: the three send modes, ipc vs mpiwrap ===============
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=160)
modes      = ["A\npack + copy\n+ unpack", "C\nremote-pack\n+ unpack",
              "B\ndirect field writes\n(no pack, no unpack)"]
ipc_times  = [DATA["ipc"]["elapsed"], DATA["ipc_rp"]["elapsed"],
              DATA["direct"]["elapsed"]]
wrap_times = [DATA["mpiwrap"]["elapsed"], DATA["mpiwrap_rp"]["elapsed"], None]

x = range(3)
w = 0.32
b1 = ax.bar([i - w/2 for i in x], ipc_times, width=w, color=BLUE,
            edgecolor=SURFACE, linewidth=2, label="hand-written CUDA IPC")
b2 = ax.bar([i + w/2 for i in x[:2]], wrap_times[:2], width=w, color=GREEN,
            edgecolor=SURFACE, linewidth=2,
            label="mpiwrap (MPI windows + LD_PRELOAD interposer)")

for bar, t in zip(b1, ipc_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.035, f"{t:.3f} s",
            ha="center", fontsize=9.5, color=INK)
for bar, t in zip(b2, wrap_times[:2]):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.035, f"{t:.3f} s",
            ha="center", fontsize=9.5, color=INK)
ax.text(2 + w/2, 0.55, "hand-written\nIPC only", ha="center",
        va="center", fontsize=8.5, color=INK2, style="italic")
ax.set_xlim(-0.55, 2.62)

ax.set_xticks(list(x))
ax.set_xticklabels(modes, fontsize=9.5)
ax.set_ylabel("elapsed (s) - lower is better", fontsize=10)
ax.set_ylim(0, 1.95)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(bottom=False)
ax.set_title("Matched CUDA IPC and interposed MPI-window paths\n"
             "(job 60150, 8 ranks / 8x H200 SXM, -s 45, default UCX)",
             fontsize=11.5, color=INK, loc="left", pad=14)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plots/lulesh_modes_sxm.png", bbox_inches="tight")
plt.close(fig)
print("wrote plots/lulesh_variants_sxm.png and plots/lulesh_modes_sxm.png")
