#!/usr/bin/env python3
"""LULESH README charts from the SXM verification run (job 46979).

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
# categorical slots 1-4, fixed order
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# ---- data: h200x8-03 (SXM), -s 45, 3145 iterations to t=0.01 ----
variants = [  # name, mode, elapsed(s), speedup
    ("direct",     "B", 1.28, 2.78),
    ("ipc_rp",     "C", 1.47, 2.42),
    ("mpiwrap_rp", "C", 1.48, 2.41),
    ("ipc",        "A", 1.75, 2.03),
    ("nvshmem",    "A", 1.76, 2.02),
    ("mpiwrap",    "A", 1.76, 2.02),
    ("shmwin",     "-", 2.09, 1.70),
    ("staged",     "-", 3.56, 1.00),
]
MODE_COLOR = {"B": BLUE, "C": GREEN, "A": MAGENTA, "-": YELLOW}
MODE_LABEL = {
    "B": "mode B - direct field writes",
    "C": "mode C - remote-pack",
    "A": "mode A - pack + copy",
    "-": "two-sided / host window",
}

# =============== chart 1: all defensible variants ===============
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
names   = [v[0] for v in variants][::-1]
times   = [v[2] for v in variants][::-1]
speeds  = [v[3] for v in variants][::-1]
colors  = [MODE_COLOR[v[1]] for v in variants][::-1]

bars = ax.barh(names, times, height=0.62, color=colors,
               edgecolor=SURFACE, linewidth=2)
for bar, t, s in zip(bars, times, speeds):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{t:.2f} s" + (f"   ({s:.2f}x)" if s != 1.0 else "   (baseline)"),
            va="center", ha="left", fontsize=9.5, color=INK)

ax.set_xlabel("elapsed (s) - lower is better", fontsize=10)
ax.set_xlim(0, 4.55)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(left=False, labelsize=10.5)
ax.set_title("LULESH halo exchange - full sedov run, 8 ranks / 8x H200 SXM "
             "(-s 45, 3145 iterations)", fontsize=11.5, color=INK,
             loc="left", pad=14)

handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[m]) for m in "BCA-"]
ax.legend(handles, [MODE_LABEL[m] for m in "BCA-"], loc="upper right",
          frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plots/lulesh_variants_sxm.png", bbox_inches="tight")
plt.close(fig)

# =============== chart 2: the three send modes, ipc vs mpiwrap ===============
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=160)
modes      = ["A\npack + copy\n+ unpack", "C\nremote-pack\n+ unpack",
              "B\ndirect field writes\n(no pack, no unpack)"]
ipc_times  = [1.75, 1.47, 1.28]   # ipc, ipc_rp, direct (hand-written IPC family)
wrap_times = [1.76, 1.48, None]   # mpiwrap, mpiwrap_rp, (no counterpart)

x = range(3)
w = 0.32
b1 = ax.bar([i - w/2 for i in x], ipc_times, width=w, color=BLUE,
            edgecolor=SURFACE, linewidth=2, label="hand-written CUDA IPC")
b2 = ax.bar([i + w/2 for i in x[:2]], wrap_times[:2], width=w, color=GREEN,
            edgecolor=SURFACE, linewidth=2,
            label="mpiwrap (MPI windows + LD_PRELOAD interposer)")

for bar, t in zip(b1, ipc_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.045, f"{t:.2f} s",
            ha="center", fontsize=9.5, color=INK)
for bar, t in zip(b2, wrap_times[:2]):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.045, f"{t:.2f} s",
            ha="center", fontsize=9.5, color=INK)
ax.text(2 + w/2, 0.62, "no interposer\ncounterpart\n(yet)", ha="center",
        va="center", fontsize=8.5, color=INK2, style="italic")
ax.set_xlim(-0.55, 2.62)

ax.set_xticks(list(x))
ax.set_xticklabels(modes, fontsize=9.5)
ax.set_ylabel("elapsed (s) - lower is better", fontsize=10)
ax.set_ylim(0, 2.15)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(bottom=False)
ax.set_title("Send modes: the interposer costs nothing at any rung",
             fontsize=11.5, color=INK, loc="left", pad=14)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plots/lulesh_modes_sxm.png", bbox_inches="tight")
plt.close(fig)
print("wrote plots/lulesh_variants_sxm.png and plots/lulesh_modes_sxm.png")
