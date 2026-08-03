#!/usr/bin/env python3
"""Stencil figure for the paper. Writes plots/stencil_mpi_ipc.png.

Data source: job 59853 (h200x8-04, 2026-08-03) -- the consolidated re-baseline.
UCX defaults (UCX_TLS deliberately unset), STENCIL_WARMUP=20, uniform
-O3 -gencode arch=compute_90,code=sm_90, 100 timed iterations, rank counts
2/4/8 all from the same node. See results/stencil_results.txt, section
"RE-BASELINE (authoritative)".

Three explicitly-named series per review feedback -- "Wrapper IPC",
"GPU-aware MPI", "Host-staged MPI" -- replacing the earlier two-series
"IPC"/"MPI" version, which omitted the parity result that is now the central
stencil finding. GPU-aware MPI is drawn in the same purple used for that series
in the transpose figure, so the two figures read consistently.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
LBL = {"ipc": "Wrapper IPC", "gpu": "GPU-aware MPI", "stg": "Host-staged MPI"}
RANK_C = {2: "#dcd0ec", 4: "#a97fc9", 8: "#6a3d9a"}

fig, axes = plt.subplots(1, 3, figsize=(17.9, 5.12), dpi=150)
fig.suptitle("Five-point stencil — H200 GPUs, default UCX, 100 timed "
             "iterations after 20 untimed",
             fontsize=14, fontweight="bold", y=0.99)

# =============== panel 1: throughput ===============
ax = axes[0]
ax.plot(x, G4["ipc"], "-o",  color=IPC_C, lw=2, ms=7, label=LBL["ipc"])
ax.plot(x, G4["gpu"], "-.D", color=GPU_C, lw=2, ms=6.5, label=LBL["gpu"])
ax.plot(x, G4["stg"], "--s", color=STG_C, lw=2, ms=7, label=LBL["stg"])
ax.set_title("Stencil Throughput (4 GPUs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Billion cells/sec", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.legend(loc="upper left", fontsize=10.5)
ax.grid(True, alpha=0.3)

# =============== panel 2: time ===============
ax = axes[1]
w = 0.27
ax.bar(x - w, T[4]["ipc"], w, color=IPC_C, label=LBL["ipc"])
ax.bar(x,     T[4]["gpu"], w, color=GPU_C, label=LBL["gpu"])
ax.bar(x + w, T[4]["stg"], w, color=STG_C, label=LBL["stg"])
ax.set_title("Stencil Time (4 GPUs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Time (ms)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.legend(loc="upper left", fontsize=10.5)
ax.grid(True, axis="y", alpha=0.3)

# =============== panel 3: parity with GPU-aware MPI ===============
# How much slower GPU-aware MPI is than wrapper IPC. Log y because the values
# span 0.1%-75%: small grids are latency-bound and noisy, large grids converge
# to parity. All values are plotted -- nothing is clipped.
ax = axes[2]
for i, np_ in enumerate((2, 4, 8)):
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    ax.bar(x + (i - 1) * w, d, w, color=RANK_C[np_], label=f"{np_} GPUs")
ax.axhline(1.0, color="#8a8a8a", ls=":", lw=1.2)
ax.text(-0.48, 1.06, "1%", fontsize=8.5, color="#6a6a6a", ha="left")
ax.set_yscale("log")
ax.set_title("GPU-aware MPI overhead vs Wrapper IPC",
             fontsize=13, fontweight="bold")
ax.set_ylabel("% slower than Wrapper IPC (log scale)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.legend(loc="upper right", fontsize=10.5)
ax.grid(True, axis="y", alpha=0.3, which="both")
for i, np_ in enumerate((2, 4, 8)):     # label the two largest grids
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    for k in (4, 5):
        ax.text(k + (i - 1) * w, d[k] * 1.12, f"{d[k]:.1f}",
                ha="center", fontsize=7.5, color="#3a3a3a")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("plots/stencil_mpi_ipc.png", bbox_inches="tight")
plt.close(fig)

for np_ in (2, 4, 8):
    d = [(g / p - 1.0) * 100.0 for p, g in zip(T[np_]["ipc"], T[np_]["gpu"])]
    print(f"np={np_} GPU-aware vs wrapper IPC: " +
          ", ".join(f"{s}={v:+.1f}%" for s, v in zip(SIZES, d)))
print("wrote plots/stencil_mpi_ipc.png")
