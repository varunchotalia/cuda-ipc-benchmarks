#!/usr/bin/env python3
"""Stencil README/paper chart. Regenerates plots/stencil_mpi_ipc.png.

Data source: job 59853 (h200x8-04, 2026-08-03) -- the consolidated re-baseline.
UCX defaults (UCX_TLS deliberately unset), STENCIL_WARMUP=20, uniform
-O3 -gencode arch=compute_90,code=sm_90, ranks 2/4/8 from one node.
100 timed iterations. Supersedes the Apr-2026 version of this figure, which
predated the re-baseline; see results/stencil_results.txt.

Layout and palette deliberately match the previous version of this figure so
it drops into the same slot -- only the numbers changed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZES = ["1024²", "2048²", "4096²", "8192²", "16384²", "32768²"]
x = np.arange(len(SIZES))

# ---- job 59853, time in ms / throughput in Gcell/s ------------------------
# 4 ranks
IPC4_T  = [3.95, 4.02, 6.07, 12.92, 39.00, 143.04]
MPI4_T  = [7.80, 9.48, 13.20, 23.42, 57.50, 161.44]
IPC4_G  = [26.57, 104.23, 276.36, 519.26, 688.26, 750.68]
MPI4_G  = [13.45, 44.26, 127.12, 286.58, 466.86, 665.12]
# 2 ranks (speedup panel only)
IPC2_T  = [2.26, 3.10, 6.54, 19.66, 71.52, 278.78]
MPI2_T  = [3.61, 5.08, 9.50, 23.72, 77.74, 287.43]

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
PURPLE_D, PURPLE_L = "#7b3294", "#c5b0d5"

sp4 = [(m / i - 1.0) * 100.0 for i, m in zip(IPC4_T, MPI4_T)]
sp2 = [(m / i - 1.0) * 100.0 for i, m in zip(IPC2_T, MPI2_T)]
avg4 = float(np.mean(sp4))

fig, axes = plt.subplots(1, 3, figsize=(17.9, 5.12), dpi=150)
fig.suptitle("CUDA IPC vs MPI Benchmarks — H200 GPUs",
             fontsize=15, fontweight="bold", y=0.99)

# =============== panel 1: throughput ===============
ax = axes[0]
ax.plot(x, IPC4_G, "-o", color=BLUE, linewidth=2, markersize=7, label="IPC")
ax.plot(x, MPI4_G, "--s", color=ORANGE, linewidth=2, markersize=7, label="MPI")
ax.set_title("Stencil Throughput (4 GPUs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Billion cells/sec", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3)

# =============== panel 2: time ===============
ax = axes[1]
w = 0.38
ax.bar(x - w/2, IPC4_T, w, color=BLUE, label="IPC")
ax.bar(x + w/2, MPI4_T, w, color=ORANGE, label="MPI")
ax.set_title("Stencil Time (4 GPUs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Time (ms)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)

# =============== panel 3: speedup ===============
ax = axes[2]
b2 = ax.bar(x - w/2, sp2, w, color=PURPLE_L, label="2 GPUs")
b4 = ax.bar(x + w/2, sp4, w, color=PURPLE_D, label="4 GPUs")
avgline = ax.axhline(avg4, color=PURPLE_L, linestyle="--", linewidth=1.4)
ax.set_title("Stencil: IPC Speedup over MPI", fontsize=13, fontweight="bold")
ax.set_ylabel("IPC Speedup (%)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
# explicit order: avg line, 2 GPUs, 4 GPUs -- as in the previous figure
ax.legend([avgline, b2, b4], [f"4-GPU avg: {avg4:.1f}%", "2 GPUs", "4 GPUs"],
          loc="upper left", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("plots/stencil_mpi_ipc.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote plots/stencil_mpi_ipc.png  (4-GPU avg speedup {avg4:.1f}%)")
