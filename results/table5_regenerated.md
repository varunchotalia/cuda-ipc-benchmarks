# Table V regenerated from a single job

> **Status (2026-09-23).** Written against an earlier draft of the paper; its table, figure and section numbers differ from the final version. Applied. The final paper's Table III and Fig. 3 come from job 90161 and state 2.44×. "Table V" below is that table's draft number.

Proposal, not applied. Table V currently stitches job 28917 (2026-05-05) for
its IPC/MPI columns to job 61541 (2026-08-07) for NVSHMEM -- three months
apart, across the `UCX_TLS` re-baseline in `05d0e2b`. Job 90161 measured
every mode in one allocation on one node, medians of 3 reps.

| | IPC SK | IPC buf. | GPU MPI | Host MPI |
|---|--:|--:|--:|--:|
| **B += A^T** | | | | |
| 1024² | 563.8 | 215.1 | 231.1 | 66.1 |
| 4096² | 878.1 | 857.1 | 883.1 | 101.8 |
| 16384² | 884.8 | 1064.1 | 1080.3 | 85.0 |
| **B = A^T** | | | | |
| 1024² | 710.7 | 225.4 | 242.7 | 70.4 |
| 4096² | 1322.5 | 916.7 | 945.4 | 102.5 |
| 16384² | 1301.1 | 1155.8 | 1174.6 | 85.4 |

## What changes against the published table

| | IPC SK | IPC buf. | GPU MPI | Host MPI |
|---|--:|--:|--:|--:|
| **B += A^T** | | | | |
| 1024² | -5.3% | +22.6% | +15.5% | -18.9% |
| 4096² | -0.5% | +2.1% | +3.8% | -22.7% |
| 16384² | -0.0% | +0.4% | +0.3% | -26.0% |
| **B = A^T** | | | | |
| 1024² | -1.6% | +26.9% | +17.3% | -6.1% |
| 4096² | -2.0% | +4.7% | +4.1% | -8.9% |
| 16384² | -0.1% | +0.4% | +0.3% | -32.9% |

IPC single-kernel and the two large-order columns move by under 5%. The
small-order buffered and GPU-aware cells move by 15-27%, and host-staged by
-19% to -33%. Host-staged is already documented in Section V-I as
environment sensitive. The others are the three-month gap: the May job
measured a slower GPU-aware MPI than this cluster now delivers.

## The consequence that needs a decision

The abstract's headline is single-kernel IPC against GPU-aware MPI at the
small orders. Regenerating the table moves it:

| order | published | regenerated |
|--:|--:|--:|
| 1024² | 2.98× | **2.44×** |
| 2048² | 2.14× | **1.90×** |
| 4096² | 1.04× | **0.99×** |

**"up to 3.0× higher throughput than GPU-aware MPI" becomes 2.4×.** Both
terms move against us: single-kernel is 5% slower than the May figure and
GPU-aware MPI is 15% faster.

Two options, and they are not equally defensible:

1. **Regenerate and restate as 2.4×.** One job, one node, one allocation,
   3 reps per point. Retires the stitched provenance and the Section V-I
   reproducibility caveat at the same time. A reviewer who reruns gets ~2.4×.
2. **Keep the published table.** Keeps 3.0×, but the provenance stays
   stitched, and Section V-I's "within 5% across the sweep" stays false for
   the buffered cells at 1024², which are +12.3% and +7.2%.

Recommendation: option 1. The headline is weaker but the number survives
someone checking it, and 2.4× at 1024² with the crossover shown honestly is
a stronger claim than 3.0× that does not reproduce.

Source: job 90161, h200x4-04, 2026-09-09, 13 orders x 8 modes x 2 accumulate
x 3 reps, all 624 runs validating. Reproducibility against 63328 is in
`results/transpose_v3_reproducibility.md`.
