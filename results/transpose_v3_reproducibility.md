# Transpose sweep, job 90161: what reproduces and what does not

> **Status (2026-09-23).** Written against an earlier draft of the paper; its table, figure and section numbers differ from the final version. This job supplies the final paper's Fig. 3 and Table III (`results/transpose_90161.csv`). "Section V-I" below refers to a draft.

Raw log `transpose_fig_v3_90161.out` is gitignored; this is the committed
record. Figure: `plots/transpose_v3_sweep.pdf`.

| field | value |
|---|---|
| job | 90161, h200x4-04, 2026-09-09, 1h48m |
| coverage | 13 orders (1024-65536, incl. non-powers-of-two) x 8 modes x 2 accumulate x 3 reps = 624 runs |
| settings | 4 GPUs, 100 timed iterations after 20 untimed, UCX defaults |
| validation | 624 of 624 report "Solution validates" |
| reference | job 63328 (2026-08-13, same node class, 5 orders) |

An earlier attempt (87249) was OOM-killed at order 65536: the verifier mirrors
each rank's whole B block to the host, 8 GB per rank at that order, against the
44 GB partition default. Fixed with `--mem=256G`.

## 1. Cross-job agreement, all 10 overlapping cells per mode

Both accumulate settings x the 5 orders 63328 also measured. Median deviation.

| mode | vs 63328 | verdict |
|---|--:|---|
| WinIPC direct, single-kernel | +0.6% | reproduces |
| WinIPC direct, per-phase | +2.6% | reproduces |
| WinIPC buffered | +2.6% | reproduces |
| GPU-aware MPI | +3.6% | reproduces |
| Host-staged MPI | -15.6% | environment-sensitive, expected |
| NVSHMEM single-kernel | **-93.2%** | **broken** |
| NVSHMEM direct | **-94.8%** | **broken** |
| NVSHMEM buffered | **-92.3%** | **broken** |

Host-staged MPI is documented in the paper (Section V-I) as environment
sensitive, ranging 96-126 GB/s under UCX defaults, so -15.6% is inside its
known behaviour rather than a regression.

## 2. Within-job stability, 3 reps per cell

| mode | max CoV | max spread |
|---|--:|--:|
| WinIPC buffered | 2.0% | 4.1% |
| WinIPC direct, per-phase | 4.4% | 9.3% |
| GPU-aware MPI | 6.3% | 12.8% |
| Host-staged MPI | 6.3% | 14.5% |
| WinIPC direct, single-kernel | 34.6% | 59.1% |
| NVSHMEM single-kernel | 124.6% | 2076% |
| NVSHMEM direct | 98.0% | 477% |
| NVSHMEM buffered | 130.6% | 3499% |

68 of 208 cells vary by more than 10% across their reps. **64 of those 68 are
`nv*`.** Of the four that are ours, three are mild (staged 14.5% and 10.2%,
gpumpi 12.8%, each a single low first rep). The fourth is the single-kernel
number above: `accum=1 order=3072` gave 1057.1 / 1059.5 / 434.7 GB/s, one cold
third rep against two that agree to 0.2%. The plotted median takes the
converged value.

## 3. The NVSHMEM failure is size-dependent, and unexplained

Not random flakiness and not a slow link:

| orders | nv* runs | above 200 GB/s |
|---|--:|--:|
| >= 24576 | 72 | 60 |
| < 24576 | 162 | 9 |

That is the shape of a large FIXED per-iteration cost that only amortises once
the problem is big enough. Job 63328 measured nvsingle at 415.1 GB/s at order
1024; job 90161 gives roughly 16-24.

What has been ruled out by inspection: the source is byte-identical
(`transpose_nvshmem.cu` md5 aeff8603216ec4bc0b03aaf0cadb8b6a in both jobs), the
build flags are identical, the launch and its `-x` forwarding are identical,
the node class is the same, no NVSHMEM warning appears anywhere in the log, and
every affected run still validates.

One hypothesis was tested and rejected: that reps 2 and 3 were contaminated by
the previous rep's teardown, since several cells show a correct rep 1 followed
by a collapse. Restricting the comparison to rep 1 alone still gives -89% to
-94%, so that is not it.

Job 92224 runs the two size extremes with `NVSHMEM_DEBUG=INFO` to see whether
the small order selects a different transport path.

## 4. Consequence for the figure

`plots/transpose_v3_sweep.pdf` plots the WinIPC and MPI series only. NVSHMEM is
measured in this job but excluded until it is explained; a 45x-wrong curve
beside correct ones is worse than a missing panel. Restore the NVSHMEM panel
once job 92224 or its successor accounts for the loss.

---

## 5. Resolved: NVSHMEM on this cluster is bimodal, not slow

Job 92232 (h200x4-02, clean timed runs, no debug output, 3 reps x 5 orders x 2
modes, accumulate off). Raw log gitignored.

| binary | order | the three reps (GB/s) | best | 63328 | best/ref |
|---|--:|--:|--:|--:|--:|
| nvsingle | 1024 | 0.6 / 15.5 / 44.0 | 44.0 | 415.1 | 11% |
| nvsingle | 2048 | 20.8 / 221.3 / **1085.0** | 1085.0 | 1061.4 | **102%** |
| nvsingle | 4096 | 65.6 / 230.4 / 269.7 | 269.7 | 1215.6 | 22% |
| nvsingle | 16384 | 151.7 / 171.9 / 175.4 | 175.4 | 1295.0 | 14% |
| nvbuffered | 1024 | 0.2 / 11.0 / 18.1 | 18.1 | 120.0 | 15% |
| nvbuffered | 4096 | 3.3 / 34.0 / 326.7 | 326.7 | 751.5 | 43% |
| nvbuffered | 16384 | 50.8 / 96.6 / 99.2 | 99.2 | 1167.0 | 9% |

**30 runs, every one validates, spanning 0.2 to 1085 GB/s — a 5421x range
within one job on one node.** Only 1 of 24 comparable runs lands within 20% of
the 63328 reference. That one run, nvsingle at order 2048 reaching 102% of
reference, is the important one: the hardware and the library CAN deliver the
expected rate. They usually do not.

### What this is not

- **Not node-specific.** Reproduces on h200x4-02 here and h200x4-04 in 90161.
- **Not transport selection.** Job 92230 with `NVSHMEM_DEBUG=INFO` shows
  `P2P list: 0 1 2 3` on every rank and IBRC explicitly skipped ("neither
  nv_peer_mem, or nvidia_peermem detected"). All four GPUs use P2P. Correct.
- **Not our code.** `transpose_nvshmem.cu` is md5-identical to the version that
  measured 415-1295 GB/s in 63328, with identical build flags and launch.
- **Not the reps.** Tested: restricting 90161 to rep 1 alone still gives -89%
  to -94%, and here the good run is rep 3 at one order and rep 1 at another.
- **Not a correctness problem.** Every affected run validates.

### Why 63328 looked fine

**63328 ran the nv\* modes with ONE repetition per cell.** A single sample
cannot show a bimodal distribution. It landed in the fast mode and was recorded
as the measurement. 90161 added repetitions and exposed the spread; 92232
confirms it on another node.

### Consequence for the paper

The intra-node NVSHMEM numbers behind Fig. 3c come from single runs (job 61541
per the provenance note in `transpose_results.md`). If NVSHMEM here is bimodal,
those are single samples of a distribution spanning two orders of magnitude,
not measurements with an error bar. Before the camera-ready either

  - re-measure NVSHMEM with enough repetitions to state a distribution, and
    report the median with its spread rather than a point; or
  - state plainly that the NVSHMEM comparison is a single run and that its
    run-to-run variation on this cluster was not characterised.

The WinIPC and MPI series are unaffected: they reproduce to within a few
percent across both jobs (section 1) and their within-job CoV is under 6.3%
(section 2). The instability is confined to NVSHMEM.

Root cause is still unknown. What is established is that it is not the
transport, not the node, not our source, and not the repetition structure.
