# Transpose sweep, job 90161: what reproduces and what does not

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
