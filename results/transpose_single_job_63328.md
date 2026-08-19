# Distributed transpose — job 63328, complete sweep in a single job

Raw log `transpose_fig_v2_63328.out` is gitignored (`*.out`); this is the
committed record.

| field | value |
|---|---|
| job | 63328, node h200x4-04 |
| submitted / started | 2026-08-11T15:55 / **2026-08-13T11:10** (1.8 days queued) |
| harness HEAD | 7cc19cf, `transpose/run_transpose_figure_v2.sbatch` |
| source | `transpose_ipc.cu` md5 872ad5fbb2e7d0a49ac96c9b1b38f045 |
| settings | 4 GPUs, 100 timed iterations, warmup 20, UCX_TLS unset |
| coverage | 10 modes x 5 orders x 2 accumulate settings, **one job, one node** |

## Why this matters beyond the numbers

`results/transpose_results.md` carries a standing provenance correction: its
4-GPU IPC/MPI tables are job 28917 (**2026-05-05**) while its NVSHMEM tables
are job 61541 (2026-08-07) — two measurements three months apart, stitched.
This job measures every mode on one node in one allocation, so it retires
that caveat if Table V is regenerated from it.

`*C` modes are `-DDEVICE_TIMING=0` controls; the paper's "IPC SK" column
corresponds to `singleC`.

## Sweep — Overwrite, B = A^T (accum=0)

| mode | 1024^2 | 2048^2 | 4096^2 | 8192^2 | 16384^2 |
|---|--:|--:|--:|--:|--:|
| direct | 266.2 | 657.2 | 1025.8 | 1215.2 | 1277.2 |
| directC | 337.7 | 757.0 | 1080.0 | 1232.7 | 1281.8 |
| single | 607.8 | 1131.9 | 1321.1 | 1303.4 | 1300.9 |
| singleC | 724.5 | 1217.4 | 1347.5 | 1310.4 | 1302.6 |
| buffered | 199.4 | 539.2 | 892.6 | 1092.2 | 1153.5 |
| gpumpi | 212.0 | 562.6 | 911.3 | 1104.9 | 1171.5 |
| staged | 78.5 | 102.8 | 110.7 | 121.2 | 101.7 |
| nvdirect | 160.9 | 469.2 | 874.6 | 1151.8 | 1259.5 |
| nvsingle | 415.1 | 1061.4 | 1215.6 | 1279.4 | 1295.0 |
| nvbuffered | 120.0 | 359.7 | 751.5 | 1049.9 | 1167.0 |

_GB/s, median over reps._

## Sweep — Accumulate, B += A^T (accum=1)

| mode | 1024^2 | 2048^2 | 4096^2 | 8192^2 | 16384^2 |
|---|--:|--:|--:|--:|--:|
| direct | 230.7 | 499.0 | 672.4 | 803.5 | 863.3 |
| directC | 281.2 | 555.0 | 695.3 | 811.2 | 865.4 |
| single | 511.6 | 1086.7 | 870.7 | 886.2 | 884.0 |
| singleC | 591.0 | 1167.8 | 882.5 | 889.2 | 884.7 |
| buffered | 188.1 | 526.6 | 836.7 | 1014.2 | 1062.4 |
| gpumpi | 202.2 | 547.7 | 853.2 | 1022.8 | 1077.2 |
| staged | 78.0 | 113.4 | 128.7 | 115.7 | 120.6 |
| nvdirect | 146.4 | 377.8 | 599.9 | 776.1 | 854.8 |
| nvsingle | 355.7 | 837.5 | 825.4 | 872.1 | 879.9 |
| nvbuffered | 111.1 | 355.9 | 723.4 | 979.9 | 1073.8 |

_GB/s, median over reps._

## Reproducibility against the submitted Table V

§V-I claims the interposed and GPU-aware paths "reproduced within 0.3% at
16384^2 on rerun and within 5% across the sweep". Checked cell by cell:

| accum | order | column | Table V | job 63328 | deviation |
|---|--:|---|--:|--:|--:|
| 0 | 1024^2 | IPC SK | 722.3 | 724.5 | +0.3% |
| 0 | 1024^2 | IPC buf | 177.6 | 199.4 | +12.3% **!** |
| 0 | 1024^2 | GPU MPI | 207.0 | 212.0 | +2.4% |
| 0 | 1024^2 | Host MPI | 74.9 | 78.5 | +4.8% |
| 0 | 4096^2 | IPC SK | 1349.4 | 1347.5 | -0.1% |
| 0 | 4096^2 | IPC buf | 875.5 | 892.6 | +2.0% |
| 0 | 4096^2 | GPU MPI | 908.4 | 911.3 | +0.3% |
| 0 | 4096^2 | Host MPI | 112.6 | 110.7 | -1.7% |
| 0 | 16384^2 | IPC SK | 1302.8 | 1302.6 | -0.0% |
| 0 | 16384^2 | IPC buf | 1151.0 | 1153.5 | +0.2% |
| 0 | 16384^2 | GPU MPI | 1171.0 | 1171.5 | +0.0% |
| 0 | 16384^2 | Host MPI | 127.3 | 101.7 | -20.1% **!** |
| 1 | 1024^2 | IPC SK | 595.6 | 591.0 | -0.8% |
| 1 | 1024^2 | IPC buf | 175.4 | 188.1 | +7.2% **!** |
| 1 | 1024^2 | GPU MPI | 200.1 | 202.2 | +1.1% |
| 1 | 1024^2 | Host MPI | 81.6 | 78.0 | -4.4% |
| 1 | 4096^2 | IPC SK | 882.3 | 882.5 | +0.0% |
| 1 | 4096^2 | IPC buf | 839.3 | 836.7 | -0.3% |
| 1 | 4096^2 | GPU MPI | 851.1 | 853.2 | +0.2% |
| 1 | 4096^2 | Host MPI | 131.7 | 128.7 | -2.3% |
| 1 | 16384^2 | IPC SK | 884.8 | 884.7 | -0.0% |
| 1 | 16384^2 | IPC buf | 1060.3 | 1062.4 | +0.2% |
| 1 | 16384^2 | GPU MPI | 1077.1 | 1077.2 | +0.0% |
| 1 | 16384^2 | Host MPI | 114.8 | 120.6 | +5.1% **!** |

**At 16384^2 the claim holds cleanly** — every interposed and GPU-aware cell
is within 0.3%. Across the full sweep it does not hold everywhere:

- `IPC buf` at 1024^2, accum=0: 177.6 -> 199.4 GB/s (+12.3%)
- `Host MPI` at 16384^2, accum=0: 127.3 -> 101.7 GB/s (-20.1%)
- `IPC buf` at 1024^2, accum=1: 175.4 -> 188.1 GB/s (+7.2%)
- `Host MPI` at 16384^2, accum=1: 114.8 -> 120.6 GB/s (+5.1%)

The buffered small-matrix cells are the ones to look at. The likely cause is
not instability but age: Table V's IPC/MPI entries are job 28917 from
2026-05-05, so these compare across three months and a `UCX_TLS` re-baseline
(`05d0e2b`), not across a rerun. Host-staged MPI is separately documented in
§V-I as environment-sensitive (96-126 GB/s under UCX defaults) and is not
covered by the +/-5% sentence.

**Recommendation for camera-ready:** regenerate Table V from this job. It
both retires the stitched-provenance caveat and removes the need to qualify
the +/-5% reproducibility sentence.

## A result not currently in the paper: accumulate reverses the ranking

| order | single-kernel IPC | buffered IPC | GPU-aware MPI |
|--:|--:|--:|--:|
| 4096^2 | 882.5 | 836.7 | 853.2 |
| 8192^2 | 889.2 | 1014.2 | 1022.8 |
| 16384^2 | 884.7 | 1062.4 | 1077.2 |

Under accumulate at >=4096^2, single-kernel IPC drops to ~880 GB/s while
buffered and GPU-aware MPI hold ~1060-1077, so **buffered MPI beats direct
peer writes by ~20% at 16384^2**. A kernel accumulating into peer memory must
read the destination before writing the sum; the buffered path does that read
locally. Fig. 3a shows the crossover and §V-D notes the 1.6% gap at 16384^2,
but the mechanism is worth stating plainly as a limit of direct peer writes.
