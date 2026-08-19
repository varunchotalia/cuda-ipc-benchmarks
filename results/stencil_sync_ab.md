# Stencil completion handshake: global barrier vs neighbour-only token

Evidence behind the `VALIDATED 2026-08-13, job 63329` note in
`stencil/stencil_ipc.cu`. The raw log `stencil_sync_ab_63329.out` is not
in the repo (`*.out` is gitignored), so this file is the committed record.

| field | value |
|---|---|
| job | 63329, node h200x8-03, 2026-08-13T11:33 -> 11:49 |
| source | `stencil/stencil_ipc.cu` md5 9b8e57f3dcf56ff209bcaefaa3345a00 |
| harness | `stencil/run_stencil_sync_ab.sbatch`, HEAD 7cc19cf |
| settings | STENCIL_WARMUP=20, 100 timed iterations, UCX_TLS unset |
| coverage | np 2/4/8 x 6 grid sizes, barrier x1 rep + neighbour x5 reps |

## 1. Correctness — the reason this job exists

The first token attempt (445c39a) was reverted as racy: at 8 ranks it gave
L2 3.7550293292 at 1024^2 and 5.0563202266 at 4096^2 against the correct
5.1449605829. The second attempt adds double-buffered receive slots
(`slot = iter & 1`). This job is the A/B that tests whether that closed it.

**All 144 RESULT rows report the same L2 norm: `5.1449605829`.**
Distinct L2 values observed across the whole job: 1. Zero mismatches
between the barrier and neighbour arms at any np/size. The race is closed.

## 2. Performance — the token is not a general win

Median ms over reps. `nbr gain` is barrier/neighbour-1: positive means the
neighbour-only handshake is faster.

| np | grid | MPI | IPC barrier | IPC neighbour | IPC/MPI | nbr gain |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 1024^2 | 3.45 | 2.17 | 2.21 | 1.56x | -1.8% |
| 2 | 2048^2 | 4.86 | 2.98 | 2.99 | 1.63x | -0.3% |
| 2 | 4096^2 | 9.71 | 6.22 | 6.21 | 1.56x | +0.2% |
| 2 | 8192^2 | 21.50 | 18.58 | 18.60 | 1.16x | -0.1% |
| 2 | 16384^2 | 72.44 | 67.92 | 67.94 | 1.07x | -0.0% |
| 2 | 32768^2 | 275.22 | 265.76 | 265.88 | 1.04x | -0.0% |
| 4 | 1024^2 | 7.26 | 3.33 | 3.33 | 2.18x | +0.0% |
| 4 | 2048^2 | 9.09 | 4.21 | 4.26 | 2.13x | -1.2% |
| 4 | 4096^2 | 14.24 | 6.12 | 6.18 | 2.30x | -1.0% |
| 4 | 8192^2 | 26.99 | 12.04 | 12.10 | 2.23x | -0.5% |
| 4 | 16384^2 | 64.15 | 36.86 | 36.90 | 1.74x | -0.1% |
| 4 | 32768^2 | 153.64 | 135.28 | 135.36 | 1.14x | -0.1% |
| 8 | 1024^2 | 7.45 | 3.34 | 3.34 | 2.23x | +0.0% |
| 8 | 2048^2 | 9.16 | 4.03 | 3.83 | 2.39x | +5.2% |
| 8 | 4096^2 | 13.12 | 5.02 | 4.80 | 2.73x | +4.6% |
| 8 | 8192^2 | 23.27 | 8.25 | 8.21 | 2.83x | +0.5% |
| 8 | 16384^2 | 49.39 | 20.65 | 20.61 | 2.40x | +0.2% |
| 8 | 32768^2 | 88.18 | 70.03 | 69.97 | 1.26x | +0.1% |

The token wins only at 8 ranks in the mid sizes -- **+5.2% at 2048^2 and
+4.6% at 4096^2** -- where barrier latency is a real fraction of the step.
Everywhere else it is a wash or marginally worse. The earlier "NOT faster"
note in the source was measured at 4 ranks, where it still holds
(16384^2: 36.86 barrier vs 36.90 token); it simply does not generalise.

Two slots suffice: a neighbour cannot reach iteration i+2's write without
first receiving this rank's iteration-i+1 token, which is only sent after
the `cudaDeviceSynchronize` that retires iteration i's unpack.

## 3. IPC vs NVSHMEM at 16384^2

| np | IPC neighbour (ms) | NVSHMEM neighbour (ms) | IPC advantage |
|--:|--:|--:|--:|
| 2 | 67.94 | 68.24 | +0.4% |
| 4 | 36.90 | 37.27 | +1.0% |
| 8 | 20.61 | 20.77 | +0.8% |

## 4. Caveat for the paper

Every stencil number in the submitted paper (job 59853) is a **barrier**
run, and the source default remains `barrier` for that reason. Do not mix
arms when quoting. The neighbour path is validated and available; adopting
it in the paper means re-measuring the sweep, not swapping one column.
