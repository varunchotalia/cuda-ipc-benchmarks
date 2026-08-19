# LULESH halo-exchange lifecycle — job 61540 (second variance sample)

Raw log `lulesh_lifecycle_61540.out` is gitignored (`*.out`); this is the
committed record.

| field | value |
|---|---|
| job | 61540, node h200x8-03 |
| submitted / started | 2026-08-06T23:06 / **2026-08-13T09:46** (6.5 days queued) |
| harness HEAD | 7cc19cf |
| problem | `-s 45`, 3145 iterations, 8 ranks on 8 H200 SXM GPUs |
| coverage | 9 variants x 5 reps at np=8, plus a np=1 control |
| env | UCX_TLS unset, UCX defaults |

## Why this file is separate from `results/lulesh_variance.csv`

`lulesh_variance.csv` holds jobs 60796-60800 and is what
`plots/make_lulesh_plots.py` medians to draw Fig. 5. That file has a `jobid`
column and would accept these rows, **but the plot script medians over the
whole file**, so appending would silently move the published bars. This job
is therefore recorded separately as an independent second sample. Merging is
a deliberate decision to make after review, not a side effect.

## 1. Reproducibility against the committed numbers

Median FOM in zones/s over 5 reps. `CoV` is within this job; `drift` is
against `lulesh_results.csv` (job 60150, 2026-08-04, single run).

| variant | median FOM | CoV | vs lulesh_results.csv | vs lulesh_variance.csv |
|---|--:|--:|--:|--:|
| direct | 1,801,684 | 0.08% | -1.75% | -1.65% |
| ipc_rp | 1,677,896 | 0.11% | -0.41% | -0.61% |
| mpiwrap_rp | 1,674,087 | 0.05% | -0.69% | — |
| ipc | 1,424,222 | 0.06% | -1.09% | -1.06% |
| mpiwrap | 1,418,812 | 0.14% | -1.16% | — |
| nvshmem | 1,301,089 | 0.09% | -1.06% | — |
| gpumpi | 1,248,485 | 0.21% | -0.15% | -0.38% |
| staged | 1,178,744 | 0.26% | +0.75% | — |
| shmwin | 1,097,739 | 0.10% | -0.88% | — |

Variant ordering is identical to both committed sources. Within-session CoV
is <=0.26%, but **cross-session drift reaches 1.6%** (`direct`: 1,801,684 here
vs 1,831,897 in the variance file). Fig. 5's min/max whiskers are
within-session and therefore understate true run-to-run spread.

## 2. The paper's core claim, re-tested

§V-F states the interposed-vs-handwritten paired difference "changes sign and
never exceeds 0.46%". Independently reproduced here:

| mode | handwritten | interposed | paired difference |
|---|--:|--:|--:|
| A (pack+copy) | 1,424,222 | 1,418,812 | -0.38% |
| C (remote-pack) | 1,677,896 | 1,674,087 | -0.23% |

Both inside +/-0.46%. Pointer acquisition through the window costs nothing in
steady state.

## 3. Where the interposer does cost something — setup

Not in `lulesh_variance.csv`: this job carries the lifecycle phase counters.
Median ms at np=8.

| variant | setup | of which `MPI_Win_allocate` | peer query | free |
|---|--:|--:|--:|--:|
| ipc | 126.6 | 0.0 | 0.000 | 121.1 |
| mpiwrap | 223.3 | 221.8 | 0.017 | 57.6 |
| ipc_rp | 128.0 | 0.0 | 0.000 | 131.2 |
| mpiwrap_rp | 227.1 | 226.0 | 0.017 | 57.2 |
| nvshmem | 2.3 | 0.0 | 0.000 | 55.7 |
| gpumpi | 2.3 | 0.0 | 0.000 | 0.7 |
| staged | 2.3 | 0.0 | 0.000 | 8.2 |
| shmwin | 8.7 | 0.0 | 0.000 | 56.6 |
| direct | 167.3 | 0.0 | 0.000 | 283.1 |

The interposer's whole cost is a one-time **+97 ms** in
`MPI_Win_allocate` (handle export + collective allgather), amortised over a
3145-iteration solve. Peer query is ~0.02 ms because mappings are cached.

## 4. Correctness

- np=8: all 45 runs report `Final Origin Energy = 1.482403e+06`.
- np=1: all runs report `4.234875e+05`.
- `fallback_peers` is 0 on every run — no peer ever fell back to MPI.

## 5. Known log noise

The log is 848 KB, of which 5,156 lines are a single repeated UCX message,
`cuda_iface.h: UCX ERROR cuCtxGetApiVersion(ctx, &version) failed: invalid
device context`, confined entirely to the five `np=8 variant=gpumpi` reps.
The prior run had 5,160 of them; results validate regardless. Cosmetic, but
worth a `UCX_LOG_LEVEL` bump for that variant so real errors are not buried.

Separately, every `not IPC-reachable` line in the log reads `0 of 0 peers` —
the message fires unconditionally and is a positive confirmation, not a
warning. The job footer tells the reader to grep for that string, which is
misleading on its own.
