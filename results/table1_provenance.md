# Table I: which source snapshot produced each number

Table I counts **source lines and call sites**, so its provenance is a tree
snapshot, not a job date. Every number in it reproduces at commit `3df6bf2`
(2026-08-05, "Correct shared_query's reported size and disp_unit; route
internals through PMPI"), verified by checking out that commit into a worktree
and re-running the repository's own scripts.

| Table I row | published | reproduced at 3df6bf2 | how |
|---|--:|--:|---|
| cudaIpc* call sites, packed exchange | 3 | **3** | `scripts/check_no_ipc_calls.sh` |
| cudaIpc* call sites, direct field writes | 8 | **8** | same |
| of which retained from the packed path | 3 | **3** | same |
| Collective handle exchanges in app code | 1 | 1 | by inspection |
| Setup + cleanup, handwritten | 62 | **62** | `scripts/count_mechanism_lines.sh` |
| Setup + cleanup, interposed | 43 | **43** | same (47 as committed, −4 instrumentation) |
| Shared steady-state lines | 206 | **206** | same |
| Library lines supporting all cases | 358 | **358** | `wc -l mpi-intercept/mpiwrap_ipc.cc` |

`mpiwrap` and `mpiwrap_rp` report 0 call sites at that snapshot, as published.

## One inconsistency worth fixing in the caption

The caption states "Line counts are effective lines, excluding blank and
comment-only lines". That is true of 62, 43 and 206, which come from the
counting script's awk state machine. **It is not true of 358**, which is a raw
`wc -l` including blank and comment lines. Effective lines for the library at
the same snapshot are **277**, not 358.

So Table I mixes two counting rules. Either restate 358 as 277 to match the
caption, or qualify that row as total file length. The 277 figure strengthens
the paper's point rather than weakening it, since the claim is that the
mechanism is small.

## The same counts today, for the artifact

The library has grown since, entirely from the peer-delivery probe and the
reachability gate it replaced:

| snapshot | date | raw `wc -l` | effective |
|---|---|--:|--:|
| `3df6bf2` (produced Table I) | 2026-08-05 | 358 | 277 |
| `8cd8653` (WinIPC rebrand) | 2026-08-07 | 380 | 282 |
| `95a4607` (reachability gate) | 2026-09-03 | 426 | 303 |
| `63df376` (delivery probe) | 2026-09-10 | 572 | 391 |
| `a502e55` (current main) | 2026-09-22 | 572 | 391 |

The other Table I rows are unchanged on current main: 62 / 43 / 206 and the
same call-site counts all still reproduce, because the probe lives in the
library rather than in the application backends.

## Experiment revisions remain inferred

Separately from Table I, the interposer revision in force for each published
*measurement* is an inference from commit timestamps against job dates, not a
recorded fact. Only job 61541 logs its revision (`8cd8653`); the
`building libmpiwrap.so (<commit>)` line came from `scripts/build_mpiwrap.sh`,
which was added later. Report those as "the revision in force on that date":

| experiment | job | date | inferred revision |
|---|---|---|---|
| Transpose Fig. 3 / Table V (IPC, MPI) | 28917 | 2026-05-05 | `26a68cc` (inferred) |
| Stencil Table VI / Fig. 4 | 59853 | 2026-08-03 | `5fce046` (inferred) |
| LULESH Table V-F | 60150 | 2026-08-04 | `5fce046` (inferred) |
| LULESH Fig. 5 | 60796-60800 | 2026-08-06 | `3df6bf2` (inferred) |
| Transpose Fig. 3c (NVSHMEM) | 61541 | 2026-08-07 | `8cd8653` (**recorded**) |

All predate the reachability gate (2026-09-03) and the delivery probe
(2026-09-10).
