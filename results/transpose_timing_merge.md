# Transpose — what the timed-region merge (PR #1) and the NVSHMEM race fix changed

**Status: authoritative.** This file is the committed evidence for three claims
that are otherwise only visible in gitignored `*.out` logs. Anything writing
about the transpose benchmark should cite this file, not the logs.

| | |
|---|---|
| Node | `h200x4-01` (4x H200, one node) |
| Config | UCX defaults (`UCX_TLS` **unset**), `TRANSPOSE_WARMUP=20`, 100 timed iterations, `-O3 -gencode arch=compute_90,code=sm_90`, 4 ranks |
| Jobs | 60636 (2026-08-05), 61344 (2026-08-07), 61541 (2026-08-07) |
| Commits | PRE = `6f674cb`, POST = `7c77c37` (merge of PR #1), NVSHMEM barrier = `7ec29b5` |

`UCX_TLS` is deliberately unset: exporting it fakes a CUDA-aware-MPI setup cost
and cuts staged MPI ~2.6x. All three jobs share this config, so they are
directly comparable.

---

## Claim 1 — the "~40x inflated single-kernel rate" claim is refuted

PR #1's code comment asserts that the missing stop-side barrier "inflates the
reported single-kernel rate ~40x". Its PR description asserts the opposite —
that the CUDA events *confirm* the rate is real. Both cannot hold.

**Job 61344** settles it by building identical binaries from `6f674cb` (PRE, no
stop sync) and `7c77c37` (POST, with it) and running them in one job, 3 repeats
per cell. Mean of 3 reps; `PRE_spread` is (max-min)/mean across those reps.

| order | mode | PRE MB/s | POST MB/s | POST/PRE | PRE spread |
|------:|:-----|---------:|----------:|---------:|-----------:|
| 1024 | direct | 276,939 | 215,429 | 0.778 | 4.65% |
| 1024 | gpumpi | 205,148 | 204,891 | **0.999** | 0.56% |
| 1024 | single | 590,158 | 484,630 | 0.821 | 3.52% |
| 1024 | staged | 77,490 | 78,639 | 1.015 | 1.21% |
| 4096 | direct | 695,762 | 664,192 | 0.955 | 0.14% |
| 4096 | gpumpi | 856,258 | 854,874 | **0.998** | 0.06% |
| 4096 | single | 882,458 | 864,860 | 0.980 | 0.11% |
| 4096 | staged | 113,252 | 119,348 | 1.054 | 10.37% |
| 16384 | direct | 865,538 | 862,399 | 0.996 | 0.04% |
| 16384 | gpumpi | 1,077,670 | 1,077,323 | **1.000** | 0.02% |
| 16384 | single | 884,814 | 883,558 | **0.999** | 0.02% |
| 16384 | staged | 110,864 | 111,796 | 1.008 | 3.70% |

**`gpumpi` is the decisive row.** Reading the code, `COMM_MODE=2` and `3` are the
only modes that gain a *new* stop-side global barrier from the merge — the
`SINGLE_KERNEL` block has ended every iteration with `cudaStreamSynchronize` +
`MPI_Barrier` since `6ed0bb8` (2026-03-29), months before any published number.
`gpumpi` is flat to three digits at every size (0.998-1.000). `staged` scatters
+-6% either way, which is its own run-to-run noise (10.4% spread within the PRE
arm alone at 4096), not a barrier effect.

**No single-kernel number changes.** The `single` row is 0.999 at 16384. The
`~40x` figure in the code comment is not supported at any size; the PR
description is correct and the comment is wrong.

### Why `direct` and `single` moved anyway — instrumentation, not bandwidth

`direct` and `single` are exactly and only the two `COMM_MODE=0` binaries that
the merge also added CUDA-event timing to. Converting the ratios to absolute
per-iteration time using the logged `wall/iter`:

| order | mode | POST wall/iter | PRE wall/iter | delta/iter | delta per event call |
|------:|:-----|---------------:|--------------:|-----------:|---------------------:|
| 1024 | direct | 77.88 us | 60.58 us | 17.30 us | 5.77 us |
| 4096 | direct | 404.15 us | 385.81 us | 18.34 us | 6.11 us |
| 16384 | direct | 4980.25 us | 4962.19 us | 18.06 us | 6.02 us |
| 1024 | single | 34.62 us | 28.43 us | 6.19 us | 6.19 us |
| 4096 | single | 310.38 us | 304.19 us | 6.19 us | 6.19 us |
| 16384 | single | 4860.99 us | 4854.09 us | 6.90 us | 6.90 us |

The cost is **constant in absolute time and independent of matrix size**, which
is why it reads as -22% at 1024^2 and -0.4% at 16384^2.

Two independent checks identify it as `cudaEventElapsedTime` — a blocking driver
call the merge placed *inside* the timed loop, once per timed iteration per comm
kernel:

1. **Ratio.** `direct`'s overhead is 2.9x `single`'s. `direct` records one event
   pair per remote phase (P-1 = 3 at np=4); `single` records one per iteration.
   2.9 ~ 3. A once-per-region stop barrier would cost both modes the *same*.
2. **Per-call cost.** Dividing by the call count per iteration gives 5.77-6.90 us
   across all six cells and both modes — one consistent constant.

**Consequence for the paper.** The wall-clock rates from device-timed
`COMM_MODE=0` binaries are not comparable to un-instrumented ones at small
matrices, and the reported `barrier+host overhead` column is inflated by the
measurement itself. `dev_us` (device-side) is unaffected. Fixed in
`transpose_ipc.cu` by recording into a per-iteration event pool and moving every
`cudaEventElapsedTime` read past the stop barrier; `-DDEVICE_TIMING=0` builds a
provably uninstrumented binary. Job `transpose_fig_v2` re-runs the figure with
both builds as an in-job control.

---

## Claim 2 — the NVSHMEM correctness cost applies to the buffered series ONLY

`7ec29b5` adds a post-unpack `nvshmem_barrier_all()` to `transpose_nvshmem.cu`.
It was expected to move all three NVSHMEM series down. It does not.

60636 (before) vs 61541 (after), same node and config:

| order | acc | nvdirect | nvsingle | nvbuffered |
|------:|:---:|---------:|---------:|-----------:|
| 1024 | 1 | 1.001 | 1.000 | **0.806** |
| 2048 | 1 | 1.002 | 1.000 | **0.842** |
| 4096 | 1 | 1.004 | 1.000 | *(60636 FAILED — see Claim 3)* |
| 8192 | 1 | 0.999 | 0.999 | **0.968** |
| 16384 | 1 | 1.000 | 1.000 | **0.991** |
| 1024 | 0 | 0.994 | 0.996 | **0.799** |
| 2048 | 0 | 1.003 | 0.948 | **0.838** |
| 4096 | 0 | 1.006 | 0.999 | **0.915** |
| 8192 | 0 | 0.999 | 1.000 | **0.965** |
| 16384 | 0 | 1.000 | 1.000 | **0.990** |

`nvdirect` spans 0.994-1.006 and `nvsingle` 0.948-1.000 — unchanged. Only
`nvbuffered` moves, from 0.80 at 1024^2 converging to 0.99 at 16384^2.

**Why:** the barrier sits inside the `recv_buf` / `unpack_kernel` path, which
only the buffered variant executes. The direct and single-kernel NVSHMEM paths
never reach it.

The single `nvsingle` outlier (0.948 at 2048^2, accum=0) is a one-rep cell in
both jobs with no repeats, and is not reproduced at any neighbouring size.
Do not read it as an effect.

**Write it as "the buffered NVSHMEM series", not "the NVSHMEM series".**

### Action taken (2026-08-08) — the figure was wrong, and is now fixed

This claim required a figure regeneration, not just careful wording.
`plots/make_transpose_plots.py` reads `results/transpose_results.md`, and that
file's NVSHMEM tables were **job 21711 (2026-04-04)** — not 60636 as first
assumed, and four months stale. They predated the `UCX_TLS` re-baseline, the
`TRANSPOSE_WARMUP=20` convention *and* the race fix, and carried a literal
`(rerun pending)` placeholder at 2048² accum that the plot script rendered as
a gap.

`transpose_ipc_vs_nvshmem.pdf` plots the **no-accumulation** bucket, so that is
where the error lived. NVSHMEM-buffered, old vs job 61541:

| order | was (21711) | now (61541) | overstatement |
|------:|------------:|------------:|--------------:|
| 1024 | 152.3 | 119.3 | +27.7% |
| 2048 | 428.6 | 359.0 | +19.4% |
| 4096 | 808.5 | 752.7 | +7.4% |
| 8192 | 1085.1 | 1052.2 | +3.1% |
| 16384 | 1178.3 | 1167.2 | +1.0% |

The published curve was up to **28% too fast at the small grids, in the
paper's favour**, sourced from a configuration that failed validation outright
at 4096² accum in job 60636. Both NVSHMEM tables have been replaced with 61541
and the figure regenerated.

The IPC/MPI half of that file is still 2026-05-05 data and is *not* fixed by
this — see the provenance block at the top of `results/transpose_results.md`.

---

## Claim 3 — the race fix has stronger evidence than its own commit message

`7ec29b5`'s comment justifies the barrier by a validation failure "at 32 PEs on
GB200 NVL72", passing at 16. That understates it. In **job 60636 on h200x4 at
4 PEs**:

```
RESULT transpose np=4 order=4096 accum=1 mode=nvbuffered mbps=FAIL validates=0
```

That is the only validation failure in the job — 79 of 80 cells validate. In
**job 61541**, with the barrier in place, the same cell validates and reports
725,967 MB/s, and the job is **80/80 with zero failures**.

So the race is reproducible locally at 4 PEs on a single H200 node, not only at
32 PEs on multi-node NVLink. The barrier is required for correctness on the same
hardware the transpose figure is taken on.

*Caveat for the writer:* this is a single observed failure, and the race is
timing-dependent, so it is not a deterministic reproducer. State it as "a
validation failure was observed at 4 PEs (job 60636)" rather than "fails at
4 PEs".

---

## Provenance

Raw logs `transpose_figure_60636.out`, `stopsync_ab_61344.out` and
`transpose_figure_61541.out` are held locally per the `results/raw/README.md`
convention (`.gitignore` excludes `*.out` repo-wide). Every number above was
extracted from `RESULT` / `DEVLINE` lines in those logs; the `PRE wall/iter`
column is derived as `POST wall/iter x (POST/PRE rate ratio)`, not measured
directly, because the PRE binaries do not emit a wall/iter line.
