# Transpose Benchmark — H200 GPUs, 100 iterations

> **READ FIRST (added 2026-08-01, jobs 59067 + 59070).** The IPC/MPI tables
> below (measured 2026-05-05 under UCX *default* transport selection) appear to
> be CORRECT. Do not "fix" them. Two later-discovered effects can make them look
> wrong if you re-measure with `UCX_TLS` pinned — see
> "Appendix: UCX_TLS pinning" at the end of this file. Short version:
> **do not export `UCX_TLS`; use UCX defaults.**

> **PROVENANCE — THIS FILE IS NOT ONE MEASUREMENT (corrected 2026-08-08).**
> The note above said "the tables below (measured 2026-05-05)". That was only
> ever true of the IPC/MPI tables. Traced cell by cell against the job logs:
>
> | section | job | date |
> |---|---|---|
> | 4-GPU IPC / MPI (both accumulate settings) | 28917 | 2026-05-05 |
> | 4-GPU NVSHMEM (both accumulate settings) | **61541** | **2026-08-07** |
>
> The NVSHMEM tables previously held job **21711 (2026-04-04)** data — four
> months old, predating the `UCX_TLS` re-baseline (`05d0e2b`), the
> `TRANSPOSE_WARMUP=20` convention, *and* the NVSHMEM post-unpack race fix
> (`7ec29b5`). They also carried a literal `(rerun pending)` placeholder at
> 2048² accum NVSHMEM-buffered, which the plot script renders as a gap.
> Replaced here with job 61541: one job, all ten cells validating, warmup 20,
> UCX defaults, post-race-fix. See `results/transpose_timing_merge.md`.
>
> **What moved and why.** Only the *buffered* NVSHMEM series changes materially
> — it is the only one that executes the `recv_buf`/`unpack` path the new
> `nvshmem_barrier_all()` sits in. It drops most at small matrices (1024² accum
> 139.2 → 111.9 GB/s, −20%) and converges by 16384² (1083.1 → 1073.8, −0.9%).
> That is a correctness cost, not a regression: the old numbers came from a
> configuration that failed validation outright at 4096² accum in job 60636.
> `NVSHMEM direct` and `single-K` shift only by run-to-run noise.
>
> **STILL STALE: the IPC/MPI half.** Those are 2026-05-05 numbers and have not
> been re-taken on a current build. Job **61945** (`transpose_fig_v2`) will
> supply a clean single-provenance replacement for them. Do not splice job
> 61541's `direct`/`single` columns in as an interim fix — they are inflated by
> in-loop `cudaEventElapsedTime` instrumentation, ~6 µs/iteration, which is
> −22% at 1024² (see `results/transpose_timing_merge.md`, Claim 1).

## 4 GPUs — With Accumulation (B += A^T)

### IPC / MPI — All Modes
Matrix Size | Mode                    | GB/s
------------|-------------------------|-------
1024²       | IPC direct (per-phase)  |  284.3
1024²       | IPC direct (single-K)   |  595.6
1024²       | IPC buffered            |  175.4
1024²       | GPU-aware MPI           |  200.1
1024²       | Staged MPI              |   81.6
2048²       | IPC direct (per-phase)  |  559.9
2048²       | IPC direct (single-K)   | 1171.7
2048²       | IPC buffered            |  499.8
2048²       | GPU-aware MPI           |  542.2
2048²       | Staged MPI              |  117.1
4096²       | IPC direct (per-phase)  |  696.3
4096²       | IPC direct (single-K)   |  882.3
4096²       | IPC buffered            |  839.3
4096²       | GPU-aware MPI           |  851.1
4096²       | Staged MPI              |  131.7
8192²       | IPC direct (per-phase)  |  812.1
8192²       | IPC direct (single-K)   |  889.6
8192²       | IPC buffered            | 1011.7
8192²       | GPU-aware MPI           | 1023.7
8192²       | Staged MPI              |  121.2
16384²      | IPC direct (per-phase)  |  865.7
16384²      | IPC direct (single-K)   |  884.8
16384²      | IPC buffered            | 1060.3
16384²      | GPU-aware MPI           | 1077.1
16384²      | Staged MPI              |  114.8

### NVSHMEM
Matrix Size | Mode              | GB/s
------------|-------------------|-------
1024²       | NVSHMEM direct    |  147.4
1024²       | NVSHMEM single-K  |  359.3
1024²       | NVSHMEM buffered  |  111.9
2048²       | NVSHMEM direct    |  383.7
2048²       | NVSHMEM single-K  |  881.8
2048²       | NVSHMEM buffered  |  358.8
4096²       | NVSHMEM direct    |  608.1
4096²       | NVSHMEM single-K  |  826.9
4096²       | NVSHMEM buffered  |  726.0
8192²       | NVSHMEM direct    |  776.3
8192²       | NVSHMEM single-K  |  871.5
8192²       | NVSHMEM buffered  |  979.0
16384²      | NVSHMEM direct    |  855.0
16384²      | NVSHMEM single-K  |  879.9
16384²      | NVSHMEM buffered  | 1073.8

## 4 GPUs — Without Accumulation (B = A^T)

### IPC / MPI — All Modes
Matrix Size | Mode                    | GB/s
------------|-------------------------|-------
1024²       | IPC direct (per-phase)  |  342.1
1024²       | IPC direct (single-K)   |  722.3
1024²       | IPC buffered            |  177.6
1024²       | GPU-aware MPI           |  207.0
1024²       | Staged MPI              |   74.9
2048²       | IPC direct (per-phase)  |  759.3
2048²       | IPC direct (single-K)   | 1240.9
2048²       | IPC buffered            |  501.9
2048²       | GPU-aware MPI           |  551.8
2048²       | Staged MPI              |  119.8
4096²       | IPC direct (per-phase)  | 1085.5
4096²       | IPC direct (single-K)   | 1349.4
4096²       | IPC buffered            |  875.5
4096²       | GPU-aware MPI           |  908.4
4096²       | Staged MPI              |  112.6
8192²       | IPC direct (per-phase)  | 1233.5
8192²       | IPC direct (single-K)   | 1310.6
8192²       | IPC buffered            | 1095.3
8192²       | GPU-aware MPI           | 1099.2
8192²       | Staged MPI              |  140.0
16384²      | IPC direct (per-phase)  | 1282.4
16384²      | IPC direct (single-K)   | 1302.8
16384²      | IPC buffered            | 1151.0
16384²      | GPU-aware MPI           | 1171.0
16384²      | Staged MPI              |  127.3

### NVSHMEM
Matrix Size | Mode              | GB/s
------------|-------------------|-------
1024²       | NVSHMEM direct    |  161.4
1024²       | NVSHMEM single-K  |  414.6
1024²       | NVSHMEM buffered  |  119.3
2048²       | NVSHMEM direct    |  471.1
2048²       | NVSHMEM single-K  | 1013.8
2048²       | NVSHMEM buffered  |  359.0
4096²       | NVSHMEM direct    |  874.6
4096²       | NVSHMEM single-K  | 1218.2
4096²       | NVSHMEM buffered  |  752.7
8192²       | NVSHMEM direct    | 1151.8
8192²       | NVSHMEM single-K  | 1278.7
8192²       | NVSHMEM buffered  | 1052.2
16384²      | NVSHMEM direct    | 1259.9
16384²      | NVSHMEM single-K  | 1295.1
16384²      | NVSHMEM buffered  | 1167.2

## 2 GPUs — With Accumulation (Buffered IPC)
Matrix Size | Mode           | GB/s
------------|----------------|------
1024²       | IPC (buffered) | 389.3
1024²       | GPU-aware MPI  | 361.9
1024²       | Staged MPI     |  92.4
2048²       | IPC (buffered) | 612.2
2048²       | GPU-aware MPI  | 581.3
2048²       | Staged MPI     | 104.7
4096²       | IPC (buffered) | 727.0
4096²       | GPU-aware MPI  | 693.7
4096²       | Staged MPI     | 108.4
8192²       | IPC (buffered) | 769.1
8192²       | GPU-aware MPI  | 738.6
8192²       | Staged MPI     | 105.6
16384²      | IPC (buffered) | 772.4
16384²      | GPU-aware MPI  | 742.0
16384²      | Staged MPI     | 104.3

---

## Appendix: `UCX_TLS` pinning is harmful — use UCX defaults
(jobs 59067, 59070; h200x4; 2026-07-31 / 2026-08-01)

Re-measuring the 4-GPU tables above with `UCX_TLS=self,sm,cuda_copy,cuda_ipc`
exported (a pinning introduced later, for LULESH) produced two large
discrepancies. Both trace to that one environment variable. The
`COMM_MODE==3` staged code path is unchanged since commit 9b9e6eb (verified
by diff), and build flags match `transpose/Makefile`.

### Symptom 1 — staged MPI is ~2.6x slower when UCX_TLS is set

16384², B+=A^T, staged MPI, varying ONLY `UCX_TLS`:

| UCX_TLS                          | GB/s  |
|----------------------------------|-------|
| *unset* (default selection)      | 96.0  |
| self,sm,cuda_copy,cuda_ipc       | 37.3  |
| self,sm  (host only, no CUDA)    | 37.3  |
| self,sm,cuda_ipc (no cuda_copy)  | 37.6  |

Note `self,sm` — no CUDA transports at all — is *also* 37.3. So this is NOT
`cuda_copy` mis-classifying pinned host memory (my initial hypothesis, now
refuted). ANY explicit restriction costs ~2.6x. UCX's default selection
evidently includes a transport the aliases above omit that matters for very
large host messages. The transpose staged path sends ~512 MiB per exchange
(Bo × order × 8 = 4096 × 16384 × 8 at 16384²/4 ranks), which is why this is
severe here. The stencil's staged halo is only ~128 KiB per exchange, and its
staged numbers do NOT show a comparable penalty — so this effect is
large-message-specific.

Buffered/direct IPC and GPU-aware MPI are UNAFFECTED by UCX_TLS
(~1,062,000 / ~1,073,000 MB/s in every setting).

`UCX_TLS=self,sm,cuda_ipc` (cuda_ipc without cuda_copy) additionally emits
"no copy across memory types transport ... Destination is unreachable" errors
for all modes — cuda_copy is required for any CUDA staging. Do not use it.

### Symptom 2 — one warmup iteration is NOT enough for GPU-aware MPI (when pinned)

GPU-aware MPI (COMM_MODE=2), MB/s, warmup run in DESCENDING order so that
warmup=1 is not confounded with being first-in-job:

| warmup | 4096²    | 16384²    |
|--------|----------|-----------|
| 50     | 820,940  | 1,073,179 |
| 20     | 819,664  | 1,073,289 |
| 5      | 818,428  | 1,073,371 |
| 1      | 117,616  |   714,738 |
| 1 (repeated LAST) | 132,671 | 716,773 |
| 50 (repeated last)| 821,292 | — |

warmup=1 is slow BOTH times, including when it runs last, so this is genuine
warmup insufficiency, not a first-run artifact. Controls are flat to ~4
significant figures across all warmup values: ipc_direct ~696,000 (4096²) /
~865,500 (16384²), ipc_buffered ~1,062,000, staged ~37,300. Only GPU-aware
moves. Matches the stencil result (job 57012/58445): under pinned UCX_TLS,
GPU-aware MPI needs >=5 untimed iterations; the built-in PRK single warmup
(transpose_ipc.cu, `iter == warmup`, default 1) is insufficient.

### Why the 2026-05-05 tables are believed correct

Those runs predate the UCX_TLS pinning, i.e. they used defaults. Under
defaults with warmup=20, this job measured GPU-aware MPI at 16384² =
1,077,634 MB/s vs the recorded **1077.1 GB/s** — a near-exact match. Staged
under defaults is 96.0 GB/s vs recorded 114.8, still 1.20x apart and NOT
fully explained (candidates: node-to-node variation, or another environment
difference not yet identified).

### Actions

1. **Do not export `UCX_TLS`.** It was introduced to work around LULESH's
   "broken gpumpi", which is now known to be a one-time setup cost and not a
   transport fault at all (see `LULESH/cuda/README.md` appendix). The pinning
   is therefore both unnecessary and actively harmful.
2. `TRANSPOSE_WARMUP` (env, default 1 = original behavior) was added to
   `transpose/transpose_ipc.cu` on 2026-07-30 so warmup can be raised. Use
   >=5 for any GPU-aware measurement.
3. ~~Not yet measured: GPU-aware MPI at warmup=1 under UCX *defaults*.~~
   **RESOLVED (job 59853, 2026-08-03).** Under UCX defaults, GPU-aware MPI is
   already converged at warmup=1 — 16384²: 1,073,277 (wu=1) / 1,074,014 (wu=2)
   / 1,073,900 (wu=5) / 1,074,167 (wu=20) MB/s. So the warmup sensitivity in
   Symptom 2 is *entirely* an artifact of the `UCX_TLS` pinning, and the
   built-in single PRK warmup was never inadequate under the configuration the
   2026-05-05 tables were taken in. Those tables stand.
4. `LULESH/run_lulesh.sbatch` and `run_lulesh_verify.sbatch` still export
   `UCX_TLS`; the LULESH README's claim that pinning protects host-staged MPI
   is contradicted by Symptom 1 and should be revisited.
