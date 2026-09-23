# Commit hashes changed on 2026-09-23

On 2026-09-23 the history of this repository was rewritten to remove
details of a non-public system that had been committed by mistake. Every
commit listed below (from 6-7 August 2026 onward) received a new hash. Earlier
commits, including `3df6bf2` (the source revision behind Table I in the
paper), are unchanged.

Raw job logs, `results/transpose_90161.csv` and several notes in `results/`
record the old hashes. This table maps each old hash to its new one. The tag
`v1.0-camera-ready` now points at the rewritten commit.

| date | old | new | subject |
|---|---|---|---|
| 2026-08-06 | `7c77c37` | `b4aac49` | Merge pull request #1 from jeffhammond/transpose-timing-cuda-events |
| 2026-08-07 | `7ec29b5` | `88498b3` | NVSHMEM: barrier after transpose unpack, warmup in the large stencil |
| 2026-08-07 | `8cd8653` | `71b9503` | REBRAND: mpiwrap is now WinIPC |
| 2026-08-07 | `8cdc884` | `f58a96f` | Merge pull request #2 from jeffhammond/lulesh-atomicadd-race-fix |
| 2026-08-07 | `a189700` | `9b41da5` | Repo hygiene: drop CloverLeaf, track the transpose harness |
| 2026-08-07 | `ae6b27f` | `bb4cf41` | NVL72: clear CUDA errors in the interposer, pin one HPC-X tree |
| 2026-08-07 | `c89bf58` | `253fef8` | STENCIL: add speedup-over-host-staged figure |
| 2026-08-07 | `c8e7408` | `24920bc` | LULESH: check setup-path CUDA calls, clamp comBufSize at np=1 |
| 2026-08-07 | `fddc5ba` | `21303ea` | E1: commit the five-run LULESH variance table |
| 2026-08-08 | `03f23c8` | `b8fcb35` | Stencil: second attempt at neighbour-only sync, OFF by default and unrun |
| 2026-08-08 | `27bfb8c` | `9286539` | Stencil: harness to validate the neighbour-only handshake (job 61948) |
| 2026-08-08 | `62ac2e7` | `f3fbb34` | Figures: GB200 NVL scale-out transpose and stencil, 16 and 32 GPUs |
| 2026-08-08 | `8a0886f` | `86b5f3e` | Fig 6: plot LULESH figure of merit instead of elapsed time |
| 2026-08-08 | `97fbda4` | `b6a2a49` | Fig 3(c): the NVSHMEM curves were April data; replace with job 61541 |
| 2026-08-08 | `bdf3105` | `073a18a` | Transpose: take cudaEventElapsedTime off the timed path |
| 2026-08-08 | `c949dba` | `fce9e22` | Repo hygiene: never commit the scratch benchmark duplicate |
| 2026-08-08 | `efbc081` | `583889c` | Results: commit the evidence for what PR #1 and the NVSHMEM barrier changed |
| 2026-08-11 | `7cc19cf` | `87e22d5` | Harness: derive MPI_HOME instead of assuming the module exports it |
| 2026-08-11 | `a75de54` | `c43fd52` | Fig: key all seven series in the IPC vs NVSHMEM panel |
| 2026-08-19 | `08efe7f` | `470a71b` | Stencil: commit the 63329 A/B evidence and fail loudly on a bad STENCIL_SYNC |
| 2026-08-19 | `14d4e79` | `f4c4113` | Results: record the 08-13 jobs; keep manuscripts out of git |
| 2026-08-19 | `57d80f2` | `799414f` | Transpose: negative control proving the verifier inspects remote data |
| 2026-08-21 | `4fb8ca7` | `38033d9` | Transpose: report verification margin on every run, not just pass/fail |
| 2026-08-21 | `a055de1` | `3c76814` | Control harness: unset TRANSPOSE_SKIP_PEER on baseline arms instead of "NA" |
| 2026-08-21 | `be33b85` | `e49721c` | Control: TRANSPOSE_SKIP_PEER names a rank globally, not per-rank |
| 2026-08-23 | `58e21f7` | `75a17db` | Stencil: add STENCIL_SYNC=ssend so the advisor's proposal can be measured |
| 2026-08-23 | `6cf58a0` | `e83fa78` | Results: the transpose verifier does inspect remote data (job 68524) |
| 2026-09-03 | `2b75444` | `6edb763` | Transpose: denser order sweep, non-power-of-two points, reps on every series |
| 2026-09-03 | `95a4607` | `490f287` | WinIPC: refuse peers the hardware cannot reach, instead of trusting the open |
| 2026-09-03 | `e2e9f20` | `c931f4c` | NVL72 suite: order sweep, repetitions, both stencil sync arms, real geometry |
| 2026-09-03 | `f84c7df` | `e5a90ce` | Stencil A/B: np=8 only, 5 reps, settle the node before timing |
| 2026-09-08 | `072d631` | `935e96d` | Poster figure: show the direct transpose variant; drop gaps and scaffolding |
| 2026-09-08 | `45eee38` | `0f0d6f5` | Poster: panel (c) drawn the way the paper draws it, not just from the same data |
| 2026-09-08 | `55ecb39` | `8487cc2` | Poster: name the NVSHMEM variant in (b), trim the caption |
| 2026-09-08 | `9f26378` | `043938e` | Poster: panel (c) IS the paper's LULESH figure, not a second version of it |
| 2026-09-08 | `a5be2e4` | `68a99db` | Poster (b): add the per-phase direct variant, the one that explains the others |
| 2026-09-08 | `e887c22` | `1916be7` | Poster: rename output to poster_figure, reword two caption lines |
| 2026-09-10 | `0990184` | `36af9d6` | NVSHMEM here is bimodal, and that undercuts the paper's single-run comparison |
| 2026-09-10 | `1a932b8` | `dd68e8b` | Reproducer: does a raw CUDA IPC write land across an island boundary? |
| 2026-09-10 | `391d58b` | `0008479` | NVSHMEM: transport selection is not the cause; measure rates cleanly instead |
| 2026-09-10 | `63df376` | `3b35256` | WinIPC: gate peers on delivery, not on reachability |
| 2026-09-10 | `6dc568e` | `3f10575` | Record what job 90161 reproduces, and queue the NVSHMEM diagnostic |
| 2026-09-10 | `be5ffa3` | `9efa6b6` | NVSHMEM diagnostic: stop truncating away the results |
| 2026-09-10 | `d076e13` | `197cfb2` | Plot the job 90161 transpose sweep; leave NVSHMEM out of it |
| 2026-09-15 | `15cddb7` | `d1520b6` | Table V regenerated from one job, and the NVSHMEM distribution queued |
| 2026-09-15 | `6b79ee0` | `05735de` | Record the cross-island finding; isolate the MPI fallback |
| 2026-09-17 | `92b961a` | `13fc6da` | LULESH: relabel the mode letters into mechanism order |
| 2026-09-17 | `a69d5ca` | `5ab8c4e` | NVSHMEM: not the toolchain either; stop investigating and document it |
| 2026-09-17 | `b965fff` | `02b6585` | NVSHMEM: we cannot reproduce the published numbers, and it is not just bimodal |
| 2026-09-17 | `e64126a` | `4b22b2a` | Test whether NVSHMEM recovers when built against the driver's CUDA version |
| 2026-09-22 | `1dcdb18` | `8e7ca8c` | Table I provenance: every number reproduces at 3df6bf2, and 358 is raw lines |
| 2026-09-22 | `a502e55` | `9c6c843` | LULESH chart 2: relabel the x-axis, which 13fc6da missed |
| 2026-09-23 | `0422f69` | `e873314` | Public repo: drop GB200 system details, match README to the paper, date stale notes |
| 2026-09-23 | `b810ac9` | `6aea1cb` | Fig. 3 and Table III from job 90161; GB200 transpose figure to the paper's values |
