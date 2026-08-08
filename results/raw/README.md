# Raw job logs — deliberately NOT tracked in git

`.gitignore` excludes `*.out` repo-wide, and that convention is intact: no
raw log is tracked anywhere. This directory is a local audit trail, kept
until publication, so that committed results files can be checked against
the original output for transcription errors and for unexpected runtime
messages that a consolidated table would drop.

Authoritative committed results live in `results/*.csv` and `results/*.md`,
which carry node, command, environment, correctness values and provenance in
their header blocks. Figures are generated from those files, not from
hard-coded values, so a figure cannot drift from its data.

Currently held here (not in git):

| log | job | what it backs |
|-----|-----|---------------|
| `lulesh_verify_60150.out` | 60150 | `results/lulesh_results.csv` — all nine LULESH variants, UCX defaults, h200x8-03. 1335 lines, of which 1032 are gpumpi teardown `cuCtxGetApiVersion` noise. |
| `transpose_figure_60636.out` | 60636 | `results/transpose_timing_merge.md` — all eight series at `6f674cb`, i.e. before PR #1. Holds the one `validates=0` cell (`order=4096 accum=1 nvbuffered`) that Claim 3 rests on. |
| `stopsync_ab_61344.out` | 61344 | `results/transpose_timing_merge.md` Claim 1 — PRE/POST A/B, 3 reps per cell. `DEVLINE` rows carry the `wall/iter` values behind the overhead table. |
| `transpose_figure_61541.out` | 61541 | `results/transpose_timing_merge.md` Claim 2 — all eight series after PR #1 and the NVSHMEM barrier. 80/80 validate. |

These three are kept at the repo root as well as here (Slurm writes them there);
the copies in this directory are the audit trail.

If a log is ever needed in-repo, prefer extending the consolidated results
file over force-adding the log past `.gitignore`.
