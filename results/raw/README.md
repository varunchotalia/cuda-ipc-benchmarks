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

If a log is ever needed in-repo, prefer extending the consolidated results
file over force-adding the log past `.gitignore`.
