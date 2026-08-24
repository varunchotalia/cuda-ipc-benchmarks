# Does the transpose verifier inspect remotely delivered data? Yes.

Settles NVL_SUMMARY 6.Q4 ("the single-kernel transpose variant reports
validation success on work it does not appear to perform") for the
validation half. Raw log `transpose_val_ctrl_68524.out` is gitignored.

| field | value |
|---|---|
| job | 68524, node h200x4-02, 2026-08-22T02:56 -> 02:57 (96 s) |
| harness | `transpose/run_transpose_validator_control.sbatch` |
| build | COMM_MODE=0 SINGLE_KERNEL=1, 4 GPUs, 50 iterations, warmup 1 |
| control | `TRANSPOSE_SKIP_PEER=<rank>` suppresses every other rank's write to that rank |

## Why this was in doubt

Every transpose run prints "Solution validates", which is equally consistent
with a verifier that checks remote arrivals and one blind to them. A test
that has never failed cannot separate the two. The doubt was load-bearing:
the abstract's 3.0x headline is IPC single-kernel at 1024^2, and the
multi-node single-kernel variant was CUT from the GB200 tables on the
strength of this suspicion.

## Result

| accum | order | skip | verdict | mismatched | ratio to threshold |
|--:|--:|---|---|--:|--:|
| 0 | 1024^2 | NA | Solution validates | 0 | 0.000e+00 |
| 0 | 4096^2 | NA | Solution validates | 0 | 0.000e+00 |
| 1 | 1024^2 | NA | Solution validates | 0 | 0.000e+00 |
| 1 | 4096^2 | NA | Solution validates | 0 | 0.000e+00 |
| 0 | 1024^2 | 1 | NEGATIVE CONTROL PASSED | 196,608 | 1.065e+13 |
| 0 | 1024^2 | 2 | NEGATIVE CONTROL PASSED | 196,608 | 9.014e+12 |
| 0 | 1024^2 | 3 | NEGATIVE CONTROL PASSED | 196,608 | 7.380e+12 |
| 0 | 4096^2 | 1 | NEGATIVE CONTROL PASSED | 3,145,728 | 1.704e+14 |
| 0 | 4096^2 | 2 | NEGATIVE CONTROL PASSED | 3,145,728 | 1.442e+14 |
| 0 | 4096^2 | 3 | NEGATIVE CONTROL PASSED | 3,145,728 | 1.180e+14 |
| 1 | 1024^2 | 1 | NEGATIVE CONTROL PASSED | 196,608 | 5.430e+14 |
| 1 | 1024^2 | 2 | NEGATIVE CONTROL PASSED | 196,608 | 4.597e+14 |
| 1 | 1024^2 | 3 | NEGATIVE CONTROL PASSED | 196,608 | 3.764e+14 |
| 1 | 4096^2 | 1 | NEGATIVE CONTROL PASSED | 3,145,728 | 8.690e+15 |
| 1 | 4096^2 | 2 | NEGATIVE CONTROL PASSED | 3,145,728 | 7.354e+15 |
| 1 | 4096^2 | 3 | NEGATIVE CONTROL PASSED | 3,145,728 | 6.018e+15 |

**4/4 baselines validate; 12/12 controls correctly fail.**

## The confirmation that matters: the mismatch count is exact

With P=4 and block width Bo=order/P, each rank's B block is order x Bo and
is assembled from one LOCAL transposed sub-block plus three REMOTE ones,
each Bo x Bo. Withholding one rank should therefore corrupt exactly three
sub-blocks -- its local contribution must survive.

| order | Bo | sub-block | 3 remote sub-blocks | observed `mismatched` |
|--:|--:|--:|--:|--:|
| 1024^2 | 256 | 65,536 | 196,608 | 196,608 |
| 4096^2 | 1024 | 1,048,576 | 3,145,728 | 3,145,728 |

Exact, at both sizes and for every skipped rank. This rules out the loose
readings: the suppression is neither partial nor over-broad, the verifier
sees precisely the withheld elements, and the skipped rank's own local
transpose still lands (3 sub-blocks corrupted, not 4).

## Margins

Baselines report `abserr_total = 0.000000e+00` with `mismatched=0` --
bit-exact, as expected since the expected values are integers well inside
double precision. Controls exceed the 1.0e-8 threshold by **12 to 15 orders
of magnitude** (ratio 7.4e12 to 8.7e15), confirming the static estimate that
a missing block could not slip under it.

## What this does and does not establish

Established: the verifier inspects remotely delivered data, so
"Solution validates" is meaningful, and the intra-node single-kernel numbers
behind Fig. 3 and Table V rest on a check with teeth.

NOT established: that the multi-node single-kernel rates are correct. This
job runs at 4 GPUs on one node. The reason that variant was cut was its
comm-kernel time falling with rank count at 16/32 GPUs -- a separate claim,
with an innocent explanation (per-rank work falls as 1/P for a fixed global
order, while the per-phase variant pays P-1 barriers). Since the validation
premise behind the cut is now disproven, a single-kernel arm at 16/32 GPUs
on current source is the remaining step; it is item 2 in the GB200 request.
