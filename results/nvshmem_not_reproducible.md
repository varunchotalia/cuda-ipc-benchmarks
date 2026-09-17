# We cannot currently reproduce the paper's NVSHMEM numbers

Job 96770, h200x4, 2026-09-16. 350 runs, every one validating. Seven Fig. 3c
series x 5 orders x 10 reps, overwrite (accum=0), one job so every series
shares a node, a build and an allocation. Raw log gitignored.

## 1. Our own series are solid

With 10 reps per point:

| series | order | median GB/s | CoV | max/min |
|---|--:|--:|--:|--:|
| WinIPC single-kernel | 1024 | 704.9 | 3% | 1.1x |
| WinIPC single-kernel | 4096 | 1320.8 | 0% | 1.0x |
| WinIPC single-kernel | 16384 | 1301.1 | 4% | 1.1x |
| WinIPC per-phase | 4096 | 1055.7 | 0% | 1.0x |
| WinIPC buffered | 4096 | 918.4 | 0% | 1.0x |
| GPU-aware MPI | 4096 | 946.0 | 1% | 1.0x |
| GPU-aware MPI | 16384 | 1174.6 | 0% | 1.0x |

GPU-aware MPI at 1024 is the one blemish: 9 runs near 244 GB/s and one at 46.4,
giving a 26% CoV. One outlier in ten, and the median is unaffected.

## 2. NVSHMEM is not reproducible, and it is worse than bimodal

| series | order | published | median | max | runs reaching 80% of published |
|---|--:|--:|--:|--:|--:|
| nvsingle | 1024 | 415.1 | 19.6 | 433.5 | **4 of 10** |
| nvsingle | 4096 | 1215.6 | 118.8 | 336.7 | **0 of 10** |
| nvsingle | 16384 | 1295.0 | 194.5 | 265.5 | **0 of 10** |
| nvdirect | 1024 | 160.9 | 13.3 | 62.2 | 0 of 10 |
| nvdirect | 4096 | 874.6 | 56.8 | 79.6 | 0 of 10 |
| nvdirect | 16384 | 1259.5 | 108.7 | 174.2 | 0 of 10 |
| nvbuffered | 1024 | 120.0 | 20.4 | 60.6 | 0 of 10 |
| nvbuffered | 4096 | 751.5 | 37.0 | 113.9 | 0 of 10 |
| nvbuffered | 16384 | 1167.0 | 109.6 | 138.2 | 0 of 10 |

**4 of 90 runs reach 80% of the published value. Medians sit at 8-10% of it.**

The earlier "bimodal" reading was too generous. At order 1024 the fast mode is
real and reproduces: nvsingle peaks at 433.5 against a published 415.1, four
times in ten. **At 4096 and 16384 the fast mode does not appear at all.** The
best of ten runs reaches 27% and 20% of the published figure. Those values are
not a rare draw from a distribution we are sampling; we are not reaching them.

## 3. A candidate, unproven

NVSHMEM reports `CUDA API 12080, CUDA Runtime 12080, CUDA Driver 13020` -- a
13.2 driver under a library built for 12.8 (build timestamp Feb 2025). A major-
version gap is a plausible cause.

It is not established. No log from before 2026-09-10 records a driver version,
so there is nothing to compare against and no way to show the driver changed
after job 61541 measured the published values on 2026-08-07. **Every harness
should print `nvidia-smi --query-gpu=driver_version` from now on**; this
question would have been answerable in one grep.

## 4. What this means for the paper

The Fig. 3c NVSHMEM curves and the NVSHMEM column of Table V come from single
runs in job 61541. We can no longer reproduce them, and at the two larger
orders we cannot get within 4x of them.

That cuts both ways, which is why none of the options is comfortable:

- **Reporting today's medians would be unfair to NVSHMEM.** "WinIPC is 10x
  faster than NVSHMEM" is almost certainly an artefact of this cluster's
  current state, not a property of the library, and Section V-D's careful
  "competitive performance" wording would become an overclaim in our favour.
- **Leaving the published numbers unqualified is also wrong.** They are single
  samples we cannot now reproduce.

Defensible choices:

1. **Keep the August numbers, state the reproduction attempt.** Say plainly
   that a September re-measurement on the same cluster reached 8-10% of those
   values, that the cause is unidentified, and that the NVSHMEM comparison
   should be read as indicative. Honest, costs nothing, and is the option that
   survives a reviewer trying to reproduce.
2. **Drop the NVSHMEM comparison.** Defensible, but discards a comparison the
   paper is stronger for having.
3. **Fix the environment first.** Best outcome if the driver mismatch is the
   cause and a 13.x-matched NVSHMEM build is available. Needs a rebuild against
   the installed driver and a re-run -- worth trying before falling back to 1.

Recommendation: attempt 3, and fall back to 1 rather than 2. Do not report
today's NVSHMEM medians as a result.

Table VIII's GB200 NVSHMEM numbers come from Jeff's system and are untouched by
this. Our own WinIPC and MPI numbers are unaffected -- section 1 above, and the
cross-job agreement in `results/transpose_v3_reproducibility.md`.

---

## 5. Not the toolchain either (job 854688)

The CUDA-version hypothesis was tested and rejected. Same source built two ways
and interleaved on one node, 6 reps per point, driver 595.71.05:

  A = HPC SDK 25.3 / CUDA 12.8 (what we had been running)
  B = HPC SDK 25.9 / CUDA 13.0 (matched to the 13.x driver)

| mode | order | published | A median | A max | B median | B max |
|---|--:|--:|--:|--:|--:|--:|
| nvsingle | 1024 | 415.1 | 43.6 | 430.6 | 18.4 | 422.0 |
| nvsingle | 4096 | 1215.6 | 21.5 | 75.0 | 30.5 | 304.8 |
| nvsingle | 16384 | 1295.0 | 196.0 | 232.9 | 208.8 | 272.7 |
| nvbuffered | 1024 | 120.0 | 25.8 | 60.6 | 6.0 | 30.5 |
| nvbuffered | 4096 | 751.5 | 52.6 | 129.2 | 45.9 | 158.8 |
| nvbuffered | 16384 | 1167.0 | 103.5 | 131.0 | 116.5 | 119.9 |

**Both arms reach 80% of published in 1 of 36 runs.** Medians land at 3-16% of
published either way. Rebuilding against the driver's CUDA version changes
nothing, so the 12.8-library-under-13.x-driver mismatch is not the cause,
however well the installation dates lined up.

Worth noting what still works: at order 1024 both arms occasionally hit the
published value (A max 430.6, B max 422.0 against 415.1). The fast mode is real
in both builds. At 4096 and 16384 neither arm comes close -- best of twelve
runs is 304.8 against 1215.6.

## 6. Recommendation: stop investigating, document it

Three hypotheses have now been tested and rejected:

1. reps 2-3 contaminated by the previous rep's teardown -- rep-1-only is still
   -89% to -94%
2. NVSHMEM falling back to InfiniBand -- `P2P list: 0 1 2 3`, IBRC skipped
3. CUDA toolchain mismatch with the driver -- both arms fail identically

Each round costs about a day of queue latency, the obvious causes are gone, and
nothing in the logs points anywhere. Further investigation has poor expected
value against the camera-ready deadline, and this is a local environment problem
rather than a defect in our work.

Take option 1 from section 4: **keep the August numbers and state the
reproduction attempt.** Something like:

  "The NVSHMEM comparison uses single runs measured in August 2026. A
   September re-measurement on the same cluster did not reproduce them:
   medians reached 8-10% of the recorded values, and at the two larger matrix
   orders the recorded values were not approached in any of ten runs. The
   cause was not identified; rebuilding against the installed CUDA version did
   not change it, and our own WinIPC and GPU-aware MPI measurements in the same
   jobs remained stable to within 4%. The NVSHMEM comparison should therefore
   be read as indicative."

That is honest, it costs nothing, and it is what survives a reviewer who tries
to reproduce. Do not publish the September medians as a result -- they would
say WinIPC is 10x faster than NVSHMEM, which this cluster's behaviour does not
support as a claim about the library.
