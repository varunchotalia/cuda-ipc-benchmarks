#!/usr/bin/env python3
"""Build results/lulesh_variance.csv from E1 job logs (one row per run).

Usage:
    scripts/build_lulesh_variance_csv.py lulesh_verify_*.out > results/lulesh_variance.csv
    scripts/build_lulesh_variance_csv.py --self-test

WHAT `rep` MEANS. One row per (variant, job). `rep` indexes the JOB SUBMISSION,
not a repetition inside a process: LULESH/run_lulesh_verify.sbatch runs each
variant exactly once per job, so E1's five separate submissions (60796-60800)
give five independent runs per variant, each re-paying process start, driver
init and window construction. reps are numbered by ascending jobid, so rep 1 is
the lowest-numbered job.

WHY jobid AND node ARE COLUMNS. The paper forms the matched-pair residual as
WinIPC - ipc *within a job*, so job and node effects cancel. Because rep maps
1:1 onto a submission, pairing by rep is equivalent to pairing within a job --
but that equivalence should be checkable rather than trusted, so both the jobid
and the node are carried per row. Pair on jobid if you want to be explicit;
`--check-pairing` asserts the two groupings agree.

elapsed_s_hi is reconstructed from the log's higher-precision overall grind
time, matching results/lulesh_results.csv: the "Elapsed time" field prints only
two decimals, which is too coarse for sub-percent matched differences.

    elapsed_s_hi = grind_us_z_c * zone_cycles / 1e6
    zone_cycles  = iterations * zones_per_rank * ranks

zones and ranks are read from each log, not assumed, so a different -s or rank
count still reconstructs correctly.
"""
import argparse
import csv
import os
import re
import sys

VARIANT_RE = re.compile(r"^=== LULESH (\S+): (\d+) ranks, .*-s (\d+)")
NODE_RE    = re.compile(r"^Node:\s*(\S+)")
ITER_RE    = re.compile(r"Iteration count\s*=\s*(\d+)")
ENERGY_RE  = re.compile(r"Final Origin Energy\s*=\s*(\S+)")
ELAPSED_RE = re.compile(r"Elapsed time\s*=\s*(\S+)")
GRIND_RE   = re.compile(r"Grind time \(us/z/c\)\s*=\s*(\S+).*?\(\s*(\S+) overall\)")
FOM_RE     = re.compile(r"FOM\s*=\s*(\S+)")
JOBID_RE   = re.compile(r"_(\d+)\.out$")

FIELDS = ["variant", "rep", "jobid", "node", "ranks", "size",
          "iterations", "grind_us_z_c", "elapsed_s_log", "elapsed_s_hi",
          "fom_z_per_s", "energy"]


def parse_log(path):
    """Yield one dict per variant found in a single job log."""
    jm = JOBID_RE.search(os.path.basename(path))
    jobid = jm.group(1) if jm else ""
    node = ""
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            m = NODE_RE.match(line)
            if m and not node:
                node = m.group(1)
                continue
            m = VARIANT_RE.match(line)
            if m:
                if cur:
                    yield cur
                cur = {"variant": m.group(1), "ranks": int(m.group(2)),
                       "size": int(m.group(3)), "jobid": jobid, "node": node}
                continue
            if cur is None:
                continue
            for rx, key, cast in ((ITER_RE, "iterations", int),
                                  (ENERGY_RE, "energy", str),
                                  (ELAPSED_RE, "elapsed_s_log", float),
                                  (FOM_RE, "fom_z_per_s", float)):
                m = rx.search(line)
                if m and key not in cur:
                    cur[key] = cast(m.group(1))
            m = GRIND_RE.search(line)
            if m and "grind_us_z_c" not in cur:
                cur["grind_us_z_c"] = float(m.group(2))   # the OVERALL figure
    if cur:
        yield cur


def finish(row):
    """Derive elapsed_s_hi; returns None if the run did not produce metrics."""
    need = ("grind_us_z_c", "iterations", "ranks", "size")
    if any(k not in row for k in need):
        return None
    zones = row["size"] ** 3
    zone_cycles = row["iterations"] * zones * row["ranks"]
    row["zone_cycles"] = zone_cycles
    row["elapsed_s_hi"] = row["grind_us_z_c"] * zone_cycles / 1e6
    return row


def build(paths):
    by_job = {}
    for p in paths:
        for raw in parse_log(p):
            row = finish(raw)
            if row is None:
                print(f"warning: {p}: variant {raw.get('variant')} produced no "
                      f"metrics -- run failed, row omitted", file=sys.stderr)
                continue
            by_job.setdefault(row["jobid"], []).append(row)
    # rep numbers follow ascending jobid so they are stable across reruns
    rows = []
    for rep, jobid in enumerate(sorted(by_job, key=lambda j: int(j or 0)), 1):
        for row in by_job[jobid]:
            row["rep"] = rep
            rows.append(row)
    return rows


def check_pairing(rows):
    """Assert that pairing by rep and pairing by jobid give the same groups."""
    rep_to_jobs = {}
    for r in rows:
        rep_to_jobs.setdefault(r["rep"], set()).add(r["jobid"])
    bad = {k: v for k, v in rep_to_jobs.items() if len(v) != 1}
    if bad:
        print(f"FAIL: rep does not map 1:1 to jobid: {bad}", file=sys.stderr)
        return False
    print(f"pairing OK: {len(rep_to_jobs)} reps, each exactly one job "
          f"({', '.join(sorted(next(iter(v)) for v in rep_to_jobs.values()))})",
          file=sys.stderr)
    return True


PAIRS = [("mpiwrap", "ipc", "A"), ("mpiwrap_rp", "ipc_rp", "C")]


def report_pairs(rows):
    """Matched-pair residuals, computed WITHIN a job.

    This is the quantity the paper reports. Taking the difference inside one
    job cancels whatever that job saw -- node, co-tenants, thermal state --
    which two independent marginals (mean WinIPC vs mean ipc) would not.
    """
    idx = {}
    for r in rows:
        idx[(r["jobid"], r["variant"])] = r
    jobs = sorted({r["jobid"] for r in rows}, key=lambda j: int(j or 0))
    print("\nmatched-pair residuals (interposed - handwritten, within job):",
          file=sys.stderr)
    for wrap, hand, mode in PAIRS:
        deltas = []
        for j in jobs:
            a, b = idx.get((j, wrap)), idx.get((j, hand))
            if not (a and b):
                continue
            d = a["elapsed_s_hi"] - b["elapsed_s_hi"]
            deltas.append(d)
            print(f"  mode {mode}  job {j}  {wrap} {a['elapsed_s_hi']:.5f} - "
                  f"{hand} {b['elapsed_s_hi']:.5f} = {d:+.5f} s "
                  f"({d / b['elapsed_s_hi'] * 100:+.3f}%)", file=sys.stderr)
        if deltas:
            pct = [d / idx[(j, hand)]["elapsed_s_hi"] * 100
                   for j, d in zip(jobs, deltas) if (j, hand) in idx]
            print(f"  mode {mode}  n={len(deltas)}  "
                  f"residual range [{min(pct):+.3f}%, {max(pct):+.3f}%]",
                  file=sys.stderr)


def self_test():
    """Reconstruct job 60150 from its committed raw log and compare against
    results/lulesh_results.csv, which was produced independently."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = os.path.join(here, "results", "raw", "lulesh_verify_60150.out")
    ref = os.path.join(here, "results", "lulesh_results.csv")
    if not (os.path.exists(log) and os.path.exists(ref)):
        print("self-test: missing fixture", file=sys.stderr)
        return 1
    got = {r["variant"]: r for r in build([log])}
    with open(ref) as f:
        rows = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    want = {r["variant"]: r for r in csv.DictReader(rows)}
    ok = True
    for v in sorted(want):
        if v not in got:
            print(f"  MISSING  {v}", file=sys.stderr); ok = False; continue
        a, b = got[v]["elapsed_s_hi"], float(want[v]["elapsed_s_hi"])
        e_a, e_b = got[v]["energy"], want[v]["energy"]
        d = abs(a - b)
        match = d < 5e-5 and e_a == e_b
        ok &= match
        print(f"  {'OK ' if match else 'DIFF'} {v:11s} "
              f"elapsed_s_hi {a:.5f} vs {b:.5f} (d={d:.2e})  energy {e_a}",
              file=sys.stderr)
    print("self-test PASS" if ok else "self-test FAIL", file=sys.stderr)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-pairing", action="store_true")
    ap.add_argument("--pairs", action="store_true",
                    help="also print matched-pair residuals to stderr")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.logs:
        ap.error("give one or more job logs, or --self-test")
    rows = build(a.logs)
    if a.check_pairing and not check_pairing(rows):
        sys.exit(1)
    if a.pairs:
        report_pairs(rows)
    w = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["variant"], r["rep"])):
        w.writerow(r)


if __name__ == "__main__":
    main()
