#!/usr/bin/env python3
"""Build results/transpose_90161.csv from the job 90161 log (one row per run).

Usage:
    scripts/build_transpose_90161_csv.py transpose_fig_v3_90161.out > results/transpose_90161.csv

The raw log is gitignored (*.out), so this CSV is the committed record behind
Fig. 3 and Table III. plots/make_transpose_plots.py reads it and takes the
median over reps; the reps are kept here so that median can be checked.

The run settings are read from the log's own header, not assumed, and the
script refuses to write anything if the header disagrees with what the paper
states (20 untimed + 100 timed iterations, UCX_TLS unset), if any run failed to
validate, or if any cell has other than three reps.
"""
import csv
import re
import sys
from collections import Counter

HDR_NODE = re.compile(r"^=== node=(\S+) job=(\d+) date=(\S+) ===")
HDR_CFG = re.compile(r"^=== UCX_TLS=(\S+) warmup=(\d+) iters=(\d+) reps=(\d+) ===")
HDR_HEAD = re.compile(r"^=== HEAD=(\w+) ")
HDR_MD5 = re.compile(r"^=== transpose_ipc\.cu md5=(\w+) ===")
LIB = re.compile(r"^--- building libmpiwrap\.so \((\w+) ")
ROW = re.compile(r"^rep=(\d+) RESULT transpose (.*)$")

EXPECT = {"ucx_tls": "<unset>", "warmup": 20, "iters": 100, "reps": 3}


def main(path):
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            if m := HDR_NODE.match(line):
                meta.update(node=m[1], job=m[2], date=m[3])
            elif m := HDR_CFG.match(line):
                meta.update(ucx_tls=m[1], warmup=int(m[2]), iters=int(m[3]),
                            reps=int(m[4]))
            elif m := HDR_HEAD.match(line):
                meta["head"] = m[1]
            elif m := HDR_MD5.match(line):
                meta["src_md5"] = m[1]
            elif m := LIB.match(line):
                meta["libmpiwrap"] = m[1]
            elif m := ROW.match(line):
                d = dict(re.findall(r"(\w+)=([\w.+-]+)", m[2]))
                rows.append(d | {"rep": m[1]})

    for k, v in EXPECT.items():
        if meta.get(k) != v:
            sys.exit(f"header {k}={meta.get(k)!r}, expected {v!r}; not writing")
    bad = [r for r in rows if r["validates"] != "1" or r["mbps"] == "FAIL"]
    if bad:
        sys.exit(f"{len(bad)} runs failed or did not validate; not writing")
    per_cell = Counter((r["accum"], r["mode"], r["order"]) for r in rows)
    if set(per_cell.values()) != {EXPECT["reps"]}:
        sys.exit(f"rep counts per cell: {Counter(per_cell.values())}; not writing")

    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["job", "node", "date", "head", "libmpiwrap", "src_md5",
                "warmup", "iters", "np", "accum", "mode", "order", "rep",
                "gbps"])
    for r in rows:
        w.writerow([meta["job"], meta["node"], meta["date"], meta["head"],
                    meta["libmpiwrap"], meta["src_md5"], meta["warmup"],
                    meta["iters"], r["np"], r["accum"], r["mode"], r["order"],
                    r["rep"], f"{float(r['mbps']) / 1000.0:.4f}"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
