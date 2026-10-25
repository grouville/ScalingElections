"""Alternate isolated processes to compare complete Mojo GPU calls without sharing GPU pools."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("/workspace/kemeny-complete"))
parser.add_argument("--worker", type=Path)
parser.add_argument("--repeats", type=int, default=9)
parser.add_argument("--control", action="store_true")
args = parser.parse_args()
cases = [(30, "uint64")] if args.control else [(n, dtype) for dtype in ("uint32", "uint64") for n in (24, 28, 30)]

if args.worker:
    sys.path.insert(0, str(args.worker / "build"))
    sys.path.insert(0, str(args.worker))
    os.chdir(args.worker)
    import numpy as np

    import scalingelections as se

    rows = []
    for n, dtype in cases:
        counts = np.random.default_rng(42 + n).integers(0, 100, (n, n), dtype=np.uint32)
        np.fill_diagonal(counts, 0)
        se.compute_kemeny_ranking(counts, implementation="mojo", backend="gpu", score_type=dtype)
        start = time.perf_counter_ns()
        result = se.compute_kemeny_ranking(counts, implementation="mojo", backend="gpu", score_type=dtype)
        elapsed = time.perf_counter_ns() - start
        rows.append(
            {
                "n": n,
                "dtype": dtype,
                "elapsed_ns": elapsed,
                "result": {
                    "ranking": list(result.ranking),
                    "score": int(result.score),
                    "winners": list(result.winners),
                    "multiplicity": str(result.multiplicity),
                },
            }
        )
    print(json.dumps(rows))
    sys.exit(0)

report = {
    "seed": 42,
    "baseline_commit": "25bf88568cff938c96a777b8776be41677fb1f27",
    "scope": "complete public Mojo GPU calls, one warm call before each sample, alternating isolated processes",
    "cases": [],
    "pairs": [],
}
for label in ("baseline", "candidate"):
    source = args.root / ("baseline" if args.control else label)
    report[label + "_sources"] = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("kemeny.mojo", "scalingelections.mojo", "ballots.mojo")
    }
report["same_code_control"] = args.control
output = args.root / ("gpu-control.json" if args.control else "gpu-benchmark.json")
for repeat in range(args.repeats):
    pair = {}
    order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
    for label in order:
        source = args.root / ("baseline" if args.control else label)
        result = subprocess.run(
            [
                str(source / ".venv/bin/python"),
                str(Path(__file__).resolve()),
                "--worker",
                str(source),
                *(["--control"] if args.control else []),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pair[label] = json.loads(result.stdout.strip().splitlines()[-1])
    for b, c in zip(pair["baseline"], pair["candidate"], strict=True):
        assert (b["n"], b["dtype"]) == (c["n"], c["dtype"])
        assert b["result"] == c["result"], (b, c)
    report["pairs"].append(pair)
    output.write_text(json.dumps(report, indent=2))
    print(f"GPU pair {repeat + 1}/{args.repeats} verified", flush=True)

for i, (n, dtype) in enumerate(cases):
    base = [p["baseline"][i]["elapsed_ns"] for p in report["pairs"]]
    candidate = [p["candidate"][i]["elapsed_ns"] for p in report["pairs"]]
    row = {
        "n": n,
        "dtype": dtype,
        "baseline_ms": statistics.median(base) / 1e6,
        "candidate_ms": statistics.median(candidate) / 1e6,
        "paired_speedup": statistics.median(b / c for b, c in zip(base, candidate, strict=True)),
    }
    report["cases"].append(row)
    print(json.dumps(row), flush=True)
output.write_text(json.dumps(report, indent=2))
