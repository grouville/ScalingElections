"""Compare complete Python Kemeny calls with alternating baseline/candidate pairs."""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sizes", type=int, nargs="+", default=[12, 18, 22])
parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
parser.add_argument("--repeats", type=int, default=9)
args = parser.parse_args()
cache = tempfile.TemporaryDirectory(prefix="kemeny-benchmark-cache-")
os.environ["NUMBA_CACHE_DIR"] = cache.name
sys.path.insert(0, str(args.candidate))
numba = importlib.import_module("numba")
np = importlib.import_module("numpy")
ScoreType = importlib.import_module("ballots").ScoreType


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path / "kemeny.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


modules = {
    "baseline": load("kemeny_baseline_benchmark", args.baseline),
    "candidate": load("kemeny_candidate_benchmark", args.candidate),
}
report = {
    "platform": platform.platform(),
    "python": sys.version,
    "numpy": np.__version__,
    "numba": numba.__version__,
    "seed": 42,
    "scope": "complete compute_kemeny_ranking calls after JIT warmup, alternating pairs",
    "rows": [],
}

for score_type in (ScoreType.uint32, ScoreType.uint64):
    for threads in args.threads:
        numba.set_num_threads(threads)
        for n in args.sizes:
            counts = np.random.default_rng(42 + n).integers(0, 100, (n, n), dtype=np.uint32)
            np.fill_diagonal(counts, 0)
            tables = {
                label: module.compute_kemeny_costs(counts, score_type=score_type) for label, module in modules.items()
            }
            np.testing.assert_array_equal(tables["baseline"], tables["candidate"])
            del tables
            expected = modules["baseline"].compute_kemeny_ranking(counts, score_type=score_type)
            samples = {"baseline": [], "candidate": []}
            for repeat in range(args.repeats):
                order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                for label in order:
                    start = time.perf_counter_ns()
                    actual = modules[label].compute_kemeny_ranking(counts, score_type=score_type)
                    elapsed = time.perf_counter_ns() - start
                    assert actual == expected
                    samples[label].append(elapsed)
            row = {
                "n": n,
                "score_type": score_type.value,
                "threads": threads,
                "samples_ns": samples,
                "baseline_ms": statistics.median(samples["baseline"]) / 1e6,
                "candidate_ms": statistics.median(samples["candidate"]) / 1e6,
                "paired_speedup": statistics.median(
                    b / c for b, c in zip(samples["baseline"], samples["candidate"], strict=True)
                ),
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != "samples_ns"}), flush=True)
