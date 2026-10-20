"""Alternate installed Python API calls across the exact baseline and two GPU variants."""

import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(manifest["root"])
sys.path.insert(0, str(root / "groups"))

np = importlib.import_module("numpy")
se = importlib.import_module("scalingelections")

modules = {}
for label, key in [("base", "baseline_extension"), ("memory", "memory_extension"), ("groups", "groups_extension")]:
    spec = importlib.util.spec_from_file_location(f"{label}.scalingelections_cuda", manifest[key])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "gpu" in module.available_backends()
    modules[label] = module


def select(label):
    sys.modules["scalingelections_cuda"] = modules[label]


def call(counts, score_type):
    return se.compute_kemeny_ranking(counts, implementation="cpp", backend="gpu", score_type=score_type)


fixtures = []
for seed in (42, 73):
    for dtype in ("uint32", "uint64"):
        for n in (20, 24, 28):
            fixtures.append((n, dtype, seed, "random"))
for dtype in ("uint32", "uint64"):
    fixtures.append((30, dtype, 42, "random"))
for n in (1, 4, 8, 9, 12, 16):
    fixtures.append((n, "uint32", 42, "random"))
fixtures.extend([(24, "uint32", 42, "tied"), (24, "uint32", 42, "unanimous")])

samples, summary = [], []


def describe(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


metadata = dict(
    gpu=describe(["nvidia-smi", "-L"]),
    compiler=describe(["nvcc", "--version"]),
    host_compiler=describe([os.environ.get("CUDAHOSTCXX", "g++"), "--version"]),
    python=platform.python_version(),
    numpy=np.__version__,
    threads=os.environ.get("OMP_NUM_THREADS"),
    source=manifest,
    scope="complete public Python compute_kemeny_ranking calls, including validation and destruction",
    procedure="module selected before timing; nine alternating pairs; every returned field checked",
)
output = root / "api-benchmark.json"
for n, dtype, seed, profile in fixtures:
    rng = np.random.default_rng(seed)
    bound = 100 if dtype == "uint32" else 350000000
    counts = rng.integers(0, bound + 1, (n, n), dtype=dtype)
    if profile in ("tied", "unanimous"):
        counts.fill(7)
        if profile == "unanimous":
            counts = np.triu(counts, k=1)
    np.fill_diagonal(counts, 0)
    select("base")
    expected = call(counts, dtype)
    for label in ("memory", "groups"):
        select(label)
        assert call(counts, dtype) == expected
    experiments = [("memory", "base", "memory"), ("groups_added", "memory", "groups"), ("combined", "base", "groups")]
    if seed == 42 and profile == "random" and n in (20, 24, 28, 30):
        experiments.insert(0, ("control", "base", "base"))
    for kind, before, after in experiments:
        for _ in range(2):
            for label in (before, after):
                select(label)
                assert call(counts, dtype) == expected
        before_samples, after_samples, ratios = [], [], []
        repeats = 5 if kind == "control" else 9
        for repeat in range(repeats):
            pair = {}
            order = [("before", before), ("after", after)]
            if repeat % 2:
                order.reverse()
            for path, label in order:
                select(label)
                start = time.perf_counter_ns()
                actual = call(counts, dtype)
                elapsed = time.perf_counter_ns() - start
                assert actual == expected
                pair[path] = elapsed
                samples.append(
                    dict(
                        kind=kind,
                        candidates=n,
                        score_type=dtype,
                        seed=seed,
                        profile=profile,
                        repeat=repeat,
                        path=path,
                        variant=label,
                        ns=elapsed,
                    )
                )
            ratios.append(pair["before"] / pair["after"])
            before_samples.append(pair["before"] / 1e6)
            after_samples.append(pair["after"] / 1e6)
        bootstrap = np.random.default_rng(42).choice(ratios, (5000, repeats), replace=True)
        interval = np.quantile(np.median(bootstrap, axis=1), [0.025, 0.975])
        row = dict(
            kind=kind,
            candidates=n,
            score_type=dtype,
            seed=seed,
            profile=profile,
            pairs=repeats,
            baseline_ms=statistics.median(before_samples),
            candidate_ms=statistics.median(after_samples),
            paired_speedup=statistics.median(ratios),
            paired_bootstrap_interval=interval.tolist(),
        )
        summary.append(row)
        output.write_text(json.dumps(dict(metadata=metadata, summary=summary, samples=samples), indent=2) + "\n")
        print(json.dumps(row), flush=True)
print(f"Raw API benchmark: {output}", flush=True)
