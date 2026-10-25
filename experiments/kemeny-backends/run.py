"""Rebuild both source revisions and reproduce the Python and Mojo GPU measurements."""

import argparse
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="120", help="CUDA architecture for the installed GPU")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
repository = here.parents[1]
root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="kemeny-backends-"))
root.mkdir(parents=True, exist_ok=True)
sources = {
    "baseline": "25bf88568cff938c96a777b8776be41677fb1f27",
    "candidate": "3bd6ba382fb3b79f2b563acd65030f774e1fc83a",
}
env = os.environ.copy()
env.update(
    SCALING_ELECTIONS_CUDA_ARCH=args.arch,
    NUMBA_NUM_THREADS="4",
    OMP_NUM_THREADS="4",
    SCALINGELECTIONS_TESTS_SEED="42",
)


def run(command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


for label, revision in sources.items():
    source = root / label
    source.mkdir()
    data = subprocess.check_output(["git", "archive", revision], cwd=repository)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(source, filter="data")
    run(["uv", "sync", "--locked", "--extra", "cpu", "--no-install-project", "--python", "3.12"], source)
    python = str(source / ".venv/bin/python")
    run(["uv", "pip", "install", "--python", python, "setuptools>=83", "pybind11>=3.0.3"], source)
    run([python, "setup.py", "build_ext", "--inplace"], source)
    run(["pixi", "install", "--locked"], source)
    run(["pixi", "run", "build"], source)

for name in ("benchmark_gpu.py", "benchmark_python.py"):
    shutil.copy2(here / name, root / name)
candidate = root / "candidate"
python = str(candidate / ".venv/bin/python")
run(["pixi", "run", "uv", "run", "--no-sync", "pytest", "test.py", "-q"], candidate)
run(
    [
        python,
        str(root / "benchmark_python.py"),
        "--baseline",
        str(root / "baseline"),
        "--candidate",
        str(candidate),
        "--output",
        str(root / "python-final-benchmark.json"),
    ],
    candidate,
)
run(["pixi", "run", python, str(root / "benchmark_gpu.py"), "--root", str(root)], candidate)
run(
    ["pixi", "run", python, str(root / "benchmark_gpu.py"), "--root", str(root), "--control", "--repeats", "5"],
    candidate,
)
print(f"Results: {root}")
