"""Rebuild the baseline, prefetch-only and grouped CUDA extensions, then compare their Python APIs.

Example on RTX PRO 6000 Blackwell: python experiments/kemeny-cuda-groups/run.py --arch 120
Requires Python 3.12+, uv, a CUDA development toolkit and an NVIDIA GPU with enough memory for an 8 GiB table.
Use --prepare-only to inspect the exact source snapshots without building or running anything.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        default=os.environ.get("SCALING_ELECTIONS_CUDA_ARCH"),
        help="Numeric CUDA target, such as 120 for RTX PRO 6000 Blackwell or 90 for H100",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.arch or "native" in args.arch:
        parser.error("Supply a numeric --arch target; the baseline predates the native-target build fix")
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    sources = json.loads((here / "sources.json").read_text())
    root = Path(tempfile.mkdtemp(prefix="scalingelections-gpu-"))
    print(f"Sources, build logs and results: {root}", flush=True)
    environment = dict(
        os.environ,
        SCALING_ELECTIONS_CUDA_ARCH=args.arch,
        OMP_NUM_THREADS="4",
        OMP_THREAD_LIMIT="4",
        NUMBA_NUM_THREADS="4",
    )
    if not args.prepare_only and shutil.which("nvcc") is None:
        candidate = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
        if not candidate.is_file():
            raise RuntimeError("A CUDA development toolkit with nvcc is required")
        environment["PATH"] = str(candidate.parent) + os.pathsep + os.environ["PATH"]
    for key in ("baseline", "prefetch", "candidate"):
        revision = sources[key]
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode:
            subprocess.run(["git", "fetch", "origin", revision], cwd=repository, check=True)
    archive = subprocess.check_output(["git", "archive", sources["baseline"]], cwd=repository)
    manifest = dict(root=str(root), source_revisions=sources, cuda_arch=args.arch, threads=4, source_sha256={})
    for label, source_key in [("base", "baseline"), ("memory", "prefetch"), ("groups", "candidate")]:
        checkout = root / label
        checkout.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
            snapshot.extractall(checkout, filter="data")
        for name in ("kemeny.cuh", "test.py"):
            data = subprocess.check_output(["git", "show", f"{sources[source_key]}:{name}"], cwd=repository)
            (checkout / name).write_bytes(data)
        manifest["source_sha256"][label] = {
            name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
            for name in ("kemeny.cuh", "test.py", "setup.py", "ballots.cuh", "pixi.lock")
        }
        if args.prepare_only:
            continue
        print(f"Building {label}...", flush=True)
        with (root / f"{label}-build.log").open("w") as log:
            result = subprocess.run(
                ["uv", "sync", "--frozen", "--python", "3.12"],
                cwd=checkout,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{label} build failed; see {root / (label + '-build.log')}")
        extension = subprocess.check_output(
            ["uv", "run", "--no-sync", "python", "-c", "import scalingelections_cuda as cpp; print(cpp.__file__)"],
            cwd=checkout,
            env=environment,
            text=True,
        ).strip()
        manifest[{"base": "baseline_extension", "memory": "memory_extension", "groups": "groups_extension"}[label]] = (
            extension
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared source manifest: {path}", flush=True)
        return
    command = [str(root / "groups/.venv/bin/python"), str(here / "benchmark_api.py"), str(path)]
    subprocess.run(command, cwd=root / "groups", env=environment, check=True)


if __name__ == "__main__":
    main()
