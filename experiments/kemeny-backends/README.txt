Kemeny backend comparison, October 25, 2026

Baseline: 25bf88568cff938c96a777b8776be41677fb1f27
Candidate: 3bd6ba382fb3b79f2b563acd65030f774e1fc83a
Review commit: 0988826f628e59ef0f7304eb6f9664bcc5ab8c8f

The review commit has the identical source tree. Only the commit messages
were expanded to explain the intent and checks.

The baseline includes Ash's device-buffer, GPU traceback and HIP changes.
The candidate preserves that commit, merges current main, adds grouped Mojo
GPU execution and optimizes the narrower Python score paths and subset sums.

Test platform:
  NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
  AMD EPYC 9555, four Numba/OpenMP threads
  CUDA 13.0.88, GCC 13.3.0, Mojo 1.1.0, MAX 26.6.0

Reproduction needs Python 3.12, uv, pixi, a supported GPU and its CUDA SDK.
With nvcc on PATH, run from this checkout:

  python experiments/kemeny-backends/run.py --arch 120 --output /tmp/kemeny-backends-run

The runner checks out the pinned source trees into that empty output directory,
builds both native libraries, runs the project tests and records new samples.
It does not create or manage any cloud resources.

gpu-benchmark.json contains nine alternating baseline/candidate process pairs.
Each case warms a complete public API call before the timed call. Isolated
processes keep the two versions' GPU memory pools independent. Ranking, score,
winners and multiplicity must agree for every measured pair.

gpu-control.json runs the same baseline source on both sides for five pairs.
Its median paired speedup is 1.00x.

python-final-benchmark.json contains nine alternating call pairs after JIT warmup.
The entire cost table is compared for every case before timing. The script uses
an isolated Numba cache so loading the two modules does not alter project caches.

Times are medians; speedups are medians of paired ratios. Those statistics need
not divide exactly. Small Python fields are roughly flat; larger uint32 cases
improve, and the uint64 path retains its vectorizable candidate loop.

The sanitizer transcripts cover complete Mojo GPU cost tables at 1, 7, 8, 9,
15, 16 and 17 candidates, with ties and strict profiles, at every score width.
This validation used NVIDIA hardware; AMD and Apple execution was not measured.

Documentation follow-up: 01d8252d30e068c937b4191762366a9b08813523
This restores launch-order wording without changing the calculation code.
