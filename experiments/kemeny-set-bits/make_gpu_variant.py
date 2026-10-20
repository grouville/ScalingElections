"""Create a GPU-only experimental header from the actual baseline header."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'kemeny.cuh'
text = source.read_text()
start = text.index('__global__ void kemeny_layer_cuda_')
old = '''    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        subset_mask_t const bit = subset_mask_t {1} << candidate;
        if (!(subset & bit)) continue;'''
new = '''    for (subset_mask_t members = subset; members; members &= members - 1) {
        candidate_index_t const candidate = static_cast<candidate_index_t>(__ffsll(members) - 1);
        subset_mask_t const bit = subset_mask_t {1} << candidate;'''
assert text[start:].count(old) == 1
changed = text[:start] + text[start:].replace(old, new, 1)
destination = root / 'gpu-set-bits' / 'kemeny.cuh'
destination.parent.mkdir(exist_ok=True)
destination.write_text(changed)
print('Created experimental GPU-only header:', destination, file=sys.stderr)
