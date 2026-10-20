"""Summarize complete CUDA solves, retaining the raw paired samples."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

path = Path(sys.argv[1])
groups = defaultdict(dict)
for row in csv.DictReader(path.open()):
    key = (row['kind'], int(row['candidates']))
    groups[key].setdefault(int(row['repeat']), {})[row['path']] = int(row['nanoseconds'])
assert len(groups) == 8, 'Incomplete benchmark output'
records = []
for (kind, n), group in sorted(groups.items()):
    assert len(group) == 7 and all(set(pair) == {'baseline', 'candidate'} for pair in group.values())
    pairs = list(group.values())
    ratios = [pair['baseline'] / pair['candidate'] for pair in pairs]
    record = {'kind': kind, 'candidates': n,
              'baseline_median_ms': statistics.median(pair['baseline'] for pair in pairs) / 1e6,
              'candidate_median_ms': statistics.median(pair['candidate'] for pair in pairs) / 1e6,
              'median_paired_speedup': statistics.median(ratios),
              'paired_speedup_range': [min(ratios), max(ratios)]}
    records.append(record)
    print(f"{kind:10s} n={n:2d} baseline={record['baseline_median_ms']:.3f}ms "
          f"candidate={record['candidate_median_ms']:.3f}ms "
          f"paired={record['median_paired_speedup']:.3f}x")
path.with_suffix('.summary.json').write_text(json.dumps(records, indent=2) + '\n')
