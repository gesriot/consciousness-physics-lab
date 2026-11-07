"""Recompute E015 flags and exact-repeat equality from retained artifacts."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import collocation_map as m


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def audit(first, second):
    a, b = [read(p/'map.json.gz') for p in (first, second)]
    fields = ('experiment', 'points', 'batches', 'refinements', 'summary',
              'protocol_sha256', 'source_sha256')
    for field in fields:
        assert a[field] == b[field], field
    assert a['summary'] == m.summarize(a['points'])
    assert read(first/'boundaries.json.gz') == read(second/'boundaries.json.gz')
    for row in a['points']:
        check = m.classify([complex(*v) for v in row['values']], complex(*row['reference']))
        for key, value in check.items():
            assert row[key] == value, (row['coordinate'], key)
    for level in range(4):
        name = f'reference-level-{level}.json.gz'
        assert read(first/name) == read(second/name), name
    assert read(first/'controls.json') == read(second/'controls.json')
    for path, digest in a['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256(Path(m.PROTOCOL).read_bytes()).hexdigest() == a['protocol_sha256']
    return dict(exact_scientific_repeat=True, points=len(a['points']),
                classifications_checked=7*len(a['points']),
                reference_batches_equal=4, source_hashes_match=True,
                excluded_fields=['seconds', 'git_head', 'environment', 'runtime_override', 'protocol_commit'])


if __name__ == '__main__':
    first = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/E015-collocation-map')
    second = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('results/E015-collocation-map-repeat')
    print(json.dumps(audit(first, second), indent=2))
