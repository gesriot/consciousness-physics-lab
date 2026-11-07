#!/usr/bin/env python3
"""Compare exact repeats or cross-platform decisions with audited provenance."""
import argparse
import json
import math
from pathlib import Path

# Only top-level execution provenance is separated. Protocol identity, input
# digests, seeds, decisions, labels and counters remain in the scientific data.
PROVENANCE = {'environment', 'git_head', 'source_sha256', 'runtime_override', 'replay'}


def stable(value, ignore_git_head=False):
    if isinstance(value, dict):
        return {key: stable(item, ignore_git_head) for key, item in value.items()
                if not key.endswith('seconds') and not (ignore_git_head and key == 'git_head')}
    if isinstance(value, list):
        return [stable(item, ignore_git_head) for item in value]
    return value


def differences(left, right, relative_tolerance=0.0, path='$'):
    """Integers and booleans are exact; only finite floats use relative error."""
    if type(left) is not type(right):
        return [path + ': type differs']
    if isinstance(left, dict):
        result = [path + '.' + key + ': missing key' for key in sorted(left.keys() ^ right.keys())]
        for key in sorted(left.keys() & right.keys()):
            result.extend(differences(left[key], right[key], relative_tolerance, path + '.' + key))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [path + ': length differs']
        return [issue for index, (a, b) in enumerate(zip(left, right))
                for issue in differences(a, b, relative_tolerance, f'{path}[{index}]')]
    if isinstance(left, float):
        equal = (math.isfinite(left) and math.isfinite(right) and
                 math.isclose(left, right, rel_tol=relative_tolerance, abs_tol=0.0))
    else:
        equal = left == right
    return [] if equal else [path + ': value differs']


def compare(original, repeat, mode='exact', relative_tolerance=1e-12, ignore_git_head=False):
    if mode not in ('exact', 'decisions'):
        raise ValueError('Unknown comparison mode')
    if not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError('Relative tolerance must be finite and in [0, 1)')
    left, right = stable(original, ignore_git_head), stable(repeat, ignore_git_head)
    provenance = differences({k: left[k] for k in PROVENANCE if k in left},
                             {k: right[k] for k in PROVENANCE if k in right})
    if mode == 'decisions':
        left = {k: v for k, v in left.items() if k not in PROVENANCE}
        right = {k: v for k, v in right.items() if k not in PROVENANCE}
    return differences(left, right, relative_tolerance if mode == 'decisions' else 0.0), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--mode', choices=('exact', 'decisions'), default='exact')
    parser.add_argument('--rtol', type=float, default=1e-12,
                        help='Relative tolerance for floats in decisions mode; absolute tolerance is zero')
    parser.add_argument('--ignore-git-head', action='store_true',
                        help='Only for reproduction from a later documentation/results commit')
    args = parser.parse_args()
    original = json.loads(args.original.read_text())
    repeat = json.loads(args.repeat.read_text())
    try:
        failures, provenance = compare(original, repeat, args.mode, args.rtol, args.ignore_git_head)
    except ValueError as error:
        parser.error(str(error))
    print('Provenance: ' + ('IDENTICAL' if not provenance else 'DIFFERENT'))
    for issue in provenance:
        print('  ' + issue)
    if failures:
        for issue in failures[:50]:
            print('  ' + issue)
        raise SystemExit(f'FAIL: {len(failures)} non-timing differences')
    if args.mode == 'exact':
        print('PASS: exact non-timing comparison' + (' excluding git_head.' if args.ignore_git_head else '.'))
    else:
        print(f'PASS: decisions identical; floats rtol={args.rtol:g}, atol=0; integers, booleans, '
              'strings, structure, protocol identity and input digests exact.')
        print('Execution provenance compared separately above; this is not an exact reproduction claim.')


if __name__ == '__main__':
    main()
