#!/usr/bin/env python3
"""Compare exact repeats or cross-platform decisions with audited provenance."""
import argparse
import json
import math
import hashlib
from pathlib import Path

PROVENANCE = {'environment', 'git_head', 'source_sha256', 'runtime_override', 'replay',
              'reference_audit_sha256', 'intake_sha256'}
# These summaries describe the full diagnostic path, never work used to decide.
SHADOW_SUMMARIES = {'accurate_shadow_accept_lost', 'futility_accurate_shadow_accept_lost',
                    'futility_lost_shadow_violations', 'shadow_reversals'}


def stable(value, ignore_git_head=False):
    if isinstance(value, dict):
        return {key: stable(item, ignore_git_head) for key, item in value.items()
                if not key.endswith('seconds') and not (ignore_git_head and key == 'git_head')}
    if isinstance(value, list):
        return [stable(item, ignore_git_head) for item in value]
    return value


def differences(left, right, relative_tolerance=0.0, path='$', absolute_tolerance=0.0):
    """Integers and booleans are exact; finite floats use atol + rtol * scale."""
    if type(left) is not type(right):
        return [path + ': type differs']
    if isinstance(left, dict):
        result = [path + '.' + key + ': missing key' for key in sorted(left.keys() ^ right.keys())]
        for key in sorted(left.keys() & right.keys()):
            result.extend(differences(left[key], right[key], relative_tolerance,
                                      path + '.' + key, absolute_tolerance))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [path + ': length differs']
        return [issue for index, (a, b) in enumerate(zip(left, right))
                for issue in differences(a, b, relative_tolerance, f'{path}[{index}]', absolute_tolerance)]
    if isinstance(left, float):
        equal = (math.isfinite(left) and math.isfinite(right) and
                 abs(left - right) <= absolute_tolerance + relative_tolerance * max(abs(left), abs(right)))
    else:
        equal = left == right
    return [] if equal else [path + ': value differs']


def separate_diagnostics(data):
    """Separate only registered non-operational paths; retain unknown fields.

    Operational steps (including convergence flags and fallback indicators),
    accuracy labels, accepted/verdict fields, stop reasons and work counters
    remain authoritative. A diagnostic difference does not establish full
    scientific equivalence, even when all operational decisions match.
    """
    diagnostic = {}
    experiment = data.get('experiment')
    def walk(value, path='$'):
        if isinstance(value, dict):
            kept = {}
            for key, item in value.items():
                current = path + '.' + key
                selected = (
                    (path == '$' and key == 'coverage_steps' and experiment in ('E007', 'E009')) or
                    (path.startswith('$.runs[') and key == 'shadow' and experiment in ('E007', 'E009')) or
                    (path.startswith('$.tasks[') and key == 'shadow_values' and experiment in ('E007', 'E009')) or
                    (path.startswith('$.summary.') and experiment in ('E007', 'E009') and
                     (key == 'coverage' or key in SHADOW_SUMMARIES)))
                if selected:
                    diagnostic[current] = item
                else:
                    kept[key] = walk(item, current)
            return kept
        if isinstance(value, list):
            return [walk(item, f'{path}[{index}]') for index, item in enumerate(value)]
        return value
    return walk(data), diagnostic


def canonical_inputs(data):
    """Exact scientific intake, independent of file formatting and environment."""
    if 'intake_channel' in data:
        return data['intake_channel']
    return [{key: row[key] for key in ('task_id', 'cohort', 'request', 'request_sha256',
                                     'sampled_magnitude', 'sampled_phase') if key in row}
            for row in data.get('tasks', [])]


def input_digest(data):
    raw = json.dumps(canonical_inputs(data), sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def decision_view(value, path='$'):
    """Ignore measured float magnitudes, never inputs, labels, counts or structure."""
    if isinstance(value, dict):
        return {k: (v if k in ('request', 'intake_channel') else decision_view(v, path + '.' + k))
                for k, v in value.items()}
    if isinstance(value, list):
        return [decision_view(v, path + '[]') for v in value]
    if isinstance(value, float):
        return ('measured floating-point quantity',)
    return value


def numeric_summary(left, right):
    """Maxima per field path (array indices collapsed), with nonfinite counts."""
    result = {}
    def walk(a, b, path='$'):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(a.keys() & b.keys()):
                if not k.endswith('seconds'):
                    walk(a[k], b[k], path + '.' + k)
        elif isinstance(a, list) and isinstance(b, list):
            for x, y in zip(a, b):
                walk(x, y, path + '[]')
        elif isinstance(a, float) and isinstance(b, float):
            row = result.setdefault(path, dict(count=0, changed=0, max_abs=0.0,
                                               max_rel=0.0, nonfinite=0))
            row['count'] += 1
            row['changed'] += a != b
            if math.isfinite(a) and math.isfinite(b):
                delta = abs(a-b)
                scale = max(abs(a), abs(b))
                row['max_abs'] = max(row['max_abs'], delta)
                row['max_rel'] = max(row['max_rel'], delta/scale if scale else 0.0)
            else:
                row['nonfinite'] += 1
    walk(left, right)
    return result


def compare_details(original, repeat, mode='exact', relative_tolerance=1e-12,
                    ignore_git_head=False, absolute_tolerance=1e-12):
    if mode not in ('exact', 'decisions'):
        raise ValueError('Unknown comparison mode')
    if not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError('Relative tolerance must be finite and in [0, 1)')
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError('Absolute tolerance must be finite and nonnegative')
    left, right = stable(original, ignore_git_head), stable(repeat, ignore_git_head)
    provenance = differences({k: left[k] for k in PROVENANCE if k in left},
                             {k: right[k] for k in PROVENANCE if k in right})
    diagnostic = []
    if mode == 'decisions':
        left = {k: v for k, v in left.items() if k not in PROVENANCE}
        right = {k: v for k, v in right.items() if k not in PROVENANCE}
        left, left_diagnostic = separate_diagnostics(left)
        right, right_diagnostic = separate_diagnostics(right)
        diagnostic = differences(left_diagnostic, right_diagnostic, 0.0, '$diagnostic', 0.0)
        input_failures = differences(canonical_inputs(left), canonical_inputs(right),
                                     path='$canonical_inputs')
        left, right = decision_view(left), decision_view(right)
    else:
        input_failures = []
    return (input_failures + differences(left, right, relative_tolerance if mode == 'decisions' else 0.0,
                        absolute_tolerance=absolute_tolerance if mode == 'decisions' else 0.0),
            provenance, diagnostic)


def compare(original, repeat, mode='exact', relative_tolerance=1e-12,
            ignore_git_head=False, absolute_tolerance=1e-12):
    """Compatibility interface; use compare_details to inspect diagnostics too."""
    failures, provenance, _ = compare_details(original, repeat, mode, relative_tolerance,
                                              ignore_git_head, absolute_tolerance)
    return failures, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--mode', choices=('exact', 'decisions'), default='exact')
    parser.add_argument('--rtol', type=float, default=1e-12,
                        help='Legacy compatibility option; decisions never fail on measured floats')
    parser.add_argument('--atol', type=float, default=1e-12,
                        help='Legacy compatibility option; decisions never fail on measured floats')
    parser.add_argument('--ignore-git-head', action='store_true',
                        help='Only for reproduction from a later documentation/results commit')
    args = parser.parse_args()
    original = json.loads(args.original.read_text())
    repeat = json.loads(args.repeat.read_text())
    try:
        failures, provenance, diagnostic = compare_details(
            original, repeat, args.mode, args.rtol, args.ignore_git_head, args.atol)
    except ValueError as error:
        parser.error(str(error))
    print('Provenance: ' + ('IDENTICAL' if not provenance else 'DIFFERENT'))
    for issue in provenance:
        print('  ' + issue)
    if args.mode == 'decisions':
        print('Canonical input digests: ' + input_digest(original) + ' / ' + input_digest(repeat))
        print('Numerical differences by field:')
        for field, summary in numeric_summary(original, repeat).items():
            print('  ' + field + ': ' + json.dumps(summary, sort_keys=True))
        print(f'Diagnostic differences: {len(diagnostic)} (not operational failures)')
        print('E007/E009 full shadow trajectories and coverage summaries are diagnostic. '
              'Order estimation is unstable near rounded differences; shadow verdicts, '
              'convergence flags and coverage fallback counts may change without altering '
              'operational decisions. Other differences still require scientific review.')
        for issue in diagnostic:
            print('  ' + issue)
    if failures:
        for issue in failures[:50]:
            print('  ' + issue)
        raise SystemExit(f'FAIL: {len(failures)} non-timing authoritative differences')
    if args.mode == 'exact':
        print('PASS: exact non-timing comparison' + (' excluding git_head.' if args.ignore_git_head else '.'))
    else:
        print('PASS: operational decisions identical; measured floats reported without a pass/fail '
              'threshold; authoritative integers, booleans, strings, structure, protocol '
              'identity and canonical inputs exact.')
        print('Diagnostic differences and execution provenance are reported separately; '
              'this is not an exact reproduction or complete scientific-equivalence claim.')


if __name__ == '__main__':
    main()
