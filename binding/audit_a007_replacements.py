"""Count archived post-alarm replacements without rerunning or changing A007."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from attention.compare_a004 import Archive


def audit(path):
    groups = defaultdict(lambda: Counter({
        name: 0 for name in ('replacements', 'guarded_accept',
        'guarded_violation_detected', 'guarded_inconclusive', 'unguarded_accept',
        'unguarded_violation_detected', 'unguarded_inconclusive', 'closure_accept',
        'closure_inconclusive', 'closure_violation_detected', 'guard_passed',
        'replacement_cost', 'closures_requiring_replacement')}))
    total = Counter()
    packages = 0
    for package in Archive(path).packages():
        packages += 1
        for fault, fractions in package['conditions'].items():
            for fraction, policies in fractions.items():
                for policy, run in policies.items():
                    key = (fault, fraction, policy)
                    counts = groups[key]
                    counts['runs'] += 1
                    suspect = set()
                    for index, row in enumerate(run['actions']):
                        if row.get('sample'):
                            if row['monitor']['mismatch']:
                                suspect.add(row['channel'])
                            continue
                        if not row.get('replacement'):
                            continue
                        assert row['channel'] in suspect, (key, index, 'no prior alarm')
                        obs = row['observation']
                        assert obs == row['original_observation'] == row['hooked_observation']
                        assert row['channel'].split(':')[0] in ('v2', 'collocation')
                        assert obs['grids'] == [4096, 8192, 16384, 32768]
                        counts['replacements'] += 1
                        counts['guarded_' + obs['verdict']] += 1
                        counts['unguarded_' + obs['unguarded_verdict']] += 1
                        counts['guard_passed'] += int(obs['safety']['passed'])
                        counts['replacement_cost'] += row['cost']
                    assert suspect == set(run['suspect'])
                    for outcome in run['inputs']:
                        if outcome['required_replacements']:
                            counts['closures_requiring_replacement'] += 1
                            counts['closure_' + outcome['verdict']] += 1
                    assert counts['guarded_accept'] + counts['guarded_violation_detected'] + counts['guarded_inconclusive'] == counts['replacements']
    rows = []
    for (fault, fraction, policy), counts in sorted(groups.items()):
        total.update(counts)
        rows.append(dict(fault=fault, fraction=fraction, policy=policy, **dict(sorted(counts.items()))))
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return dict(source=str(path), source_sha256=sha.hexdigest(), packages=packages,
                unit='Dependent action records across policies, faults and budgets; not independent trials',
                checks=dict(prior_channel_alarm=True, clean_unspoiled_replacement=True,
                            guarded_uniform_grids=True, final_suspect_state=True),
                total=dict(sorted(total.items())), groups=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(packages=result['packages'], total=result['total']), sort_keys=True))


if __name__ == '__main__':
    main()
