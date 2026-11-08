"""Bounded series-at-a-time comparison of decisions and diagnostics."""
import argparse
import json
from pathlib import Path
from attention.io import read_json
from attention.a007 import atomic_json


def decision_projection(value):
    if isinstance(value, list):
        return [decision_projection(v) for v in value]
    if isinstance(value, dict):
        if 'verdict' in value and 'step' in value:
            return dict(verdict=value['verdict'], passed=value.get('safety', {}).get('passed'))
        return {k: decision_projection(v) for k, v in value.items()
                if k not in ('standard_error', 'precision', 'recall')}
    return value


def compare(left, right):
    left, right = Path(left), Path(right)
    lm, rm = read_json(left/'manifest.json'), read_json(right/'manifest.json')
    if lm['intake_sha256'] != rm['intake_sha256'] or lm['freeze_sha256'] != rm['freeze_sha256']:
        raise ValueError('Mismatched intake or freeze')
    decisions, diagnostics = [], []
    for i in range(12):
        for fault in ('none', 'permissive', 'underestimated'):
            for package in range(4):
                filename = f'{fault}-{package}.json.gz'
                a, b = (read_json(root/f'series-{i:02d}'/filename)['result'] for root in (left, right))
                def upstream(r):
                    return dict(inputs=[{k:x[k] for k in ('verdict', 'closure_action_index', 'required_replacements')} for x in r['inputs']],
                        actions=[dict(input_index=x['input_index'], channel=x['channel'], sample=x['sample'],
                                      replacement=x['replacement'], verdict=x['observation']['verdict'],
                                      terminal=x['observation'].get('terminal'), cost=x['cost']) for x in r['actions']],
                        alerts=r['alerts'], suspect=r['suspect'])
                if upstream(a) != upstream(b): decisions.append([i, filename])
                elif a != b: diagnostics.append([i, filename])
        for filename in ('recalls.json.gz', 'ledger.json.gz', 'summary.json'):
            a, b = read_json(left/f'series-{i:02d}'/filename), read_json(right/f'series-{i:02d}'/filename)
            # Raw trace hashes include floating diagnostics and are reported separately.
            if filename == 'ledger.json.gz':
                ah, bh = a.pop('package_hashes'), b.pop('package_hashes')
                if ah != bh: diagnostics.append([i, 'upstream_trace_hashes'])
            if decision_projection(a) != decision_projection(b): decisions.append([i, filename])
            elif a != b: diagnostics.append([i, filename])
    return dict(decisions_pass=not decisions, differing_decisions=decisions,
                diagnostics_pass=not diagnostics, differing_diagnostics=diagnostics,
                series_compared=12, intake_pass=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('left'); p.add_argument('right'); p.add_argument('--output', required=True)
    a = p.parse_args(); result = compare(a.left, a.right); atomic_json(a.output, result)
    print(json.dumps(result, indent=2))
    if not result['decisions_pass']: raise SystemExit(1)
