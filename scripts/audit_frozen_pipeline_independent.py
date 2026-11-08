#!/usr/bin/env python3
"""Independently replay all archived model-gate decisions using fresh probes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import geometry_invariants as geometry
from lab import matched_mixture, frozen_pipeline
from lab.guard import Request
from lab.executor import request_hash


def run():
    original_fv = geometry.fv_test_hook
    original_modal = geometry.modal_test_hook
    reports = []
    for name, intake_key in (('E011-extrapolation', 'tasks'),
                             ('E013-matched-mixture', 'intake_channel')):
        path = ROOT / 'results' / name / 'results.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        tasks = {r['task_id']: r for r in data[intake_key]}
        seen, details, differences = set(), [], []
        for row in data['runs']:
            model = row.get('model', row.get('condition'))
            model = 'sphere' if model == 'clean' else model
            key = (row['task_id'], model)
            if key in seen:
                continue
            seen.add(key)
            fields = dict(tasks[row['task_id']]['request'])
            fields['z'] = complex(*fields['z'])
            request = Request(**fields)
            def uniform(solver, z, t, n, d):
                if model == 'mixture':
                    return sum(w * solver(z, t*x, n, 1)
                               for w, x, _ in matched_mixture.components('mixture'))
                return solver(z, t*(1000 if model == 'model_scale' else 1), n,
                              1 if model == 'slab' else 3)
            with patch('lab.geometry_invariants.modal_test_hook',
                       side_effect=lambda z,t,n,d: uniform(original_modal,z,t,n,d)), \
                 patch('lab.geometry_invariants.fv_test_hook',
                       side_effect=lambda z,t,n,d: uniform(original_fv,z,t,n,d)):
                probes = geometry.probes(request, request.mean_time, 3)
            digest = request_hash(request)
            independent = frozen_pipeline.checks_from_probes(probes, digest, row['original_digest'])
            assembled = frozen_pipeline.checks_from_probes(
                frozen_pipeline.model_probes(request, model), digest, row['original_digest'])
            expected = row['checks']
            record = dict(task_id=row['task_id'], model=model, expected=expected,
                          independent_uniform=independent, assembled=assembled,
                          passed=independent == expected == assembled)
            details.append(record)
            if not record['passed']:
                differences.append(record)
        reports.append(dict(archive=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(raw).hexdigest(), archived_rows=len(data['runs']),
            unique_task_model_inputs=len(seen), differences=len(differences),
            mismatches=differences, comparisons=details))
    v2 = audit_v2()
    return dict(v2=v2, purpose='Fresh original uniform E008/E010 probes and assembled module versus archived gates',
                comparisons=reports, passed=all(r['differences'] == 0 for r in reports) and v2['discrete_mismatches'] == 0,
                scope='Fresh archived model gates and E011 V2; no held-out generation')


def audit_v2():
    source = ROOT / 'results/E011-extrapolation/results.json'
    expected_path = ROOT / 'results/E014-convergence-candidates/first/cost-replay.json.gz'
    data = json.loads(source.read_bytes())
    with gzip.open(expected_path, 'rt') as handle:
        expected_data = json.load(handle)
    expected = {(r['task_id'], r['fault']): r['candidates']['C1']
                for r in expected_data['rows'] if r['experiment'] == 'E011'}
    tasks = {r['task_id']: r for r in data['tasks']}
    results = []
    for row in data['runs']:
        fields = dict(tasks[row['task_id']]['request'])
        fields['z'] = complex(*fields['z'])
        request = Request(**fields)
        actual = frozen_pipeline.check_v2(request, complex(*row['outcome']['value']), row['model'])
        old = expected[row['task_id'], row['fault']]
        issues = [key for key in ('verdict', 'stop_reason', 'cells_solved') if actual[key] != old[key]]
        if len(actual['steps']) != len(old['steps']):
            issues.append('step_count')
        numerical = []
        for index, (a, b) in enumerate(zip(actual['steps'], old['steps'])):
            issues.extend(f'steps[{index}].{key}' for key in
                          ('last_grid', 'converging', 'verdict', 'futile', 'blocked_rounding')
                          if a[key] != b[key])
            for key in ('uncertainty_estimate', 'discrepancy'):
                delta = float(abs(a[key] - b[key]))
                numerical.append(dict(step=index, field=key, absolute_difference=delta,
                    within_tolerance=bool(delta <= 1e-12 + 1e-12 * max(abs(a[key]), abs(b[key])))))
        results.append(dict(task_id=row['task_id'], fault=row['fault'], model=row['model'],
            expected_verdict=old['verdict'], actual_verdict=actual['verdict'],
            expected_cells=old['cells_solved'], actual_cells=actual['cells_solved'],
            discrete_differences=issues, numerical_comparisons=numerical))
    numeric = [n for r in results for n in r['numerical_comparisons']]
    return dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_source=str(expected_path.relative_to(ROOT)),
        expected_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        rows=len(results), discrete_mismatches=sum(bool(r['discrete_differences']) for r in results),
        numerical_fields=len(numeric), nonexact_numerical_fields=sum(n['absolute_difference'] != 0 for n in numeric),
        numerical_fields_outside_tolerance=sum(not n['within_tolerance'] for n in numeric),
        max_absolute_difference=max((n['absolute_difference'] for n in numeric), default=0),
        comparisons=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; preserve earlier audit')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons', 'v2')}, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('comparisons', 'mismatches')}
                      for r in result['comparisons']], indent=2))
    print(json.dumps({k:v for k,v in result['v2'].items() if k != 'comparisons'}, indent=2))
    if not result['passed']:
        raise SystemExit(1)
