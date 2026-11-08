#!/usr/bin/env python3
"""Independently replay all archived model-gate decisions using fresh probes."""
import argparse
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
    return dict(purpose='Fresh original uniform E008/E010 probes and assembled module versus archived gates',
                comparisons=reports, passed=all(r['differences'] == 0 for r in reports),
                scope='Model/request gate booleans; neither held-out generation nor numerical V2 or third replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; preserve earlier audit')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'comparisons'}, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('comparisons', 'mismatches')}
                      for r in result['comparisons']], indent=2))
    if not result['passed']:
        raise SystemExit(1)
