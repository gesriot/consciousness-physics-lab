"""Independently check E011 accounting, endpoint arithmetic and exact replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.extrapolation import summarize

TIMING_KEYS = {'seconds', 'reference_audit_seconds', 'base_execution_seconds', 'decision_seconds', 'hash_seconds',
    'injection_seconds', 'shadow_seconds', 'initial_stale_seconds',
    'operational_and_diagnostic_wall_seconds'}
SCIENTIFIC_KEYS = ('experiment', 'task_seed', 'order_seed', 'paired_tasks', 'sampling_unit',
    'reference_audit', 'intake_channel', 'tasks', 'coverage_steps', 'runs', 'summary', 'work')


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def audit(path):
    data = json.loads(path.read_text())
    rows, tasks, steps = data['runs'], data['tasks'], data['coverage_steps']
    assert len(tasks) == 192 and len(rows) == 1536 and len(steps) == 2304
    assert data['summary'] == summarize(rows, steps, tasks)
    by_task = {t['task_id']: t for t in tasks}
    coverage = {(s['task_id'], s['model'], s['grid']): s for s in steps}
    for row in rows:
        task = by_task[row['task_id']]
        tolerance = task['request']['tolerance']
        error = abs(complex(*row['outcome']['value']) - complex(*task['reference']))
        assert error == row['absolute_error'] and row['accurate'] == (error <= tolerance)
        numerical = row['independent']
        assert numerical['cells_solved'] == sum(numerical['grids'])
        assert numerical['grids'][0] == 64 and numerical['steps'][0]['last_grid'] == 512
        for step in numerical['steps']:
            expected = ('accept' if step['converging'] and step['discrepancy'] + step['uncertainty_estimate'] <= tolerance
                else 'violation_detected' if step['converging'] and step['discrepancy'] - step['uncertainty_estimate'] > tolerance
                else 'inconclusive')
            assert step['verdict'] == expected
            assert step['uncertainty_estimate'] == coverage[row['task_id'], row['model'], step['last_grid']]['uncertainty']
        expected_accept = numerical['verdict'] == 'accept' and row['resolution']['accepted'] and all(row['checks'].values())
        assert row['accepted'] == expected_accept
        assert (row['verdict'] == 'accept') == expected_accept
    for task in tasks:
        for model, values in task['shadow_values'].items():
            reference = complex(*task['model_references'][model])
            for index, grid in enumerate((64,128,256,512,1024,2048,4096)):
                if grid < 512:
                    continue
                value, previous = complex(*values[index]), complex(*values[index-1])
                u = 2*abs(value-previous) + 32*2**-52*grid**2*max(1,abs(value))
                error = abs(value-reference)
                step = coverage[task['task_id'], model, grid]
                assert (u, error, error <= u) == (step['uncertainty'], step['error'], step['covered'])
    assert data['work']['decision_cells'] == sum(r['independent']['cells_solved'] for r in rows)
    assert data['work']['shadow_cells'] == 192*3*8128
    for group, modal, fv in [('old',8192,1024),('second_moment',8192,8192),('high_frequency',4096,4096)]:
        assert data['work'][group]['modal_terms'] == 1536*modal
        assert data['work'][group]['finite_volume_cells'] == 1536*fv
    audit_file = path.with_name(path.stem+'-reference-audit.json')
    intake_file = path.with_name(path.stem+'-intake.json')
    assert hashlib.sha256(audit_file.read_bytes()).hexdigest() == data['reference_audit_sha256']
    assert hashlib.sha256(intake_file.read_bytes()).hexdigest() == data['intake_sha256']
    assert json.loads(audit_file.read_text()) == data['reference_audit']
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    args = parser.parse_args()
    original, repeat = audit(args.original), audit(args.repeat)
    scientific_original = without_timing({k: original[k] for k in SCIENTIFIC_KEYS})
    scientific_repeat = without_timing({k: repeat[k] for k in SCIENTIFIC_KEYS})
    assert scientific_original == scientific_repeat, 'Scientific non-timing replay mismatch'
    print('PASS: both runs satisfy accounting, accuracy, coverage and verdict arithmetic.')
    print('PASS: every scientific field matches exactly, including diagnostics and reference audit.')
    print('Compared top-level fields:', ', '.join(SCIENTIFIC_KEYS))
    print('Excluded timing keys:', ', '.join(sorted(TIMING_KEYS)))
    print('Excluded top-level provenance:', ', '.join(sorted(set(original) - set(SCIENTIFIC_KEYS))))
