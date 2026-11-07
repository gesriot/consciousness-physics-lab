"""Retrospective E005/E006 analysis; no held-out claims."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from lab.scientist import ROOT, analytic


def verdict(step, tolerance):
    if step['converging']:
        if step['discrepancy'] + step['uncertainty_estimate'] <= tolerance:
            return 'accept'
        if step['discrepancy'] - step['uncertainty_estimate'] > tolerance:
            return 'violation_detected'
    return 'inconclusive'


def run():
    expected = ('3.12.14', '2.3.5', '1.17.0')
    actual = (platform.python_version(), np.__version__, scipy.__version__)
    if actual != expected:
        raise RuntimeError(f'Pinned environment required: {expected}, got {actual}')
    sources = ['results/tolerance-allocation/results.json', 'results/fault-heldout/results.json']
    data = [json.loads((ROOT / p).read_text()) for p in sources]
    # Deduplicate identical FV grid evidence across conditions and injected values.
    unique = {}
    for dataset in data:
        for row in dataset['runs']:
            independent = row.get('independent') if dataset['experiment']=='E005' else row['decision']['independent']
            if not independent:
                continue
            req = row['request']
            oracle = complex(analytic(complex(*req['z']), req['mean_time']))
            steps = independent.get('steps', [dict(independent, last_grid=512)])
            for step in steps:
                grid = step['last_grid']
                fv = complex(*independent['values'][independent['grids'].index(grid)])
                error = abs(fv-oracle)
                uncertainty = step['uncertainty_estimate']
                key = (dataset['experiment'], row['task_id'], grid)
                record = dict(experiment=key[0],task_id=key[1],grid=grid,
                              converging=step['converging'],error=error,uncertainty=uncertainty,
                              ratio=error/uncertainty,covered=bool(error<=uncertainty))
                if key in unique and unique[key] != record:
                    raise AssertionError('Repeated FV evidence differs')
                unique[key] = record
    coverage = list(unique.values())
    rows = [r for r in data[1]['runs'] if r['condition']=='D' and r['decision']['resolution']['accepted']]
    replay=[]
    for row in rows:
        independent=row['decision']['independent']
        chosen=independent['steps'][-1]
        for step in independent['steps']:
            if verdict(step,row['request']['tolerance'])!='inconclusive':
                chosen=step;break
        replay.append(dict(task_id=row['task_id'],fault=row['fault'],accurate=row['accurate'],
                           verdict=verdict(chosen,row['request']['tolerance']),
                           original_cells=independent['cells_solved'],
                           replay_cells=sum(g for g in independent['grids'] if g<=chosen['last_grid'])))
    task11=next(r for r in data[1]['runs'] if r['condition']=='D' and r['fault']=='clean' and r['task_id']==11)
    ind=task11['decision']['independent'];ref=complex(analytic(complex(*task11['request']['z']),task11['request']['mean_time']))
    history=[dict(grid=g,error=abs(complex(*v)-ref)) for g,v in zip(ind['grids'],ind['values'])]
    last=ind['steps'][-1]
    rejected=[r for r in data[1]['runs'] if r['condition']=='D' and r['accurate'] and not r['decision']['accepted']]
    return dict(analysis='development E005/E006; retrospective, not preregistered',
                environment=dict(zip(('python','numpy','scipy'),actual)),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                coverage=dict(unique_steps=len(coverage),misses=sum(not s['covered'] for s in coverage),
                              maximum=max(coverage,key=lambda s:s['ratio']),steps=coverage),
                task11=dict(absolute_error=task11['absolute_error'],tolerance=task11['request']['tolerance'],
                            error_fraction=task11['absolute_error']/task11['request']['tolerance'],
                            fv_history=history,uncertainty=last['uncertainty_estimate'],
                            required_maximum_U=task11['request']['tolerance']-last['discrepancy']),
                replay_summary=dict(guard_pass_rows=len(rows),inaccurate=sum(not r['accurate'] for r in replay),
                    inaccurate_violation=sum(not r['accurate'] and r['verdict']=='violation_detected' for r in replay),
                    accurate_violation=sum(r['accurate'] and r['verdict']=='violation_detected' for r in replay),
                    accurate_inconclusive=sum(r['accurate'] and r['verdict']=='inconclusive' for r in replay),
                    original_cells=sum(r['original_cells'] for r in replay),replay_cells=sum(r['replay_cells'] for r in replay)),
                accurate_rejections=dict(total=len(rejected),clean=sum(r['fault']=='clean' for r in rejected),
                    corrupted=sum(r['fault']!='clean' for r in rejected),
                    corrupted_guard=sum(r['fault']!='clean' and not r['decision']['resolution']['accepted'] for r in rejected)),
                replay=replay)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','replay')},indent=2))
    print(json.dumps({k:v for k,v in result['coverage'].items() if k!='steps'},indent=2))
