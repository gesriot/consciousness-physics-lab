"""E014 registered development selection; production arithmetic is unchanged."""
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from lab import coverage_map as base, surface_order
from lab.experiment_audit import metadata
from lab.extrapolation import combine_verdict
from lab.three_way import GRIDS, classify, ratio

PROTOCOL = 'experiments/E014-convergence-candidates.md'
SOURCES = ['lab/convergence_candidates.py', 'lab/coverage_map.py', 'lab/surface_order.py',
           'lab/three_way.py', 'lab/extrapolation.py', 'lab/experiment_audit.py',
           'lab/finite_volume.py', 'lab/scientist.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py']
CANDIDATES = ('C0','C1','C2','C3')
ARCHIVES = ('results/surface-order/results.json','results/E011-extrapolation/results.json')


def coordinate(key):
    x,y = key
    logq,phase = -6+x/320, -math.pi/2+y*math.pi/576
    return logq,phase,10**logq*complex(math.cos(phase),math.sin(phase))


def base_cells():
    xs,ys = [0,*range(8,5760,16),5760],[0,*range(8,576,16),576]
    return [(a,b,c,d) for a,c in zip(xs,xs[1:]) for b,d in zip(ys,ys[1:])]


def flags(values, grid, layer_ratio):
    ds = [abs(b-a) for a,b in zip(values,values[1:])]
    rounding = 32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
    ordinary,previous,rounded = ds[-1]<=.5*ds[-2],ds[-2]<=.5*ds[-3],ds[-1]<=rounding
    return {k:bool(v) for k,v in zip(CANDIDATES,
        (ordinary or rounded, ordinary, ordinary or (rounded and previous),
         ordinary or (rounded and layer_ratio>=1)))}


def evidence(values,grid,value,tolerance,layer_ratio,candidate):
    ds = [abs(b-a) for a,b in zip(values,values[1:])]
    rounding = 32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
    uncertainty = 2*ds[-1]+rounding
    converging = flags(values,grid,layer_ratio)[candidate]
    verdict = classify(converging,abs(value-values[-1]),uncertainty,tolerance)
    futile = (grid in (1024,2048) and verdict=='inconclusive' and uncertainty>=4*tolerance
              and ratio(ds[-1],ds[-2])>=.5 and ratio(ds[-2],ds[-3])>=.5)
    return dict(last_grid=grid,converging=converging,uncertainty_estimate=uncertainty,
        discrepancy=abs(value-values[-1]),verdict=verdict,futile=bool(futile),
        blocked_rounding=bool(ds[-1]<=rounding and not converging))


def replay_path(values,value,tolerance,layer_ratios,candidate):
    steps=[]
    for i,n in enumerate(GRIDS):
        if n<512: continue
        step=evidence(values[:i+1],n,value,tolerance,layer_ratios[n],candidate)
        steps.append(step)
        if step['verdict']!='inconclusive' or step['futile']: break
    last=steps[-1]
    reason=('decisive' if last['verdict']!='inconclusive' else 'futility' if last['futile']
            else 'rounding-limit' if last['blocked_rounding'] else 'grid_limit')
    return dict(verdict=last['verdict'],steps=steps,stop_reason=reason,
                cells_solved=sum(n for n in GRIDS if n<=last['last_grid']))


def needs_refinement(cell,records):
    vectors=[tuple((s['candidates'][c],s['covered']) for s in records[k]['steps']
                   for c in CANDIDATES) for k in base.corners(cell)]
    return any(v!=vectors[0] for v in vectors[1:])


def reference_batch(keys,rng):
    # E012 reference selection uses an integer-coordinate threshold internally.
    # Translate only that threshold test; calculation retains shifted coordinates.
    with patch.object(base,'coordinate',coordinate):
        low=[k for k in keys if k[0]<=4480]
        chosen={low[i] for i in rng.choice(len(low),size=len(low)//10,replace=False)}
        chosen.update(k for k in keys if k[0]>4480)
        refs={k:complex(base.analytic(coordinate(k)[2],1.)) for k in keys}
        audits={}
        def check(k):
            hp=base.precise(coordinate(k)[2])
            with base.mp.workdps(60):
                err=float(abs(base.mp.mpc(refs[k].real,refs[k].imag)-base.mp.mpc(*hp)))
            limit=1e-13+1e-12*abs(complex(*map(float,hp)))
            return dict(coordinate=list(k),high_precision=hp,analytic=base.pair(refs[k]),
                        absolute_error=err,passed=bool(math.isfinite(err) and err<=limit))
        for k in sorted(chosen): audits[k]=check(k)
        fallback=not all(r['passed'] for r in audits.values())
        if fallback:
            for k in keys:
                if k not in audits: audits[k]=check(k)
                refs[k]=complex(*map(float,audits[k]['high_precision']))
        return refs,dict(fallback=fallback,random_subset_size=len(low)//10,audited=len(audits),
            failed=sum(not r['passed'] for r in audits.values()),
            max_error=max((r['absolute_error'] for r in audits.values()),default=0),
            records=list(audits.values()))


def evaluate(key,level,reference):
    with patch.object(base,'coordinate',coordinate):
        row=base.evaluate(key,level,reference)
    values=[complex(*v) for v in row['values']]
    for i,s in enumerate(row['steps'],3):
        s['candidates']=flags(values[:i+1],GRIDS[i],s['layer_ratio'])
    return row


def check_archived_step(actual, archived, context=None):
    """Keep discrete decisions exact; allow only finite rounding-scale drift."""
    for key in ('last_grid', 'converging', 'verdict', 'futile'):
        assert type(actual[key]) is type(archived[key]) and actual[key] == archived[key], (context, key)
    for key in ('uncertainty_estimate', 'discrepancy'):
        a, b = actual[key], archived[key]
        assert (math.isfinite(a) and math.isfinite(b)
                and abs(a-b) <= 1e-12 + 1e-12*max(abs(a), abs(b))), (context, key)


def cost_replay():
    rows=[];hashes={};baseline_checks=0
    for name in ARCHIVES:
        p=Path(name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        data=json.loads(p.read_text());experiment=data['experiment']
        tasks={t['task_id']:t for t in data['tasks']}
        for old in data['runs']:
            if experiment=='E009' and old['cell']!='surface/frozen': continue
            task=tasks[old['task_id']];request=task['request'];z=complex(*request['z']);t=request['mean_time']
            model=old.get('model','sphere');pathkey='surface' if experiment=='E009' else model
            values=[complex(*v) for v in task['shadow_values'][pathkey]]
            kappa=1/((3 if model=='slab' else 15000 if model=='model_scale' else 15)*t)
            layers={n:math.sqrt(kappa/abs(z))/(1-surface_order.faces(n)[-2]) for n in GRIDS}
            candidates={}
            for candidate in CANDIDATES:
                replay=replay_path(values,complex(*old['outcome']['value']),request['tolerance'],layers,candidate)
                numerical=replay['verdict']
                if experiment=='E011':
                    combined,reason=combine_verdict(numerical,old['resolution']['accepted'],old['checks'])
                    accepted=combined=='accept'
                else:
                    combined=numerical;reason='numerical'
                    accepted=numerical=='accept' and old['resolution']['accepted']
                candidates[candidate]=dict(replay,combined_verdict=combined,accepted=accepted,combined_reason=reason)
            c0=candidates['C0'];ind=old['independent']
            count=len(ind['values'])
            assert values[:count]==[complex(*v) for v in ind['values']], 'Archived shadow/operational prefix differs'
            assert c0['verdict']==ind['verdict'] and c0['cells_solved']==ind['cells_solved']
            assert c0['accepted']==old['accepted'] and c0['combined_verdict']==old['verdict']
            assert len(c0['steps']) == len(ind['steps']), 'Archived step count differs'
            for a,b in zip(c0['steps'],ind['steps']):
                check_archived_step(a, b, (experiment, old['task_id']))
            baseline_checks+=1
            rows.append(dict(experiment=experiment,task_id=old['task_id'],cohort=old['cohort'],
                fault=old['fault'],accurate=old['accurate'],candidates=candidates))
    summaries={}
    for experiment in ('E009','E011','combined'):
        subset=[r for r in rows if experiment=='combined' or r['experiment']==experiment]
        summaries[experiment]={}
        for c in CANDIDATES:
            def got(r): return r['candidates'][c]
            def old(r): return r['candidates']['C0']
            summaries[experiment][c]=dict(rows=len(subset),
                accurate_accepts=sum(r['accurate'] and got(r)['accepted'] for r in subset),
                lost_accurate_accepts=sum(r['accurate'] and old(r)['accepted'] and not got(r)['accepted'] for r in subset),
                gained_accurate_accepts=sum(r['accurate'] and not old(r)['accepted'] and got(r)['accepted'] for r in subset),
                false_accepts=sum(not r['accurate'] and got(r)['accepted'] for r in subset),
                new_false_accepts=sum(not r['accurate'] and not old(r)['accepted'] and got(r)['accepted'] for r in subset),
                violation_verdict_changes=sum((got(r)['combined_verdict']=='violation_detected')!=(old(r)['combined_verdict']=='violation_detected') for r in subset),
                numerical_violation_changes=sum((got(r)['verdict']=='violation_detected')!=(old(r)['verdict']=='violation_detected') for r in subset),
                decision_cells=sum(got(r)['cells_solved'] for r in subset))
    return dict(source_sha256=hashes,baseline_rows_exact=baseline_checks,
        baseline_comparison=dict(discrete='exact', numerical_atol=1e-12, numerical_rtol=1e-12,
            legacy_counter='baseline_rows_exact counts exact discrete decisions, not exact floats'),
        rows=rows,summary=summaries)


def summarize(points):
    result={}
    for c in CANDIDATES:
        result[c]={}
        for n in GRIDS[3:]:
            rows=[s for p in points for s in p['steps'] if s['grid']==n]
            result[c][str(n)]=dict(steps=len(rows),coverage_misses=sum(not s['covered'] for s in rows),
                dangerous_misses=sum(s['candidates'][c] and not s['covered'] for s in rows),
                covered_not_converging=sum(s['covered'] and not s['candidates'][c] for s in rows))
    return result


def select(summary,cost):
    eligible=[c for c in CANDIDATES if sum(s['dangerous_misses'] for s in summary[c].values())==0]
    simplicity=('C1','C3','C2','C0')
    return min(eligible,key=lambda c:(cost['combined'][c]['lost_accurate_accepts'],simplicity.index(c))) if eligible else None


def run(directory, runtime_override=None):
    output=Path(directory)
    if output.exists(): raise ValueError('Output exists')
    meta=metadata(PROTOCOL,SOURCES,runtime_override);output.mkdir(parents=True)
    dimensions=base.dimensionless_check();base.write(output/'dimensionless.json',dimensions)
    if not all(r['passed'] for r in dimensions): raise RuntimeError('Dimensionless check failed')
    rng=np.random.default_rng(20261127);records={};active=base_cells();batches=[];refinements=[]
    started=time.perf_counter()
    original=surface_order.faces
    with patch.object(surface_order,'faces',lru_cache(maxsize=7)(original)):
        for level in range(4):
            keys=sorted({k for c in active for k in base.corners(c)}-records.keys())
            refs,audit=reference_batch(keys,rng);base.write(output/f'reference-level-{level}.json.gz',audit)
            for k in keys: records[k]=evaluate(k,level,refs[k])
            batches.append(dict(level=level,new_points=len(keys),audited=audit['audited'],failed=audit['failed'],fallback=audit['fallback']))
            print(json.dumps(batches[-1]),flush=True)
            flagged=[c for c in active if needs_refinement(c,records)] if level<3 else []
            refinements.append(dict(level=level,cells_tested=len(active),refined_cells=flagged))
            active=[child for c in flagged for child in base.children(c)]
    points=[records[k] for k in sorted(records)];summary=summarize(points)
    base.write(output/'map.json.gz',dict(experiment='E014',**meta,points=points,batches=batches,
        refinements=refinements,summary=summary,seconds=time.perf_counter()-started))
    # Retain the completed map even if the independent cost audit fails.
    cost=cost_replay()
    selected=select(summary,cost['summary'])
    base.write(output/'cost-replay.json.gz',cost)
    result=dict(points=len(points),batches=batches,map=summary,cost=cost['summary'],selected_V2=selected,
                baseline_rows_exact=cost['baseline_rows_exact'])
    base.write(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory')
    p.add_argument('--runtime-override')
    args=p.parse_args()
    print(json.dumps(run(args.directory,args.runtime_override),indent=2))
