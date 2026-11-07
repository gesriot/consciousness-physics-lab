"""E009 registered 2 by 2 experiment; oracle is evaluated after decisions."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from lab.scientist import analytic
from lab.guard import Request, final_guard
from lab.executor import execute
from lab.allocation import propose_modes
from lab.fault_holdout import FAULTS, inject
from lab.three_way import GRIDS
from lab.surface_order import CELLS, faces, solver, evidence, check, shadow
from lab.verifier_stress import stratum
from lab.experiment_audit import metadata
from lab.replay import load_replay, protect_output

PROTOCOL = 'experiments/E009-surface-order.md'
SOURCES = ['lab/surface_experiment.py','lab/surface_order.py','lab/three_way.py',
           'lab/experiment_audit.py','lab/replay.py','lab/fault_holdout.py','lab/allocation.py',
           'lab/finite_volume.py','lab/guard.py','lab/executor.py','lab/scientist.py',
           'lab/verifier_stress.py','vendor/sphere-thermal-cli/sphere_thermal/response.py']


def summary(rows, coverage, tasks):
    result = {}
    for cell in CELLS:
        group = [r for r in rows if r['cell'] == cell]
        steps = [s for s in coverage if s['cell'] == cell]
        decisive_ids = {(r['task_id'],s['last_grid']) for r in group for s in r['independent']['steps']
                        if s['verdict'] != 'inconclusive'}
        decisive = [s for s in steps if (s['task_id'],s['grid']) in decisive_ids]
        strata = {}
        for cohort in ('broad','high'):
            for bin_name in ('[0,.5)','[.5,1)','[1,2)','[2,4)','[4,infinity)'):
                selected = [s for s in steps if s['cohort']==cohort and s['stratum']==bin_name]
                strata[cohort+':'+bin_name] = dict(steps=len(selected),misses=sum(not s['covered'] for s in selected),
                    max_ratio=max((s['ratio'] for s in selected),default=None))
        clean = {}
        for cohort in ('broad','high'):
            selected = [r for r in group if r['fault']=='clean' and r['cohort']==cohort]
            clean[cohort] = dict(tasks=len(selected),accurate=sum(r['accurate'] for r in selected),
                accurate_accepted=sum(r['accurate'] and r['accepted'] for r in selected),
                inconclusive=sum(r['verdict']=='inconclusive' for r in selected))
        result[cell] = dict(records=len(group), coverage=dict(unique_steps=len(steps),
            misses=sum(not s['covered'] for s in steps), max_ratio=max(s['ratio'] for s in steps),
            decisive_steps=len(decisive), decisive_misses=sum(not s['covered'] for s in decisive), strata=strata,
            fallback_steps=sum(s['uncertainty_fallback'] for s in steps)),
            false_acceptances=sum(r['accepted'] and not r['accurate'] for r in group),
            accurate_violation=sum(r['accurate'] and r['verdict']=='violation_detected' for r in group),
            accurate_violation_guard_pass=sum(r['accurate'] and r['resolution']['accepted'] and r['verdict']=='violation_detected' for r in group),
            clean=clean, inconclusive=sum(r['verdict']=='inconclusive' for r in group),
            accurate_inconclusive=sum(r['accurate'] and r['verdict']=='inconclusive' for r in group),
            decision_cells=sum(r['independent']['cells_solved'] for r in group),
            full_path_equivalent_cells=len(group)*sum(GRIDS),
            futility_stops=sum(r['independent']['stop_reason']=='futility' for r in group),
            accurate_shadow_accept_lost=sum(r['accurate'] and r['resolution']['accepted'] and not r['accepted'] and r['shadow']['any_accept'] for r in group),
            futility_accurate_shadow_accept_lost=sum(r['accurate'] and r['resolution']['accepted'] and r['independent']['stop_reason']=='futility' and r['shadow']['any_accept'] for r in group),
            futility_lost_shadow_violations=sum(r['independent']['stop_reason']=='futility' and r['shadow']['any_violation'] for r in group),
            shadow_reversals=sum(r['shadow']['decisive_reversal'] for r in group),
            verification_seconds=sum(r['verification_seconds'] for r in group))
    return dict(cells=result,work=dict(shadow_cells=sum(t['shadow_cells'] for t in tasks),
        decision_cells=sum(r['independent']['cells_solved'] for r in rows),
        total_actual_fv_cells=sum(t['shadow_cells'] for t in tasks)+sum(r['independent']['cells_solved'] for r in rows),
        execution_modal_terms=sum(t['base_modes'] for t in tasks),
        additional_injection_modes=sum(t['additional_injection_modes'] for t in tasks),
        shadow_seconds=sum(t['shadow_seconds'] for t in tasks)))


def run(replay=None, runtime_override=None):
    audit = metadata(PROTOCOL,SOURCES,runtime_override)
    saved = None
    if replay is not None:
        saved, requests, audit['replay'] = load_replay(replay, 'E009', 192)
    rng, order = np.random.default_rng(20261118), np.random.default_rng(20261119)
    initial = Request(1,1,.01)
    previous = execute(initial,propose_modes(initial,.5)).value
    rows,tasks,coverage = [],[],[]
    for task_id in range(192):
        cohort = 'broad' if task_id<96 else 'high'
        if saved is None:
            t = 10**rng.uniform(-2,2)
            magnitude = 10**rng.uniform(-6 if cohort=='broad' else 5,6)
            phase = rng.uniform(-np.pi/2,np.pi/2)
            tolerance = 10**rng.uniform(-4,-1)
            z = magnitude*np.exp(1j*phase)/t
            request = Request(z,t,tolerance)
        else:
            record = saved['tasks'][task_id]
            if record['cohort'] != cohort:
                raise ValueError('Replay cohort mismatch')
            request = requests[task_id]
            z, t, tolerance = request.z, request.mean_time, request.tolerance
            magnitude = record['sampled_magnitude']
            if not (math.isfinite(magnitude) and magnitude > 0 and
                    math.isclose(magnitude, abs(z*t), rel_tol=1e-12)):
                raise ValueError('Replay sampled magnitude mismatch')
        receipt = execute(request,propose_modes(request,.5))
        local=[]; injection_modes=0
        for fault in order.permutation(FAULTS):
            started=time.perf_counter()
            outcome,extra=inject(request,receipt,str(fault),previous)
            injection_seconds=time.perf_counter()-started
            injection_modes+=extra
            for cell in order.permutation(CELLS):
                mesh,estimator=str(cell).split('/')
                started=time.perf_counter()
                resolution=final_guard(request,outcome)
                checked=check(request,complex(*outcome['value']),mesh,estimator)
                seconds=time.perf_counter()-started
                local.append(dict(task_id=task_id,cohort=cohort,cell=str(cell),fault=str(fault),
                    outcome=outcome,resolution=resolution,independent=checked,
                    verdict=checked['verdict'],accepted=bool(resolution['accepted'] and checked['accepted']),
                    verification_seconds=seconds,injection_seconds=injection_seconds,
                    additional_injection_modes=extra))
        # No oracle/shadow values enter the operational paths above.
        started=time.perf_counter()
        paths={mesh:[solver(mesh)(z,t,n) for n in GRIDS] for mesh in ('uniform','surface')}
        shadow_seconds=time.perf_counter()-started
        reference=complex(analytic(z,t))
        for cell in CELLS:
            mesh,estimator=cell.split('/')
            values=paths[mesh]
            for i,n in enumerate(GRIDS):
                if n<512:continue
                step=evidence(values[:i+1],n,receipt.value,tolerance,estimator)
                error=abs(values[i]-reference); uncertainty=step['uncertainty_estimate']
                hs=1/n if mesh=='uniform' else 1-faces(n)[-2]
                dh=1/math.sqrt(15*magnitude)/hs
                coverage.append(dict(task_id=task_id,cohort=cohort,cell=cell,grid=n,
                    surface_width=hs,delta_over_h=dh,stratum=stratum(dh),error=error,
                    uncertainty=uncertainty,ratio=error/uncertainty,covered=bool(error<=uncertainty),
                    converging=step['converging'],observed_order=step['observed_order'],
                    uncertainty_fallback=step['uncertainty_fallback']))
        for row in local:
            mesh,estimator=row['cell'].split('/')
            value=complex(*row['outcome']['value']);error=abs(value-reference)
            row.update(absolute_error=error,accurate=bool(error<=tolerance),
                       shadow=shadow(request,value,paths[mesh],estimator))
        rows.extend(local)
        tasks.append(dict(task_id=task_id,cohort=cohort,request=dict(z=[z.real,z.imag],mean_time=t,tolerance=tolerance),
            sampled_magnitude=magnitude,base_modes=receipt.modes,base_execution_seconds=receipt.kernel_seconds,
            additional_injection_modes=injection_modes,
            shadow_values={k:[[v.real,v.imag] for v in vs] for k,vs in paths.items()},
            shadow_cells=2*sum(GRIDS),shadow_seconds=shadow_seconds))
        previous=receipt.value
    return dict(experiment='E009',**audit,task_seed=20261118,order_seed=20261119,paired_tasks=192,
        sampling_unit='task; input conditions, factorial cells and repeated grids are dependent',
        accounting='One base execution per task; one injection per input; one shadow per mesh; no decision shadow cache.',
        initial_stale_modes=propose_modes(initial,.5),tasks=tasks,coverage_steps=coverage,runs=rows,
        summary=summary(rows,coverage,tasks))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay', type=Path, help='Read exact requests from archived results.json')
    parser.add_argument('--runtime-override', help='Explicit audit reason for using an unpinned runtime')
    args=parser.parse_args()
    protect_output(args.output,args.replay)
    result=run(args.replay,args.runtime_override);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
