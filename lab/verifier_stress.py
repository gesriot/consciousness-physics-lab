"""E007: preregistered boundary-layer stress and early three-way decisions."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from lab.scientist import analytic
from lab.guard import Request,final_guard
from lab.executor import execute
from lab.allocation import propose_modes
from lab.fault_holdout import FAULTS,inject,verify
from lab.three_way import GRIDS,full_path,shadow_evidence,check_three_way,step_evidence
from lab.experiment_audit import metadata

PROTOCOL='experiments/E007-verifier-stress.md'
SOURCES=['lab/verifier_stress.py','lab/three_way.py','lab/experiment_audit.py',
         'lab/fault_holdout.py','lab/allocation.py','lab/finite_volume.py','lab/guard.py',
         'lab/executor.py','vendor/sphere-thermal-cli/sphere_thermal/response.py']


def stratum(x):
    for bound,name in ((.5,'[0,.5)'),(1,'[.5,1)'),(2,'[1,2)'),(4,'[2,4)')):
        if x<bound:return name
    return '[4,infinity)'


def summarize(rows,coverage,tasks):
    policies={}
    for condition in ('D','D3'):
        group=[r for r in rows if r['condition']==condition]
        policies[condition]=dict(records=len(group),accurate=sum(r['accurate'] for r in group),
            false_acceptances=sum(r['accepted'] and not r['accurate'] for r in group),
            accurate_accepted=sum(r['accepted'] and r['accurate'] for r in group),
            clean_accurate_rejected=sum(r['accurate'] and not r['accepted'] and r['fault']=='clean' for r in group),
            corrupted_accurate_rejected=sum(r['accurate'] and not r['accepted'] and r['fault']!='clean' for r in group),
            accurate_guard_rejected=sum(r['accurate'] and not r['resolution']['accepted'] for r in group),
            inaccurate_violation=sum(not r['accurate'] and r['verdict']=='violation_detected' for r in group),
            accurate_violation=sum(r['accurate'] and r['verdict']=='violation_detected' for r in group),
            accurate_violation_guard_pass=sum(r['accurate'] and r['resolution']['accepted'] and r['verdict']=='violation_detected' for r in group),
            accurate_inconclusive=sum(r['accurate'] and r['verdict']=='inconclusive' for r in group),
            decision_cells=sum(r['independent']['cells_solved'] for r in group),
            full_path_equivalent_cells=len(group)*sum(GRIDS),
            saved_vs_full_cells=sum(sum(GRIDS)-r['independent']['cells_solved'] for r in group),
            verification_seconds=sum(r['verification_seconds'] for r in group),
            futility_stops=sum(r['independent'].get('stop_reason')=='futility' for r in group),
            accurate_shadow_accept_lost=sum(r['accurate'] and r['resolution']['accepted'] and not r['accepted'] and r['shadow']['any_accept'] for r in group),
            fv_accurate_shadow_accept_lost=sum(r['accurate'] and not r['independent']['accepted'] and r['shadow']['any_accept'] for r in group),
            futility_lost_shadow_violations=sum(r['independent'].get('stop_reason')=='futility' and r['shadow']['any_violation'] for r in group))
    strata={}
    for cohort in ('broad','high'):
        for bin_name in ('[0,.5)','[.5,1)','[1,2)','[2,4)','[4,infinity)'):
            selected=[s for s in coverage if s['cohort']==cohort and s['stratum']==bin_name]
            strata[cohort+':'+bin_name]=dict(steps=len(selected),misses=sum(not s['covered'] for s in selected),
                max_ratio=max((s['ratio'] for s in selected),default=None))
    decisive={}
    for condition in ('D','D3'):
        indices={(r['task_id'],s['last_grid']) for r in rows if r['condition']==condition
                 for s in r['independent']['steps']
                 if (s.get('verdict') in ('accept','violation_detected') or s.get('accepted',False))}
        selected=[s for s in coverage if (s['task_id'],s['grid']) in indices]
        decisive[condition]=dict(unique_task_grids=len(selected),misses=sum(not s['covered'] for s in selected))
    return dict(policies=policies,coverage=dict(unique_steps=len(coverage),misses=sum(not s['covered'] for s in coverage),
        max_ratio=max(s['ratio'] for s in coverage),strata=strata,decisive=decisive),
        shadow_cells=sum(t['shadow_cells'] for t in tasks),shadow_seconds=sum(t['shadow_seconds'] for t in tasks),
        shadow_reversals=sum(r['shadow']['decisive_reversal'] for r in rows if r['condition']=='D3'),
        total_actual_fv_cells=sum(t['shadow_cells'] for t in tasks)+sum(r['independent']['cells_solved'] for r in rows))


def run():
    audit=metadata(PROTOCOL,SOURCES)
    rng=np.random.default_rng(20261114);order=np.random.default_rng(20261115)
    initial=Request(1,1,.01);previous=execute(initial,propose_modes(initial,.5)).value
    rows=[];tasks=[];coverage=[]
    for task_id in range(192):
        cohort='broad' if task_id<96 else 'high'
        mean_time=10**rng.uniform(-2,2)
        magnitude=10**rng.uniform(-6 if cohort=='broad' else 5,6)
        phase=rng.uniform(-np.pi/2,np.pi/2);tolerance=10**rng.uniform(-4,-1)
        z=magnitude*np.exp(1j*phase)/mean_time;request=Request(z,mean_time,tolerance)
        receipt=execute(request,propose_modes(request,.5))
        local=[]
        for fault in order.permutation(FAULTS):
            started=time.perf_counter();outcome,extra=inject(request,receipt,str(fault),previous)
            injection_seconds=time.perf_counter()-started
            for condition in order.permutation(('D','D3')):
                started=time.perf_counter()
                if condition=='D':
                    checked=verify(request,outcome,'D')
                    resolution=checked['resolution'];independent=checked['independent']
                    verdict='accept' if independent['accepted'] else 'binary_reject'
                else:
                    resolution=final_guard(request,outcome)
                    independent=check_three_way(request,complex(*outcome['value']))
                    verdict=independent['verdict']
                seconds=time.perf_counter()-started
                local.append(dict(task_id=task_id,cohort=cohort,fault=str(fault),condition=str(condition),
                    outcome=outcome,resolution=resolution,independent=independent,verdict=verdict,
                    accepted=bool(resolution['accepted'] and independent['accepted']),verification_seconds=seconds,
                    additional_injection_modes=extra,injection_seconds=injection_seconds))
        # Shadow and oracle are evaluated only after ALL operational decisions.
        started=time.perf_counter();values=full_path(request);shadow_seconds=time.perf_counter()-started
        reference=complex(analytic(z,mean_time))
        for i,grid in enumerate(GRIDS):
            if grid<512:continue
            evidence=step_evidence(values[:i+1],grid,receipt.value,tolerance)
            error=abs(values[i]-reference);u=evidence['uncertainty_estimate']
            dh=grid/math.sqrt(15*magnitude)
            coverage.append(dict(task_id=task_id,cohort=cohort,grid=grid,delta_over_h=dh,stratum=stratum(dh),
                error=error,uncertainty=u,ratio=error/u,covered=bool(error<=u),converging=evidence['converging']))
        for row in local:
            value=complex(*row['outcome']['value'])
            error=abs(value-reference)
            row.update(absolute_error=error,accurate=bool(error<=tolerance),
                       shadow=shadow_evidence(request,value,values))
        rows.extend(local)
        tasks.append(dict(task_id=task_id,cohort=cohort,request=dict(z=[z.real,z.imag],mean_time=mean_time,tolerance=tolerance),
            base_modes=receipt.modes,base_execution_seconds=receipt.kernel_seconds,
            shadow_values=[[v.real,v.imag] for v in values],shadow_cells=sum(GRIDS),shadow_seconds=shadow_seconds))
        previous=receipt.value
    return dict(experiment='E007',**audit,task_seed=20261114,order_seed=20261115,paired_tasks=192,
        sampling_unit='task; input conditions, policies and repeated grids are dependent',
        accounting='Base execution once per task; injection shared across policies; shadow once per task, never used for decisions.',
        initial_stale_modes=propose_modes(initial,.5),tasks=tasks,coverage_steps=coverage,runs=rows,
        summary=summarize(rows,coverage,tasks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
