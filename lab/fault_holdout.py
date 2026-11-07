"""E006: frozen D verification on new held-out tasks and boundary fault injection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from lab.scientist import ROOT, analytic, modal_partial_sums
from lab.guard import Request, final_guard
from lab.executor import execute, as_outcome
from lab.allocation import propose_modes, refine_value

FAULTS=('clean','conjugate','renormalized','time_units','near_tolerance','stale_value')


def inject(request,receipt,fault,stale):
    """Explicit untrusted artifact, never a forged Receipt object."""
    outcome=as_outcome(request,receipt)
    value=receipt.value
    additional_modes=0
    if fault=='conjugate':value=value.conjugate()
    elif fault=='renormalized':
        total=complex(modal_partial_sums(0,request.mean_time,[receipt.modes])[0])
        value/=total
        additional_modes=receipt.modes
    elif fault=='time_units':
        value=complex(modal_partial_sums(request.z,1000*request.mean_time,[receipt.modes])[0])
        additional_modes=receipt.modes
    elif fault=='near_tolerance':value+=1.25*request.tolerance
    elif fault=='stale_value':value=stale
    elif fault!='clean':raise ValueError('unregistered fault')
    outcome['value']=[value.real,value.imag]
    return outcome,additional_modes


def verify(request,outcome,condition):
    resolution=final_guard(request,outcome)
    independent=None
    if condition=='D':independent=refine_value(request,complex(*outcome['value']))
    elif condition!='resolution_only':raise ValueError('unregistered condition')
    return {'resolution':resolution,'independent':independent,
            'accepted':bool(resolution['accepted'] and (independent is None or independent['accepted']))}


def run():
    rng=np.random.default_rng(20261112); order=np.random.default_rng(20261113)
    initial=Request(1,1,.01)
    previous=execute(initial,propose_modes(initial,.5)).value
    rows=[]
    for task_id in range(96):
        mean_time=10**rng.uniform(-2,2)
        magnitude=10**rng.uniform(-6,5)
        phase=rng.uniform(-np.pi/2,np.pi/2)
        tolerance=10**rng.uniform(-4,-1)
        z=magnitude*np.exp(1j*phase)/mean_time
        request=Request(z,mean_time,tolerance)
        receipt=execute(request,propose_modes(request,.5))
        for fault in order.permutation(FAULTS):
            started=time.perf_counter()
            supplied,additional_modes=inject(request,receipt,str(fault),previous)
            injection_seconds=time.perf_counter()-started
            for condition in order.permutation(('resolution_only','D')):
                started=time.perf_counter()
                decision=verify(request,supplied,str(condition))
                verification_seconds=time.perf_counter()-started
                error=abs(complex(*supplied['value'])-complex(analytic(z,mean_time)))
                rows.append({'task_id':task_id,'request':{'z':[z.real,z.imag],'mean_time':mean_time,'tolerance':tolerance},
                    'fault':str(fault),'condition':str(condition),'artifact_status':'explicit_untrusted_boundary_input',
                    'outcome':copy.deepcopy(supplied),'decision':decision,'absolute_error':error,'accurate':bool(error<=tolerance),
                    'generation':{'base_execution_modes':receipt.modes,'base_execution_seconds':receipt.kernel_seconds,
                        'additional_injection_modes':additional_modes,'injection_seconds':injection_seconds,
                        'shared_between_ablations':True},'verification_seconds':verification_seconds})
        previous=receipt.value
    summary={}
    for condition in ('resolution_only','D'):
        summary[condition]={}
        for fault in FAULTS:
            group=[r for r in rows if r['condition']==condition and r['fault']==fault]
            summary[condition][fault]={'tasks':len(group),'accurate':sum(r['accurate'] for r in group),
                'accurate_accepted':sum(r['accurate'] and r['decision']['accepted'] for r in group),
                'false_acceptances':sum(not r['accurate'] and r['decision']['accepted'] for r in group),
                'accurate_rejected':sum(r['accurate'] and not r['decision']['accepted'] for r in group),
                'declared_modes':sum(r['outcome']['modes_spent'] for r in group),
                'finite_volume_cells':sum(r['decision']['independent']['cells_solved'] if r['decision']['independent'] else 0 for r in group),
                'additional_injection_modes':sum(r['generation']['additional_injection_modes'] for r in group),
                'verification_seconds':sum(r['verification_seconds'] for r in group)}
    sources=['lab/fault_holdout.py','lab/allocation.py','lab/executor.py','lab/finite_volume.py','lab/guard.py',
             'vendor/sphere-thermal-cli/sphere_thermal/response.py']
    return {'experiment':'E006','task_seed':20261112,'order_seed':20261113,'paired_tasks':96,
         'generation_accounting':'base execution reused across six faults; each injected artifact shared by both ablations; do not sum duplicated counters as execution cost',
         'initial_stale_request':{'z':[1,0],'mean_time':1,'tolerance':.01},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
         'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
         'summary':summary,'runs':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
