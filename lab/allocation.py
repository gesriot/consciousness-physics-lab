"""E005 factorial allocation: modal accuracy margin versus verifier refinement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from lab.scientist import ROOT, analytic
from lab.guard import Request, final_guard
from lab.executor import execute, as_outcome
from lab.finite_volume import response, check_value

CONDITIONS={'A':(1.,False),'B':(.5,False),'C':(1.,True),'D':(.5,True)}


def propose_modes(request, fraction):
    if fraction not in (.5,1.):
        raise ValueError('registered tolerance fractions are .5 and 1')
    if not math.isfinite(request.tolerance) or request.tolerance<=0:
        raise ValueError('positive finite tolerance required')
    ratio=6/math.pi**2/(request.tolerance*fraction)
    return request.budget if not math.isfinite(ratio) or ratio>=request.budget else max(1,math.ceil(ratio-.5))


def refine_value(request,value,max_cells=4096):
    if max_cells not in (512,1024,2048,4096):
        raise ValueError('maximum grid must be a registered dyadic size')
    grids=[]; values=[]; steps=[]
    cells=64
    while cells<=max_cells:
        grids.append(cells)
        values.append(response(request.z,request.mean_time,cells))
        if len(grids)>=4:
            differences=[abs(b-a) for a,b in zip(values,values[1:])]
            rounding=32*np.finfo(float).eps*cells**2*max(1,abs(values[-1]))
            converging=differences[-1]<=max(.5*differences[-2],rounding)
            uncertainty=2*differences[-1]+rounding
            discrepancy=abs(value-values[-1])
            accepted=bool(converging and math.isfinite(discrepancy) and discrepancy+uncertainty<=request.tolerance)
            steps.append({'last_grid':cells,'converging':bool(converging),'uncertainty_estimate':uncertainty,
                          'discrepancy':discrepancy,'accepted':accepted})
            if accepted:break
        cells*=2
    return {'accepted':steps[-1]['accepted'],'grids':grids,'values':[[v.real,v.imag] for v in values],
            'cells_solved':sum(grids),'steps':steps}


def evaluate(request,condition):
    fraction,adaptive=CONDITIONS[condition]
    modes=propose_modes(request,fraction)
    receipt=execute(request,modes)
    outcome=as_outcome(request,receipt)
    started=time.perf_counter()
    resolution=final_guard(request,outcome)
    if adaptive:
        independent=refine_value(request,receipt.value)
    else:
        independent=check_value(request.z,request.mean_time,request.tolerance,receipt.value)
        independent['cells_solved']=sum(independent['grids'])
    accepted=resolution['accepted'] and independent['accepted']
    verification_seconds=time.perf_counter()-started
    # No oracle is used in proposing, execution, or acceptance.
    return {'condition':condition,'target_fraction':fraction,'modes':modes,
            'value':[receipt.value.real,receipt.value.imag],'resolution':resolution,
            'independent':independent,'accepted':bool(accepted),
            'execution_seconds':receipt.kernel_seconds,'verification_seconds':verification_seconds}


def run():
    rng=np.random.default_rng(20261110); order=np.random.default_rng(20261111)
    rows=[]
    for task_id in range(96):
        mean_time=10**rng.uniform(-2,2)
        magnitude=10**rng.uniform(-5,4)
        phase=rng.uniform(-np.pi/2,np.pi/2)
        tolerance=10**rng.uniform(-4,-1)
        z=magnitude*np.exp(1j*phase)/mean_time
        request=Request(z,mean_time,tolerance)
        for condition in order.permutation(list(CONDITIONS)):
            result=evaluate(request,str(condition))
            error=abs(complex(*result['value'])-complex(analytic(z,mean_time)))
            result.update(task_id=task_id,request={'z':[z.real,z.imag],'mean_time':mean_time,'tolerance':tolerance},
                          absolute_error=error,accurate=bool(error<=tolerance))
            rows.append(result)
    summary={}
    for condition in CONDITIONS:
        group=[r for r in rows if r['condition']==condition]
        summary[condition]={'tasks':len(group),'accurate':sum(r['accurate'] for r in group),
             'accurate_accepted':sum(r['accurate'] and r['accepted'] for r in group),
             'false_acceptances':sum(not r['accurate'] and r['accepted'] for r in group),
             'accurate_rejected':sum(r['accurate'] and not r['accepted'] for r in group),
             'modes':sum(r['modes'] for r in group),
             'finite_volume_cells':sum(r['independent']['cells_solved'] for r in group),
             'execution_seconds':sum(r['execution_seconds'] for r in group),
             'verification_seconds':sum(r['verification_seconds'] for r in group)}
    sources=['lab/allocation.py','lab/executor.py','lab/finite_volume.py','lab/guard.py',
             'vendor/sphere-thermal-cli/sphere_thermal/response.py']
    return {'experiment':'E005','task_seed':20261110,'order_seed':20261111,'paired_tasks':96,
         'limits':{'mode_terms':4096,'finite_volume_cells':8128},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
         'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
         'summary':summary,'runs':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
