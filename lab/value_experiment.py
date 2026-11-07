"""E004 paired fault injection and verification ablations."""
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
from lab.scientist import ROOT, solve, analytic, modal_partial_sums
from lab.guard import Request, final_guard
from lab.executor import execute, as_outcome
from lab.finite_volume import check_value

FAULTS=('clean','forged_count','value_offset','wrong_time','planner_report')


def run():
    rows=[]
    for mean_time in (.1,1.,10.):
        for zt in (0,.01,1,10j,100,100j):
            for tolerance in (.001,.01):
                z=zt/mean_time
                request=Request(z,mean_time,tolerance)
                planned=solve(z,mean_time,tolerance)
                n=planned['trace'][-1]['modes']
                for fault in FAULTS:
                    supplied=copy.deepcopy(planned)
                    if fault=='forged_count':
                        value=complex(modal_partial_sums(z,mean_time,[8])[0])
                        supplied['value']=[value.real,value.imag]
                    elif fault in ('value_offset','planner_report'):
                        supplied['value'][0]+=.05
                    elif fault=='wrong_time':
                        value=complex(modal_partial_sums(z,2*mean_time,[n])[0])
                        supplied['value']=[value.real,value.imag]
                    for condition in ('resolution_only','value_check','reexecute_and_value'):
                        execution_seconds=0.
                        outcome=supplied
                        if condition=='reexecute_and_value':
                            receipt=execute(request,n)
                            outcome=as_outcome(request,receipt)
                            execution_seconds=receipt.kernel_seconds
                        started=time.perf_counter()
                        resolution=final_guard(request,outcome)
                        independent=None
                        if condition!='resolution_only':
                            independent=check_value(z,mean_time,tolerance,complex(*outcome['value']))
                        accepted=resolution['accepted'] and (independent is None or independent['accepted'])
                        check_seconds=time.perf_counter()-started
                        # Oracle evaluated strictly after decision.
                        error=abs(complex(*outcome['value'])-complex(analytic(z,mean_time)))
                        rows.append({'request':{'z':[complex(z).real,complex(z).imag], 'mean_time':mean_time,'tolerance':tolerance},
                            'fault':fault,'condition':condition,'outcome':outcome,
                            'resolution':resolution,'independent':independent,'accepted':bool(accepted),
                            'absolute_error':error,'accurate':bool(error<=tolerance),
                            'additional_execution_modes':n if condition=='reexecute_and_value' else 0,
                            'additional_execution_seconds':execution_seconds,'verification_seconds':check_seconds})
    summary={}
    for condition in ('resolution_only','value_check','reexecute_and_value'):
        summary[condition]={}
        for fault in FAULTS:
            group=[r for r in rows if r['condition']==condition and r['fault']==fault]
            summary[condition][fault]={'cases':len(group),'accurate':sum(r['accurate'] for r in group),
                'accepted':sum(r['accepted'] for r in group),
                'false_acceptances':sum(r['accepted'] and not r['accurate'] for r in group),
                'accurate_rejected':sum(not r['accepted'] and r['accurate'] for r in group),
                'additional_execution_modes':sum(r['additional_execution_modes'] for r in group),
                'verification_seconds':sum(r['verification_seconds'] for r in group),
                'finite_volume_cells_solved':sum(sum(r['independent']['grids']) if r['independent'] else 0 for r in group)}
    paths=['lab/value_experiment.py','lab/finite_volume.py','lab/executor.py','lab/guard.py','lab/scientist.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py']
    return {'experiment':'E004','split':'development_fault_injection','paired_tasks':36,
       'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
       'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
       'summary':summary,'runs':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
