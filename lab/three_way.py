"""E007 three-way FV check. No analytic oracle or shadow inputs here."""
import math
import numpy as np
from lab.finite_volume import response

GRIDS=(64,128,256,512,1024,2048,4096)


def classify(converging, discrepancy, uncertainty, tolerance):
    if not all(math.isfinite(x) for x in (discrepancy,uncertainty,tolerance)):
        return 'inconclusive'
    if converging and discrepancy+uncertainty<=tolerance:
        return 'accept'
    if converging and discrepancy-uncertainty>tolerance:
        return 'violation_detected'
    return 'inconclusive'


def ratio(numerator,denominator):
    return numerator/denominator if denominator else (0. if not numerator else math.inf)


def step_evidence(values,grid,value,tolerance):
    diffs=[abs(b-a) for a,b in zip(values,values[1:])]
    rounding=32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
    converging=bool(diffs[-1]<=max(.5*diffs[-2],rounding))
    uncertainty=2*diffs[-1]+rounding
    discrepancy=abs(value-values[-1])
    verdict=classify(converging,discrepancy,uncertainty,tolerance)
    futile=(grid in (1024,2048) and verdict=='inconclusive'
            and uncertainty>=4*tolerance
            and ratio(diffs[-1],diffs[-2])>=.5
            and ratio(diffs[-2],diffs[-3])>=.5)
    return dict(last_grid=grid,converging=converging,uncertainty_estimate=float(uncertainty),
                discrepancy=discrepancy,verdict=verdict,futile=bool(futile))


def check_three_way(request,value,solver=response):
    values=[];steps=[];grids=[]
    for grid in GRIDS:
        grids.append(grid);values.append(solver(request.z,request.mean_time,grid))
        if grid>=512:
            step=step_evidence(values,grid,value,request.tolerance);steps.append(step)
            if step['verdict']!='inconclusive' or step['futile']:
                break
    verdict=steps[-1]['verdict']
    reason='decisive' if verdict!='inconclusive' else ('futility' if steps[-1]['futile'] else 'grid_limit')
    return dict(accepted=verdict=='accept',verdict=verdict,stop_reason=reason,
                grids=grids,values=[[v.real,v.imag] for v in values],steps=steps,cells_solved=sum(grids))


def full_path(request,solver=response):
    return [solver(request.z,request.mean_time,n) for n in GRIDS]


def shadow_evidence(request,value,values):
    steps=[step_evidence(values[:i+1],n,value,request.tolerance) for i,n in enumerate(GRIDS) if n>=512]
    decisive=next((s for s in steps if s['verdict']!='inconclusive'),steps[-1])
    return dict(steps=steps,any_accept=any(s['verdict']=='accept' for s in steps),
                any_violation=any(s['verdict']=='violation_detected' for s in steps),
                first_decisive=decisive['verdict'],first_decisive_grid=decisive['last_grid'],
                terminal_verdict=steps[-1]['verdict'],
                decisive_reversal=any(s['verdict'] not in ('inconclusive',decisive['verdict']) for s in steps))
