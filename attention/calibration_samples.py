"""Frozen known-answer inputs and live check execution."""
import copy
from attention import agent
from lab import frozen_pipeline as P
from lab.convergence_candidates import classify, ratio
from lab.executor import request_hash
from lab.guard import Request, final_guard
from lab.radial_collocation import check as third_check

FAMILIES=('collocation','second_moment','v2')
FACTOR=100.


def request(sample):
    fields=dict(sample['request']);fields['z']=complex(*fields['z'])
    return Request(**fields)


def spoil_v2(original, tolerance):
    out=copy.deepcopy(original)
    if 'step' not in out:return out
    step=out['step'];step['uncertainty_estimate']/=FACTOR
    step['verdict']=classify(step['converging'],step['discrepancy'],step['uncertainty_estimate'],tolerance)
    step['futile']=bool(step['last_grid'] in (1024,2048) and step['verdict']=='inconclusive'
        and step['uncertainty_estimate']>=4*tolerance
        and ratio(step['last_difference'],step['previous_difference'])>=.5
        and original['step'].get('antecedent_ratio',0.)>=.5)
    out.update(verdict=step['verdict'],terminal=bool(step['verdict']!='inconclusive' or step['futile'] or step['last_grid']==agent.GRIDS[-1]))
    return out


def v2_observation(values, grid, candidate, tolerance):
    out=dict(verdict='inconclusive',terminal=False,value=[values[-1].real,values[-1].imag])
    if grid>=P.CONFIG.first_decisive_grid:
        step=P.v2_step(values,grid,candidate,tolerance)
        differences=[abs(b-a) for a,b in zip(values,values[1:])]
        step['antecedent_ratio']=ratio(differences[-2],differences[-3])
        out.update(step=step,verdict=step['verdict'],terminal=bool(step['verdict']!='inconclusive' or step['futile'] or grid==agent.GRIDS[-1]))
    return out


def maximum_cost(sample, weights):
    r=request(sample);family=sample['family']
    if family=='v2':return sum(agent.GRIDS)*weights['fv_cell']
    return agent.cost(family,r,{'cost_weights':weights})


def execute_sample(sample, weights, fault='none'):
    """No stored expected verdict is consulted when executing a check."""
    r=request(sample);family=sample['family'];model=sample.get('model','sphere')
    trace=[]
    if family=='v2':
        values=[]
        for grid in agent.GRIDS:
            values.append(P.model_response('fv',r.z,r.mean_time,grid,model))
            original=v2_observation(values,grid,complex(*sample['candidate']),r.tolerance)
            observed=spoil_v2(original,r.tolerance) if fault=='underestimated' else copy.deepcopy(original)
            trace.append(dict(action='v2_'+str(grid),cost=grid*weights['fv_cell'],original=original,observed=observed))
            if observed['terminal']:break
        original,observed=trace[-1]['original'],trace[-1]['observed']
        charge=sum(x['cost'] for x in trace)
    else:
        if family=='collocation':
            original=third_check(r.z,r.mean_time,complex(*sample['candidate']),r.tolerance)
            original={k:v for k,v in original.items() if k!='seconds'}
        elif family in agent.FEATURES:original=agent.feature(r,family,model)
        elif family=='binding':original=dict(verdict='accept' if request_hash(r)==sample['digest'] else 'violation_detected')
        elif family=='modal':
            guard=final_guard(r,sample['outcome']);original=dict(verdict='accept' if guard['accepted'] else 'inconclusive',resolution=guard)
        else:raise ValueError('Unknown sample family')
        observed=copy.deepcopy(original)
        if (fault=='permissive' and family=='collocation') or (fault=='uninformative' and family=='second_moment'):
            observed={'verdict':'accept'}
        charge=maximum_cost(sample,weights)
    return dict(verdict=observed['verdict'],original=original,observed=observed,trace=trace,cost=charge)
