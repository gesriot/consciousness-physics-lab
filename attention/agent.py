"""Lazy actions and a scheduler restricted to observed evidence."""
import copy
import math
import numpy as np
from lab import frozen_pipeline as P
from lab.executor import execute, request_hash
from lab.fault_holdout import FAULTS, inject
from lab.guard import final_guard
from lab.radial_collocation import check as third_check

GRIDS = P.CONFIG.grids
FEATURES = ('normalization', 'first_moment', 'second_moment', 'high_frequency')
ORDER = ('modal',) + tuple(f'v2_{n}' for n in GRIDS) + ('binding',) + FEATURES + ('collocation',)
POLICIES = ('B', 'C1', 'C2', 'C3', 'D', 'Dprime')
FRACTIONS = (.3, .6, 1.)


def cost(action, request, model):
    w = model['cost_weights']
    if action == 'modal':
        return P.plan_modes(request) * w['modal_term']
    if action.startswith('v2_'):
        return int(action[3:]) * w['fv_cell']
    if action == 'binding':
        return w['binding']
    if action in ('normalization', 'first_moment'):
        return 4096*w['modal_term'] + 512*w['fv_cell']
    if action == 'second_moment':
        return 8192*w['modal_term'] + 8192*w['fv_cell']
    if action == 'high_frequency':
        return 4096*w['modal_term'] + 4096*w['fv_cell']
    if action == 'collocation':
        return sum(n**3 for n in P.CONFIG.third_degrees)*w['collocation_degree_cubed']
    raise ValueError('Unknown action')


def corrupt(model, policy):
    result = copy.deepcopy(model)
    if policy in ('D', 'Dprime'):
        profiles = result['profiles' if policy == 'D' else 'unused_profiles']
        for profile in profiles.values():
            left, right = profile['normalization'], profile['high_frequency']
            key = 'detection_probability'
            left[key], right[key] = right[key], left[key]
    return result


def band(request):
    x = abs(request.z*request.mean_time)
    return str(next((i for i,b in enumerate((.1,1e3,1e6)) if x <= b),3))


def available(observed):
    done = set(observed)
    options = [a for a in ('modal', 'binding') + FEATURES if a not in done]
    if 'modal' not in done:
        return options
    if 'collocation' not in done:
        options.append('collocation')
    for i,n in enumerate(GRIDS):
        a = f'v2_{n}'
        if a in done:
            if observed[a].get('terminal'):
                break
            continue
        if i == 0 or f'v2_{GRIDS[i-1]}' in done:
            options.append(a)
        break
    return options


def verdict(observed):
    if any(v['verdict'] == 'violation_detected' for v in observed.values()):
        return 'violation_detected'
    required = ('modal', 'binding') + FEATURES + ('collocation',)
    v2 = [v for a,v in observed.items() if a.startswith('v2_') and v.get('terminal')]
    if all(a in observed and observed[a]['verdict']=='accept' for a in required) and v2 and v2[-1]['verdict']=='accept':
        return 'accept'
    return 'inconclusive'


def choose(policy, request, observed, remaining, model, permutation):
    """No artifact, condition, model hook, truth or unobserved evidence argument."""
    options = [a for a in available(observed) if cost(a,request,model) <= remaining]
    if not options:
        return None
    if policy == 'C1':
        return min(options,key=ORDER.index)
    if policy == 'C2':
        return min(options,key=lambda a:(cost(a,request,model),ORDER.index(a)))
    if policy == 'C3':
        return min(options,key=permutation.index)
    profile = model['profiles'][band(request)]
    def score(a):
        entry = profile['v2_64' if a=='modal' else a]
        accept = entry['accept_probability']
        if a == 'collocation' and request.tolerance < model['collocation_accept_floor']:
            accept = 0.
        return (entry['detection_probability'] + model['score_accept_weight']*accept)/cost(a,request,model)
    return min(options,key=lambda a:(-score(a),ORDER.index(a)))


def feature(request, name, model):
    """Individual original P probes without executing unrelated probes."""
    records = {}
    for kind in ('modal','fv'):
        size = 512 if name in ('normalization','first_moment') and kind=='fv' else 4096
        solver = lambda z: P.probe_response(kind,z,request.mean_time,size,model)
        if name == 'normalization':
            value = solver(0)
            error = abs(value-1)
            passed = error <= P.CONFIG.normalization_allowance
            records[kind] = dict(value=[value.real,value.imag],error=float(error))
        elif name == 'first_moment':
            h = 1e-6/request.mean_time
            value = solver(1j*h)
            mean = -value.imag/h
            error = abs(mean/request.mean_time-1)
            passed = error <= P.CONFIG.first_moment_allowance
            records[kind] = dict(value=[value.real,value.imag],error=float(error))
        elif name == 'second_moment':
            h = .01/request.mean_time
            zero, shifted = solver(0),solver(1j*h)
            moment = 2*(zero.real-shifted.real)/(h*request.mean_time)**2
            error = abs(moment/(20/7)-1)
            passed = error <= P.CONFIG.second_moment_allowance
            records[kind] = dict(zero=[zero.real,zero.imag],shifted=[shifted.real,shifted.imag],error=float(error))
        elif name == 'high_frequency':
            value = solver(1e4/request.mean_time)
            error = abs(100*value/np.sqrt(3/5)-1)
            passed = error <= P.CONFIG.high_frequency_allowance
            records[kind] = dict(value=[value.real,value.imag],error=float(error))
        else:
            raise ValueError('Unknown feature')
        records[kind]['accepted'] = bool(passed)
    return dict(verdict='accept' if all(r['accepted'] for r in records.values()) else 'violation_detected',probes=records)


class Executor:
    """Private test harness: hidden hooks never passed into choose()."""
    def __init__(self,request,digest,condition,previous):
        self.request,self.digest,self.condition,self.previous=request,digest,condition,previous
        self.model=condition if condition in ('model_scale','slab','mixture') else 'sphere'
        self.outcome=None
        self.values=[]
        self.observed={}

    def perform(self,action):
        if action not in available(self.observed):
            raise ValueError('Unavailable or repeated action')
        r=self.request
        if action=='modal':
            modes=P.plan_modes(r)
            if self.condition in FAULTS:
                receipt=execute(r,modes)
                self.outcome,_=inject(r,receipt,self.condition,self.previous)
            else:
                value=P.model_response('modal',r.z,r.mean_time,modes,self.model)
                self.outcome=dict(value=[value.real,value.imag],calls=1,modes_spent=modes,trace=[dict(call=1,modes=modes)])
            resolution=final_guard(r,self.outcome)
            result=dict(verdict='accept' if resolution['accepted'] else 'inconclusive',resolution=resolution,outcome=self.outcome)
        elif action=='binding':
            result=dict(verdict='accept' if request_hash(r)==self.digest else 'violation_detected')
        elif action in FEATURES:
            result=feature(r,action,self.model)
        elif action=='collocation':
            result=third_check(r.z,r.mean_time,complex(*self.outcome['value']),r.tolerance)
            result={k:v for k,v in result.items() if k!='seconds'}
        else:
            grid=int(action[3:])
            value=P.model_response('fv',r.z,r.mean_time,grid,self.model)
            self.values.append(value)
            result=dict(verdict='inconclusive',terminal=False,value=[value.real,value.imag])
            if grid >= P.CONFIG.first_decisive_grid:
                step=P.v2_step(self.values,grid,complex(*self.outcome['value']),r.tolerance)
                result.update(step=step,verdict=step['verdict'],terminal=bool(step['verdict']!='inconclusive' or step['futile'] or grid==GRIDS[-1]))
        self.observed[action]=result
        return copy.deepcopy(result)


def run_policy(executor,policy,budget,model,permutation,full=False):
    observed={}
    actions=[]
    spent=0.
    while True:
        if not full and verdict(observed)=='violation_detected':
            break
        action=choose('C1' if full else policy,executor.request,observed,budget-spent,model,permutation)
        if action is None:
            break
        observation=executor.perform(action)
        charge=cost(action,executor.request,model)
        spent+=charge
        observed[action]=observation
        actions.append(dict(action=action,cost=charge,observation=observation))
    result=verdict(observed)
    actions.append(dict(action='declare',cost=0.,verdict=result))
    return dict(verdict=result,spent=spent,actions=actions,
                early_violation=bool(result=='violation_detected' and available(observed)),
                futile_v2=any(v.get('step',{}).get('futile',False) for v in observed.values()))
