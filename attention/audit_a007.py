"""Independent reconstruction of A007 accounting, hooks and decisions.

This module never calls the experimental scheduler, monitor or summarizer.
Fresh physical replay is a separate test; saved numerical traces are checked
algebraically and all working categories and accounting are exact.
"""
import argparse
import copy
import math
import subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from attention.audit_a005 import Audit, FEATURES, GRIDS, ORDER, available, base_verdict, price
from attention.compare_a004 import Archive
from attention.compare_a007 import decision_view
from attention.io import read_json, write_json
from attention import s3
from lab.guard import Request
from lab.replay import canonical_sha256
from lab.scientist import ROOT

POLICIES=('BA','C2A','C2R','BU','C2','D','Dprime')
FRACTIONS=('0.2','0.4','0.6','1.0')
FAULTS=('none','permissive','underestimated','compressed')
CHANNELS=tuple('v2:'+str(i) for i in range(6))+tuple('collocation:'+str(i) for i in range(6))+FEATURES+('binding','modal')
V2_EDGES=(0.0009344424533473637,0.7622440218285347,9.724297155975574,16360.24821532068,691475293.9820151)
TOL_EDGES=(1.166881434659866e-05,0.0002790990323118471,0.0015705174092262033,0.007414398085483424,0.019005124124090125)


def decode(fields):
    fields=dict(fields);fields['z']=complex(*fields['z']);return Request(**fields)


def channel(action,request):
    if action.startswith('v2'):
        magnitude=abs(request.z*request.mean_time);edges=V2_EDGES;family='v2'
    elif action.startswith('collocation'):
        magnitude=request.tolerance;edges=TOL_EDGES;family='collocation'
    else:return action
    return family+':'+str(next((i for i,b in enumerate(edges) if magnitude<=b),5))


def classify(converging,d,u,tolerance):
    if not all(math.isfinite(x) for x in (d,u,tolerance)):return 'inconclusive'
    if converging and d+u<=tolerance:return 'accept'
    if converging and d-u>tolerance:return 'violation_detected'
    return 'inconclusive'


def ratio(a,b):return a/b if b else (math.inf if a else 0.)


def verify_hook(check,action,original,observed,tolerance,spoiled,kind,tag):
    expected=copy.deepcopy(original)
    if spoiled and kind=='permissive' and action=='collocation':expected['verdict']='accept'
    elif spoiled and action.startswith('v2') and 'step' in expected:
        step=expected['step']
        if kind=='underestimated':step['uncertainty_estimate']/=100.
        elif kind=='compressed':step['discrepancy']/=100.
        step['verdict']=classify(step['converging'],step['discrepancy'],step['uncertainty_estimate'],tolerance)
        step['futile']=bool(step['last_grid'] in (1024,2048) and step['verdict']=='inconclusive'
            and step['uncertainty_estimate']>=4*tolerance
            and ratio(step['last_difference'],step['previous_difference'])>=.5
            and original['step'].get('antecedent_ratio',0.)>=.5)
        expected.update(verdict=step['verdict'],terminal=bool(step['verdict']!='inconclusive' or step['futile'] or step['last_grid']==GRIDS[-1]))
    check.equal(observed,expected,tag+': exact local hook')


def verify_v2_trace(check,trace,candidate,tolerance,weights,spoiled,kind,tag):
    values=[];total=0.
    for index,row in enumerate(trace):
        grid=int(row['action'][3:]);check.equal(grid,GRIDS[index],tag+': grid sequence')
        original=row['original'];values.append(complex(*original['value']))
        check.near(row['cost'],grid*weights['fv_cell'],tag+': grid price');total+=row['cost']
        if grid>=512:
            ds=[abs(b-a) for a,b in zip(values,values[1:])]
            rounding=32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
            d=abs(candidate-values[-1]);u=2*ds[-1]+rounding;converging=ds[-1]<=.5*ds[-2]
            step=original['step'];v=classify(converging,d,u,tolerance)
            for key,want in {'discrepancy':d,'uncertainty_estimate':u,'rounding':rounding,
                'last_difference':ds[-1],'previous_difference':ds[-2]}.items():check.near(step[key],want,tag+': '+key)
            check.equal(step['converging'],converging,tag+': convergence')
            check.equal(step['verdict'],v,tag+': clean step verdict')
            check.equal(original['verdict'],v,tag+': clean verdict')
        verify_hook(check,row['action'],original,row['observed'],tolerance,spoiled,kind,tag)
        check.check(index==len(trace)-1 or not row['observed']['terminal'],tag+': no step after terminal')
    check.check(bool(trace),tag+': nonempty trace')
    return total


def sample_max(sample,weights):
    return sum(GRIDS)*weights['fv_cell'] if sample['family']=='v2' else price(sample['family'],decode(sample['request']),weights)


def verify_guard(check,result,request,candidate,tag):
    values=[complex(*x) for x in result['values']];ds=[abs(b-a) for a,b in zip(values,values[1:])]
    check.equal(result['grids'],[4096,8192,16384,32768],tag+': independent grids')
    rounding=32*np.finfo(float).eps*32768**2*max(1,abs(values[-1]))
    u=2*ds[-1]+rounding;d=abs(candidate-values[-1]);converging=ds[-1]<=.5*ds[-2]
    raw=classify(converging,d,u,request.tolerance)
    margin=request.tolerance-d-u if raw=='accept' else d-u-request.tolerance if raw=='violation_detected' else None
    contraction=.5*ds[-2]-ds[-1]
    expected=raw if margin is not None and margin>4*rounding and contraction>4*rounding else 'inconclusive'
    check.equal(result['verdict'],expected,tag+': guarded replacement verdict')
    check.equal(result['unguarded_verdict'],raw,tag+': unguarded verdict')
    check.equal(result['step']['verdict'],raw,tag+': replacement step verdict')
    safety=result['safety']
    check.equal(safety['multiplier'],4,tag+': fixed guard multiplier')
    check.equal(safety['decision_margin'],margin,tag+': decision margin')
    check.near(safety['contraction_margin'],contraction,tag+': contraction margin')
    check.near(safety['threshold'],4*rounding,tag+': guard threshold')
    check.equal(safety['passed'],expected!='inconclusive',tag+': guard flag')
    for key,want in {'discrepancy':d,'uncertainty_estimate':u,'rounding':rounding,
        'last_difference':ds[-1],'previous_difference':ds[-2]}.items():check.near(result['step'][key],want,tag+': '+key)
    return expected



def learned_states(requests):
    """Compile categorical smoothing directly, without the online planner."""
    runtime=s3.Runtime(s3.load_model())
    likelihood=np.zeros((len(runtime.raw),len(ORDER),4))
    for j,a in enumerate(ORDER):
        k=4 if a.startswith('v2_') else 3 if a in ('modal','collocation') else 2
        present=runtime.raw[:,j]>=0
        likelihood[present,j,:k]=.02/k
        likelihood[np.flatnonzero(present),j,runtime.raw[present,j]]+=.98
    states=[]
    for r in requests:
        neighbors,_=runtime.neighbors(r);ids=np.flatnonzero(np.isin(runtime.groups,neighbors))
        prior=np.full(len(ids),1/len(ids))
        states.append(dict(weights=prior.copy(),prior=prior,likelihood=likelihood[ids],accepted=runtime.full[ids]==0))
    return states


def forecast(state,action):
    j=ORDER.index(action);k=4 if action.startswith('v2_') else 3 if action in ('modal','collocation') else 2
    probabilities=state['weights']@state['likelihood'][:,j];mass=float(probabilities.sum())
    return [1/k]*k if mass<=0 or not math.isfinite(mass) else (probabilities[:k]/mass).tolist()


def audit_run(check,run,infos,records,spec,fraction,weights,library,tag,noise=False):
    policy=run['policy'];requests=[decode(records[i['task_id']]['request']) for i in infos]
    samples={s['sample_id']:s for s in library['samples']}
    learned=learned_states(requests) if policy in ('BA','D','Dprime') else None
    pools={c:[] for c in CHANNELS}
    for c in CHANNELS:
        for kind in ('boundary','nonaccept','accept'):
            selected=[s for s in library['samples'] if s['channel']==c and s['standard_kind']==kind]
            if selected and selected[0]['sample_id'] not in [s['sample_id'] for s in pools[c]]:pools[c].append(selected[0])
    maximum=max(sample_max(s,weights) for pool in pools.values() for s in pool)
    requested={'.2':2,'.4':4,'.6':6,'1.0':10}.get(str(fraction))
    if requested is None:requested={.2:2,.4:4,.6:6,1.:10}[float(fraction)]
    quota=min(requested,int(run['budget']//maximum)) if policy!='C2' else 0
    allocation=run['sample_allocation']
    check.equal(allocation['quota'],quota,tag+': quota')
    check.equal(allocation['requested'],requested,tag+': requested quota')
    check.near(allocation['maximum_price'],maximum,tag+': maximum sample price')
    check.near(allocation['initial_reserve'],quota*maximum,tag+': initial reserve')
    ordinary_budget=run['budget']-quota*maximum
    check.near(allocation['ordinary_budget'],ordinary_budget,tag+': ordinary budget')
    v2_traces=[[] for _ in infos]
    evidence=[{} for _ in infos];extras=[{} for _ in infos];spent_by=[0. for _ in infos]
    spent=sample_cost=replacement_cost=ordinary_cost=0.
    ordinary=done_samples=post_samples=target_exposures=target_samples=0
    touched=set();post_touched=set();suspect=set();alerts=[];history=[]
    counts={c:0 for c in CHANNELS};last={c:0 for c in CHANNELS}
    detection=None;preonset=False;closures={};last_sample_ordinary=-1
    onset=spec['onset_action'];target=spec['target_channel'];kind=spec['kind']
    rng=np.random.default_rng(spec['random_seed'])
    final_inputs=run['inputs'];rows=run['actions']
    def capture_closures(index):
        for i,result in enumerate(final_inputs):
            if result['closure_action_index']==index:
                required=sorted(set(channel(f,requests[i]) for f in ('v2','collocation')+FEATURES+('binding','modal'))&suspect)
                check.equal(result['required_replacements'],required,tag+': closure required channels')
                check.equal(result['suspect_at_closure'],required,tag+': closure snapshot')
                check.equal(result['closure_counter'],ordinary,tag+': closure ordinary counter')
                check.equal(result['closure_after_onset'],ordinary>=onset,tag+': closure phase')
                base=base_verdict(evidence[i]);v=base
                if base=='accept':
                    if any(x['verdict']=='violation_detected' for x in extras[i].values()):v='violation_detected'
                    elif not all(extras[i].get(c,{}).get('verdict')=='accept' for c in required):v='inconclusive'
                check.equal(result['verdict'],v,tag+': strict final verdict')
                closures[i]=index
    for index,row in enumerate(rows):
        capture_closures(index)
        rowtag=tag+': action '+str(index)
        check.equal(row['ordinary_before'],ordinary,rowtag+': counter before')
        check.equal(row['counter_before'],ordinary,rowtag+': counter alias')
        check.near(row['reserve_before'],(quota-done_samples)*maximum,rowtag+': reserved remainder')
        active=ordinary>=onset and kind!='none'
        check.equal(row['fault_active'],active,rowtag+': active fault')
        remaining=max(0.,run['budget']-spent-(quota-done_samples)*maximum)
        insurance=[]
        for j,ej in enumerate(evidence):
            if j in closures or base_verdict(ej)!='accept':continue
            required=set(channel(f,requests[j]) for f in ('v2','collocation')+FEATURES+('binding','modal'))&suspect
            for c in required-set(extras[j]):
                fam=c.split(':')[0];charge=61440*weights['fv_cell'] if fam in ('v2','collocation') else price(fam,requests[j],weights)
                if charge<=remaining:insurance.append((charge,j,c))
        if insurance:
            choice=min(insurance,key=lambda v:(v[0],v[1],CHANNELS.index(v[2])));is_extra=True
        else:
            pairs=[(price(a,requests[j],weights),j,a) for j,ej in enumerate(evidence)
                if j not in closures and base_verdict(ej)=='inconclusive' for a in available(ej)
                if price(a,requests[j],weights)<=remaining]
            choice=min(pairs,key=lambda v:(v[0],v[1],ORDER.index(v[2]))) if pairs else None;is_extra=False
        which=row['channel'];spoiled=active and which==target and not row['replacement']
        check.equal(row['fault_applied'],spoiled,rowtag+': local exposure')
        if row['sample']:
            sample=samples[row['sample_id']];r=decode(sample['request'])
            check.equal(which,sample['channel'],rowtag+': sample channel')
            check.equal(which,channel(sample['family'],r),rowtag+': sample routed')
            chosen=pools[which][counts[which]%len(pools[which])]
            check.equal(sample['sample_id'],chosen['sample_id'],rowtag+': frozen sample cycle')
            if policy=='C2A':check.equal(which,CHANNELS[(spec['circle_phase']+done_samples)%18],rowtag+': circular channel')
            if policy=='C2R':check.equal(which,CHANNELS[int(rng.integers(18))],rowtag+': random channel')
            check.equal(row['expected'],sample['expected'],rowtag+': frozen expectation')
            if sample['family']=='v2':
                actual_cost=verify_v2_trace(check,row['sample_trace'],complex(*sample['candidate']),r.tolerance,weights,spoiled,kind,rowtag)
                check.equal(row['observation'],row['sample_trace'][-1]['observed'],rowtag+': sample final observation')
                check.equal(row['original_observation'],row['sample_trace'][-1]['original'],rowtag+': sample final clean')
            else:
                actual_cost=sample_max(sample,weights)
                verify_hook(check,sample['family'],row['original_observation'],row['observation'],r.tolerance,spoiled,kind,rowtag)
            check.near(row['cost'],actual_cost,rowtag+': sample price')
            check.check(row['cost']<=maximum,rowtag+': reserved maximum')
            deadline=done_samples/quota*ordinary_budget
            check.near(row['sample_deadline'],deadline,rowtag+': deadline')
            anticipated=policy in ('BA','BU','D','Dprime') and ordinary>last_sample_ordinary and choice is not None and not is_extra and channel(choice[2],requests[choice[1]])==which
            check.check(ordinary_cost>=deadline or choice is None or anticipated,rowtag+': sample timing eligible')
            reason='deadline' if ordinary_cost>=deadline else 'exhausted' if choice is None else 'anticipated_dependency'
            check.equal(row['sample_reason'],reason,rowtag+': sample timing reason')
            check.near(row['ordinary_cost_before'],ordinary_cost,rowtag+': ordinary spend')
            details=row['score_details']
            if details is not None:
                raw=details['raw']
                if policy in ('BA','BU','D','Dprime'):
                    usage={c:0. for c in CHANNELS}
                    for j,ej in enumerate(evidence):
                        if j in closures:continue
                        full=1. if policy=='BU' else float(learned[j]['weights']@learned[j]['accepted'])
                        if noise and policy!='BU' and full>0:full=float(np.nextafter(full,np.inf))
                        for fam in ('v2','collocation')+FEATURES+('binding','modal'):
                            action=fam
                            if fam=='v2':
                                pending=[a for a in available(ej) if a.startswith('v2_')];paid=[a for a in ej if a.startswith('v2_')]
                                action=pending[0] if pending else paid[-1] if paid else 'v2_64'
                            probability=1. if policy=='BU' else forecast(learned[j],action)[0]
                            if noise and policy!='BU' and probability>0:probability=float(np.nextafter(probability,-np.inf))
                            c=channel(fam,requests[j]);charge=61440*weights['fv_cell'] if fam in ('v2','collocation') else price(fam,requests[j],weights)
                            usage[c]+=full*probability*charge*(1+len(ej)/len(ORDER))
                    check.equal(raw,usage,rowtag+': independently reconstructed risk or usage')
                shifted={CHANNELS[(j+9)%18]:raw[c] for j,c in enumerate(CHANNELS)}
                used=shifted if policy=='D' else raw
                check.equal(details['used'],used,rowtag+': relevant permutation')
                magnitude=dict(changed=sum(raw[c]!=shifted[c] for c in CHANNELS),l1=sum(abs(raw[c]-shifted[c]) for c in CHANNELS))
                inter=details['intervention'];zero=dict(changed=0,l1=0.)
                for key in ('counterfactual_active','counterfactual_unused'):check.equal(inter[key],magnitude,rowtag+': same-snapshot magnitude')
                check.equal(inter['active'],magnitude if policy=='D' else zero,rowtag+': active intervention')
                check.equal(inter['unused'],magnitude if policy=='Dprime' else zero,rowtag+': unused intervention')
                scores={c:round(used[c]*(1+ordinary-last[c])/max(sample_max(s,weights) for s in pools[c]),9) for c in CHANNELS}
                check.equal(row['sample_scores'],scores,rowtag+': quantized scores')
                eligible=[c for c in CHANNELS if c not in suspect] or list(CHANNELS)
                check.equal(which,min(eligible,key=lambda c:(-scores[c],CHANNELS.index(c))),rowtag+': deterministic maximum')
            done_samples+=1;counts[which]+=1;last[which]=ordinary
            if ordinary>=onset:post_samples+=1
            if spoiled:target_samples+=1
            mismatch=row['observation']['verdict']!=sample['expected'];event=row['monitor']
            for key,want in dict(channel=which,sample_id=sample['sample_id'],expected=sample['expected'],observed=row['observation']['verdict'],mismatch=mismatch,
                ordinary_before=ordinary,action_index=index,samples=done_samples,postonset_samples=post_samples,
                target_detection=bool(mismatch and which not in suspect and spoiled and not preonset and detection is None),postonset_inputs=len(post_touched),postonset_actions=max(0,ordinary-onset),fault_active=active,target=which==target).items():check.equal(event[key],want,rowtag+': event '+key)
            history.append(event)
            if mismatch and which not in suspect:
                suspect.add(which);alerts.append(event)
                if which==target and ordinary<onset:preonset=True
                if spoiled and not preonset and detection is None:
                    detection=dict(inputs=len(post_touched),samples=post_samples,actions=max(0,ordinary-onset),action_index=index,ordinary_before=ordinary)
            sample_cost+=row['cost'];last_sample_ordinary=ordinary
        else:
            i=row['input_index'];r=requests[i];a=row['action']
            check.check(choice is not None,rowtag+': ordinary choice exists')
            if choice is not None:
                check.equal((i,a),(choice[1],choice[2]),rowtag+': cheapest deterministic operation')
                check.equal(row['replacement'],is_extra,rowtag+': insurance priority')
            if done_samples<quota:check.check(ordinary_cost<done_samples/quota*ordinary_budget,rowtag+': no overdue sample skipped')
            check.check(i not in closures,rowtag+': closed input never revisited')
            touched.add(i)
            if row['replacement']:
                check.equal(which,a,rowtag+': replacement channel')
                check.check(which in suspect,rowtag+': suspected replacement')
                check.check(which not in extras[i],rowtag+': no repeated replacement')
                check.equal(base_verdict(evidence[i]),'accept',rowtag+': replacement after strict evidence')
                family=a.split(':')[0]
                actual_cost=61440*weights['fv_cell'] if family in ('v2','collocation') else price(family,r,weights)
                if family in ('v2','collocation'):verify_guard(check,row['observation'],r,complex(*infos[i]['outcome']['value']),rowtag)
                check.equal(row['observation'],row['original_observation'],rowtag+': clean replacement')
                check.equal(row['observation'],row['hooked_observation'],rowtag+': replacement unspoiled')
                extras[i][which]=row['observation'];replacement_cost+=row['cost']
            else:
                check.check(a in available(evidence[i]),rowtag+': available ordinary action')
                check.equal(which,channel(a,r),rowtag+': ordinary routed')
                actual_cost=price(a,r,weights)
                verify_hook(check,a,row['original_observation'],row['hooked_observation'],r.tolerance,spoiled,kind,rowtag)
                if a=='modal':check.equal(row['original_observation']['outcome'],infos[i]['outcome'],rowtag+': candidate identity')
                if a.startswith('v2_'):
                    original=row['original_observation'];values=[complex(*t['original']['value']) for t in v2_traces[i]]+[complex(*original['value'])]
                    grid=int(a[3:])
                    if grid>=512:
                        ds=[abs(b-a) for a,b in zip(values,values[1:])];rounding=32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
                        d=abs(complex(*infos[i]['outcome']['value'])-values[-1]);u=2*ds[-1]+rounding
                        step=original['step'];converging=ds[-1]<=.5*ds[-2]
                        for key,want in {'discrepancy':d,'uncertainty_estimate':u,'rounding':rounding,'last_difference':ds[-1],'previous_difference':ds[-2]}.items():check.near(step[key],want,rowtag+': clean V2 '+key)
                        check.equal(step['converging'],converging,rowtag+': clean V2 convergence')
                        check.equal(original['verdict'],classify(converging,d,u,r.tolerance),rowtag+': clean V2 verdict')
                    v2_traces[i].append(dict(action=a,original=original,observed=row['hooked_observation'],cost=row['cost']))
                expected={'verdict':row['hooked_observation']['verdict']}
                if a.startswith('v2_'):expected['terminal']=row['hooked_observation']['terminal']
                check.equal(row['observation'],expected,rowtag+': delivered categories')
                if learned is not None:
                    state=learned[i];wanted_forecast=forecast(state,a)
                    for actual,want in zip(row['forecast']['probabilities'],wanted_forecast):check.near(actual,want,rowtag+': independent forecast')
                    verdict=expected['verdict'];category=0 if verdict=='accept' else 1 if verdict=='violation_detected' else 3 if a.startswith('v2_') and expected['terminal'] else 2
                    updated=state['weights']*state['likelihood'][:,ORDER.index(a),category];mass=float(updated.sum())
                    state['weights']=updated/mass if mass>0 and math.isfinite(mass) else state['prior'].copy()
                evidence[i][a]=row['observation']
                if ordinary>=onset:post_touched.add(i)
                ordinary+=1
                if spoiled:target_exposures+=1
            check.near(row['cost'],actual_cost,rowtag+': operation price')
            ordinary_cost+=row['cost'];spent_by[i]+=row['cost']
        spent+=row['cost']
        check.check(spent+(quota-done_samples)*maximum<=run['budget']+1e-12,rowtag+': budget with reserve')
        check.equal(row['ordinary_after'],ordinary,rowtag+': counter after')
    capture_closures(len(rows))
    check.equal(len(closures),len(infos),tag+': all finalizations traced')
    for i,result in enumerate(final_inputs):
        check.equal(result['observed'],evidence[i],tag+': final evidence')
        check.equal(result['replacements'],extras[i],tag+': final replacements')
        check.near(result['spent'],spent_by[i],tag+': input spent')
        error=abs(complex(*infos[i]['outcome']['value'])-complex(*infos[i]['reference']))
        accurate=error<=requests[i].tolerance
        check.near(infos[i]['absolute_error'],error,tag+': physical error')
        check.equal(infos[i]['accurate'],accurate,tag+': physical label')
        correct=(accurate and result['verdict']=='accept') or (not accurate and result['verdict']=='violation_detected')
        check.equal(result['correct'],correct,tag+': correct verdict')
        check.equal(result['false_accept'],not accurate and result['verdict']=='accept',tag+': false acceptance')
    for key,want in dict(spent=spent,sample_cost=sample_cost,ordinary_cost=ordinary_cost,replacement_cost=replacement_cost).items():check.near(run[key],want,tag+': total '+key)
    for key,want in dict(ordinary_actions=ordinary,touched_inputs=len(touched),postonset_inputs=len(post_touched),postonset_samples=post_samples,
        target_exposures=target_exposures,target_samples=target_samples,suspect=sorted(suspect),alerts=alerts,monitor_history=history,
        preonset_target_alert=preonset,detected=detection is not None if kind!='none' else False,
        censored=detection is None if kind!='none' else False,delay=detection).items():check.equal(run[key],want,tag+': total '+key)
    check.equal(done_samples,quota,tag+': exact executed quota')
    check.equal(allocation['executed'],quota,tag+': saved executed quota')
    check.equal(allocation['counts_after'],counts,tag+': channel counts')
    return dict(correct=sum(x['correct'] for x in final_inputs),
        false_accepts=sum(x['false_accept'] for x in final_inputs),
        false_accepts_before=sum(x['false_accept'] and not x['closure_after_onset'] for x in final_inputs),
        false_accepts_after=sum(x['false_accept'] and x['closure_after_onset'] for x in final_inputs),
        detected=detection is not None,penalized_inputs=detection['inputs'] if detection else 46,
        penalized_samples=detection['samples'] if detection else quota+1,alerts=len(alerts),
        spent=spent,sample_cost=sample_cost,replacement_cost=replacement_cost,samples=quota)


def parity_view(run):
    """Only placement of an unused intervention differs by design."""
    data=copy.deepcopy(run);data.pop('policy',None)
    for row in data['actions']:
        if row.get('score_details') is not None:row['score_details'].pop('intervention',None)
    return decision_view(data)


def audit_paths(path,negative_control=False):
    archive=Archive(path);check=Audit();model=s3.load_model();weights=model['action_model']['cost_weights']
    library=read_json(ROOT/'attention/sample_library_a007.json');totals={};first=None
    for package in archive.packages():
        if first is None:first=copy.deepcopy(package)
        root=archive.metadata;records=root['intake_channel'];pid=package['package_id']
        check.equal(len(package['inputs']),45,'package input count')
        check.equal(package['fault_specs'],root['fault_specs'][pid],'frozen package faults')
        reference_cost=sum(i['full']['spent'] for i in package['inputs'])
        check.near(package['reference_cost'],reference_cost,'reference cost')
        for fault in FAULTS:
            for fraction in FRACTIONS:
                spec=dict(package['fault_specs'][fault],random_seed=package['fault_specs'][fault]['random_seed']+int(float(fraction)*100))
                quota=None
                for policy in POLICIES:
                    run=package['conditions'][fault][fraction][policy];tag=f'{pid}/{fault}/{fraction}/{policy}'
                    check.equal(run['policy'],policy,tag+': policy')
                    check.near(run['budget'],float(fraction)*reference_cost,tag+': budget fraction')
                    measures=audit_run(check,run,package['inputs'],records,spec,fraction,weights,library,tag,root['last_bit_noise'])
                    for key in ('correct','false_accepts_before','false_accepts_after'):check.equal(run[key],measures[key],tag+': measured '+key)
                    check.equal(measures['false_accepts_before'],0,tag+': no pre-onset false acceptance')
                    if policy!='C2':
                        if quota is None:quota=run['sample_allocation']['quota']
                        check.equal(run['sample_allocation']['quota'],quota,tag+': matched active quota')
                    key=(fault,fraction,policy)
                    totals.setdefault(key,[]).append((pid,measures,run['detection'],
                        len(run['alerts']),sum(not a.get('target_detection',False) for a in run['alerts']),run['ordinary_cost'],
                        run['fault_exposure'],sum(a.get('replacement') and a['observation'].get('safety',{}).get('passed') is False for a in run['actions'])))
                runs=package['conditions'][fault][fraction]
                check.equal(parity_view(runs['BA']),parity_view(runs['Dprime']),f'{pid}/{fault}/{fraction}: irrelevant intervention parity')
    root=archive.metadata
    check.equal(archive.package_count,12,'12 packages')
    check.equal(root['experiment'],'A007','experiment')
    check.equal(len(root['intake_channel']),60,'60 requests')
    check.equal(root['intake_sha256'],canonical_sha256([root['intake_channel'],root['fault_specs']]),'intake digest')
    check.equal(root['sample_library_sha256'],canonical_sha256(library),'frozen library digest')
    check.equal(root['self_model_sha256'],canonical_sha256(model),'frozen model digest')
    check.equal(root['full_pipeline_parity_failures'],[],'strict full-pipeline parity')
    for name,entry in root['request_overlap'].items():check.equal(entry['exact_request_matches'],0,'no overlap '+name)
    faults_rng=np.random.default_rng(20261203);phases=np.random.default_rng(20261205)
    for pid,row in enumerate(root['fault_specs']):
        onset=int(faults_rng.integers(20,61));phase=int(phases.integers(18))
        for kind in FAULTS:
            band=int(faults_rng.integers(6));family='collocation' if kind in ('none','permissive') else 'v2'
            check.equal(row[kind],dict(kind=kind,target_channel=family+':'+str(band),onset_action=onset,circle_phase=phase,random_seed=20261204+1000*pid),'seeded fault specification')
    for (fault,fraction,policy),runs in totals.items():
        saved=root['summary'][fault][fraction][policy];tag=f'summary/{fault}/{fraction}/{policy}'
        sums={k:sum(x[1][k] for x in runs) for k in ('correct','false_accepts_before','false_accepts_after','spent','sample_cost','replacement_cost','samples')}
        sums.update(alerts=sum(x[3] for x in runs),false_alerts=sum(x[4] for x in runs),ordinary_cost=sum(x[5] for x in runs),
            detected_packages=sum(x[2]['detected'] for x in runs),censored_packages=sum(x[2]['censored'] for x in runs),
            delay_inputs_penalized=sum(x[1]['penalized_inputs'] for x in runs) if fault!='none' else 0,
            delay_samples_penalized=sum(x[1]['penalized_samples'] for x in runs) if fault!='none' else 0,
            guarded_replacements=sum(x[7] for x in runs),
            changed_ordinary_categories=sum(x[6]['changed_ordinary_categories'] for x in runs),
            changed_sample_categories=sum(x[6]['changed_sample_categories'] for x in runs),
            target_observations=sum(x[6]['ordinary_target_actions'] for x in runs),target_samples=sum(x[6]['target_samples'] for x in runs))
        sums.update(correct_by_package=[x[1]['correct'] for x in runs],false_accepts_by_package=[x[1]['false_accepts_after'] for x in runs],
            detection_by_package=[x[2] for x in runs],observed_delays=[dict(package_id=x[0],**x[2]['delay']) for x in runs if x[2]['detected']])
        for key,want in sums.items():check.equal(saved[key],want,tag+': '+key)
    passing=0
    for row,fraction in zip(root['stopping_rule']['budgets'],FRACTIONS[:3]):
        combined={p:{m:sum(root['summary'][f][fraction][p][m] for f in FAULTS[1:]) for m in ('false_accepts_after','delay_inputs_penalized')} for p in ('BA','BU','C2A')}
        wins={m:combined['BA'][m]<min(combined['BU'][m],combined['C2A'][m]) for m in ('false_accepts_after','delay_inputs_penalized')}
        passed=any(wins.values());passing+=passed
        check.equal(row,dict(fraction=fraction,totals=combined,wins=wins,passes=passed),'registered stopping budget')
    check.equal(root['stopping_rule']['passing_limited_budgets'],passing,'passing budgets')
    check.equal(root['stopping_rule']['series_closed'],passing<2,'registered closure')
    protocol=ROOT/'attention/protocols/A007-local-channel-calibration.md'
    import hashlib
    check.equal(root['protocol_sha256'],hashlib.sha256(protocol.read_bytes()).hexdigest(),'protocol hash')
    result=subprocess.run(['git','merge-base','--is-ancestor',root['protocol_commit'],root['git_head']],cwd=ROOT,capture_output=True)
    check.equal(result.returncode,0,'protocol precedes execution commit')
    prior=subprocess.check_output(['git','ls-tree',root['protocol_commit'],'attention/channel_monitor.py'],cwd=ROOT,text=True)
    check.equal(prior,'','confirmatory implementation absent at preregistration')
    controls=[]
    if negative_control and first is not None:
        baseline=first['conditions']['permissive']['0.4']['BA'];spec=dict(first['fault_specs']['permissive'],random_seed=first['fault_specs']['permissive']['random_seed']+40)
        mutations=[('operation_price',lambda r:r['actions'][0].__setitem__('cost',r['actions'][0]['cost']+1)),
            ('final_verdict',lambda r:r['inputs'][0].__setitem__('verdict','accept' if r['inputs'][0]['verdict']!='accept' else 'inconclusive')),
            ('sample_channel',lambda r:next(a for a in r['actions'] if a['sample']).__setitem__('channel','not-a-channel')),
            ('detection',lambda r:r.__setitem__('detected',not r['detected']))]
        for name,mutate in mutations:
            changed=copy.deepcopy(baseline);mutate(changed);probe=Audit()
            try:audit_run(probe,changed,first['inputs'],root['intake_channel'],spec,'0.4',weights,library,'mutation',root['last_bit_noise'])
            except (KeyError,ValueError,IndexError,TypeError):probe.check(False,'Malformed mutation rejected')
            detected=bool(probe.errors);controls.append(dict(mutation=name,detected=detected))
            check.check(detected,'negative control '+name)
    return dict(passed=not check.errors,checks=check.checks,errors=check.errors,packages=archive.package_count,
        negative_controls=controls,scope='Independent action accounting, cheapest ordinary and replacement choice, sample timing, weighted-usage scores, categorical hooks, guarded replacement, truth labels, quotas, closure timing, summaries and stopping rule. Learned posterior risk and model-free usage are independently reconstructed from frozen categorical trajectories; no experimental planner is called. Fresh physics is checked by replay.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('--output',required=True)
    parser.add_argument('--negative-control',action='store_true');args=parser.parse_args()
    result=audit_paths(args.source,args.negative_control);write_json(args.output,result)
    print(result)
    if not result['passed']:raise SystemExit(1)
