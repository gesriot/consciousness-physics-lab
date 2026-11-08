"""Paid local-channel self-checks with matched ordinary scheduling."""
import copy
import math
import numpy as np
from attention import agent, s3, monitor_planning as planning
from attention.build_a007_library import CHANNELS, channel
from attention.calibration_samples import execute_sample, maximum_cost, request, v2_observation
from attention.monitor import combined
from lab import frozen_pipeline as P
from lab.convergence_candidates import classify, ratio
from lab.finite_volume import response as uniform_response

POLICIES=('BA','C2A','C2R','BU','C2','D','Dprime')
ACTIVE=tuple(p for p in POLICIES if p!='C2')
QUOTAS={.2:2,.4:4,.6:6,1.:10}
FAMILIES=('v2','collocation')+agent.FEATURES+('binding','modal')
REPLACEMENT_GRIDS=(4096,8192,16384,32768)


def family(action):
    return 'v2' if action.startswith('v2') else action


def dependencies(r):
    return tuple(channel(f,r) for f in FAMILIES)


def hook(original,tolerance,kind):
    """A private injected channel changes diagnostics, never the scheduler."""
    observed=copy.deepcopy(original)
    if kind=='permissive':
        observed['verdict']='accept'
    elif kind in ('underestimated','compressed') and 'step' in observed:
        step=observed['step']
        key='uncertainty_estimate' if kind=='underestimated' else 'discrepancy'
        step[key]/=100.
        step['verdict']=classify(step['converging'],step['discrepancy'],step['uncertainty_estimate'],tolerance)
        step['futile']=bool(step['last_grid'] in (1024,2048) and step['verdict']=='inconclusive'
            and step['uncertainty_estimate']>=4*tolerance
            and ratio(step['last_difference'],step['previous_difference'])>=.5
            and step.get('antecedent_ratio',0.)>=.5)
        observed.update(verdict=step['verdict'],terminal=bool(step['verdict']!='inconclusive' or step['futile'] or step['last_grid']==agent.GRIDS[-1]))
    return observed


def sample_execution(sample,weights,kind='none'):
    if sample['family']!='v2':
        result=execute_sample(sample,weights)
        result['observed']=hook(result['original'],sample['request']['tolerance'],kind)
        result['verdict']=result['observed']['verdict']
        return result
    r=request(sample);values=[];trace=[]
    for grid in agent.GRIDS:
        values.append(P.model_response('fv',r.z,r.mean_time,grid,sample.get('model','sphere')))
        original=v2_observation(values,grid,complex(*sample['candidate']),r.tolerance)
        observed=hook(original,r.tolerance,kind)
        trace.append(dict(action='v2_'+str(grid),cost=grid*weights['fv_cell'],original=original,observed=observed))
        if observed['terminal']:break
    return dict(verdict=observed['verdict'],original=original,observed=observed,trace=trace,cost=sum(t['cost'] for t in trace))


def replacement_price(which,r,weights):
    f=which.split(':')[0]
    return sum(REPLACEMENT_GRIDS)*weights['fv_cell'] if f in ('v2','collocation') else agent.cost(f,r,{'cost_weights':weights})


def guarded_uniform(executor):
    r=executor.request
    values=[uniform_response(r.z,r.mean_time,n) for n in REPLACEMENT_GRIDS]
    step=P.v2_step(values,REPLACEMENT_GRIDS[-1],complex(*executor.outcome['value']),r.tolerance)
    d,u=step['discrepancy'],step['uncertainty_estimate'];v=step['verdict']
    margin=r.tolerance-d-u if v=='accept' else d-u-r.tolerance if v=='violation_detected' else None
    contraction=.5*step['previous_difference']-step['last_difference']
    threshold=4*step['rounding']
    passed=bool(margin is not None and margin>threshold and contraction>threshold)
    return dict(verdict=v if passed else 'inconclusive',unguarded_verdict=v,grids=list(REPLACEMENT_GRIDS),
        values=[[z.real,z.imag] for z in values],step=step,
        safety=dict(multiplier=4,decision_margin=margin,contraction_margin=contraction,threshold=threshold,passed=passed))


def replacement_result(which,executor,weights):
    f=which.split(':')[0]
    if f in ('v2','collocation'):result=guarded_uniform(executor)
    else:
        # Re-execute on an isolated clean channel with the same paid inputs.
        clean=copy.copy(executor);clean.observed=copy.deepcopy(executor.observed)
        clean.observed.pop(f,None)
        result=clean.perform(f)
    return result,replacement_price(which,executor.request,weights)


def sample_pools(library):
    pools={c:[] for c in CHANNELS}
    for c in CHANNELS:
        seen=set()
        for kind in ('boundary','nonaccept','accept'):
            choices=[s for s in library['samples'] if s['channel']==c and s['standard_kind']==kind]
            if choices:
                sample=choices[0]
                if sample['sample_id'] not in seen:pools[c].append(sample);seen.add(sample['sample_id'])
        if not pools[c]:raise ValueError('Missing calibration channel: '+c)
    return pools


def score_channels(requests,evidence,posteriors,closed,weights,pools,last,ordinary,policy,suspect,noise=False,cache=None):
    """Cache only per-input contributions, retaining the original sum order.

    A cache belongs to one run with fixed policy, noise, requests and model.
    Its logical key is input index plus evidence length; real evidence only
    grows, and the posterior changes exactly when that input gains evidence.
    Passing None retains the uncached reference computation.
    """
    raw={c:0. for c in CHANNELS}
    for i,(r,e) in enumerate(zip(requests,evidence)):
        if i in closed:continue
        entry=None if cache is None else cache.get(i)
        if entry is not None and entry[0]==len(e):
            contributions=entry[1]
        else:
            contributions=[]
            full=1. if policy=='BU' else posteriors[i].snapshot()['full_probabilities']['accept']
            if noise and policy!='BU' and full>0:full=float(np.nextafter(full,np.inf))
            for f in FAMILIES:
                a=f
                if f=='v2':
                    available=[x for x in agent.available(e) if x.startswith('v2_')]
                    paid=[x for x in e if x.startswith('v2_')]
                    a=available[0] if available else paid[-1] if paid else 'v2_64'
                probability=1. if policy=='BU' else posteriors[i].forecast(a)['probabilities'][0]
                if noise and policy!='BU' and probability>0:probability=float(np.nextafter(probability,-np.inf))
                c=channel(f,r)
                contribution=full*probability*replacement_price(c,r,weights)*(1+len(e)/len(agent.ORDER))
                contributions.append((c,contribution))
            if cache is not None:cache[i]=(len(e),contributions)
        # Do not reduce cached rows as vectors or alter summation association.
        for c,contribution in contributions:raw[c]+=contribution
    shifted={CHANNELS[(j+9)%len(CHANNELS)]:raw[c] for j,c in enumerate(CHANNELS)}
    magnitude=dict(changed=sum(raw[c]!=shifted[c] for c in CHANNELS),l1=sum(abs(raw[c]-shifted[c]) for c in CHANNELS))
    used=shifted if policy=='D' else raw
    scores={c:round(used[c]*(1+ordinary-last[c])/max(maximum_cost(s,weights) for s in pools[c]),9) for c in CHANNELS}
    eligible=[c for c in CHANNELS if c not in suspect] or list(CHANNELS)
    chosen=min(eligible,key=lambda c:(-scores[c],CHANNELS.index(c)))
    intervention=dict(active=magnitude if policy=='D' else dict(changed=0,l1=0.),
        unused=magnitude if policy=='Dprime' else dict(changed=0,l1=0.),counterfactual_active=magnitude,counterfactual_unused=copy.deepcopy(magnitude),equal_snapshot_magnitude=True)
    return chosen,scores,dict(raw=raw,used=used,intervention=intervention)


def run_batch(executors,policy,budget,model,library,package_id,fault_spec,fraction,noise=False):
    if policy not in POLICIES:raise ValueError('Unknown policy')
    weights=model['action_model']['cost_weights'];requests=[e.request for e in executors]
    pools=sample_pools(library);max_sample=max(maximum_cost(s,weights) for pool in pools.values() for s in pool)
    requested=QUOTAS[fraction];quota=min(requested,int(budget//max_sample)) if policy in ACTIVE else 0
    ordinary_budget=budget-quota*max_sample
    runtime=planning.Runtime(model) if policy in ('BA','D','Dprime') else None
    posteriors=[planning.Posterior(r,runtime,True) for r in requests] if runtime is not None else None
    evidence=[{} for _ in executors];extras=[{} for _ in executors]
    prices=[{a:agent.cost(a,r,model['action_model']) for a in agent.ORDER} for r in requests]
    counts={c:0 for c in CHANNELS};last={c:0 for c in CHANNELS};suspect=set()
    closed={};closure={};closure_required={};closure_indices={};spent_by=[0.]*len(executors);rows=[];alerts=[];history=[]
    touched=set();post_touched=set();ordinary=0;done_samples=0;post_samples=0;target_samples=0;target_exposures=0
    spent=0.;sample_cost=0.;ordinary_cost=0.;replacement_cost=0.;last_sample_ordinary=-1
    onset=fault_spec['onset_action'];target=fault_spec['target_channel'];kind=fault_spec['kind']
    rng=np.random.default_rng(fault_spec['random_seed']);detection=None;preonset_target=False
    pending_random=None;score_cache={}
    while True:
        insurance=[]
        for i,e in enumerate(evidence):
            if i in closed:continue
            v=agent.verdict(e);required=set(dependencies(requests[i]))&suspect
            if v=='violation_detected':
                closed[i]=v;closure[i]=ordinary;closure_required[i]=sorted(required);closure_indices[i]=len(rows);continue
            if v=='accept':
                missing=required-set(extras[i]);result=combined(e,extras[i],required)
                if result!='inconclusive' or not missing:
                    closed[i]=result;closure[i]=ordinary;closure_required[i]=sorted(required);closure_indices[i]=len(rows);continue
                insurance.extend((replacement_price(c,requests[i],weights),i,c) for c in missing)
        reserve=(quota-done_samples)*max_sample;remaining=max(0.,budget-spent-reserve)
        affordable=[x for x in insurance if x[0]<=remaining]
        is_extra=bool(affordable);choice=None
        if affordable:
            charge,i,a=min(affordable,key=lambda x:(x[0],x[1],CHANNELS.index(x[2])));choice=(i,a)
        else:
            pairs=[(prices[i][a],i,a) for i,e in enumerate(evidence) if i not in closed and agent.verdict(e)=='inconclusive'
                   for a in agent.available(e) if prices[i][a]<=remaining]
            if pairs:
                charge,i,a=min(pairs,key=lambda x:(x[0],x[1],agent.ORDER.index(x[2])));choice=(i,a)
        if done_samples<quota:
            scores=details=None
            if policy=='C2A':which=CHANNELS[(fault_spec['circle_phase']+done_samples)%len(CHANNELS)]
            elif policy=='C2R':
                if pending_random is None:pending_random=CHANNELS[int(rng.integers(len(CHANNELS)))]
                which=pending_random
            else:which,scores,details=score_channels(requests,evidence,posteriors,closed,weights,pools,last,ordinary,policy,suspect,noise,score_cache)
            deadline=done_samples/quota*ordinary_budget
            due=ordinary_cost>=deadline or choice is None
            anticipated=policy in ('BA','BU','D','Dprime') and ordinary>last_sample_ordinary and choice is not None and not is_extra and channel(a,requests[i])==which
            if due or anticipated:
                sample=pools[which][counts[which]%len(pools[which])]
                active=ordinary>=onset and kind!='none';spoiled=active and which==target
                result=sample_execution(sample,weights,kind if spoiled else 'none')
                charge=result['cost']
                if charge>max_sample or charge>budget-spent:raise AssertionError('Sample exceeds reserved cost')
                done_samples+=1;counts[which]+=1;last[which]=ordinary
                if ordinary>=onset:post_samples+=1
                if spoiled:target_samples+=1
                mismatch=result['verdict']!=sample['expected']
                target_detection=bool(mismatch and which not in suspect and spoiled and not preonset_target and detection is None)
                event=dict(channel=which,sample_id=sample['sample_id'],expected=sample['expected'],observed=result['verdict'],mismatch=mismatch,
                    package_id=package_id,ordinary_before=ordinary,action_index=len(rows),samples=done_samples,postonset_samples=post_samples,
                    target_detection=target_detection,postonset_inputs=len(post_touched),postonset_actions=max(0,ordinary-onset),fault_active=active,target=which==target)
                history.append(event)
                if mismatch and which not in suspect:
                    suspect.add(which);alerts.append(copy.deepcopy(event))
                    if which==target and ordinary<onset:preonset_target=True
                    if spoiled and not preonset_target and detection is None:
                        detection=dict(inputs=len(post_touched),samples=post_samples,actions=max(0,ordinary-onset),action_index=len(rows),ordinary_before=ordinary)
                rows.append(dict(input_index=None,action=sample['family'],channel=which,sample=True,replacement=False,
                    sample_id=sample['sample_id'],standard_kind=sample['standard_kind'],cost=charge,observation=result['observed'],original_observation=result['original'],
                    sample_trace=result['trace'],expected=sample['expected'],monitor=event,sample_scores=scores,score_details=details,
                    sample_reason='deadline' if ordinary_cost>=deadline else 'exhausted' if choice is None else 'anticipated_dependency',
                    sample_deadline=deadline,ordinary_cost_before=ordinary_cost,reserve_before=reserve,ordinary_before=ordinary,counter_before=ordinary,ordinary_after=ordinary,
                    fault_active=active,fault_applied=spoiled))
                spent+=charge;sample_cost+=charge;last_sample_ordinary=ordinary;pending_random=None
                continue
        if choice is None:break
        before=ordinary;which=a if is_extra else channel(a,requests[i]);active=before>=onset and kind!='none'
        spoiled=bool(active and which==target and not is_extra)
        touched.add(i)
        if before>=onset and not is_extra:post_touched.add(i)
        if is_extra:
            original,charge=replacement_result(a,executors[i],weights);hooked=observed=original;extras[i][a]=observed
            replacement_cost+=charge;forecast=None
        else:
            forecast=posteriors[i].forecast(a) if posteriors is not None else None
            original=executors[i].perform(a)
            if a.startswith('v2_'):
                original=v2_observation(executors[i].values,int(a[3:]),complex(*executors[i].outcome['value']),requests[i].tolerance)
            hooked=hook(original,requests[i].tolerance,kind if spoiled else 'none')
            executors[i].observed[a]=copy.deepcopy(hooked)
            observed=s3.observation(a,s3.token(a,hooked));evidence[i][a]=observed
            if posteriors is not None:posteriors[i].update(a,observed)
            ordinary+=1
            if spoiled:target_exposures+=1
        spent+=charge;ordinary_cost+=charge;spent_by[i]+=charge
        rows.append(dict(input_index=i,action=a,channel=which,sample=False,replacement=is_extra,cost=charge,
            observation=copy.deepcopy(observed),original_observation=copy.deepcopy(original),hooked_observation=copy.deepcopy(hooked),
            forecast=forecast,reserve_before=reserve,ordinary_before=before,counter_before=before,ordinary_after=ordinary,fault_active=active,fault_applied=spoiled))
    if done_samples!=quota:raise AssertionError('Unfulfilled matched sample quota')
    inputs=[]
    for i,e in enumerate(evidence):
        required=set(dependencies(requests[i]))&suspect
        inputs.append(dict(input_index=i,verdict=closed.get(i,combined(e,extras[i],required)),observed=e,replacements=extras[i],spent=spent_by[i],
            already_closed=i in closed,closure_ordinary=closure.get(i,ordinary),closure_counter=closure.get(i,ordinary),
            closure_action_index=closure_indices.get(i,len(rows)),suspect_at_closure=closure_required.get(i,sorted(required)),
            closure_after_onset=closure.get(i,ordinary)>=onset,required_replacements=closure_required.get(i,sorted(required))))
    return dict(policy=policy,budget=budget,spent=spent,inputs=inputs,actions=rows,alerts=alerts,monitor_history=history,suspect=sorted(suspect),
        ordinary_actions=ordinary,touched_inputs=len(touched),postonset_inputs=len(post_touched),postonset_samples=post_samples,
        target_exposures=target_exposures,target_samples=target_samples,onset_action=onset,preonset_target_alert=preonset_target,
        detected=detection is not None if kind!='none' else False,delay=detection,censored=detection is None if kind!='none' else False,
        detection=dict(detected=detection is not None if kind!='none' else False,delay=detection,
            censored=detection is None if kind!='none' else False,preonset_target_alert=preonset_target),
        sample_allocation=dict(quota=quota,requested=requested,maximum_price=max_sample,initial_reserve=quota*max_sample,
            ordinary_budget=ordinary_budget,executed=done_samples,counts_before={c:0 for c in CHANNELS},counts_after=counts),
        sample_cost=sample_cost,replacement_cost=replacement_cost,ordinary_cost=ordinary_cost,
        fault_exposure=dict(ordinary_target_actions=target_exposures,target_samples=target_samples,
            changed_ordinary_categories=sum(not r['sample'] and not r['replacement'] and r['original_observation']['verdict']!=r['hooked_observation']['verdict'] for r in rows),
            changed_sample_categories=sum(r['sample'] and r['original_observation']['verdict']!=r['observation']['verdict'] for r in rows)),
        support_audit=dict(prior_resets=sum(p.fallback_count for p in posteriors) if posteriors else 0,
            real_prediction_extensions=sum(p.real_extensions for p in posteriors) if posteriors else 0))
