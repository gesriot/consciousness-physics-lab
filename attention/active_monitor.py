"""Paid active calibration with matched, outcome-independent sample quotas."""
import copy
from attention import agent, s3, monitor_planning as planning
from attention.monitor import Monitor, combined, replacement, replacement_cost
from attention.calibration_samples import FAMILIES, execute_sample, maximum_cost, spoil_v2, v2_observation

POLICIES=('B','BM','BA','C2A','C2','D','Dprime')
ACTIVE=('BA','C2A','D','Dprime')


def family(action):
    return 'v2' if action.startswith('v2_') else action


def target(fault):
    return {'uninformative':'second_moment','underestimated':'v2'}.get(fault,'collocation')


class State(Monitor):
    def __init__(self,threshold,policy,library):
        super().__init__(threshold,'BM')
        self.active_policy=policy
        self.samples={f:[s for s in library['samples'] if s['family']==f] for f in FAMILIES}
        self.counts={f:0 for f in FAMILIES};self.last={f:0 for f in FAMILIES}
        self.ordinary=0;self.total_samples=0
        self.expectations={s['sample_id']:s['expected'] for s in library['samples']}
        self.unused_expectations=copy.deepcopy(self.expectations)
        destination=self.expectations if policy=='D' else self.unused_expectations if policy=='Dprime' else None
        if destination is not None:
            for s in library['samples']:
                if s['family'] in ('collocation','second_moment'):destination[s['sample_id']]='accept'

    def next_sample(self,which):
        choices=self.samples[which];counter=self.counts[which]
        positive=[s for s in choices if s['expected']=='accept']
        negative=[s for s in choices if s['expected']!='accept']
        pool=(positive if counter%2==0 else negative) or choices
        return pool[(counter//2)%len(pool)]

    def observe_sample(self,sample,result,location):
        which=sample['family'];expected=self.expectations[sample['sample_id']]
        mismatch=expected!=result['verdict']
        row=dict(action=which,sample_id=sample['sample_id'],expected=expected,observed=result['verdict'],mismatch=mismatch,**location)
        self.counts[which]+=1;self.total_samples+=1;self.last[which]=self.ordinary
        self.history.append(row)
        if mismatch and which not in self.suspect:
            self.suspect.add(which);self.alerts.append(copy.deepcopy(row))
        return row


def sample_choice(schedule,state,closed,weights):
    scores={}
    for which in FAMILIES:
        loss=0.
        for i,e in enumerate(schedule.evidence):
            if i in closed or agent.verdict(e)=='violation_detected':continue
            a=which
            if which=='v2':
                possible=[x for x in agent.available(e) if x.startswith('v2_')]
                paid=[x for x in e if x.startswith('v2_')]
                a=possible[0] if possible else paid[-1] if paid else 'v2_512'
            p=schedule.posteriors[i]
            acceptance=p.forecast(a)['probabilities'][0]
            full=p.snapshot()['full_probabilities']['accept']
            replacement_price=(sum(agent.GRIDS)*weights['fv_cell'] if which=='v2' else replacement_cost(which,schedule.requests[i],{'cost_weights':weights}))
            loss+=acceptance*full*replacement_price*(1+len(e)/len(agent.ORDER))
        maxprice=max(maximum_cost(s,weights) for s in state.samples[which])
        scores[which]=round(loss*(1+state.ordinary-state.last[which])/maxprice,9)
    if any(f not in state.suspect for f in FAMILIES):
        scores={f:(0. if f in state.suspect else v) for f,v in scores.items()}
    chosen=min(FAMILIES,key=lambda f:(-scores[f],FAMILIES.index(f)))
    return chosen,scores


def ordinary_observation(executor,action,fault):
    original=executor.perform(action)
    if action.startswith('v2_') and 'step' in original:
        original=v2_observation(executor.values,int(action[3:]),complex(*executor.outcome['value']),executor.request.tolerance)
    if fault=='underestimated' and action.startswith('v2_'):
        hooked=spoil_v2(original,executor.request.tolerance)
    elif (fault=='permissive' and action=='collocation') or (fault=='uninformative' and action=='second_moment'):
        hooked=dict(verdict='accept')
    else:hooked=copy.deepcopy(original)
    # Private availability must follow the delivered V2 terminal decision.
    executor.observed[action]=copy.deepcopy(hooked)
    observed=s3.observation(action,s3.token(action,hooked))
    return original,hooked,observed


def extra_price(action,executor,weights):
    if action=='v2':return sum(agent.GRIDS)*weights['fv_cell']
    return replacement_cost(action,executor.request,{'cost_weights':weights})


def extra_result(action,executor,weights):
    if action=='v2':
        from lab.common_mode import encode
        recipe=dict(family='v2',request=encode(executor.request),candidate=executor.outcome['value'],model=executor.model)
        result=execute_sample(recipe,weights)
        return dict(verdict=result['verdict'],trace=result['trace'],cost=result['cost']),result['cost']
    return replacement(executor,action),extra_price(action,executor,weights)


def run_batch(executors,policy,budget,model,state,package_id,fault,onset,fraction):
    schedule=planning.Scheduler([e.request for e in executors],model,'U0T')
    schedule.requests=[e.request for e in executors]
    prices=schedule.prices;weights=model['action_model']['cost_weights']
    active=policy in ACTIVE;passive=policy=='BM'
    max_sample=max(maximum_cost(s,weights) for ss in state.samples.values() for s in ss)
    requested={.2:3,.4:6,.6:9,1.:15}[fraction]
    quota=min(requested,int(budget//max_sample)) if active else 0
    ordinary_budget=budget-quota*max_sample
    done_samples=0;sample_cost=0.;ordinary_cost=0.;last_sample_ordinary=-1
    closed={};extras=[{} for _ in executors];spent_by=[0.]*len(executors)
    rows=[];spent=0.;touched=set();target_exposures=0;target_samples=0
    alerts_start=len(state.alerts);history_start=len(state.history)
    start_counts=copy.deepcopy(state.counts)
    while True:
        required=set(state.suspect) if active or passive else set()
        insurance=[]
        for i,e in enumerate(schedule.evidence):
            if i in closed:continue
            v=agent.verdict(e)
            if v=='violation_detected':closed[i]=v;continue
            if v=='accept':
                missing=[a for a in required if a not in extras[i]]
                result=combined(e,extras[i],required)
                if result!='inconclusive' or not missing:closed[i]=result;continue
                insurance.extend((extra_price(a,executors[i],weights),i,a) for a in missing)
        reserve=(quota-done_samples)*max_sample
        remaining=max(0.,budget-spent-reserve)
        affordable=[x for x in insurance if x[0]<=remaining]
        is_extra=bool(affordable);choice=None
        if is_extra:
            charge,i,a=min(affordable,key=lambda x:(x[0],x[1],agent.ORDER.index(x[2]) if x[2] in agent.ORDER else len(agent.ORDER)));plan=None
            choice=(i,a,plan)
        elif policy in ('C2','C2A'):
            pairs=[(prices[i][a],i,a) for i,e in enumerate(schedule.evidence) if i not in closed and agent.verdict(e)=='inconclusive' for a in agent.available(e) if prices[i][a]<=remaining]
            if pairs:
                charge,i,a=min(pairs,key=lambda x:(x[0],x[1],s3.INDEX[x[2]]));plan=None;choice=(i,a,plan)
        else:
            choice=schedule.choose(remaining)
            if choice is not None:i,a,plan=choice;charge=prices[i][a]
        if active and done_samples<quota:
            if policy=='C2A':which=FAMILIES[state.total_samples%len(FAMILIES)];scores=None
            else:which,scores=sample_choice(schedule,state,closed,weights)
            deadline=done_samples/quota*ordinary_budget
            due=ordinary_cost>=deadline or choice is None
            anticipated=policy!='C2A' and state.ordinary>last_sample_ordinary and choice is not None and (family(a)==which or bool(insurance))
            if due or anticipated:
                sample=state.next_sample(which)
                result=execute_sample(sample,weights,fault if package_id>=onset else 'none')
                if result['cost']>max_sample or result['cost']>budget-spent:raise AssertionError('Sample exceeds reserve')
                target_samples+=int(which==target(fault))
                location=dict(package_id=package_id,input_index=None,action_index=len(rows),touched_inputs=len(touched),target_exposures=target_exposures,
                    samples=done_samples+1,target_samples=target_samples,ordinary_actions=sum(not r['sample'] for r in rows))
                observation=state.observe_sample(sample,result,location)
                rows.append(dict(input_index=None,action=which,sample=True,replacement=False,cost=result['cost'],sample_id=sample['sample_id'],
                    observation=result['observed'],original_observation=result['original'],sample_trace=result['trace'],expected=observation['expected'],
                    monitor=observation,sample_scores=scores,sample_reason='deadline' if ordinary_cost>=deadline else 'exhausted' if choice is None else 'anticipated_dependency',
                    sample_deadline=deadline,ordinary_cost_before=ordinary_cost,reserve_before=reserve,plan=None,forecast=None))
                spent+=result['cost'];sample_cost+=result['cost'];done_samples+=1;last_sample_ordinary=state.ordinary
                continue
        if choice is None:break
        touched.add(i)
        if is_extra:
            original,charge=extra_result(a,executors[i],weights);observed=original;hooked=original
            extras[i][a]=observed;forecast=None;mon=None
        else:
            forecast=schedule.posteriors[i].forecast(a)
            original,hooked,observed=ordinary_observation(executors[i],a,fault if package_id>=onset else 'none')
            schedule.update(i,a,observed)
            target_exposures+=int(family(a)==target(fault))
            mon=state.observe(a,agent.band(executors[i].request),forecast['probabilities'][0],observed,
                dict(package_id=package_id,input_index=i,action_index=len(rows),touched_inputs=len(touched),target_exposures=target_exposures,
                    samples=done_samples,target_samples=target_samples,ordinary_actions=sum(not r['sample'] for r in rows)+1)) if passive else None
        state.ordinary+=1
        spent+=charge;ordinary_cost+=charge;spent_by[i]+=charge
        rows.append(dict(input_index=i,action=a,sample=False,replacement=is_extra,cost=charge,observation=copy.deepcopy(observed),
            original_observation=copy.deepcopy(original),hooked_observation=copy.deepcopy(hooked),plan=plan,forecast=forecast,monitor=mon,reserve_before=reserve))
    if done_samples!=quota:raise AssertionError('Unfulfilled matched quota')
    inputs=[];required=set(state.suspect) if active or passive else set()
    for i,e in enumerate(schedule.evidence):
        inputs.append(dict(input_index=i,verdict=closed.get(i,combined(e,extras[i],required)),observed=e,replacements=extras[i],spent=spent_by[i],already_closed=i in closed))
    return dict(policy=policy,budget=budget,spent=spent,inputs=inputs,actions=rows,
        alerts=copy.deepcopy(state.alerts[alerts_start:]),suspect=sorted(state.suspect),monitor_history=copy.deepcopy(state.history[history_start:]),
        touched_inputs=len(touched),target_exposures=target_exposures,target_samples=target_samples,
        sample_allocation=dict(quota=quota,requested=requested,maximum_price=max_sample,initial_reserve=quota*max_sample,ordinary_budget=ordinary_budget,
            executed=done_samples,counts_before=start_counts,counts_after=copy.deepcopy(state.counts)),
        fault_exposure=dict(ordinary_target_actions=target_exposures,target_samples=target_samples,
            changed_ordinary_categories=sum(not r['sample'] and not r['replacement'] and r['original_observation']['verdict']!=r['hooked_observation']['verdict'] for r in rows),
            changed_sample_categories=sum(r['sample'] and r['original_observation']['verdict']!=r['observation']['verdict'] for r in rows),
            changed_ordinary_v2_steps=sum(bool(not r['sample'] and not r['replacement'] and r['original_observation'].get('step',{}).get('uncertainty_estimate')!=r['hooked_observation'].get('step',{}).get('uncertainty_estimate')) for r in rows),
            changed_sample_v2_steps=sum(bool(t['original'].get('step',{}).get('uncertainty_estimate')!=t['observed'].get('step',{}).get('uncertainty_estimate')) for r in rows if r['sample'] for t in r['sample_trace'])),
        sample_cost=sample_cost,replacement_cost=sum(r['cost'] for r in rows if r['replacement']),ordinary_cost=ordinary_cost,
        support_audit=dict(prior_resets=sum(p.fallback_count for p in schedule.posteriors),minimum_support=min(p['support_count'] for p in schedule.predictions),
            real_prediction_extensions=sum(p.real_extensions for p in schedule.posteriors),hypothetical_prediction_extensions=sum(p.hypothetical_extensions for p in schedule.posteriors)))
