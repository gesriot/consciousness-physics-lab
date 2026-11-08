"""Prospective monitoring of paid categorical observations."""
import copy
from attention import agent, s3, monitor_planning as planning
from lab import frozen_pipeline as P
from lab.finite_volume import response as uniform_response

TARGETS = ('collocation', 'second_moment')
POLICIES = ('C2', 'B', 'BM', 'D', 'Dprime')


class Monitor:
    def __init__(self, threshold, policy):
        self.threshold, self.policy = threshold, policy
        self.sums, self.suspect, self.alerts = {}, set(), []
        self.active_l1 = self.unused_l1 = 0.
        self.active_count = self.unused_count = 0
        self.history = []

    def observe(self, action, regime, probability, observed, location):
        unused = probability
        if action in TARGETS:
            if self.policy == 'D':
                self.active_l1 += abs(.999-probability)
                self.active_count += int(probability != .999)
                probability = .999
            if self.policy == 'Dprime':
                self.unused_l1 += abs(.999-unused)
                self.unused_count += int(unused != .999)
                unused = .999
        key = action+':'+regime
        value = max(0., self.sums.get(key, 0.) + int(observed['verdict']=='accept') - probability - .02)
        self.sums[key] = value
        row = dict(action=action, regime=regime, probability=probability,
                   unused_probability=unused, statistic=value, **location)
        self.history.append(row)
        if value > self.threshold and action not in self.suspect:
            self.suspect.add(action)
            self.alerts.append(copy.deepcopy(row))
        return row


def replacement_cost(action, request, model):
    if action == 'collocation':
        return 61440 * model['cost_weights']['fv_cell']
    return agent.cost(action, request, model)


def replacement(executor, action):
    r=executor.request
    if action == 'collocation':
        grids=(4096,8192,16384,32768)
        values=[uniform_response(r.z,r.mean_time,n) for n in grids]
        out=P.v2_step(values,grids[-1],complex(*executor.outcome['value']),r.tolerance)
        return dict(verdict=out['verdict'], grids=list(grids),
                    values=[[v.real,v.imag] for v in values],step=out)
    if action in agent.FEATURES:
        return agent.feature(r,action,executor.model)
    # Preserve paid prerequisites; recompute exactly one action on a clean channel.
    clean=copy.copy(executor)
    clean.observed=copy.deepcopy(executor.observed)
    clean.observed.pop(action,None)
    if action.startswith('v2_'):
        grid=int(action[3:])
        clean.observed={a:v for a,v in clean.observed.items() if not a.startswith('v2_') or int(a[3:])<grid}
        clean.values=[complex(*clean.observed[f'v2_{n}']['value']) for n in agent.GRIDS if n<grid and f'v2_{n}' in clean.observed]
    return clean.perform(action)



def combined(base, extra, required):
    verdict=agent.verdict(base)
    if verdict != 'accept':return verdict
    if any(v['verdict']=='violation_detected' for v in extra.values()):return 'violation_detected'
    if all(a in extra and extra[a]['verdict']=='accept' for a in required):return 'accept'
    return 'inconclusive'


def run_batch(executors, policy, budget, model, monitor, package_id, fault, onset):
    schedule=planning.Scheduler([e.request for e in executors],model,'U0T')
    prices=schedule.prices
    closed={}; extras=[{} for _ in executors]; spent_by=[0.]*len(executors)
    rows=[]; spent=0.; touched=set(); target_exposures=0
    alerts_start=len(monitor.alerts); history_start=len(monitor.history)
    while True:
        active=policy in ('BM','D','Dprime')
        required=set(monitor.suspect) if active else set()
        insurance=[]
        for i,e in enumerate(schedule.evidence):
            if i in closed:continue
            v=agent.verdict(e)
            if v=='violation_detected':closed[i]=v;continue
            if v=='accept':
                missing=[a for a in required if a not in extras[i]]
                result=combined(e,extras[i],required)
                if result!='inconclusive' or not missing:
                    closed[i]=result;continue
                insurance.extend((replacement_cost(a,executors[i].request,model['action_model']),i,a) for a in missing)
        affordable=[x for x in insurance if x[0]<=budget-spent]
        is_extra=bool(affordable)
        if is_extra:
            charge,i,a=min(affordable,key=lambda x:(x[0],x[1],s3.INDEX[x[2]]));plan=None
        else:
            if policy=='C2':
                pairs=[(prices[i][a],i,a) for i,e in enumerate(schedule.evidence) if i not in closed and agent.verdict(e)=='inconclusive' for a in agent.available(e) if prices[i][a]<=budget-spent]
                if not pairs:break
                charge,i,a=min(pairs,key=lambda x:(x[0],x[1],s3.INDEX[x[2]]));plan=None
            else:
                choice=schedule.choose(budget-spent)
                if choice is None:break
                i,a,plan=choice;charge=prices[i][a]
        touched.add(i)
        if is_extra:
            original=observed=replacement(executors[i],a)
            extras[i][a]=observed
            forecast=None;mon=None
        else:
            forecast=schedule.posteriors[i].forecast(a)
            original=executors[i].perform(a)
            spoiled=package_id>=onset and ((fault=='permissive' and a=='collocation') or (fault=='uninformative' and a=='second_moment'))
            # No true diagnostic values or hook flag cross the scheduler boundary.
            observed=s3.observation(a,0) if spoiled else s3.observation(a,s3.token(a,original))
            schedule.update(i,a,observed)
            target_exposures+=int(a==('second_moment' if fault=='uninformative' else 'collocation'))
            mon=monitor.observe(a,agent.band(executors[i].request),forecast['probabilities'][0],observed,
                dict(package_id=package_id,input_index=i,action_index=len(rows),touched_inputs=len(touched),target_exposures=target_exposures)) if active else None
        spent+=charge;spent_by[i]+=charge
        rows.append(dict(input_index=i,action=a,replacement=is_extra,cost=charge,observation=copy.deepcopy(observed),
                         original_observation=copy.deepcopy(original),plan=plan,forecast=forecast,monitor=mon))
    inputs=[]
    required=set(monitor.suspect) if policy in ('BM','D','Dprime') else set()
    for i,e in enumerate(schedule.evidence):
        inputs.append(dict(input_index=i,verdict=closed.get(i,combined(e,extras[i],required)),
            observed=e,replacements=extras[i],spent=spent_by[i],already_closed=i in closed))
    return dict(policy=policy,budget=budget,spent=spent,inputs=inputs,actions=rows,
        alerts=copy.deepcopy(monitor.alerts[alerts_start:]),suspect=sorted(monitor.suspect),
        monitor_history=copy.deepcopy(monitor.history[history_start:]),
        touched_inputs=len(touched),target_exposures=target_exposures,
        support_audit=dict(prior_resets=sum(p.fallback_count for p in schedule.posteriors),minimum_support=min(p['support_count'] for p in schedule.predictions),real_prediction_extensions=sum(p.real_extensions for p in schedule.posteriors),hypothetical_prediction_extensions=sum(p.hypothetical_extensions for p in schedule.posteriors)),
        replacement_cost=sum(r['cost'] for r in rows if r['replacement']))
