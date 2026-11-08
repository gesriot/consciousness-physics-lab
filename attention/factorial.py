"""Fixed-trace likelihoods and factorial observation-sensitive allocation."""
import bisect
import copy
import math
import time
import numpy as np
from attention import agent, s3
from attention.batch import remaining_actions
from attention.updated_batch import _option, _tie

EPSILON = .02
CORRUPTION = .25
HYBRID_THRESHOLD = .05
def alphabet_size(action):
    return 4 if action.startswith('v2_') else 3 if action in ('modal', 'collocation') else 2


POLICIES = ('U0O', 'U1O', 'U0T', 'U1T', 'H', 'D', 'Dprime', 'HD', 'HDprime')


class Runtime(s3.Runtime):
    """The frozen S3 archive, with an explicit noisy observation channel."""
    def __init__(self, model, corruption='none'):
        super().__init__(model)
        if corruption not in ('none', 'D', 'Dprime'):
            raise ValueError('Unknown observation intervention')
        self.likelihood = np.zeros((len(self.raw), len(agent.ORDER), 4))
        for action, j in s3.INDEX.items():
            k = alphabet_size(action)
            present = self.raw[:, j] >= 0
            self.likelihood[present, j, :k] = EPSILON / k
            self.likelihood[np.flatnonzero(present), j, self.raw[present, j]] += 1-EPSILON
        self.unused_likelihood = self.likelihood.copy()
        if corruption != 'none':
            destination = self.likelihood if corruption == 'D' else self.unused_likelihood
            a,b = (s3.INDEX[x] for x in s3.FEATURE_ACTIONS)
            left,right = destination[:,a].copy(),destination[:,b].copy()
            destination[:,a] = (1-CORRUPTION)*left + CORRUPTION*right
            destination[:,b] = (1-CORRUPTION)*right + CORRUPTION*left


class Posterior:
    def __init__(self, request, runtime, update):
        self.runtime = runtime
        self.neighbors,self.features = runtime.neighbors(request)
        self.ids = np.flatnonzero(np.isin(runtime.groups,self.neighbors))
        self.prior = np.full(len(self.ids),1/len(self.ids))
        self.weights = self.prior.copy()
        self.update_enabled = update
        self.step = 0
        self.evidence = {}
        self.fallback_count = 0
        self.hypothetical_queries = 0
        self.hypothetical_extensions = 0
        self.real_extensions = 0

    def snapshot(self):
        p = {v:float(self.weights @ (self.runtime.full[self.ids]==j)) for j,v in enumerate(s3.VERDICTS)}
        return dict(step=self.step,full_probabilities=p,support_count=int(np.count_nonzero(self.weights)),
            effective_support=float(1/(self.weights@self.weights)),fallback_count=self.fallback_count,
            raw_feature_violation_mass={a:float(self.weights @ (self.runtime.raw[self.ids,s3.INDEX[a]]==1)) for a in s3.FEATURE_ACTIONS})

    def forecast(self, action):
        probabilities=self.weights@self.runtime.likelihood[self.ids,s3.INDEX[action]]
        mass=float(probabilities.sum())
        extension=mass<=0 or not math.isfinite(mass)
        k=alphabet_size(action)
        values=[1/k]*k if extension else (probabilities[:k]/mass).tolist()
        return dict(eligible_mass=mass,missing_prediction_extension=extension,probabilities=values)

    def conditioned(self, weights, action, category):
        if not self.update_enabled:return weights.copy(),False
        values = weights*self.runtime.likelihood[self.ids,s3.INDEX[action],category]
        total = float(values.sum())
        if total<=0 or not math.isfinite(total):return self.prior.copy(),True
        return values/total,False

    def update(self, action, observed):
        if action in self.evidence:raise ValueError('Repeated evidence')
        category=s3.token(action,observed)
        forecast=self.forecast(action)
        self.real_extensions+=forecast['missing_prediction_extension']
        self.evidence[action]=category;self.step+=1
        self.weights,fallback=self.conditioned(self.weights,action,category)
        self.fallback_count+=fallback
        return dict(**self.snapshot(),action=action,category=category,prior_reset=fallback,
            reset_reason='zero_or_nonfinite_likelihood_mass' if fallback else None,action_forecast_before=forecast)


def branches(evidence, weights, action, posterior):
    """Absent V2 traces are excluded, never converted into observations."""
    matrix=posterior.runtime.likelihood[posterior.ids,s3.INDEX[action]]
    probabilities=weights@matrix
    mass=float(probabilities.sum())
    extension=mass<=0 or not math.isfinite(mass)
    posterior.hypothetical_queries+=1
    posterior.hypothetical_extensions+=extension
    if extension:
        k=alphabet_size(action)
        probabilities=np.array([1/k]*k+[0.]*(4-k))
    else:probabilities=probabilities/mass
    for category,probability in enumerate(probabilities):
        if probability<=0:continue
        updated,_=posterior.conditioned(weights,action,category)
        branch=dict(evidence);branch[action]=s3.observation(action,category)
        yield category,float(probability),updated,branch,extension


def reward(verdict, weights, posterior):
    if verdict=='inconclusive':return 0.
    accurate=posterior.runtime.accurate[posterior.ids]
    return float(weights @ (accurate if verdict=='accept' else ~accurate))


def basic_options(evidence, weights, prices, posterior):
    if agent.verdict(evidence)!='inconclusive':return []
    actions=agent.available(evidence);out=[]
    for action in actions:
        value=sum(probability*reward(agent.verdict(branch),updated,posterior)
            for _,probability,updated,branch,_ in branches(evidence,weights,action,posterior))
        out.append(_option('ATOMIC',action,value,prices[action]))
    if actions:
        first=min(actions,key=lambda a:(prices[a],s3.INDEX[a]))
        out.append(_option('FULL',first,float(weights@posterior.runtime.correct[posterior.ids]),
                           sum(prices[a] for a in remaining_actions(evidence))))
    return out


class Plans:
    def __init__(self,evidence,posterior,prices,tree):
        self.prices=prices
        self.direct=basic_options(evidence,posterior.weights,prices,posterior)
        self.trees=[];thresholds=[o['expected_cost'] for o in self.direct]
        if tree:
            for action in agent.available(evidence):
                parts=[]
                for category,probability,weights,branch,extension in branches(evidence,posterior.weights,action,posterior):
                    result=agent.verdict(branch)
                    options=basic_options(branch,weights,prices,posterior) if result=='inconclusive' else []
                    parts.append(dict(category=category,probability=probability,terminal=result!='inconclusive',
                        reward=reward(result,weights,posterior),options=options,missing_prediction_extension=extension,eligible_mass=float(posterior.weights@posterior.runtime.likelihood[posterior.ids,s3.INDEX[action]].sum(axis=1))))
                    thresholds.extend(prices[action]+o['expected_cost'] for o in options)
                self.trees.append(dict(action=action,branches=parts));thresholds.append(prices[action])
        self.thresholds=sorted(set(thresholds));self.key=None;self.cached=None

    def best(self,remaining):
        key=bisect.bisect_right(self.thresholds,remaining)
        if key==self.key:return self.cached
        candidates=[o for o in self.direct if o['expected_cost']<=remaining]
        for tree in self.trees:
            action=tree['action'];charge=self.prices[action]
            if charge>remaining:continue
            gain,cost,parts=0.,charge,[]
            for b in tree['branches']:
                options=[o for o in b['options'] if charge+o['expected_cost']<=remaining and o['expected_gain']>0]
                continuation=min(options,key=lambda o:(-o['score'],*_tie(o))) if options else None
                gain+=b['probability']*b['reward']
                if continuation:
                    gain+=b['probability']*continuation['expected_gain'];cost+=b['probability']*continuation['expected_cost']
                parts.append(dict(category=b['category'],probability=b['probability'],terminal=b['terminal'],
                    immediate_reward=b['reward'],continuation=continuation,missing_prediction_extension=b['missing_prediction_extension'],eligible_mass=b['eligible_mass']))
            option=_option('TREE',action,gain,cost);option['branches']=parts;candidates.append(option)
        self.cached=min(candidates,key=lambda o:(-o['score'],*_tie(o))) if candidates else None
        self.key=key;return self.cached


class Scheduler:
    """Only requests, a fixed archive, policy, and opened evidence are visible."""
    def __init__(self,requests,model,policy):
        if policy not in POLICIES:raise ValueError('Unknown factorial policy')
        self.policy=policy
        self.hybrid=policy in ('H','HD','HDprime')
        self.tree=policy in ('U0T','U1T','D','Dprime')
        corruption='D' if policy in ('D','HD') else 'Dprime' if policy in ('Dprime','HDprime') else 'none'
        self.runtime=Runtime(model,corruption)
        self.posteriors=[Posterior(r,self.runtime,not policy.startswith('U0')) for r in requests]
        self.prices=[{a:agent.cost(a,r,model['action_model']) for a in agent.ORDER} for r in requests]
        self.evidence=[{} for _ in requests];self.plans=[None for _ in requests]
        self.predictions=[dict(input_index=i,**p.snapshot()) for i,p in enumerate(self.posteriors)]
        self.skipped=[];self.planning_seconds=0.

    def choose(self,remaining):
        started=time.perf_counter()
        try:
            pairs=[(i,a) for i,e in enumerate(self.evidence) if agent.verdict(e)=='inconclusive'
                   for a in agent.available(e) if self.prices[i][a]<=remaining]
            pairs.sort(key=lambda p:(self.prices[p[0]][p[1]],p[0],s3.INDEX[p[1]]))
            if not pairs:return None
            if self.hybrid:
                for i,a in pairs:
                    p=self.posteriors[i]
                    chance=sum(prob for _,prob,_,e,_ in branches(self.evidence[i],p.weights,a,p) if agent.verdict(e)!='inconclusive')
                    if chance<HYBRID_THRESHOLD:
                        self.skipped.append(dict(after_actions=len(self.predictions)-len(self.posteriors),input_index=i,action=a,decisive_probability=chance))
                        continue
                    return i,a,dict(kind='HYBRID',action=a,decisive_probability=chance,expected_cost=self.prices[i][a])
                return None
            candidates=[]
            for i in sorted(set(i for i,_ in pairs)):
                if self.plans[i] is None:self.plans[i]=Plans(self.evidence[i],self.posteriors[i],self.prices[i],self.tree)
                option=self.plans[i].best(remaining)
                if option is not None:candidates.append((i,option))
            if not candidates or max(o['score'] for _,o in candidates)<=0:
                i,a=pairs[0];return i,a,_option('ATOMIC',a,0.,self.prices[i][a])
            i,o=min(candidates,key=lambda pair:(-pair[1]['score'],pair[0],*_tie(pair[1])))
            return i,o['action'],o
        finally:self.planning_seconds+=time.perf_counter()-started

    def update(self,index,action,observed):
        started=time.perf_counter()
        try:
            if action not in agent.available(self.evidence[index]) or agent.verdict(self.evidence[index])!='inconclusive':
                raise ValueError('Unavailable action')
            self.evidence[index][action]=observed;self.plans[index]=None
            row=dict(input_index=index,**self.posteriors[index].update(action,observed));self.predictions.append(row)
            return row
        finally:self.planning_seconds+=time.perf_counter()-started


def run_batch(executors,policy,budget,model,permutation=None):
    if math.isnan(budget) or budget<0:raise ValueError('Budget must be nonnegative')
    requests=[e.request for e in executors]
    if permutation is not None:
        expected={(i,a) for i in range(len(requests)) for a in agent.ORDER}
        if len(permutation)!=len(expected) or set(map(tuple,permutation))!=expected:raise ValueError('Invalid permutation')
    started=time.perf_counter();schedule=Scheduler(requests,model,policy)
    schedule.planning_seconds+=time.perf_counter()-started
    actions=[];rows=[[] for _ in requests];spent_by=[0. for _ in requests];spent=0.
    while True:
        selected=schedule.choose(budget-spent)
        if selected is None:break
        i,a,plan=selected;observed=executors[i].perform(a);charge=schedule.prices[i][a]
        spent+=charge;spent_by[i]+=charge;schedule.update(i,a,observed)
        row=dict(input_index=i,action=a,cost=charge,observation=copy.deepcopy(observed),plan=copy.deepcopy(plan))
        actions.append(row);rows[i].append(copy.deepcopy(row))
    inputs=[]
    for i,evidence in enumerate(schedule.evidence):
        verdict=agent.verdict(evidence)
        inputs.append(dict(input_index=i,verdict=verdict,spent=spent_by[i],actions=rows[i],observed=evidence,
            stop_reason='verdict_complete' if verdict!='inconclusive' else 'threshold_or_budget' if schedule.hybrid else 'budget_or_no_actions',
            drop_reason=None,early_violation=bool(verdict=='violation_detected' and agent.available(evidence)),
            futile_v2=any(v.get('step',{}).get('futile',False) for v in evidence.values())))
    return dict(policy=policy,budget=budget,spent=spent,actions=actions,inputs=inputs,predictions=schedule.predictions,
        skipped=schedule.skipped,support_audit=dict(prior_resets=sum(p.fallback_count for p in schedule.posteriors),
            minimum_support=min((p['support_count'] for p in schedule.predictions),default=0),
            missing_predictions=0,real_prediction_extensions=sum(p.real_extensions for p in schedule.posteriors),
            hypothetical_prediction_queries=sum(p.hypothetical_queries for p in schedule.posteriors),
            hypothetical_prediction_extensions=sum(p.hypothetical_extensions for p in schedule.posteriors)),planning_seconds=schedule.planning_seconds)


def corruption_audit(model):
    base=Runtime(model);d=Runtime(model,'D');dp=Runtime(model,'Dprime')
    delta=np.abs(d.likelihood-base.likelihood);other=np.abs(dp.unused_likelihood-base.unused_likelihood)
    return dict(epsilon=EPSILON,alpha=CORRUPTION,hybrid_threshold=HYBRID_THRESHOLD,
        D_changed_numbers=int(np.count_nonzero(delta)),Dprime_changed_numbers=int(np.count_nonzero(other)),
        D_L1=float(delta.sum()),Dprime_L1=float(other.sum()),equal_changes=bool(np.array_equal(delta,other)),
        Dprime_active_unchanged=bool(np.array_equal(base.likelihood,dp.likelihood)),
        D_unused_unchanged=bool(np.array_equal(base.unused_likelihood,d.unused_likelihood)))
