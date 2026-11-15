"""Queue-controlled reliability diagnosis with threshold-aligned repetitions."""
import copy
import math
from binding import diagnose_b005 as old
from binding import source_lab_b006 as lab
from binding import diagnose_b006 as previous

BUDGETS = (2, 8, 32)
KINDS = ('check_source', 'check_claim', 'repeat_observation')
P_CONFIGS = {
    'beta4_t80': ((4., 1.), (1., 4.), .8),
    'beta4_t95': ((4., 1.), (1., 4.), .95),
    'beta2_t80': ((2., 1.), (1., 2.), .8),
    'beta2_t95': ((2., 1.), (1., 2.), .95),
}
K_OPTIONS = {'K1of1': (1, 1), 'K2of3': (2, 3), 'K3of5': (3, 5)}
PRIMARY = ('false_accusations', 'remaining_bad', 'lost_good', 'cost')


def initial_state(config):
    tp, fp, threshold = P_CONFIGS[config]
    return dict(prior=[2., 8.], channels={k: {'tp': list(tp), 'fp': list(fp)} for k in KINDS},
                auxiliary={k: {'tp': list(tp), 'fp': list(fp)} for k in KINDS}, resolved_episodes=0)


def model_view(state, policy, true_rates=None):
    active = {k: {r: v[0] / sum(v) for r, v in row.items()} for k, row in state['channels'].items()}
    auxiliary = {k: {r: v[0] / sum(v) for r, v in row.items()} for k, row in state['auxiliary'].items()}
    if policy == 'Pstar':
        if true_rates is None:
            raise ValueError('Privileged rates required')
        active = {k: {'tp': 1-r['false_good'], 'fp': r['false_bad']} for k,r in true_rates.items()}
    if policy == 'Px':
        active = {k: {'tp': .999999, 'fp': .000001} for k in KINDS}
    if policy == 'Pxprime':
        auxiliary = {k: {'tp': .999999, 'fp': .000001} for k in KINDS}
    return {'active': active, 'auxiliary': auxiliary}


posterior_start = previous.posterior_start
likelihood = previous.likelihood
update_posterior = previous.update_posterior
best_hypothesis = previous.best_hypothesis
learn = previous.learn


def threshold_distance(posterior, current, direct_bad, threshold, direct_good=None):
    gaps = [max(0., threshold-posterior['observation'])]
    if current is not None:
        p = posterior[current['candidate']]
        if current['candidate'] in (direct_good or set()):
            gaps.append(max(0., p-(1-threshold)))
        if current['candidate'] in direct_bad:
            gaps.append(max(0., threshold-p))
    return min(gaps)


def outcomes(posterior, test, model):
    q = math.fsum(p*likelihood(h,test,model) for h,p in posterior.items())
    for bad, probability in ((True,q),(False,1-q)):
        if probability > 0:
            yield bad, probability, update_posterior(posterior,test,bad,model)


def informative(posterior, test, model):
    supported=[likelihood(h,test,model) for h,p in posterior.items() if p > 0]
    return bool(supported) and min(supported) != max(supported)


def eligible(posterior,test,current,direct_bad,direct_good,threshold,model,noise=False):
    if informative(posterior,test,model):return True
    if current is None or test['candidate']!=current['candidate']:return False
    h=current['candidate'];p=old.rounded(posterior[h],noise)
    q=math.fsum(v*likelihood(k,test,model) for k,v in posterior.items())
    return ((p>=threshold and h not in direct_bad and q>0) or
            (p<=old.rounded(1-threshold,noise) and h not in direct_good and q<1))


def information_score(posterior, test, model):
    def entropy(p):
        return -math.fsum(v*math.log(v) for v in p.values() if v > 0)
    return max(0., entropy(posterior)-math.fsum(q*entropy(p) for _,q,p in outcomes(posterior,test,model)))/test['cost']


def action_score(posterior, test, tests, current, direct_bad, threshold, model, remaining, noise=False, direct_good=None):
    direct_good=set(direct_good or ())
    gap = threshold_distance(posterior,current,direct_bad,threshold,direct_good)
    one, two, future_cost = 0., 0., 0.
    for bad,q,updated in outcomes(posterior,test,model):
        direct = direct_bad | ({test['candidate']} if bad and test['candidate']!='observation' else set())
        good=direct_good | ({test['candidate']} if not bad and test['candidate']!='observation' else set())
        g = threshold_distance(updated,current,direct,threshold,good)
        one += q*g
        next_tests = [t for t in tests if t['cost']+test['cost'] <= remaining and eligible(updated,t,current,direct,good,threshold,model,noise)] if g > 0 else []
        alternatives=[]
        for t in next_tests:
            expected = math.fsum(q2*threshold_distance(p2,current,direct | ({t['candidate']} if b2 and t['candidate']!='observation' else set()),threshold,good | ({t['candidate']} if not b2 and t['candidate']!='observation' else set()))
                                 for b2,q2,p2 in outcomes(updated,t,model))
            alternatives.append((expected,t['cost'],t['kind'],t['id'],t))
        if alternatives:
            expected,price,_,_,_ = min(alternatives,key=lambda x:(old.rounded(x[0],noise),*x[1:4]))
            two += q*expected; future_cost += q*price
        else:
            two += q*g
    value=max(max(0.,gap-one)/test['cost'],max(0.,gap-two)/(test['cost']+future_cost))
    return old.rounded(value,noise)


def choose_probabilistic(posterior, tests, current, direct_bad, threshold, model, remaining, noise=False, direct_good=None):
    direct_good=set(direct_good or ())
    allowed=[t for t in tests if t['cost'] <= remaining and eligible(posterior,t,current,direct_bad,direct_good,threshold,model,noise)]
    if not allowed:
        return None, 0.
    scored=[(action_score(posterior,t,allowed,current,direct_bad,threshold,model,remaining,noise,direct_good),t) for t in allowed]
    if max(x[0] for x in scored)<=0:
        scored=[(old.rounded(information_score(posterior,t,model),noise),t) for t in allowed]
    return min(scored,key=lambda x:(-x[0],x[1]['kind'],x[1]['id'],x[1]['candidate']))[::-1]


def simulate(episode, policy, budget, state=None, config='beta4_t80', noise=False, true_rates=None):
    if policy not in ('D','R','N','P','P0','Pstar','Px','Pxprime',*K_OPTIONS) or budget not in BUDGETS:
        raise ValueError('Unknown policy or budget')
    public = lab.public_episode(episode)
    case, ids, branch = public['case'], public['signal_ids'], public['signal_branch']
    memory = old.Memory(case, 'L', ids)
    service = lab.NoisyService(episode, noise)
    initial = [memory.grounds(c) for c in ids]
    candidates = (set.intersection(*initial) if initial and branch=='measurement'
                  else set.union(*initial) if initial else set())
    original = set(candidates)
    limit, spent, trace = budget*case['budget_unit'], 0, []
    checked, repaired, attempts = set(), [], {}
    values = {c: r['value'] for c, r in memory.rows.items()}
    decision, probability, resolved = None, None, not ids
    probabilistic = policy in ('P','P0','Pstar','Px','Pxprime')
    state = copy.deepcopy(state if state is not None else initial_state(config))
    before = copy.deepcopy(state)
    view = model_view(state, policy, true_rates)
    posterior = posterior_start(candidates, state)
    threshold = P_CONFIGS[config][2]
    direct_bad = set()
    direct_good = set()
    stop_reason = 'no_signal' if not ids else 'inactive'
    current = None

    def perform(test, score=0.):
        nonlocal spent
        if spent+test['cost'] > limit:
            raise ValueError('Unaffordable diagnostic action')
        key = (test['kind'], test['id'])
        attempt = attempts.get(key, 0)
        attempts[key] = attempt+1
        bad = (service.repeat_observation(ids) if test['kind']=='repeat_observation'
               else service.probe(test))
        spent += test['cost']
        event = dict(test, attempt=attempt, bad=bool(bad), score=score)
        trace.append(event)
        if bad and test['candidate'] != 'observation':
            direct_bad.add(test['candidate'])
        if not bad and test['candidate'] != 'observation':
            direct_good.add(test['candidate'])
        return bool(bad)

    observation = dict(kind='repeat_observation', id='|'.join(sorted(ids)),
                       candidate='observation', cost=service.repeat_cost())
    if ids and policy == 'R':
        if observation['cost'] <= limit:
            if not perform(observation):
                decision, resolved = 'observation', True
        else:
            # R's compulsory first operation cannot be bypassed.
            candidates = set()
    if ids and policy != 'N' and not resolved:
        if probabilistic:
            stop_reason = 'budget'
            while True:
                if old.rounded(posterior['observation'],noise) >= threshold:
                    decision,probability,resolved = 'observation',posterior['observation'],True
                    stop_reason='threshold'; break
                tests = old.available_tests(memory,service,candidates,checked)
                affordable = [t for t in tests if spent+t['cost'] <= limit]
                if current is None:
                    current = old.choose(memory,affordable,candidates,noise) if affordable else None
                if current is not None:
                    current={k:v for k,v in current.items() if k!='score'}
                    h=current['candidate']
                    if h in direct_good and old.rounded(posterior[h],noise) <= old.rounded(1-threshold,noise):
                        checked.add(h); candidates.discard(h); current=None; continue
                    if h in direct_bad and old.rounded(posterior[h],noise) >= threshold:
                        decision,probability,resolved=h,posterior[h],True
                        stop_reason='threshold'; break
                choices=([current] if current is not None else [])+[observation]
                test,score=choose_probabilistic(posterior,choices,current,direct_bad,threshold,view['active'],limit-spent,noise,direct_good)
                if test is None:
                    stop_reason=('budget' if not any(t['cost'] <= limit-spent for t in choices) else 'no_informative_action')
                    break
                bad=perform(test,score)
                posterior=update_posterior(posterior,test,bad,view['active'])
            if probability is None:
                probability=posterior[best_hypothesis(posterior,noise)]
        else:
            while not resolved:
                tests = old.available_tests(memory,service,candidates,checked)
                if not tests:
                    if spent+observation['cost']<=limit:
                        if not perform(observation):
                            decision, resolved = 'observation', True
                    break
                affordable = [t for t in tests if spent+t['cost']<=limit]
                if not affordable:
                    break
                test = old.choose(memory,affordable,candidates,noise)
                test = {k:v for k,v in test.items() if k!='score'}
                k,m = K_OPTIONS.get(policy,(1,1))
                votes = {True:0, False:0}
                consensus = None
                for _ in range(m):
                    if spent+test['cost']>limit:
                        break
                    bad = perform(test)
                    votes[bad] += 1
                    if votes[bad]>=k:
                        consensus=bad
                        break
                if consensus is None:
                    break
                checked.add(test['candidate'])
                if consensus:
                    decision, resolved = test['candidate'], True
                    break
                candidates.discard(test['candidate'])
    # The probability is recorded before committing a diagnosis or any repair.
    terminal = {'hypothesis':decision, 'probability':probability,
                'abstained':bool(ids and decision is None)}
    if decision is not None and decision != 'observation':
        service.commit_repair_source(decision)
        targets = sorted((c for c in memory.rows if decision in memory.grounds(c)),
                         key=lambda c:(memory.rows[c]['depth'],c))
        for cid in targets:
            cost=service.repair_cost(cid)
            if spent+cost>limit:
                continue
            value=service.repair(cid)
            spent += cost
            values[cid] = value
            repaired.append(cid)
            trace.append(dict(kind='repair',id=cid,cost=cost,value=value))
        resolved = (all(old.close(values[c], case['measurements'][c]['value']) for c in ids)
                    if branch=='measurement' else
                    all(old.close(values[a],values[b]) for a,b in case['pairs'] if a in ids or b in ids))
    reports=[t for t in trace if t['kind']!='repair']
    observation_reports = [t for t in reports if t['kind']=='repeat_observation']
    observable_resolved = bool((decision=='observation' and observation_reports and not observation_reports[-1]['bad']) or
                               (decision not in (None,'observation') and repaired and resolved))
    if policy in ('P','Pxprime') and ids and observable_resolved:
        learn(state,posterior,reports)
    suspect_sources = ({decision} if decision not in (None, 'observation') else
                       original if probabilistic else candidates)
    suspected = sorted(c for c in memory.rows if suspect_sources & memory.grounds(c)) if ids and not resolved and policy!='N' else []
    evaluation = service.evaluate()
    initial_bad, final_bad = set(evaluation['initial_bad_ids']),set(evaluation['remaining_bad_ids'])
    true_hypothesis = evaluation['culprit'] if evaluation['kind']=='true_fault' else 'observation'
    innocent = decision not in (None,'observation') and decision != evaluation['culprit']
    remaining_tests=([current] if current is not None else [])+[observation]
    premature=int(probabilistic and bool(ids) and decision is None and any(
        t['cost'] <= limit-spent and eligible(posterior,t,current,direct_bad,direct_good,threshold,view['active'],noise) for t in remaining_tests))
    metrics = dict(false_accusations=int(innocent and bool(repaired)),remaining_bad=len(final_bad),
                   lost_good=len(final_bad-initial_bad),cost=spent,initial_bad=len(initial_bad),
                   attempted_false_accusations=int(innocent),checks=len(reports),
                   repeated_checks=sum(t['attempt']>0 for t in reports),
                   observation_repeats=sum(t['kind']=='repeat_observation' for t in reports),
                   rejected_false=int(evaluation['kind']=='false_objection' and decision=='observation' and not repaired),
                   corrected_counterexample=int(evaluation['kind']=='true_fault' and decision==evaluation['culprit'] and not final_bad),
                   diagnosis_made=int(bool(ids) and decision is not None),
                   diagnosis_correct=int(bool(ids) and decision==true_hypothesis),
                   abstained=int(bool(ids) and decision is None),repaired=len(repaired),suspected=len(suspected),
                   signal_depth=max((memory.rows[c]['depth'] for c in ids),default=0),
                   uncorroborated_accusations=int(decision not in (None,'observation') and decision not in direct_bad),
                   premature_stops=premature)
    protected=dict(signal_ids=ids,initial_candidates=sorted(original),decision=decision,
                   resolved=resolved,observable_resolved=observable_resolved,abstained=terminal['abstained'],checked=sorted(checked),
                   repaired=repaired,suspected=suspected,budget=limit,cost=spent,
                   actions=[{k:v for k,v in t.items() if k not in ('score','value')} for t in trace])
    return dict(protected=protected,metrics=metrics,trace=trace,
                terminal=terminal,state_before=before,state_after=state,
                diagnostics=dict(posterior=posterior,model=view,final_values=evaluation['values'],stop_reason=stop_reason,
                                 current_candidate=current['candidate'] if current else None,direct_bad=sorted(direct_bad),direct_good=sorted(direct_good)),
                evaluation=dict(events=evaluation['events'],true_hypothesis=true_hypothesis,
                                remaining_bad_ids=evaluation['remaining_bad_ids']))


def simulate_series(series, policy, budget, config='beta4_t80', noise=False):
    state=initial_state(config)
    outputs=[]
    for episode in series['episodes']:
        rates=episode['evaluator']['rates'] if policy=='Pstar' else None
        out=simulate(episode,policy,budget,state,config,noise,rates)
        state=out['state_after']
        outputs.append(out)
    return outputs
