"""Paid fallible diagnosis with inherited provenance and explicit belief state."""
import copy
import math
from binding import diagnose_b005 as old
from binding import source_lab_b006 as lab

BUDGETS = (2, 8, 32)
KINDS = ('check_source', 'check_claim', 'repeat_observation')
P_CONFIGS = {
    'beta8_t80': ((8., 2.), (1., 9.), .8),
    'beta8_t95': ((8., 2.), (1., 9.), .95),
    'beta99_t80': ((99., 1.), (1., 99.), .8),
    'beta99_t95': ((99., 1.), (1., 99.), .95),
}
K_OPTIONS = {'K1of1': (1, 1), 'K2of3': (2, 3), 'K3of5': (3, 5)}
PRIMARY = ('false_accusations', 'remaining_bad', 'lost_good', 'cost')


def initial_state(config):
    tp, fp, threshold = P_CONFIGS[config]
    return dict(prior=[2., 8.], channels={k: {'tp': list(tp), 'fp': list(fp)} for k in KINDS},
                auxiliary={k: {'tp': list(tp), 'fp': list(fp)} for k in KINDS}, resolved_episodes=0)


def model_view(state, policy):
    active = {k: {r: v[0] / sum(v) for r, v in row.items()} for k, row in state['channels'].items()}
    auxiliary = {k: {r: v[0] / sum(v) for r, v in row.items()} for k, row in state['auxiliary'].items()}
    if policy == 'Px':
        active = {k: {'tp': .999999, 'fp': .000001} for k in KINDS}
    if policy == 'Pxprime':
        auxiliary = {k: {'tp': .999999, 'fp': .000001} for k in KINDS}
    return {'active': active, 'auxiliary': auxiliary}


def posterior_start(candidates, state):
    mass = state['prior'][0] / sum(state['prior'])
    return ({'observation': mass, **{s: (1-mass)/len(candidates) for s in sorted(candidates)}}
            if candidates else {'observation': 1.})


def likelihood(hypothesis, test, model):
    truth_bad = hypothesis != 'observation' if test['kind'] == 'repeat_observation' else hypothesis == test['candidate']
    return model[test['kind']]['tp' if truth_bad else 'fp']


def update_posterior(posterior, test, bad, model):
    weights = {h: p * (likelihood(h, test, model) if bad else 1-likelihood(h, test, model))
               for h, p in posterior.items()}
    total = math.fsum(weights.values())
    if total <= 0:
        raise ValueError('Inconsistent likelihood model')
    return {h: w/total for h, w in weights.items()}


def risk_score(posterior, test, model, noise=False):
    bad_prob = math.fsum(p*likelihood(h, test, model) for h, p in posterior.items())
    after = math.fsum(q*(1-max(update_posterior(posterior,test,bad,model).values()))
                      for bad, q in ((True,bad_prob),(False,1-bad_prob)) if q > 0)
    return old.rounded(max(0., (1-max(posterior.values()))-after)/test['cost'], noise)


def best_hypothesis(posterior, noise=False):
    return min(posterior, key=lambda h: (-old.rounded(posterior[h], noise), h))


def learn(state, posterior, reports):
    """Fractional self-training on an internally resolved episode, never truth."""
    state['prior'][0] += posterior.get('observation', 0.)
    state['prior'][1] += 1-posterior.get('observation', 0.)
    for report in reports:
        weight = (1-posterior.get('observation', 0.) if report['kind']=='repeat_observation'
                  else posterior.get(report['candidate'], 0.))
        bad = int(report['bad'])
        row = state['channels'][report['kind']]
        row['tp'][0] += weight*bad
        row['tp'][1] += weight*(1-bad)
        row['fp'][0] += (1-weight)*bad
        row['fp'][1] += (1-weight)*(1-bad)
    state['resolved_episodes'] += 1


def simulate(episode, policy, budget, state=None, config='beta8_t80', noise=False):
    if policy not in ('D','R','N','P','Px','Pxprime',*K_OPTIONS) or budget not in BUDGETS:
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
    probabilistic = policy in ('P','Px','Pxprime')
    state = copy.deepcopy(state if state is not None else initial_state(config))
    before = copy.deepcopy(state)
    view = model_view(state, policy)
    posterior = posterior_start(candidates, state)
    threshold = P_CONFIGS[config][2]

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
            while True:
                best = best_hypothesis(posterior, noise)
                if old.rounded(posterior[best], noise) >= threshold:
                    decision, probability, resolved = best, posterior[best], True
                    break
                tests = old.available_tests(memory, service, original, set()) + [observation]
                tests = [t for t in tests if spent+t['cost'] <= limit]
                if not tests:
                    break
                scored = [(risk_score(posterior,t,view['active'],noise),t) for t in tests]
                score, test = min(scored,key=lambda st:(-st[0],st[1]['kind'],st[1]['id'],st[1]['candidate']))
                if score <= 0:
                    break
                bad = perform(test, score)
                posterior = update_posterior(posterior,test,bad,view['active'])
            if probability is None:
                probability = posterior[best_hypothesis(posterior, noise)]
        else:
            while not resolved:
                tests = old.available_tests(memory,service,candidates,checked)
                if not tests:
                    if branch=='measurement' and spent+observation['cost']<=limit:
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
    if probabilistic and ids and observable_resolved:
        learn(state,posterior,reports)
    suspect_sources = ({decision} if decision not in (None, 'observation') else
                       original if probabilistic else candidates)
    suspected = sorted(c for c in memory.rows if suspect_sources & memory.grounds(c)) if ids and not resolved and policy!='N' else []
    evaluation = service.evaluate()
    initial_bad, final_bad = set(evaluation['initial_bad_ids']),set(evaluation['remaining_bad_ids'])
    true_hypothesis = evaluation['culprit'] if evaluation['kind']=='true_fault' else 'observation'
    innocent = decision not in (None,'observation') and decision != evaluation['culprit']
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
                   signal_depth=max((memory.rows[c]['depth'] for c in ids),default=0))
    protected=dict(signal_ids=ids,initial_candidates=sorted(original),decision=decision,
                   resolved=resolved,observable_resolved=observable_resolved,abstained=terminal['abstained'],checked=sorted(checked),
                   repaired=repaired,suspected=suspected,budget=limit,cost=spent,
                   actions=[{k:v for k,v in t.items() if k not in ('score','value')} for t in trace])
    return dict(protected=protected,metrics=metrics,trace=trace,
                terminal=terminal,state_before=before,state_after=state,
                diagnostics=dict(posterior=posterior,model=view,final_values=evaluation['values']),
                evaluation=dict(events=evaluation['events'],true_hypothesis=true_hypothesis,
                                remaining_bad_ids=evaluation['remaining_bad_ids']))


def simulate_series(series, policy, budget, config='beta8_t80', noise=False):
    state=initial_state(config)
    outputs=[]
    for episode in series['episodes']:
        out=simulate(episode,policy,budget,state,config,noise)
        state=out['state_after']
        outputs.append(out)
    return outputs
