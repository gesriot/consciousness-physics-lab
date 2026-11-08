"""Two-stage information-sensitive allocation under strict observed evidence."""
import bisect
import copy
import math
import time
import numpy as np
from attention import agent, s3
from attention.batch import remaining_actions

KINDS = {'ATOMIC': 0, 'TREE': 1, 'FULL': 2}


def _reward(verdict, indices, runtime):
    if verdict == 'inconclusive': return 0.
    correct = runtime.accurate[indices] if verdict == 'accept' else ~runtime.accurate[indices]
    return float(np.count_nonzero(correct)/len(indices))


def _option(kind, action, reward, cost):
    return dict(kind=kind, action=action, expected_gain=float(reward), expected_cost=float(cost),
                score=float(reward/cost) if cost else 0.)


def _tie(option):
    return (s3.INDEX[option['action']], KINDS[option['kind']])


def _branches(evidence, indices, action, runtime):
    column = runtime.mapped[indices, s3.INDEX[action]]
    if np.any(column < 0):
        raise AssertionError('A compatible trajectory lacks an available action')
    result = []
    for value in np.unique(column):
        subset = indices[column == value]
        observed = dict(evidence)
        observed[action] = s3.observation(action, int(value))
        result.append((int(value), subset, observed, len(subset)/len(indices)))
    return result


def basic_options(evidence, indices, prices, runtime):
    """Unrestricted-budget options; caller filters conservative charge bounds."""
    if not len(indices) or agent.verdict(evidence) != 'inconclusive': return []
    actions = agent.available(evidence)
    result = []
    for action in actions:
        reward = sum(probability*_reward(agent.verdict(branch), subset, runtime)
                     for _, subset, branch, probability in _branches(evidence, indices, action, runtime))
        result.append(_option('ATOMIC', action, reward, prices[action]))
    if actions:
        first = min(actions, key=lambda a: (prices[a], s3.INDEX[a]))
        reward = float(np.count_nonzero(runtime.correct[indices])/len(indices))
        price = sum(prices[a] for a in remaining_actions(evidence))
        result.append(_option('FULL', first, reward, price))
    return result


class Plans:
    """Cache a finite state envelope; budget thresholds avoid stale affordability."""
    def __init__(self, evidence, posterior, prices):
        self.prices = prices
        self.direct = basic_options(evidence, posterior.indices, prices, posterior.runtime)
        self.trees = []
        thresholds = [o['expected_cost'] for o in self.direct]
        for action in agent.available(evidence):
            branches = []
            for value, indices, branch, probability in _branches(evidence, posterior.indices, action, posterior.runtime):
                result = agent.verdict(branch)
                options = basic_options(branch, indices, prices, posterior.runtime) if result == 'inconclusive' else []
                reward = _reward(result, indices, posterior.runtime)
                branches.append(dict(category=value, probability=probability,
                    terminal=result != 'inconclusive', reward=reward, options=options))
                thresholds.extend(prices[action]+o['expected_cost'] for o in options)
            self.trees.append(dict(action=action, branches=branches))
            thresholds.append(prices[action])
        self.thresholds = sorted(set(thresholds))
        self.key = None
        self.cached = None

    def best(self, remaining):
        key = bisect.bisect_right(self.thresholds, remaining)
        if self.key == key: return self.cached
        candidates = [o for o in self.direct if o['expected_cost'] <= remaining]
        for tree in self.trees:
            action = tree['action']
            charge = self.prices[action]
            if charge > remaining: continue
            reward, expected_cost, branches = 0., charge, []
            for branch in tree['branches']:
                probability = branch['probability']
                reward += probability*branch['reward']
                options = [o for o in branch['options'] if charge+o['expected_cost'] <= remaining and o['expected_gain'] > 0]
                continuation = min(options, key=lambda o: (-o['score'], *_tie(o))) if options else None
                if continuation is not None:
                    reward += probability*continuation['expected_gain']
                    expected_cost += probability*continuation['expected_cost']
                branches.append(dict(category=branch['category'], probability=probability,
                    terminal=branch['terminal'], immediate_reward=branch['reward'], continuation=continuation))
            option = _option('TREE', action, reward, expected_cost)
            option['branches'] = branches
            candidates.append(option)
        self.cached = min(candidates, key=lambda o: (-o['score'], *_tie(o))) if candidates else None
        self.key = key
        return self.cached


class Scheduler:
    """Only requests, model, and opened evidence enter the scheduling boundary."""
    def __init__(self, requests, model):
        runtime = s3.Runtime(model)
        self.posteriors = [s3.Posterior(r, runtime=runtime) for r in requests]
        self.prices = [{a: agent.cost(a, r, model['action_model']) for a in agent.ORDER} for r in requests]
        self.evidence = [{} for _ in requests]
        self.plans = [None for _ in requests]
        self.fallback = False
        self.fallback_trigger = None
        self.expansions = []
        self.predictions = []
        self.planning_seconds = 0.
        for i, posterior in enumerate(self.posteriors):
            self.predictions.append(dict(input_index=i, **posterior.snapshot(), fallback_active=False))

    def choose(self, remaining):
        started = time.perf_counter()
        try:
            pairs = [(i,a) for i,evidence in enumerate(self.evidence) if agent.verdict(evidence)=='inconclusive'
                     for a in agent.available(evidence) if self.prices[i][a] <= remaining]
            if not pairs: return None
            cheap = min(pairs, key=lambda pair:(self.prices[pair[0]][pair[1]], pair[0], s3.INDEX[pair[1]]))
            if self.fallback:
                return (*cheap, dict(kind='C2_FALLBACK', action=cheap[1], expected_gain=None, expected_cost=self.prices[cheap[0]][cheap[1]], score=None))
            candidates = []
            for index in sorted(set(i for i,_ in pairs)):
                if self.plans[index] is None:
                    self.plans[index] = Plans(self.evidence[index], self.posteriors[index], self.prices[index])
                option = self.plans[index].best(remaining)
                if option is not None: candidates.append((index, option))
            if not candidates or max(o['score'] for _,o in candidates) <= 0:
                return (*cheap, dict(kind='ZERO_GAIN_CHEAPEST', action=cheap[1], expected_gain=0., expected_cost=self.prices[cheap[0]][cheap[1]], score=0.))
            index, option = min(candidates, key=lambda pair:(-pair[1]['score'], pair[0], *_tie(pair[1])))
            return index, option['action'], option
        finally:
            self.planning_seconds += time.perf_counter()-started

    def update(self, index, action, observed):
        started = time.perf_counter()
        try:
            self.evidence[index][action] = observed
            self.plans[index] = None
            posterior = self.posteriors[index]
            if self.fallback:
                posterior.step += 1
                record = dict(step=posterior.step, full_probabilities=None, support_count=None,
                    expanded=posterior.expanded, failed=None, action=action, category=s3.token(action, observed),
                    raw_feature_violation_mass=None, raw_feature_violation_mass_before=None,
                    expanded_now=False, fallback_active=True)
            else:
                record = posterior.update(action, observed)
                if record['expanded_now']:
                    self.expansions.append(dict(input_index=index, step=posterior.step, action=action))
                if record['failed']:
                    self.fallback = True
                    self.fallback_trigger = dict(input_index=index, step=posterior.step, action=action,
                        category=record['category'], global_action_index=len(self.predictions)-len(self.posteriors))
                record['fallback_active'] = self.fallback
            self.predictions.append(dict(input_index=index, **record))
            return record
        finally:
            self.planning_seconds += time.perf_counter()-started


def run_batch(executors, policy, budget, model, permutation=None):
    """Execute only the chosen action; hidden values do not enter Scheduler."""
    if policy not in ('B2', 'D', 'Dprime'): raise ValueError('Unknown updated policy')
    if math.isnan(budget) or budget < 0: raise ValueError('Budget must be nonnegative')
    requests = [e.request for e in executors]
    canonical = [(i,a) for i in range(len(requests)) for a in agent.ORDER]
    if permutation is not None:
        pairs = [tuple(p) for p in permutation]
        if len(pairs) != len(canonical) or set(pairs) != set(canonical):
            raise ValueError('Global permutation must contain every index/action once')
    initialization_started = time.perf_counter()
    scheduler = Scheduler(requests, model)
    scheduler.planning_seconds += time.perf_counter()-initialization_started
    actions, records = [], [[] for _ in requests]
    spent_by_input = [0. for _ in requests]
    spent = 0.
    while True:
        selected = scheduler.choose(budget-spent)
        if selected is None: break
        index, action, plan = selected
        observed = executors[index].perform(action)
        charge = scheduler.prices[index][action]
        spent += charge
        spent_by_input[index] += charge
        scheduler.update(index, action, observed)
        row = dict(input_index=index, action=action, cost=charge, observation=copy.deepcopy(observed), plan=copy.deepcopy(plan))
        actions.append(row)
        records[index].append(copy.deepcopy(row))
    inputs = []
    for i, evidence in enumerate(scheduler.evidence):
        result = agent.verdict(evidence)
        stop = ('verdict_complete' if result != 'inconclusive' else
                'no_remaining_actions' if not agent.available(evidence) else 'no_affordable_action')
        inputs.append(dict(input_index=i, verdict=result, spent=spent_by_input[i], actions=records[i],
            observed=evidence, drop_reason=None, stop_reason=stop,
            early_violation=bool(result=='violation_detected' and agent.available(evidence)),
            futile_v2=any(v.get('step',{}).get('futile',False) for v in evidence.values())))
    return dict(policy=policy, budget=budget, spent=spent, actions=actions, inputs=inputs,
        predictions=scheduler.predictions,
        support_audit=dict(expansions=scheduler.expansions, expanded_inputs=len(scheduler.expansions),
            package_fallback=scheduler.fallback, fallback_trigger=scheduler.fallback_trigger,
            missing_predictions=sum(p['full_probabilities'] is None for p in scheduler.predictions)),
        planning_seconds=scheduler.planning_seconds)
