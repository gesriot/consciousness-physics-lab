"""Shared-budget scheduling from requests and already observed evidence only."""
import copy
import math
from attention.agent import ORDER, cost, available, verdict
from attention.s2 import predict

POLICIES = ('B', 'C1', 'C2', 'C3', 'C4', 'D', 'Dprime')
FRACTIONS = (.2, .4, .6, 1.)
CHEAP = ('binding', 'normalization', 'first_moment', 'high_frequency')


def completed_inconclusive(observed):
    return any(v['verdict'] == 'inconclusive' and
               (a in ('modal', 'collocation') or
                (a.startswith('v2_') and v.get('terminal', False)))
               for a, v in observed.items())


def remaining_actions(observed):
    terminal = any(a.startswith('v2_') and v.get('terminal', False)
                   for a, v in observed.items())
    return [a for a in ORDER if a not in observed and
            not (terminal and a.startswith('v2_'))]


def choose(policy, requests, observed, remaining, model, permutation, state):
    """Return an index/action pair, with no executor or hidden-outcome access.

    State contains only cached request predictions, known prices, and scheduling
    decisions. The same initial global permutation is supplied for every budget.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown batch policy')
    if len(requests) != len(observed):
        raise ValueError('Requests and observations must align')
    dropped = state.setdefault('dropped', {})
    if 'costs' not in state:
        state['costs'] = [{a: cost(a, r, model['action_model']) for a in ORDER}
                          for r in requests]
    prices = state['costs']
    if policy in ('B', 'D', 'Dprime') and 'predictions' not in state:
        state['predictions'] = [predict(r, model=model) for r in requests]
    candidates = {}
    for i, evidence in enumerate(observed):
        if i in dropped or verdict(evidence) != 'inconclusive':
            continue
        if policy == 'C4' and completed_inconclusive(evidence):
            dropped[i] = 'completed_inconclusive'
            continue
        options = [a for a in available(evidence) if prices[i][a] <= remaining]
        if not options:
            continue
        if policy in ('B', 'D', 'Dprime'):
            prediction = state['predictions'][i]
            solvable = prediction['solvable_probability']
            if completed_inconclusive(evidence):
                solvable = max(0., solvable-prediction['full_probabilities']['accept'])
            if solvable < .10 and not any(
                    prediction['actions'][a]['detection_probability'] >= .05
                    for a in options):
                dropped[i] = 'predicted_unresolvable'
                continue
        candidates[i] = options
    if not candidates:
        return None
    pairs = [(i, a) for i, options in candidates.items() for a in options]
    tie = lambda pair: (pair[0], ORDER.index(pair[1]))
    if policy == 'C1':
        return min(pairs, key=tie)
    if policy == 'C2':
        return min(pairs, key=lambda pair: (prices[pair[0]][pair[1]], *tie(pair)))
    if policy == 'C3':
        rank = {tuple(pair): n for n, pair in enumerate(permutation)}
        return min(pairs, key=lambda pair: rank[pair])
    remaining_cost = lambda i: sum(prices[i][a] for a in remaining_actions(observed[i]))
    if policy == 'C4':
        cheap = [pair for pair in pairs if pair[1] in CHEAP]
        if cheap:
            return min(cheap, key=lambda pair: (prices[pair[0]][pair[1]], *tie(pair)))
        current = state.get('current')
        if current not in candidates:
            current = min(candidates, key=lambda i: (remaining_cost(i), i))
            state['current'] = current
        return current, min(candidates[current], key=ORDER.index)
    def task_score(i):
        prediction = state['predictions'][i]
        numerator = prediction['solvable_probability']
        if completed_inconclusive(observed[i]):
            numerator = max(0., numerator-prediction['full_probabilities']['accept'])
        return numerator/remaining_cost(i)
    index = min(candidates, key=lambda i: (-task_score(i), i))
    prediction = state['predictions'][index]
    blocked_accept = completed_inconclusive(observed[index])
    def action_score(action):
        entry = prediction['actions'][action]
        accept = entry['accept_probability']
        if blocked_accept or (action == 'collocation' and
                requests[index].tolerance < model['action_model']['collocation_accept_floor']):
            accept = 0.
        return (entry['detection_probability'] + .05*accept)/prices[index][action]
    return index, min(candidates[index], key=lambda a: (-action_score(a), ORDER.index(a)))


def run_batch(executors, policy, budget, model, permutation=None):
    """Execute selected actions only. Model corruption is supplied by the caller."""
    if policy not in POLICIES:
        raise ValueError('Unknown batch policy')
    if math.isnan(budget) or budget < 0:
        raise ValueError('Budget must be nonnegative')
    requests = [e.request for e in executors]
    canonical = [(i, a) for i in range(len(requests)) for a in ORDER]
    if permutation is None:
        if policy == 'C3':
            raise ValueError('C3 requires its registered global permutation')
        permutation = canonical
    permutation = [tuple(pair) for pair in permutation]
    if len(permutation) != len(canonical) or set(permutation) != set(canonical):
        raise ValueError('Global permutation must contain every index/action once')
    observed = [{} for _ in requests]
    records = [[] for _ in requests]
    spent_by_input = [0. for _ in requests]
    state, actions, spent = {}, [], 0.
    while True:
        selected = choose(policy, requests, observed, budget-spent, model, permutation, state)
        if selected is None:
            break
        index, action = selected
        observation = executors[index].perform(action)
        charge = state['costs'][index][action]
        spent += charge
        spent_by_input[index] += charge
        observed[index][action] = observation
        row = dict(input_index=index, action=action, cost=charge,
                   observation=copy.deepcopy(observation))
        actions.append(row)
        records[index].append(copy.deepcopy(row))
    inputs = []
    for i, evidence in enumerate(observed):
        result = verdict(evidence)
        dropped = state.get('dropped', {}).get(i)
        stop = ('verdict_complete' if result != 'inconclusive' else
                'dropped' if dropped else
                'no_remaining_actions' if not available(evidence) else 'no_affordable_action')
        inputs.append(dict(input_index=i, verdict=result, spent=spent_by_input[i],
                           actions=records[i], observed=evidence, drop_reason=dropped,
                           stop_reason=stop,
                           early_violation=bool(result == 'violation_detected' and available(evidence)),
                           futile_v2=any(v.get('step', {}).get('futile', False) for v in evidence.values())))
    return dict(policy=policy, budget=budget, spent=spent, actions=actions, inputs=inputs)
