"""Paid self-check scheduling over live, typed standing provenance."""
import copy
import math
from types import SimpleNamespace
import numpy as np
from attention import agent, s3, channel_monitor as monitor
from attention.build_a007_library import CHANNELS, channel
from attention.calibration_samples import maximum_cost, v2_observation
from lab.executor import request_hash
from lab.guard import final_guard
from binding import b002
from binding.revision_b002 import selection

POLICIES = ('S', 'Sx', 'Sxprime', 'RR', 'BU', 'N0')
QUOTAS = (1, 2, 4)
RECALL_CAP = 32
CATEGORIES = ('sample', 'ordinary', 'replacement', 'recall', 'future')


def rounded(value, noise=False):
    if noise:
        value = float(np.nextafter(float(value), np.inf))
    return round(float(value), 9)


def validate_request_link(record, requests):
    edges = [e for e in record['links'] if e['kind'] == 'request']
    if len(edges) != 1:
        raise ValueError('Exactly one canonical request root required')
    edge = edges[0]
    if edge['request_id'] != record['task_id'] or edge['request_id'] not in requests:
        raise ValueError('Canonical request endpoint mismatch')
    if edge['request_hash'] != request_hash(requests[edge['request_id']]):
        raise ValueError('Canonical request hash mismatch')
    return requests[edge['request_id']]


def visible_graph(records, markers, policy):
    """Typed edits depend only on visible support, never on the injected target."""
    rows = copy.deepcopy([r for r in records if r['standing']])
    edits = []
    if policy not in ('Sx', 'Sxprime'):
        return rows, edits
    for row in rows:
        relevant = [e for e in row['links'] if e['kind'] == 'channel'
                    and e['channel'].startswith('collocation:')
                    and e['time'] > markers[e['channel']]]
        if not relevant:
            continue
        if len(relevant) != 1:
            raise ValueError('Ambiguous collocation support')
        roots = [e for e in row['links'] if e['kind'] == 'request']
        if len(roots) != 1:
            raise ValueError('Ambiguous request-origin control')
        edge = relevant[0] if policy == 'Sx' else roots[0]
        before = edge['channel']
        edge['channel'] = before.removeprefix('independent:') if before.startswith('independent:') else 'independent:'+before
        edits.append(dict(record_id=row['id'], edge_id=edge['edge_id'], before=before,
                          after=edge['channel'], time=edge['time'], binary_distance=1))
    return rows, edits


def stakes(records, markers, prices, noise=False):
    counts = {c: 0 for c in CHANNELS}
    for row in records:
        if not row['standing']:
            continue
        used = {e['channel'] for e in row['links'] if e['kind'] == 'channel'
                and e['channel'] in counts and e['time'] > markers[e['channel']]}
        for c in used:
            counts[c] += 1
    return counts, {c: rounded(counts[c]/prices[c], noise) for c in CHANNELS}


def paired_graphs(records, markers, prices, noise=False):
    snapshots = {}
    for policy in ('S', 'Sx', 'Sxprime'):
        graph, edits = visible_graph(records, markers, policy)
        counts, scores = stakes(graph, markers, prices, noise)
        snapshots[policy] = dict(records=graph, edits=edits, numerators=counts, scores=scores)
    if snapshots['S']['scores'] != snapshots['Sxprime']['scores']:
        raise AssertionError('Irrelevant request-origin edit changed a stake')
    if [e['record_id'] for e in snapshots['Sx']['edits']] != [e['record_id'] for e in snapshots['Sxprime']['edits']]:
        raise AssertionError('Unmatched graph intervention records')
    return snapshots


def choose_sample(policy, records, markers, suspect, prices, rr_pointer, noise=False):
    """Interface deliberately contains no evaluator or injection fields."""
    eligible = [c for c in CHANNELS if c not in suspect]
    if not eligible:
        raise ValueError('All channels suspect')
    rows, edits = visible_graph(records, markers, policy)
    counts, scores = stakes(rows, markers, prices, noise)
    if policy == 'RR':
        index = next((rr_pointer+k) % len(CHANNELS) for k in range(len(CHANNELS))
                     if CHANNELS[(rr_pointer+k) % len(CHANNELS)] in eligible)
        chosen = CHANNELS[index]
        pointer = (index+1) % len(CHANNELS)
    else:
        chosen = min(eligible, key=lambda c: (-scores[c], markers[c], CHANNELS.index(c)))
        pointer = rr_pointer
    return chosen, pointer, dict(records=rows, edits=edits, numerators=counts,
        denominators=prices, scores=scores, markers=copy.deepcopy(markers),
        suspect=sorted(suspect), tie_order=list(CHANNELS),
        paired_graphs=paired_graphs(records, markers, prices, noise))


class State:
    def __init__(self, policy, quota, rr_phase):
        if policy not in POLICIES or (policy != 'N0' and quota not in QUOTAS):
            raise ValueError('Unregistered policy or quota')
        self.policy = policy
        self.quota = 0 if policy == 'N0' else quota
        self.rr_phase = int(rr_phase)
        self.rr_pointer = int(rr_phase)
        self.last_success = {c: -1 for c in CHANNELS}
        self.counts = {c: 0 for c in CHANNELS}
        self.suspect = set()
        self.records = []
        self.pending = []
        self.recalled = set()
        self.tick = 0
        self.requests = {}
        self.candidates = {}
        self.scientific_models = {}

    def snapshot(self):
        return copy.deepcopy(dict(policy=self.policy, quota=self.quota,
            rr_phase=self.rr_phase, rr_pointer=self.rr_pointer,
            last_success=self.last_success, counts=self.counts,
            suspect=sorted(self.suspect), records=self.records, pending=self.pending,
            recalled=sorted(self.recalled), tick=self.tick))

    def alarm(self, which, tick):
        if which in self.suspect:
            return []
        self.suspect.add(which)
        notice = dict(channel=which, time=tick, last_success=self.last_success[which])
        rows = [r for r in self.records if r['standing'] and r['time'] < tick]
        chosen = selection(rows, notice, 'G')
        already = self.recalled | {p['record_id'] for p in self.pending}
        new = [dict(record_id=rid, channel=which, notice=copy.deepcopy(notice))
               for rid in chosen if rid not in already]
        self.pending.extend(new)
        return new


def _cross(which, request, outcome, weights, model_name='sphere'):
    """Same qualified paid cross execution for retrospective and future checks."""
    executor = SimpleNamespace(request=request, outcome=outcome)
    if which.startswith(('collocation:', 'v2:')):
        instrument = 'graded_v2' if which.startswith('collocation:') else 'collocation'
        result = b002.replacement(instrument, request, complex(*outcome['value']))
        result['instrument'] = instrument
        result['charged_work'] = (dict(grids=list(agent.GRIDS)) if instrument == 'graded_v2'
                                  else dict(degrees=list(b002.P.CONFIG.third_degrees)))
        return result, b002.native_price(which, request, weights)
    # Revalidate the frozen candidate rather than silently substituting a new one.
    if which == 'modal':
        modes = b002.P.plan_modes(request)
        checked_value = b002.P.model_response('modal', request.z, request.mean_time, modes, model_name)
        resolution = final_guard(request, copy.deepcopy(outcome))
        fresh = dict(value=[checked_value.real, checked_value.imag], calls=1, modes_spent=modes,
                     trace=[dict(call=1, modes=modes)])
        fresh_resolution = final_guard(request, fresh)
        passed = resolution['accepted'] and fresh_resolution['accepted']
        result = dict(verdict='accept' if passed else 'inconclusive',
                      resolution=resolution, outcome=copy.deepcopy(outcome), instrument='frozen_candidate_guard',
                      reexecution_outcome=fresh, reexecution_resolution=fresh_resolution,
                      reexecution_value=fresh['value'], charged_modes=modes)
    elif which == 'binding':
        result = dict(verdict='accept', request_hash=request_hash(request), instrument='canonical_request_binding')
    else:
        result = agent.feature(request, which, model_name)
        result['instrument'] = 'clean_original_context'
        result['model'] = model_name
    return result, b002.native_price(which, request, weights)


def run_package(executors, infos, intake_rows, package_id, budget, model, library,
                state, fault_channels, noise=False, sample_seed=None):
    """Execute a package; injection membership is consulted only after selection."""
    weights = model['action_model']['cost_weights']
    pools = monitor.sample_pools(library)
    max_sample = max(maximum_cost(s, weights) for pool in pools.values() for s in pool)
    quota = state.quota
    if budget < quota*max_sample:
        raise ValueError('Registered sample quota exceeds package allowance')
    before = state.snapshot()
    evidence = [{} for _ in executors]
    extras = [{} for _ in executors]
    supports = [{} for _ in executors]
    closed = {}
    actions, alerts = [], []
    costs = {c: 0. for c in CATEGORIES}
    done_samples = 0
    ordinary_count = 0
    bu_last = {c: 0 for c in CHANNELS}
    score_cache = {}
    initial_ordinary_allowance = budget-quota*max_sample
    canonical = {row['task_id']: row for row in intake_rows}
    for executor, info in zip(executors, infos):
        task = info['task_id']
        if task not in canonical or request_hash(executor.request) != canonical[task]['request_sha256']:
            raise ValueError('Input differs from canonical request')
        state.requests[task] = executor.request

    def charge(kind, price):
        costs[kind] += price
        if sum(costs.values()) > budget+1e-9:
            raise AssertionError('Work allowance exceeded')

    def append_action(row):
        state.tick += 2
        row['tick'] = state.tick-1
        actions.append(row)
        return row['tick']

    def close_inputs():
        for i, obs in enumerate(evidence):
            if i in closed:
                continue
            verdict = agent.verdict(obs)
            required = set(supports[i]) & state.suspect
            effective = monitor.combined(obs, extras[i], required)
            if verdict == 'violation_detected' or effective == 'violation_detected':
                closed[i] = dict(input_index=i, verdict='violation_detected', required_replacements=sorted(required))
            elif effective == 'accept':
                rid = f'{package_id}:{i}'
                links = []
                for c, edge in supports[i].items():
                    links.append(dict(edge, edge_id=f'{rid}:{len(links)}', kind='channel'))
                task = infos[i]['task_id']
                links.append(dict(edge_id=f'{rid}:request', kind='request', channel='request',
                    time=state.tick, request_id=task, request_hash=canonical[task]['request_sha256']))
                row = dict(id=rid, time=state.tick, package_id=package_id, input_index=i,
                    task_id=task, standing=True, links=links, original_links=copy.deepcopy(links), history=[])
                validate_request_link(row, state.requests)
                if executors[i].outcome['value'] != infos[i]['outcome']['value']:
                    raise ValueError('Candidate differs from shared construction')
                state.candidates[rid] = copy.deepcopy(executors[i].outcome)
                state.scientific_models[rid] = getattr(executors[i], 'model', 'sphere')
                state.records.append(row)
                closed[i] = dict(input_index=i, verdict='accept', record_id=rid, required_replacements=sorted(required))

    with b002.cross_channels():
        while True:
            close_inputs()
            spent = sum(costs.values())
            reserve = (quota-done_samples)*max_sample
            remaining = max(0., budget-spent-reserve)
            ordinary_cost = costs['ordinary']+costs['replacement']+costs['future']
            ordinary_choices, future_choices = [], []
            for i, obs in enumerate(evidence):
                if i in closed:
                    continue
                if agent.verdict(obs) == 'accept':
                    for c in (set(supports[i]) & state.suspect)-set(extras[i]):
                        price = b002.native_price(c, executors[i].request, weights)
                        if price <= remaining:
                            future_choices.append((rounded(price, noise), i, CHANNELS.index(c), price, c))
                else:
                    for a in agent.available(obs):
                        price = agent.cost(a, executors[i].request, model['action_model'])
                        if price <= remaining:
                            ordinary_choices.append((rounded(price, noise), i, agent.ORDER.index(a), price, a))
            choices = future_choices or ordinary_choices
            due = done_samples < quota and (ordinary_cost >= done_samples/quota*initial_ordinary_allowance or not choices)
            if due:
                if all(c in state.suspect for c in CHANNELS):
                    raise ValueError('All channels suspect')
                sample_prices = {c: maximum_cost(pools[c][state.counts[c] % len(pools[c])], weights) for c in CHANNELS}
                if state.policy == 'BU':
                    which, scores, details = monitor.score_channels(
                        [e.request for e in executors], evidence, None, closed, weights,
                        pools, bu_last, ordinary_count, 'BU', state.suspect, noise, score_cache)
                    scores = {c: rounded(details['used'][c]*(1+ordinary_count-bu_last[c])/
                        max(maximum_cost(s, weights) for s in pools[c]), noise) for c in CHANNELS}
                    which = min((c for c in CHANNELS if c not in state.suspect),
                                key=lambda c: (-scores[c], CHANNELS.index(c)))
                    snapshot = dict(records=copy.deepcopy([r for r in state.records if r['standing']]),
                        markers=copy.deepcopy(state.last_success), suspect=sorted(state.suspect),
                        scores=scores, details=details, bu_last=copy.deepcopy(bu_last),
                        ordinary_count=ordinary_count, evidence=copy.deepcopy(evidence),
                        closed=sorted(closed), edits=[], denominators=sample_prices,
                        paired_graphs=paired_graphs(state.records, state.last_success, sample_prices, noise))
                else:
                    which, state.rr_pointer, snapshot = choose_sample(state.policy,
                        state.records, state.last_success, state.suspect, sample_prices,
                        state.rr_pointer, noise)
                    scores = snapshot['scores']
                sample = pools[which][state.counts[which] % len(pools[which])]
                spoiled = which in fault_channels
                result = monitor.sample_execution(sample, weights, 'permissive' if spoiled else 'none')
                price = result['cost']
                if price > max_sample or price > budget-spent:
                    raise AssertionError('Sample reserve invalid')
                mismatch = result['verdict'] != sample['expected']
                marker_before = state.last_success[which]
                tick = append_action(dict(kind='sample', input_index=None, channel=which,
                    cost=price, observation=result['observed'], original_observation=result['original'],
                    sample_trace=result['trace'], sample_id=sample['sample_id'], expected=sample['expected'],
                    mismatch=mismatch, fault_applied=spoiled, scores=scores,
                    selection_snapshot=snapshot, ordinary_cost_before=ordinary_cost,
                    reserve_before=reserve, sample_index=done_samples, last_success_before=marker_before))
                charge('sample', price)
                state.counts[which] += 1
                bu_last[which] = ordinary_count
                done_samples += 1
                if mismatch:
                    queue = state.alarm(which, tick)
                    alerts.append(dict(channel=which, tick=tick, package_id=package_id,
                        last_success=marker_before, queued=copy.deepcopy(queue), fault_applied=spoiled))
                else:
                    state.last_success[which] = tick
                continue
            # Recall priority is shared across all alarms and packages.
            available_review = None
            if len(state.recalled) < RECALL_CAP:
                for pending in list(state.pending):
                    row = next(r for r in state.records if r['id'] == pending['record_id'])
                    eligible = row['standing'] and any(e['kind'] == 'channel' and e['channel'] == pending['channel']
                        and e['time'] > pending['notice']['last_success'] for e in row['links'])
                    if not eligible or row['id'] in state.recalled:
                        state.pending.remove(pending)
                        continue
                    request = validate_request_link(row, state.requests)
                    price = b002.native_price(pending['channel'], request, weights)
                    if price <= remaining:
                        available_review = (pending, row, request, price)
                    # The valid queue head cannot be bypassed by cheaper reviews.
                    break
            if available_review:
                pending, row, request, price = available_review
                result, actual = _cross(pending['channel'], request, state.candidates[row['id']], weights,
                                        state.scientific_models.get(row['id'], 'sphere'))
                if actual != price:
                    raise AssertionError('Review price changed')
                tick = append_action(dict(kind='recall', input_index=None, channel=pending['channel'],
                    record_id=row['id'], cost=price, observation=result, original_observation=copy.deepcopy(result),
                    fault_applied=False, notice=copy.deepcopy(pending['notice'])))
                charge('recall', price)
                state.recalled.add(row['id'])
                state.pending.remove(pending)
                event = dict(tick=tick, channel=pending['channel'], verdict=result['verdict'], cost=price)
                row['history'].append(event)
                row['standing'] = result['verdict'] == 'accept'
                if row['standing']:
                    for edge in row['links']:
                        if edge['kind'] == 'channel' and edge['channel'] == pending['channel']:
                            edge['channel'] = 'independent:'+edge['channel']
                            edge['time'] = tick
                continue
            if not choices:
                break
            _, i, _, price, action = min(choices)
            if future_choices:
                result, actual = _cross(action, executors[i].request, executors[i].outcome, weights,
                                        getattr(executors[i], 'model', 'sphere'))
                if price != actual:
                    raise AssertionError('Future check price changed')
                tick = append_action(dict(kind='future', input_index=i, channel=action, cost=price,
                    observation=result, original_observation=copy.deepcopy(result), fault_applied=False))
                extras[i][action] = result
                if result['verdict'] == 'accept':
                    supports[i][action] = dict(channel='independent:'+action, time=tick)
                charge('future', price)
            else:
                which = channel(action, executors[i].request)
                original = executors[i].perform(action)
                if action.startswith('v2_'):
                    original = v2_observation(executors[i].values, int(action[3:]),
                        complex(*executors[i].outcome['value']), executors[i].request.tolerance)
                spoiled = which in fault_channels
                observed = monitor.hook(original, executors[i].request.tolerance, 'permissive' if spoiled else 'none')
                executors[i].observed[action] = copy.deepcopy(observed)
                evidence[i][action] = s3.observation(action, s3.token(action, observed))
                tick = append_action(dict(kind='ordinary', input_index=i, action=action, channel=which,
                    cost=price, observation=copy.deepcopy(evidence[i][action]),
                    original_observation=original, hooked_observation=observed, fault_applied=spoiled))
                if observed['verdict'] == 'accept':
                    supports[i][which] = dict(channel=which, time=tick)
                charge('ordinary', price)
                ordinary_count += 1
        close_inputs()
    if done_samples != quota:
        raise AssertionError('Unfulfilled sample quota')
    inputs = []
    for i, obs in enumerate(evidence):
        inputs.append(closed.get(i, dict(input_index=i, verdict='inconclusive',
            required_replacements=sorted(set(supports[i]) & state.suspect))))
    state.tick += 2
    spent = sum(costs.values())
    return dict(package_id=package_id, policy=state.policy, quota=quota, budget=budget,
        spent=spent, costs=costs, unused_budget=budget-spent, actions=actions,
        inputs=inputs, alerts=alerts, records=copy.deepcopy(state.records),
        state=state.snapshot(), state_before=before, sample_seed=sample_seed)
