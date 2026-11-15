"""Independent reconstruction of fallible-check diagnosis records.

Policy decisions are reconstructed here, never requested from a policy simulator.
The physical executor is shared, but state, ranking, inference and ledgers are not.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from binding import source_lab_b005 as physics
from binding import source_lab_b006 as laboratory
from binding.replay_intake import compare_regenerated


def require(condition, label):
    if not condition:
        raise ValueError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def numeric(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= 64*math.ulp(1.)*(1+abs(expected)), label)


def compare_structure(actual, expected, label, allow_numeric=False):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+' keys')
        for key in expected:
            compare_structure(actual[key], expected[key], label+'.'+str(key), allow_numeric)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+' length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_structure(a, b, label+'.'+str(i), allow_numeric)
    elif allow_numeric and isinstance(expected, float):
        numeric(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def inherited_labels(case):
    rows = {r['id']: r for r in case['records']}
    require(len(rows) == len(case['records']), 'unique claim identifiers')
    cache = {}
    def visit(cid, active):
        require(cid not in active, 'cyclic provenance')
        if cid not in cache:
            row = rows[cid]
            result = set(row['source_ids'])
            for parent in row['parents']:
                require(parent in rows, 'missing parent')
                result |= visit(parent, active | {cid})
            cache[cid] = result
        return cache[cid]
    return {cid: visit(cid, set()) for cid in rows}


class Evidence:
    """Rebuild private test reports from the archived generating specification."""
    def __init__(self, episode, noise=False):
        self.episode = episode
        self.case = episode['case']
        self.noise = noise
        self.rows = {r['id']: r for r in self.case['records']}
        self.sources = copy.deepcopy(self.case['sources'])
        self.values = {c: row['value'] for c, row in self.rows.items()}
        self.attempts = {}
        self.events = []

    def source_cost(self, sid):
        return 100 if self.sources[sid]['kind'] == 'method' else 30

    def claim_cost(self, cid):
        return physics.OPERATION_COST[self.rows[cid]['operation']]+1

    def repeat_cost(self):
        return 2*self.case['budget_unit']

    def probe(self, test):
        kind, identity = test['kind'], test['id']
        if kind == 'check_source':
            truth = self.sources[identity] != self.case['catalog'][identity]
        elif kind == 'check_claim':
            sid = self.rows[identity]['step_id']
            corrected = dict(self.sources)
            corrected[sid] = self.case['catalog'][sid]
            value = physics.rederive(identity, self.case, self.values, corrected, self.noise)[0]
            truth = not physics.close(value, self.values[identity])
        elif kind == 'repeat_observation':
            ids = self.episode['signal_ids']
            if self.episode['signal_branch'] == 'measurement':
                truth = any(not physics.close(self.values[c], self.case['truth'][c]) for c in ids)
            else:
                truth = any(not physics.close(self.values[a], self.values[b])
                            for a, b in self.case['pairs'] if a in ids or b in ids)
        else:
            raise ValueError('unknown probe kind')
        key = (kind, identity)
        attempt = self.attempts.get(key, 0)
        self.attempts[key] = attempt+1
        rate = self.episode['evaluator']['rates'][kind]['false_good' if truth else 'false_bad']
        data = json.dumps(('probe', int(self.episode['seed']), kind, str(identity), attempt),
                          separators=(',', ':'), ensure_ascii=True).encode()
        draw = int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')/2**64
        bad = bool(truth) != (draw < rate)
        self.events.append(dict(kind=kind, id=identity, attempt=attempt,
                                truth_bad=bool(truth), reported_bad=bad, error=bad != truth))
        return bad

    def install(self, sid):
        self.sources[sid] = copy.deepcopy(self.case['catalog'][sid])

    def repair_cost(self, cid):
        claims, sources = set(), set()
        def visit(c):
            if c in claims:
                return
            claims.add(c)
            sources.update(self.rows[c]['source_ids'])
            for parent in self.rows[c]['parents']:
                visit(parent)
        visit(cid)
        return len(sources)+sum(physics.OPERATION_COST[self.rows[c]['operation']] for c in claims)

    def repair(self, cid):
        values, visited = dict(self.values), set()
        def rebuild(c):
            if c in visited:
                return
            for parent in self.rows[c]['parents']:
                rebuild(parent)
            values[c] = physics.rederive(c, self.case, values, self.sources, self.noise)[0]
            visited.add(c)
        rebuild(cid)
        self.values[cid] = values[cid]
        return values[cid]


CONFIGS = {
    'beta8_t80': ((8., 2.), (1., 9.), .8),
    'beta8_t95': ((8., 2.), (1., 9.), .95),
    'beta99_t80': ((99., 1.), (1., 99.), .8),
    'beta99_t95': ((99., 1.), (1., 99.), .95),
}
KINDS = ('check_source', 'check_claim', 'repeat_observation')
CONSENSUS = {'K1of1': (1, 1), 'K2of3': (2, 3), 'K3of5': (3, 5)}


def starting_state(config):
    tp, fp, _ = CONFIGS[config]
    def channels():
        return {kind: {'tp': list(tp), 'fp': list(fp)} for kind in KINDS}
    return dict(prior=[2., 8.], channels=channels(), auxiliary=channels(), resolved_episodes=0)


def rounded(value, noise=False):
    return round(math.nextafter(float(value), math.inf) if noise else float(value), 9)


def model_state(state, policy):
    views = {}
    for name in ('channels', 'auxiliary'):
        views['active' if name == 'channels' else name] = {
            kind: {label: counts[0]/sum(counts) for label, counts in row.items()}
            for kind, row in state[name].items()}
    target = 'active' if policy == 'Px' else 'auxiliary' if policy == 'Pxprime' else None
    if target:
        views[target] = {kind: {'tp': .999999, 'fp': .000001} for kind in KINDS}
    return views


def likelihoods(posterior, test, model):
    return {h: model[test['kind']]['tp' if
            (h != 'observation' if test['kind'] == 'repeat_observation' else h == test['candidate'])
            else 'fp'] for h in posterior}


def bayes(posterior, test, outcome, model):
    probs = likelihoods(posterior, test, model)
    weighted = {h: p*(probs[h] if outcome else 1-probs[h]) for h, p in posterior.items()}
    total = math.fsum(weighted.values())
    require(total > 0, 'nonzero Bayesian evidence')
    return {h: p/total for h, p in weighted.items()}


def expected_score(posterior, test, model, noise=False):
    probs = likelihoods(posterior, test, model)
    probability = math.fsum(posterior[h]*probs[h] for h in posterior)
    expected_loss = math.fsum(weight*(1-max(bayes(posterior, test, outcome, model).values()))
                             for outcome, weight in ((True, probability), (False, 1-probability))
                             if weight > 0)
    return rounded(max(0., (1-max(posterior.values()))-expected_loss)/test['cost'], noise)


def update_state(state, posterior, reports):
    updated = copy.deepcopy(state)
    observation_weight = posterior.get('observation', 0.)
    updated['prior'][0] += observation_weight
    updated['prior'][1] += 1-observation_weight
    for report in reports:
        bad_weight = (1-observation_weight if report['kind'] == 'repeat_observation'
                      else posterior.get(report['candidate'], 0.))
        row = updated['channels'][report['kind']]
        outcome = int(report['bad'])
        for label, weight in (('tp', bad_weight), ('fp', 1-bad_weight)):
            row[label][0] += weight*outcome
            row[label][1] += weight*(1-outcome)
    updated['resolved_episodes'] += 1
    return updated


def audit_output(episode, output, policy, budget, state=None, config='beta8_t80', noise=False):
    """Independently execute the registered inference equations and paid ledger."""
    require(policy in ('D', 'R', 'N', 'P', 'Px', 'Pxprime', *CONSENSUS), 'policy identity')
    require(budget in (2, 8, 32), 'budget cohort')
    state = copy.deepcopy(starting_state(config) if state is None else state)
    compare_structure(output['state_before'], state, 'state before', True)
    case = episode['case']
    evidence = Evidence(episode, noise)
    rows, labels = evidence.rows, inherited_labels(case)
    ids, branch = episode['signal_ids'], episode['signal_branch']
    initial = [labels[c] for c in ids]
    original = (set.intersection(*initial) if initial and branch == 'measurement'
                else set.union(*initial) if initial else set())
    candidates = set(original)
    limit, spent = budget*case['budget_unit'], 0
    decision, probability, resolved = None, None, not ids
    checked, repaired, trace, attempts = set(), [], [], {}
    probabilistic = policy in ('P', 'Px', 'Pxprime')
    prior_obs = state['prior'][0]/sum(state['prior'])
    posterior = ({'observation': prior_obs, **{s: (1-prior_obs)/len(original) for s in sorted(original)}}
                 if original else {'observation': 1.})
    view = model_state(state, policy)
    def tests_for(grounds, excluded):
        tests = []
        for sid in sorted(grounds & evidence.sources.keys()):
            if sid in excluded:
                continue
            tests.append(dict(kind='check_source', id=sid, candidate=sid, cost=evidence.source_cost(sid)))
            for cid, row in rows.items():
                if row['step_id'] == sid:
                    tests.append(dict(kind='check_claim', id=cid, candidate=sid, cost=evidence.claim_cost(cid)))
        return tests
    def perform(test, score=0.):
        nonlocal spent
        require(spent+test['cost'] <= limit, 'affordable action')
        key = (test['kind'], test['id'])
        attempt = attempts.get(key, 0)
        attempts[key] = attempt+1
        bad = evidence.probe(test)
        spent += test['cost']
        trace.append(dict(test, attempt=attempt, bad=bool(bad), score=score))
        return bad
    observation = dict(kind='repeat_observation', id='|'.join(sorted(ids)),
                       candidate='observation', cost=evidence.repeat_cost())
    if ids and policy == 'R':
        if observation['cost'] <= limit:
            if not perform(observation):
                decision, resolved = 'observation', True
        else:
            candidates = set()
    if ids and policy != 'N' and not resolved:
        if probabilistic:
            while True:
                best = min(posterior, key=lambda h: (-rounded(posterior[h], noise), h))
                if rounded(posterior[best], noise) >= CONFIGS[config][2]:
                    decision, probability, resolved = best, posterior[best], True
                    break
                tests = [t for t in tests_for(original, set())+[observation] if spent+t['cost'] <= limit]
                if not tests:
                    break
                scores = [(expected_score(posterior, t, view['active'], noise), t) for t in tests]
                score, test = min(scores, key=lambda pair: (-pair[0], pair[1]['kind'], pair[1]['id'], pair[1]['candidate']))
                if score <= 0:
                    break
                posterior = bayes(posterior, test, perform(test, score), view['active'])
            if probability is None:
                best = min(posterior, key=lambda h: (-rounded(posterior[h], noise), h))
                probability = posterior[best]
        else:
            while not resolved:
                tests = tests_for(candidates, checked)
                if not tests:
                    if branch == 'measurement' and spent+observation['cost'] <= limit:
                        if not perform(observation):
                            decision, resolved = 'observation', True
                    break
                affordable = [t for t in tests if spent+t['cost'] <= limit]
                if not affordable:
                    break
                test = min(affordable, key=lambda t: (-rounded(1/t['cost'], noise),
                           t['kind'] != 'check_claim', t['id'], t['candidate']))
                k, m = CONSENSUS.get(policy, (1, 1))
                votes = {True: 0, False: 0}
                outcome = None
                for _ in range(m):
                    if spent+test['cost'] > limit:
                        break
                    report = perform(test)
                    votes[report] += 1
                    if votes[report] >= k:
                        outcome = report
                        break
                if outcome is None:
                    break
                checked.add(test['candidate'])
                if outcome:
                    decision, resolved = test['candidate'], True
                    break
                candidates.discard(test['candidate'])
    terminal = dict(hypothesis=decision, probability=probability, abstained=bool(ids and decision is None))
    if decision not in (None, 'observation'):
        evidence.install(decision)
        targets = sorted((c for c in rows if decision in labels[c]), key=lambda c: (rows[c]['depth'], c))
        for cid in targets:
            cost = evidence.repair_cost(cid)
            if spent+cost > limit:
                continue
            value = evidence.repair(cid)
            spent += cost
            repaired.append(cid)
            trace.append(dict(kind='repair', id=cid, cost=cost, value=value))
        resolved = (all(physics.close(evidence.values[c], case['measurements'][c]['value']) for c in ids)
                    if branch == 'measurement' else
                    all(physics.close(evidence.values[a], evidence.values[b]) for a, b in case['pairs'] if a in ids or b in ids))
    reports = [event for event in trace if event['kind'] != 'repair']
    observations = [t for t in reports if t['kind'] == 'repeat_observation']
    observable_resolved = (bool(observations and not observations[-1]['bad']) if decision == 'observation'
                           else bool(repaired and resolved) if decision is not None else False)
    after = update_state(state, posterior, reports) if probabilistic and ids and observable_resolved else state
    suspect_sources = ({decision} if decision not in (None, 'observation') else
                       original if probabilistic else candidates)
    suspected = sorted(c for c in rows if suspect_sources & labels[c]) if ids and not resolved and policy != 'N' else []
    final_bad = sorted(c for c in rows if not physics.close(evidence.values[c], case['truth'][c]))
    initial_bad = set(case['inaccurate'])
    true_hypothesis = case['culprit'] if episode['evaluator']['kind'] == 'true_fault' else 'observation'
    innocent = decision not in (None, 'observation') and decision != case['culprit']
    metrics = dict(false_accusations=int(innocent and bool(repaired)), remaining_bad=len(final_bad),
                   lost_good=len(set(final_bad)-initial_bad), cost=spent, initial_bad=len(initial_bad),
                   attempted_false_accusations=int(innocent), checks=len(reports),
                   repeated_checks=sum(e['attempt'] > 0 for e in reports),
                   observation_repeats=sum(e['kind'] == 'repeat_observation' for e in reports),
                   rejected_false=int(episode['evaluator']['kind'] == 'false_objection' and decision == 'observation' and not repaired),
                   corrected_counterexample=int(episode['evaluator']['kind'] == 'true_fault' and decision == case['culprit'] and not final_bad),
                   diagnosis_made=int(bool(ids) and decision is not None), diagnosis_correct=int(bool(ids) and decision == true_hypothesis),
                   abstained=int(bool(ids) and decision is None), repaired=len(repaired), suspected=len(suspected),
                   signal_depth=max((rows[c]['depth'] for c in ids), default=0))
    protected = dict(signal_ids=ids, initial_candidates=sorted(original), decision=decision,
                     resolved=resolved, observable_resolved=observable_resolved, abstained=terminal['abstained'], checked=sorted(checked),
                     repaired=repaired, suspected=suspected, budget=limit, cost=spent,
                     actions=[{k: v for k, v in e.items() if k not in ('score', 'value')} for e in trace])
    compare_structure(output['protected'], protected, 'protected')
    compare_structure(output['metrics'], metrics, 'metrics')
    compare_structure(output['terminal'], terminal, 'terminal', True)
    compare_structure(output['trace'], trace, 'trace', True)
    compare_structure(output['state_after'], after, 'state after', True)
    compare_structure(output['diagnostics'], dict(posterior=posterior, model=view, final_values=evidence.values), 'diagnostics', True)
    compare_structure(output['evaluation'], dict(events=evidence.events, true_hypothesis=true_hypothesis,
                       remaining_bad_ids=final_bad), 'evaluation')
    return after


def audit_series(series, outputs, policy, budget, config='beta8_t80', noise=False):
    require(len(outputs) == len(series['episodes']), 'episode count')
    state = starting_state(config)
    for index, (episode, output) in enumerate(zip(series['episodes'], outputs)):
        require(episode['index'] == index, 'episode order')
        state = audit_output(episode, output, policy, budget, state, config, noise)
    return len(outputs)


def verify_series_data(series, noise=False):
    """Regenerate the exogenous tape; allow rounding only in physical outputs."""
    from binding.audit_b005 import verify_series
    compare_regenerated(series['intake'], physics.intake(series['seed']))
    rebuilt = laboratory.make_series(series['seed'], series['environment'], len(series['episodes']), noise, inputs=series['intake'])
    require(series.keys() == rebuilt.keys(), 'series fields')
    for key in ('seed', 'environment', 'intake'):
        compare_structure(series[key], rebuilt[key], 'series '+key)
    for actual, expected in zip(series['episodes'], rebuilt['episodes']):
        require(actual.keys() == expected.keys(), 'episode fields')
        for key in expected:
            if key == 'case':
                verify_series(actual[key], expected[key])
            else:
                compare_structure(actual[key], expected[key], 'episode '+key)
    return True


def parse_variant(variant):
    for prefix in ('Pxprime_', 'Px_', 'P_'):
        if variant.startswith(prefix):
            config = variant[len(prefix):]
            require(config in CONFIGS, 'configuration identity')
            return prefix[:-1], config
    require(variant in ('D', 'R', 'N', *CONSENSUS), 'constant identity')
    return variant, 'beta8_t80'


def expected_variants(selection=None):
    fixed = ['D', 'R', *CONSENSUS, 'N'] if selection is None else ['D', 'R', selection['selected_K'], 'N']
    configs = CONFIGS if selection is None else [selection['selected_P']]
    return fixed+[p+'_'+c for c in configs for p in ('P', 'Px', 'Pxprime')]


def audit_archive(archive, selection=None, noise=False):
    compare_regenerated(archive['inputs'], physics.intake(archive['seed']))
    require(archive['intake_sha256'] == digest(archive['inputs']), 'archive intake hash')
    require(set(archive['environments']) == set(laboratory.ENVIRONMENTS), 'environment inventory')
    conditions = {v+'|'+str(b) for v in expected_variants(selection) for b in (2, 8, 32)}
    count = 0
    for name, body in archive['environments'].items():
        series = body['series']
        require(series['environment'] == name and series['seed'] == archive['seed'], 'series identity')
        require(series['intake'] == archive['inputs'], 'paired intake')
        require(len(series['episodes']) == laboratory.EPISODES, 'registered episode count')
        verify_series_data(series, noise)
        require(set(body['outputs']) == conditions, 'condition inventory')
        for condition, outputs in body['outputs'].items():
            variant, budget = condition.split('|')
            policy, config = parse_variant(variant)
            count += audit_series(series, outputs, policy, int(budget), config, noise)
    return count


def confidence_estimates(archives):
    counts = {}
    for archive in archives:
        for body in archive['environments'].values():
            for condition, outputs in body['outputs'].items():
                name = condition.split('|')[0]
                if name not in ('D', 'R', *CONSENSUS):
                    continue
                correct, made = counts.setdefault(name, [0, 0])
                counts[name] = [correct+sum(o['metrics']['diagnosis_correct'] for o in outputs),
                                made+sum(o['metrics']['diagnosis_made'] for o in outputs)]
    return {k: good/total if total else 0. for k, (good, total) in counts.items()}


def reconstructed_summary(archives, confidences):
    cells, details, calibration = {}, {}, {}
    def sum_rows(rows):
        return {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    for archive in archives:
        for env, body in archive['environments'].items():
            episodes = body['series']['episodes']
            for condition, outputs in body['outputs'].items():
                name, budget = condition.split('|')
                for branch in ('measurement', 'conflict'):
                    rows = [o['metrics'] for e, o in zip(episodes, outputs) if e['signal_branch'] == branch]
                    cells.setdefault(f'{env}|{name}|{budget}|{branch}', []).append(sum_rows(rows))
                groups = {}
                for episode, out in zip(episodes, outputs):
                    ev, metrics = episode['evaluator'], out['metrics']
                    key = f"{env}|{name}|{budget}|{episode['signal_branch']}|{ev['kind']}|{ev['scope']}|{metrics['signal_depth']}"
                    groups.setdefault(key, []).append(metrics)
                    entry = calibration.setdefault(name, dict(samples=[], abstentions=0, series=set()))
                    entry['series'].add(archive['seed'])
                    entry['abstentions'] += metrics['abstained']
                    if metrics['diagnosis_made']:
                        probability = out['terminal']['probability']
                        if probability is None:
                            probability = confidences.get(name)
                        if probability is not None:
                            entry['samples'].append((probability, metrics['diagnosis_correct']))
                for key, rows in groups.items():
                    details.setdefault(key, []).append(sum_rows(rows))
    def cell(rows):
        total, mean, error = {}, {}, {}
        for key in rows[0]:
            values = [r[key] for r in rows]
            total[key] = sum(values)
            mean[key] = math.fsum(values)/len(values)
            error[key] = math.sqrt(math.fsum((v-mean[key])**2 for v in values)/(len(values)*(len(values)-1))) if len(values)>1 else 0.
        return dict(series=len(rows), total=total, mean=mean, standard_error=error)
    cal = {}
    for name, entry in calibration.items():
        samples, bins = entry['samples'], []
        for i in range(10):
            selected = [(p, y) for p, y in samples if min(9, int(p*10)) == i]
            bins.append(dict(lower=i/10, upper=(i+1)/10, count=len(selected),
                             confidence=sum(p for p, y in selected)/len(selected) if selected else None,
                             accuracy=sum(y for p, y in selected)/len(selected) if selected else None))
        cal[name] = dict(decisions=len(samples), abstentions=entry['abstentions'], independent_series=len(entry['series']),
                         brier=math.fsum((p-y)**2 for p, y in samples)/len(samples) if samples else None, bins=bins)
    return dict(independent_series=len(archives), primary_cells={k: cell(v) for k, v in sorted(cells.items())},
                detail_cells={k: cell(v) for k, v in sorted(details.items())}, calibration=cal)


def verify_identity(manifest):
    """Check complete current and historical source/protocol bytes and ancestry."""
    from binding import run_b006 as runner
    runner.verify_runtime_manifest(manifest, override=True)
    require(runner.source_compatible(manifest['source_sha256'], runner.source_metadata()['source_sha256']), 'complete source inventory')
    head, protocol_commit = manifest['git_head'], manifest['protocol_commit']
    runner.git('merge-base', '--is-ancestor', protocol_commit, head)
    import subprocess
    def committed(commit, path):
        return subprocess.check_output(['git', 'show', commit+':'+path], cwd=runner.ROOT)
    for path, expected in manifest['source_sha256'].items():
        require(hashlib.sha256(committed(head, path)).hexdigest() == expected, 'committed source identity '+path)
    require(hashlib.sha256(committed(head, runner.PROTOCOL)).hexdigest() == manifest['protocol_sha256'], 'committed protocol identity')
    require(committed(head, runner.PROTOCOL) == committed(protocol_commit, runner.PROTOCOL), 'protocol registration identity')


def verified_freeze(manifest):
    from binding import run_b006 as runner
    import subprocess
    commit = manifest['freeze_commit']
    runner.git('merge-base', '--is-ancestor', commit, manifest['git_head'])
    matches = []
    for path in runner.git('ls-tree', '-r', '--name-only', commit).splitlines():
        if path.endswith('.json') and 'freeze' in path.lower():
            raw = subprocess.check_output(['git', 'show', commit+':'+path], cwd=runner.ROOT)
            if hashlib.sha256(raw).hexdigest() == manifest['freeze_sha256']:
                matches.append((path, raw))
    require(len(matches) == 1, 'unique freeze identity')
    path, raw = matches[0]
    require(subprocess.check_output(['git', 'show', manifest['git_head']+':'+path], cwd=runner.ROOT) == raw, 'unchanged freeze')
    freeze = json.loads(raw)
    require(freeze['gate']['pass_gate'] is True, 'successful qualification')
    require(freeze['gate']['selected_P'] in CONFIGS and freeze['gate']['selected_K'] in CONSENSUS, 'selected configuration')
    development = freeze['development_manifest']
    require(development['phase'] == 'development', 'freeze development origin')
    runner.git('merge-base', '--is-ancestor', development['git_head'], commit)
    for key in ('source_sha256', 'protocol_sha256', 'protocol_commit'):
        require(development[key] == manifest[key], 'freeze '+key)
    return freeze['gate']


def audit(directory):
    from binding import run_b006 as runner
    directory = Path(directory).resolve()
    manifest = runner.read(directory/'manifest.json')
    verify_identity(manifest)
    require(manifest['phase'] in ('development', 'confirmation'), 'phase identity')
    selection = verified_freeze(manifest) if manifest['phase'] == 'confirmation' else None
    seeds = tuple(range(20270501, 20270513)) if selection is None else tuple(range(20270601, 20270613))
    require(manifest['seeds'] == list(seeds), 'seed cohort')
    require(manifest['environments'] == list(laboratory.ENVIRONMENTS), 'environment cohort')
    require(manifest['variants'] == expected_variants(selection), 'variant cohort')
    require(set(manifest['files']) == {f'series-{seed}.json.gz' for seed in seeds}, 'archive inventory')
    count = 0
    intake_diagnostics = {}
    for seed in seeds:
        name = f'series-{seed}.json.gz'
        runner.verify_file(directory/name, manifest['files'][name])
        archive = runner.read(directory/name)
        require(archive['seed'] == seed, 'archive seed')
        require(manifest['intakes'][str(seed)] == archive['intake_sha256'], 'manifest intake')
        intake_diagnostics[str(seed)] = compare_regenerated(archive['inputs'], physics.intake(seed))
        count += audit_archive(archive, selection, manifest['noise'])
    archives = runner.ArchiveCollection(directory, [f'series-{seed}.json.gz' for seed in seeds])
    confidences = selection['fixed_confidences'] if selection else confidence_estimates(archives)
    summary = reconstructed_summary(archives, confidences)
    if selection:
        # All primary cells are independently reconstructed; closure has no policy execution.
        summary['closure'] = runner.closure(summary, selection)
    compare_structure(runner.read(directory/'summary.json'), summary, 'summary', True)
    if selection is None:
        require(runner.read(directory/'gate.json') == runner.development_gate(archives), 'development gate accounting')
    return dict(passed=True, series=len(seeds), conditions=count, regenerated_input_diagnostics=intake_diagnostics,
                auditor_sha256=runner.sha_file(Path(__file__)), audit_git_head=runner.git('rev-parse', 'HEAD'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(canonical(audit(args.directory)).decode())


if __name__ == '__main__':
    main()
