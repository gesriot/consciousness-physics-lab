"""Post-result A002 oracle and descriptive loss analysis, not confirmation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

FRACTIONS = ('0.2', '0.4', '0.6', '1.0')


def read(path):
    path = Path(path)
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as stream:
        return json.load(stream)


def oracle_input(info):
    """Least feasible certificate cost; no unobserved-policy implementation."""
    rows = [a for a in info['full']['actions'] if a['action'] != 'declare']
    by_action = {a['action']: a for a in rows}
    if info['accurate']:
        if info['full']['verdict'] != 'accept':
            return dict(cost=None, actions=[], reason='accurate_but_full_not_accepting')
        return dict(cost=sum(a['cost'] for a in rows), actions=[a['action'] for a in rows], reason='full_acceptance')
    certificates = []
    for row in rows:
        if row['observation']['verdict'] != 'violation_detected':
            continue
        action = row['action']
        path = [action]
        if action == 'collocation':
            path = ['modal', action]
        elif action.startswith('v2_'):
            grid = int(action[3:])
            path = ['modal'] + [a['action'] for a in rows if a['action'].startswith('v2_') and int(a['action'][3:]) <= grid]
        certificates.append((sum(by_action[a]['cost'] for a in path), tuple(path)))
    if not certificates:
        return dict(cost=None, actions=[], reason='inaccurate_without_detector')
    price, path = min(certificates)
    return dict(cost=price, actions=list(path), reason='violation_certificate')


def analyze(data):
    packages = []
    for batch in data['packages']:
        certificates = [oracle_input(i) for i in batch['inputs']]
        ordered = sorted((c['cost'], i) for i, c in enumerate(certificates) if c['cost'] is not None)
        budgets = {}
        for f in FRACTIONS:
            budget = float(f) * batch['reference_cost']
            spent, indices = 0., []
            for price, index in ordered:
                if spent + price > budget:
                    break
                spent += price
                indices.append(index)
            policies = {}
            for policy in ('B', 'C2', 'C4'):
                run = batch['policies'][f][policy]
                groups = {}
                for info, result, certificate in zip(batch['inputs'], run['inputs'], certificates):
                    label = ('correct_acceptance' if result['correct'] and result['verdict'] == 'accept' else
                             'correct_detection' if result['correct'] else
                             'unreachable_correct_verdict' if certificate['cost'] is None else 'reachable_unfinished')
                    group = groups.setdefault(label, dict(count=0, spent=0., oracle_cost=0.))
                    group['count'] += 1
                    group['spent'] += result['spent']
                    if result['correct']:
                        group['oracle_cost'] += certificate['cost']
                policies[policy] = dict(correct=run['correct'], accepts=run['accepts'],
                    correct_detections=sum(x['correct'] and x['verdict'] == 'violation_detected' for x in run['inputs']),
                    spent=run['spent'], groups=groups,
                    cohort_spending={c: sum(r['spent'] for i, r in zip(batch['inputs'], run['inputs']) if i['cohort']==c) for c in ('W','H','O','S','M')})
            budgets[f] = dict(budget=budget, oracle_correct=len(indices), oracle_spent=spent, selected_input_indices=indices, policies=policies)
        packages.append(dict(package_id=batch['package_id'], certificates=certificates, budgets=budgets))
    summary = {}
    for f in FRACTIONS:
        oracle = sum(b['budgets'][f]['oracle_correct'] for b in packages)
        policies = {}
        for policy in ('B','C2','C4'):
            runs = [b['budgets'][f]['policies'][policy] for b in packages]
            groups = {}
            for run in runs:
                for name, group in run['groups'].items():
                    total = groups.setdefault(name, dict(count=0,spent=0.,oracle_cost=0.))
                    for key in total:
                        total[key] += group[key]
            correct = sum(r['correct'] for r in runs)
            policies[policy] = dict(correct=correct, accepts=sum(r['accepts'] for r in runs),
                correct_detections=sum(r['correct_detections'] for r in runs),
                oracle_fraction=correct/oracle, headroom=oracle-correct,
                spent=sum(r['spent'] for r in runs), groups=groups,
                cohort_spending={c:sum(r['cohort_spending'][c] for r in runs) for c in ('W','H','O','S','M')})
        summary[f] = dict(oracle_correct=oracle,
            oracle_by_package=[b['budgets'][f]['oracle_correct'] for b in packages], policies=policies,
            B_minus_C2_acceptance=policies['B']['accepts']-policies['C2']['accepts'],
            B_minus_C2_detection=policies['B']['correct_detections']-policies['C2']['correct_detections'])
    full = {}
    for b in data['packages']:
        for row in b['inputs']:
            key = row['full']['verdict'] + ('_accurate' if row['accurate'] else '_inaccurate')
            full[key] = full.get(key, 0)+1
    return dict(status='post-result development analysis; no independent confirmation',
        oracle_definition='Each package independently: sort minimum feasible correct-certificate costs and take affordable prefix. Unit reward, additive nonnegative charges, no shared computation.',
        attribution_limit='The archive identifies spending and outcome differences, not a causal percentage due to absent updating versus full-path allocation. These mechanisms interact and no updated policy was defined in A002.',
        full_outcomes=full, summary=summary, packages=packages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    result = analyze(read(args.archive))
    result['source_sha256'] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(full_outcomes=result['full_outcomes'], summary=result['summary']), indent=2))
