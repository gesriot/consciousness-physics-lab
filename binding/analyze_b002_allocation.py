"""Read-only B002 archive analysis; truth labels never enter allocation scores."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from attention import agent, channel_monitor as monitor, s3
from attention.a007 import decode
from attention.build_a007_library import CHANNELS, channel
from attention.calibration_samples import maximum_cost
from binding import b002
from binding.audit import read
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'binding/results/B002'
DATASETS = ('exposure-first', 'exposure-repeat', 'exposure-noise', 'confirmation-first', 'confirmation-repeat', 'confirmation-noise')

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def scores(requests, evidence, closed, weights, pools, last, ordinary, suspect, price):
    raw = dict.fromkeys(CHANNELS, 0.)
    for i, (request, observations) in enumerate(zip(requests, evidence)):
        if i in closed:
            continue
        for family in monitor.FAMILIES:
            c = channel(family, request)
            raw[c] += price(c, request, weights) * (1 + len(observations) / len(agent.ORDER))
    result = {c: round(raw[c] * (1 + ordinary - last[c]) / max(maximum_cost(s, weights) for s in pools[c]), 9) for c in CHANNELS}
    eligible = [c for c in CHANNELS if c not in suspect] or list(CHANNELS)
    chosen = min(eligible, key=lambda c: (-result[c], CHANNELS.index(c)))
    return chosen, result, raw

def revision_analysis(directory, manifest):
    aggregate = {str(b): {m: Counter() for m in ('G', 'T')} for b in (2, 8, 32)}
    records_out = []
    for path in sorted(directory.glob('series-*/revision.json.gz')):
        manifest[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        revision = read(path)['permissive']['branches']['external']['cells']
        ledger = read(path.parent / 'ordinary/ledger-4.json.gz')
        truth = ledger['evaluator']['permissive']
        records = {r['id']: r for r in ledger['records']['permissive']}
        for budget in aggregate:
            ranks, attempts = {}, {}
            for memory in ('G', 'T'):
                cell = revision[budget][memory]['cross']
                counter = aggregate[budget][memory]
                for key in ('attempted', 'selected_affected', 'affected', 'residual_false_accepts', 'initial_false', 'unnecessary_correct_recalls'):
                    counter[key] += cell['metrics'][key]
                ranks[memory] = {a['record_id']: i + 1 for i, a in enumerate(cell['attempts'])}
                attempts[memory] = {a['record_id']: a for a in cell['attempts']}
                for a in cell['attempts']:
                    false = not truth[a['record_id']]['accurate']
                    counter['false_attempts' if false else 'correct_attempts'] += 1
                    counter['false_withdrawals' if false else 'correct_withdrawals'] += int(a['withdrawn'])
            for rid in sorted(set(ranks['G']) | set(ranks['T'])):
                row = records[rid]
                records_out.append(dict(series=path.parent.name, budget=int(budget), record_id=rid,
                    false_accept=not truth[rid]['accurate'], affected_channels=truth[rid]['affected_channels'], acceptance_time=row['time'],
                    collocation_links=[e for e in row['links'] if e['channel'].startswith('collocation:')],
                    G_rank=ranks['G'].get(rid), T_rank=ranks['T'].get(rid),
                    G_channel=attempts['G'].get(rid, {}).get('channel'), T_channel=attempts['T'].get(rid, {}).get('channel')))
    for memories in aggregate.values():
        for c in memories.values():
            c['false_attempt_fraction'] = c['false_attempts'] / c['attempted']
            c['selection_recall'] = c['selected_affected'] / c['affected']
    return dict(totals=aggregate, attempted_record_ranks=records_out)

def run(output):
    output.mkdir(parents=True, exist_ok=True)
    weights = s3.load_model()['action_model']['cost_weights']
    pools = monitor.sample_pools(read(ROOT / 'attention/sample_library_a007.json'))
    original_price = monitor.replacement_price
    groups, manifest = {}, {}
    state_count = 0
    with (output / 'states.jsonl.gz').open('wb') as raw_stream, gzip.GzipFile(fileobj=raw_stream, mode='wb', mtime=0, filename='') as stream:
        for dataset in DATASETS:
            for series in sorted((BASE / dataset).glob('series-*')):
                directory = series / 'ordinary' if dataset.startswith('confirmation') else series
                intake_path = directory / 'intake.json'
                requests_all = [decode(r['request']) for r in read(intake_path)['requests']]
                ledger_path = max(directory.glob('ledger-*.json.gz'), key=lambda p: int(p.name.split('-')[1].split('.')[0]))
                ledger = read(ledger_path)
                for p in (intake_path, ledger_path):
                    manifest[p] = hashlib.sha256(p.read_bytes()).hexdigest()
                for fault in sorted(ledger['records']):
                    offset = 0
                    records = ledger['records'][fault]
                    group = groups.setdefault(dataset + '/' + fault, dict(states=0, actual=Counter(), a007=Counter(), collocation_only_a007=Counter(),
                        changed_a007=0, changed_collocation_only=0, standing_denominator=0, standing_collocation_numerator=0,
                        states_with_standing=0, prior_package_standing_total=0, current_package_standing_total=0,
                        standing_fraction_sum=0., actual_score_max_error=0., raw_max_error=0.,
                        collocation_rank_histogram=Counter(), a007_collocation_rank_histogram=Counter()))
                    for path in sorted(directory.glob(f'{fault}-*.json.gz'), key=lambda p: int(p.name.split('-')[-1].split('.')[0])):
                        archive = read(path)
                        manifest[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                        trace = archive['result']
                        requests = [requests_all[i['task_id']] for i in archive['inputs']]
                        evidence = [{} for _ in requests]
                        last, suspect = dict.fromkeys(CHANNELS, 0), set()
                        for position, action in enumerate(trace['actions']):
                            if not action['sample']:
                                if not action['replacement']:
                                    evidence[action['input_index']][action['action']] = action['observation']
                                continue
                            closed = {i for i, r in enumerate(trace['inputs']) if r['already_closed'] and r['closure_action_index'] <= position}
                            ordinary = action['ordinary_before']
                            actual, sc, raw = scores(requests, evidence, closed, weights, pools, last, ordinary, suspect, b002.native_price)
                            old, old_sc, _ = scores(requests, evidence, closed, weights, pools, last, ordinary, suspect, original_price)
                            def coll_only(c, r, w):
                                return original_price(c, r, w) if c.startswith('collocation:') else b002.native_price(c, r, w)
                            coll, coll_sc, _ = scores(requests, evidence, closed, weights, pools, last, ordinary, suspect, coll_only)
                            assert actual == action['channel'], (path, position, actual, action['channel'])
                            error = max(abs(sc[c] - action['sample_scores'][c]) for c in CHANNELS)
                            raw_error = max(abs(raw[c] - action['score_details']['raw'][c]) for c in CHANNELS)
                            assert error == 0 and raw_error == 0, (path, position, error, raw_error)
                            standing = [r for r in records if r['time'] <= offset + 2 * position]
                            linked = [r['id'] for r in standing if any(e['channel'].startswith('collocation:') for e in r['links'])]
                            def rank(values):
                                ordered = sorted(CHANNELS, key=lambda c: (-values[c], CHANNELS.index(c)))
                                return next(i + 1 for i, c in enumerate(ordered) if c.startswith('collocation:'))
                            group['states'] += 1
                            for key, chosen in (('actual', actual), ('a007', old), ('collocation_only_a007', coll)):
                                group[key][chosen.split(':')[0]] += 1
                            group['changed_a007'] += actual != old
                            group['changed_collocation_only'] += actual != coll
                            group['actual_score_max_error'] = max(group['actual_score_max_error'], error)
                            group['raw_max_error'] = max(group['raw_max_error'], raw_error)
                            prior_count = sum(r['time'] < offset for r in standing)
                            group['prior_package_standing_total'] += prior_count
                            group['current_package_standing_total'] += len(standing) - prior_count
                            group['standing_denominator'] += len(standing)
                            group['standing_collocation_numerator'] += len(linked)
                            if standing:
                                group['states_with_standing'] += 1
                                group['standing_fraction_sum'] += len(linked) / len(standing)
                            group['collocation_rank_histogram'][rank(sc)] += 1
                            group['a007_collocation_rank_histogram'][rank(old_sc)] += 1
                            record = dict(source=str(path.relative_to(ROOT)), action_index=position, ordinary=ordinary,
                                open_inputs=[i for i in range(len(requests)) if i not in closed], evidence_lengths=[len(e) for e in evidence],
                                last_sample_ordinary=last.copy(), suspect=sorted(suspect), actual_choice=actual, a007_choice=old,
                                collocation_only_a007_choice=coll, actual_scores=sc, a007_scores=old_sc,
                                collocation_only_a007_scores=coll_sc, raw_actual=raw, standing_count=len(standing),
                                standing_collocation_count=len(linked), prior_package_standing_count=prior_count,
                                current_package_standing_count=len(standing)-prior_count, standing_collocation_record_ids=linked,
                                best_collocation_rank=rank(sc), a007_best_collocation_rank=rank(old_sc))
                            stream.write((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
                            state_count += 1
                            last[actual] = ordinary
                            if action['observation']['verdict'] != action['expected']:
                                suspect.add(actual)
                        offset += 2 * (len(trace['actions']) + 1)
    for g in groups.values():
        g['standing_collocation_fraction_pooled'] = g['standing_collocation_numerator'] / g['standing_denominator'] if g['standing_denominator'] else None
        g['standing_collocation_fraction_mean_nonempty'] = g['standing_fraction_sum'] / g['states_with_standing'] if g['states_with_standing'] else None
    revisions = {d: revision_analysis(BASE / d, manifest) for d in DATASETS if d.startswith('confirmation')}
    dump(output / 'revision-ranks.json', revisions)
    summary = dict(saved_sample_states=state_count, groups=groups, weights=weights,
        replacement_work=dict(A007_uniform_cells=sum(monitor.REPLACEMENT_GRIDS), B002_graded_cells=sum(b002.P.CONFIG.grids),
                              B002_collocation_degree_cubed=sum(n**3 for n in b002.P.CONFIG.third_degrees)),
        replacement_prices={f: dict(A007=original_price(f + ':0', requests_all[0], weights),
                                   B002=b002.native_price(f + ':0', requests_all[0], weights)) for f in ('v2', 'collocation')},
        sample_maximum_prices={c: max(maximum_cost(s, weights) for s in pools[c]) for c in CHANNELS},
        revision_totals={d: r['totals'] for d, r in revisions.items()},
        scope='All archived paid-sample score states; fixed historical evidence, closures, ages and sample timing. Not a resimulation.',
        truth_usage='Only retrospective attempt and withdrawal analysis; never allocation inputs.')
    dump(output / 'summary.json', summary)
    for relative in ('binding/analyze_b002_allocation.py', 'binding/test_analyze_b002_allocation.py', 'binding/b002.py', 'binding/b001.py',
                     'binding/revision_b002.py', 'binding/qualification.py', 'attention/channel_monitor.py', 'attention/agent.py',
                     'attention/calibration_samples.py', 'attention/build_a007_library.py', 'attention/self_model_s3.json',
                     'attention/sample_library_a007.json', 'lab/frozen_pipeline.py'):
        p = ROOT / relative
        manifest[p] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(output / 'sources.json', [dict(path=str(p.relative_to(ROOT)), sha256=h) for p, h in sorted(manifest.items())])
    checks = {}
    for phase in ('exposure', 'confirmation'):
        for suffix in ('repeat', 'noise'):
            for fault in ('none', 'permissive', 'underestimated') if phase == 'exposure' else ('none', 'permissive'):
                checks[f'{phase}-{suffix}/{fault}'] = groups[f'{phase}-first/{fault}'] == groups[f'{phase}-{suffix}/{fault}']
    for d in ('confirmation-repeat', 'confirmation-noise'):
        checks[d + '/revision_totals'] = revisions[d]['totals'] == revisions['confirmation-first']['totals']
        checks[d + '/revision_record_ranks'] = revisions[d]['attempted_record_ranks'] == revisions['confirmation-first']['attempted_record_ranks']
    g = groups['confirmation-first/permissive']
    known = dict(states=480, v2=291, collocation=0, a007_collocation=115, collocation_only=151,
                 standing=11034, linked=11034, nonempty=416, total_states=20160)
    actual = dict(states=g['states'], v2=g['actual']['v2'], collocation=g['actual'].get('collocation', 0),
                  a007_collocation=g['a007']['collocation'], collocation_only=g['collocation_only_a007']['collocation'],
                  standing=g['standing_denominator'], linked=g['standing_collocation_numerator'], nonempty=g['states_with_standing'], total_states=state_count)
    checks['recovered_matches_pre_loss_observations'] = actual == known
    checks['all_saved_scores_exact'] = all(g['actual_score_max_error'] == g['raw_max_error'] == 0 for g in groups.values())
    verification = dict(checks=checks, passed=all(checks.values()), recovered_actual=actual, pre_loss_observations=known,
        recovery='Workspace disappeared after initial completed analysis. Source was reconstructed from retained conversation; only archival calculations rerun. No solver or experimental trajectory rerun. Byte identity to lost artifacts cannot be claimed.')
    dump(output / 'verification.json', verification)
    assert verification['passed'], verification
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'allocation-analysis')
    run(parser.parse_args().output)
