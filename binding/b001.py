"""Registered independent-series provenance recall experiment and replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from attention import agent, s2, s3
from attention.a001 import reference_audit
from attention.a007 import atomic_json, decode
from attention.build_a007_library import CHANNELS, channel
from attention.channel_monitor import run_batch, guarded_uniform, replacement_price
from attention.io import read_json
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES
from lab.replay import canonical_sha256
from lab.scientist import ROOT
from binding.memory import MEMORIES, BUDGETS, recall

PROTOCOL = 'binding/protocols/B001-recall.md'
FAULTS = ('none', 'permissive', 'underestimated')


def intake(seed):
    rng, order, samples = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    rows = []
    for i in range(20):
        cohort = COHORTS[i % 5]
        lo, hi = LOG_RANGES[cohort]
        log_time = float(rng.uniform(-2, 2))
        u, v = float(rng.random()), float(rng.random())
        log_tol = float(rng.uniform(-6 if cohort == 'S' else -4, -4 if cohort == 'S' else -1))
        angle = ((-np.pi/2+.1*v if v < .5 else np.pi/2-.1*(1-v)) if cohort == 'O' else -np.pi/2+np.pi*v)
        time = 10**log_time
        request = Request(10**(lo+(hi-lo)*u)*np.exp(1j*angle)/time, time, 10**log_tol)
        rows.append(dict(task_id=i, package_id=i//5, cohort=cohort, request=encode(request),
                         request_sha256=request_hash(request), condition_order=[str(c) for c in order.permutation(CONDITIONS)]))
    return dict(seed=seed, requests=rows, sample_seeds=[int(samples.integers(0, 2**31)) for _ in range(4)])


def validate_intake(data):
    if len(data['requests']) != 20 or len(data['sample_seeds']) != 4:
        raise ValueError('Series dimensions')
    for i, row in enumerate(data['requests']):
        if (row['task_id'], row['package_id'], row['cohort']) != (i, i//5, COHORTS[i % 5]):
            raise ValueError('Series order')
        r = decode(row['request'])
        if request_hash(r) != row['request_sha256'] or sorted(row['condition_order']) != sorted(CONDITIONS):
            raise ValueError('Replay intake mismatch')


def append_ledger(trace, offset, last, records, evaluator, infos, package_id):
    """Derive only executed support, replacing dependencies rescued upstream."""
    snapshots = {}
    by_close = {}
    for i, row in enumerate(trace['inputs']):
        if row['verdict'] == 'accept':
            by_close.setdefault(row['closure_action_index'], []).append(i)
    notices = []
    for position in range(len(trace['actions'])+1):
        # Closure precedes action at the same position. Half-ticks distinguish it.
        for i in by_close.get(position, []):
            row = trace['inputs'][i]
            support = {}
            for action_pos, action in enumerate(trace['actions'][:position]):
                if action['sample'] or action['input_index'] != i:
                    continue
                obs = action['observation']
                if obs['verdict'] != 'accept':
                    continue
                ch = action['channel']
                if action['replacement']:
                    support[ch] = dict(channel='independent:'+ch, time=offset+2*action_pos+1,
                                       source_channel=ch, replaced=True, fault_applied=False)
                elif not support.get(ch, {}).get('replaced'):
                    support[ch] = dict(channel=ch, time=offset+2*action_pos+1,
                                       replaced=False, fault_applied=action['fault_applied'])
            rid = f'{package_id}:{i}'
            links = []
            affected = []
            for index, edge in enumerate(support.values()):
                if edge['fault_applied']:
                    affected.append(edge['channel'])
                links.append(dict(edge_id=f'{rid}:{index}', channel=edge['channel'],
                                  region=edge['channel'].split(':')[-1], time=edge['time'],
                                  last_success=last.get(edge['channel'], -1)))
            records.append(dict(id=rid, time=offset+2*position, links=links,
                                last_good_global=min(last.values()), last_success=copy.deepcopy(last)))
            evaluator[rid] = dict(accurate=infos[i]['accurate'], affected_channels=affected,
                                 package_id=package_id, input_index=i)
        if position == len(trace['actions']):
            break
        action = trace['actions'][position]
        if action['sample']:
            ch = action['channel']
            if action['observation']['verdict'] == action['expected']:
                last[ch] = offset+2*position+1
            elif ch.startswith(('v2:', 'collocation:')) and ch not in snapshots:
                snapshots[ch] = True
                notices.append(dict(channel=ch, time=offset+2*position+1,
                                    last_success=last[ch], last_good_global=min(last.values()), source='BU'))
    return notices


def outcomes(result, records, evaluator, notices):
    accepted = {r['id'] for r in records}
    withdrawn = set(result['withdrawn'])
    affected = {r['id'] for r in records if any(r['time'] < n['time'] and n['channel'] in evaluator[r['id']]['affected_channels'] for n in notices)}
    hits = affected & withdrawn
    return dict(residual_false_accepts=sum(not evaluator[r]['accurate'] for r in accepted-withdrawn),
                unnecessary_correct_recalls=sum(evaluator[r]['accurate'] for r in withdrawn),
                recall_cost=result['cost'], attempted=len(result['attempts']), withdrawn=len(withdrawn),
                affected=len(affected), affected_false=sum(not evaluator[r]['accurate'] for r in affected),
                true_withdrawals=len(hits), precision=len(hits)/len(withdrawn) if withdrawn else None,
                recall=len(hits)/len(affected) if affected else None,
                selected_affected=len(affected & set(result['selected'])),
                unwithdrawn_affected=len(affected-withdrawn),
                service_delays=[a['service_delay'] for a in result['attempts'] if a['withdrawn']],
                accepted=len(accepted), initial_false=sum(not evaluator[r]['accurate'] for r in accepted))


def series(data, bands, directory, noise=False):
    validate_intake(data)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'intake.json', data)
    model, static = s3.load_model(), s2.load_model()
    library = read_json(ROOT/'attention/sample_library_a007.json')
    weights = model['action_model']['cost_weights']
    requests = [decode(r['request']) for r in data['requests']]
    references, audit = reference_audit(requests)
    audit['all_tasks_audited'] = len(requests)
    atomic_json(directory/'reference.json', audit)
    previous_request = Request(1, 1, .01)
    previous = execute(previous_request, P.plan_modes(previous_request)).value
    records = {f: [] for f in FAULTS}
    evaluators = {f: {} for f in FAULTS}
    notices = {f: [] for f in FAULTS}
    lasts = {f: {c: -1 for c in CHANNELS} for f in FAULTS}
    offsets = {f: 0 for f in FAULTS}
    executors_by_id = {}
    package_hashes = {f: [] for f in FAULTS}
    for package_id in range(4):
        specs, infos = [], []
        for index in range(package_id*5, (package_id+1)*5):
            r, row = requests[index], data['requests'][index]
            for condition in row['condition_order']:
                e = agent.Executor(r, row['request_sha256'], condition, previous)
                full = agent.run_policy(e, 'C1', float('inf'), static['action_model'], list(agent.ORDER), full=True)
                original = P.evaluate(r, row['request_sha256'], e.outcome, e.model)
                if full['verdict'] != original['policies']['strict']['verdict']:
                    raise AssertionError('Full pipeline parity')
                infos.append(dict(task_id=index, condition=condition, full_cost=full['spent'],
                                  outcome=copy.deepcopy(e.outcome), reference=[references[index].real, references[index].imag],
                                  absolute_error=abs(complex(*e.outcome['value'])-references[index]), tolerance=r.tolerance,
                                  accurate=bool(abs(complex(*e.outcome['value'])-references[index]) <= r.tolerance)))
                specs.append((r, row['request_sha256'], condition, previous))
            previous = execute(r, P.plan_modes(r)).value
        cost = sum(i['full_cost'] for i in infos)
        for i, spec in enumerate(specs):
            executors_by_id[f'{package_id}:{i}'] = agent.Executor(*spec)
            executors_by_id[f'{package_id}:{i}'].outcome = copy.deepcopy(infos[i]['outcome'])
        for fault in FAULTS:
            target = ('v2:' if fault == 'underestimated' else 'collocation:')+str(bands['underestimated' if fault == 'underestimated' else 'permissive'])
            spec = dict(kind=fault, target_channel=target, onset_action=0 if package_id else 10**9,
                        circle_phase=0, random_seed=data['sample_seeds'][package_id])
            result = run_batch([agent.Executor(*s) for s in specs], 'BU', cost, model, library,
                               package_id, spec, 1., noise)
            if result['spent'] > cost+1e-8:
                raise AssertionError('Package budget')
            ns = append_ledger(result, offsets[fault], lasts[fault], records[fault], evaluators[fault], infos, package_id)
            seen = {n['channel'] for n in notices[fault]}
            notices[fault].extend(n for n in ns if n['channel'] not in seen)
            offsets[fault] += 2*(len(result['actions'])+1)
            archive = dict(package_id=package_id, inputs=infos, fault_spec=spec, result=result)
            atomic_json(directory/f'{fault}-{package_id}.json.gz', archive)
            package_hashes[fault].append(canonical_sha256(archive))
        print('Saved', directory.name, 'package', package_id+1, flush=True)
    cached = {}
    def replace(rid):
        if rid not in cached:
            executor = executors_by_id[rid]
            cached[rid] = guarded_uniform(executor)
        return copy.deepcopy(cached[rid])
    price = replacement_price('v2:0', requests[0], weights)
    results, summary = {}, {}
    for scenario in ('none', 'false_notice', 'permissive', 'underestimated'):
        fault = 'none' if scenario == 'false_notice' else scenario
        target = ('v2:' if fault == 'underestimated' else 'collocation:')+str(bands['underestimated' if fault == 'underestimated' else 'permissive'])
        external = [] if scenario == 'none' else [dict(channel=target, time=offsets[fault],
                       last_success=lasts[fault][target], last_good_global=min(lasts[fault].values()), source='external')]
        results[scenario], summary[scenario] = {}, {}
        for branch, ns in (('external', external), ('BU', notices[fault])):
            results[scenario][branch], summary[scenario][branch] = {}, {}
            for budget in BUDGETS:
                key = str(budget)
                results[scenario][branch][key], summary[scenario][branch][key] = {}, {}
                for memory in MEMORIES:
                    result = recall(records[fault], ns, memory, budget, replace, price, noise)
                    result['memory_schema'] = ['id', 'time'] if memory in ('T', 'A', 'N') else ['id', 'time', 'links', 'last_success', 'last_good_global']
                    result['record_count'] = len(records[fault])
                    result['metrics'] = outcomes(result, records[fault], evaluators[fault], ns)
                    results[scenario][branch][key][memory] = result
                    summary[scenario][branch][key][memory] = result['metrics']
    atomic_json(directory/'ledger.json.gz', dict(records=records, evaluator=evaluators, BU_notices=notices,
                                                last_success=lasts, package_hashes=package_hashes))
    atomic_json(directory/'recalls.json.gz', results)
    atomic_json(directory/'summary.json', summary)
    return summary


def aggregate(rows):
    out = {}
    for scenario in rows[0]:
        out[scenario] = {}
        for branch in rows[0][scenario]:
            out[scenario][branch] = {}
            for budget in rows[0][scenario][branch]:
                out[scenario][branch][budget] = {}
                for memory in MEMORIES:
                    values = [r[scenario][branch][budget][memory] for r in rows]
                    metrics = {}
                    for key in ('residual_false_accepts', 'unnecessary_correct_recalls', 'recall_cost', 'attempted',
                                'withdrawn', 'affected', 'affected_false', 'true_withdrawals', 'initial_false', 'accepted'):
                        v = np.array([x[key] for x in values], dtype=float)
                        metrics[key] = dict(total=float(sum(v)), mean=float(v.mean()),
                                            standard_error=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None,
                                            by_series=v.tolist())
                    out[scenario][branch][budget][memory] = metrics
                for other in ('T', 'A', 'N', 'Gx', 'Gxprime'):
                    out[scenario][branch][budget][other]['paired_minus_G'] = {}
                    for key in ('residual_false_accepts', 'unnecessary_correct_recalls', 'recall_cost'):
                        delta = np.array([r[scenario][branch][budget][other][key]-r[scenario][branch][budget]['G'][key] for r in rows])
                        out[scenario][branch][budget][other]['paired_minus_G'][key] = dict(mean=float(delta.mean()),
                            standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                            by_series=delta.tolist())
    return out


def freeze(rows):
    candidates = {}
    for fault in ('permissive', 'underestimated'):
        scores = []
        for band in range(6):
            values = [rows[i][fault]['external']['32']['N'] for i in range(12) if i % 6 == band]
            scores.append(dict(band=band, affected=sum(v['affected'] for v in values),
                               affected_false=sum(v['affected_false'] for v in values)))
        chosen = min(scores, key=lambda x: (-x['affected_false'], -x['affected'], x['band']))
        candidates[fault] = dict(candidates=scores, selected=chosen['band'],
                                 exposure_gate=chosen['affected'] >= 10 and chosen['affected_false'] >= 5)
    return dict(experiment='B001', development_seeds=list(range(20261206, 20261218)),
                candidates=candidates, bands={f: x['selected'] for f, x in candidates.items()},
                all_exposure_gates_pass=all(x['exposure_gate'] for x in candidates.values()))


def run(output, phase, freeze_path=None, replay=None, noise=False, override=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    sources = sorted(str(p.relative_to(ROOT)) for base in ('binding', 'attention', 'lab')
                     for p in (ROOT/base).glob('*.py'))+['attention/sample_library_a007.json', 'attention/self_model_s2.json', 'attention/self_model_s3.json', 'attention/self_model.json']
    audit = metadata(PROTOCOL, sources, override)
    if phase == 'confirmation' and freeze_path is None:
        raise ValueError('Confirmation requires committed freeze')
    frozen = read_json(freeze_path) if freeze_path else None
    if freeze_path:
        import subprocess
        relative = str(Path(freeze_path).resolve().relative_to(ROOT))
        subprocess.check_call(['git', 'ls-files', '--error-unmatch', relative], cwd=ROOT, stdout=subprocess.DEVNULL)
        audit['freeze_sha256'] = hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest()
    if replay:
        saved = read_json(Path(replay)/'manifest.json')
        if saved['phase'] != phase or saved['freeze_sha256'] != audit.get('freeze_sha256'):
            raise ValueError('Replay phase/freeze mismatch')
        datasets = [read_json(Path(replay)/f'series-{i:02d}'/'intake.json') for i in range(12)]
        if canonical_sha256(datasets) != saved['intake_sha256']:
            raise ValueError('Replay intake digest')
    else:
        start = 20261206 if phase == 'development' else 20261218
        datasets = [intake(start+i) for i in range(12)]
    output.mkdir(parents=True)
    manifest = dict(experiment='B001', phase=phase, **{**audit, 'freeze_sha256': audit.get('freeze_sha256')},
                    last_bit_noise=noise, intake_sha256=canonical_sha256(datasets),
                    sampling_unit='12 independent series; packages, conditions, memories and budgets paired within series')
    atomic_json(output/'manifest.json', manifest)
    rows = []
    for i, data in enumerate(datasets):
        bands = frozen['bands'] if frozen else dict(permissive=i % 6, underestimated=i % 6)
        rows.append(series(data, bands, output/f'series-{i:02d}', noise))
        atomic_json(output/'progress.json', dict(completed_series=len(rows), summary=aggregate(rows)))
    summary = aggregate(rows)
    atomic_json(output/'summary.json', summary)
    if phase == 'development':
        atomic_json(output/'freeze.json', freeze(rows))
    else:
        cells = []
        for fault in ('permissive', 'underestimated'):
            for budget in map(str, BUDGETS):
                s = summary[fault]['external'][budget]
                no_worse = all(s['T'][k]['total'] <= s['G'][k]['total'] for k in ('residual_false_accepts', 'unnecessary_correct_recalls'))
                cells.append(dict(fault=fault, budget=budget, T_no_worse=no_worse))
        atomic_json(output/'closure.json', dict(cells=cells, links_unnecessary_here=all(c['T_no_worse'] for c in cells),
                                               exposure_gates=frozen['candidates']))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'confirmation'), required=True)
    parser.add_argument('--freeze')
    parser.add_argument('--replay')
    parser.add_argument('--last-bit-noise', action='store_true')
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    run(args.output, args.phase, args.freeze, args.replay, args.last_bit_noise, args.runtime_override)
