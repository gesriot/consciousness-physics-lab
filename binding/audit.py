"""Independent, bounded-memory verification of B001 archived evidence.

Does not import the runner or its memory, selection, or outcome implementation.
Numerical reference values and instrument observations are archived evidence, not
new independent physical measurements. Corrupted projections are reconstructed
because the archive contains edit logs rather than complete corrupted records.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from attention.build_a007_library import CHANNELS
from lab import frozen_pipeline as P
from attention.audit_a006_portability import diagnostic_comparison
from lab.pipeline_holdout import CONDITIONS, COHORTS

MEMORIES = ('G', 'T', 'A', 'N', 'Gx', 'Gxprime')
FAULTS = ('none', 'permissive', 'underestimated')
LIMITATIONS = [
    'Reference accuracy uses archived analytic/high-precision reference values; no new physical reference computation.',
    'Upstream verdicts and executed observations are evidence; scheduler optimality and numerical instrument execution are not rerun.',
    'Development archives omit visible snapshots; confirmation snapshots are required and checked against reconstructed projections.',
]


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as stream:
        return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class AuditError(ValueError):
    pass


def equal(actual, expected, label):
    if actual != expected:
        raise AuditError(label)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
        raise AuditError(label)


def intake_check(data):
    equal(len(data['requests']), 20, 'intake request count')
    equal(len(data['sample_seeds']), 4, 'intake sample count')
    for i, row in enumerate(data['requests']):
        equal((row['task_id'], row['package_id'], row['cohort']), (i, i//5, COHORTS[i % 5]), 'intake order')
        equal(sorted(row['condition_order']), sorted(CONDITIONS), 'condition count/order')
        h = hashlib.sha256(json.dumps(row['request'], sort_keys=True, allow_nan=False).encode()).hexdigest()
        equal(h, row['request_sha256'], 'request digest')


def reconstruct(packages, intake, reference):
    """Rebuild closure support, calibration chronology and evaluator from actions."""
    equal(reference['all_tasks_audited'], 20, 'reference count')
    equal(len(reference['records']), 20, 'reference rows')
    for i, ref in enumerate(reference['records']):
        equal(ref['task_id'], i, 'reference task')
        precise = complex(*map(float, ref['high_precision']))
        error = abs(complex(*ref['analytic'])-precise)
        close(ref['absolute_difference'], error, 'reference error')
        equal(ref['passed'], error <= 1e-13+1e-12*abs(precise), 'reference gate')
    equal(reference['fallback_all'], not all(x['passed'] for x in reference['records']), 'reference fallback')
    records, evaluator, notices, info_by_id = [], {}, [], {}
    last = dict.fromkeys(CHANNELS, -1)
    offset = 0
    hashes = []
    for package_id, archive in enumerate(packages):
        hashes.append(digest(archive))
        equal(archive['package_id'], package_id, 'package index')
        infos, trace = archive['inputs'], archive['result']
        equal(len(infos), 45, 'package input count')
        equal(len(trace['inputs']), 45, 'trace input count')
        expected = [(r['task_id'], c) for r in intake['requests'][package_id*5:package_id*5+5] for c in r['condition_order']]
        equal([(i['task_id'], i['condition']) for i in infos], expected, 'package condition sequence')
        close(trace['budget'], sum(i['full_cost'] for i in infos), 'full clean budget')
        close(trace['spent'], sum(a['cost'] for a in trace['actions']), 'upstream spent')
        if trace['spent'] > trace['budget']+1e-8:
            raise AuditError('upstream budget exceeded')
        for name, predicate in [('sample', lambda a:a['sample']), ('replacement', lambda a:a['replacement']), ('ordinary', lambda a:not a['sample'])]:
            close(trace[name+'_cost'], sum(a['cost'] for a in trace['actions'] if predicate(a)), name+' cost')
        for i, info in enumerate(infos):
            ref = reference['records'][info['task_id']]
            value = list(map(float, ref['high_precision'])) if reference['fallback_all'] else ref['analytic']
            equal(info['reference'], value, 'input reference')
            tol = intake['requests'][info['task_id']]['request']['tolerance']
            equal(info['tolerance'], tol, 'input tolerance')
            error = abs(complex(*info['outcome']['value'])-complex(*value))
            close(info['absolute_error'], error, 'input error')
            equal(info['accurate'], error <= tol, 'input accurate')
            info_by_id[f'{package_id}:{i}'] = info
        closes = {}
        for i, row in enumerate(trace['inputs']):
            equal(row['input_index'], i, 'input index')
            pos = row['closure_action_index']
            if not 0 <= pos <= len(trace['actions']):
                raise AuditError('closure bounds')
            if row['verdict'] == 'accept':
                closes.setdefault(pos, []).append(i)
        spec = archive['fault_spec']
        equal(spec['onset_action'], 10**9 if package_id == 0 else 0, 'registered onset')
        equal(spec['random_seed'], intake['sample_seeds'][package_id], 'sample seed')
        support = {i: {} for i in range(45)}
        ordinary_count = 0
        for pos in range(len(trace['actions'])+1):
            for i in closes.get(pos, []):
                rid = f'{package_id}:{i}'
                edges = []
                affected = []
                for j, (ch, (when, replaced, fault)) in enumerate(support[i].items()):
                    name = 'independent:'+ch if replaced else ch
                    edges.append(dict(edge_id=f'{rid}:{j}', channel=name, region=ch.split(':')[-1], time=when, last_success=last.get(name, -1)))
                    if fault and not replaced:
                        affected.append(name)
                records.append(dict(id=rid, time=offset+2*pos, links=edges, last_good_global=min(last.values()), last_success=last.copy()))
                evaluator[rid] = dict(accurate=info_by_id[rid]['accurate'], affected_channels=affected, package_id=package_id, input_index=i)
            if pos == len(trace['actions']):
                break
            action = trace['actions'][pos]
            ch, when = action['channel'], offset+2*pos+1
            if action['sample']:
                if action['observation']['verdict'] == action['expected']:
                    last[ch] = when
                elif ch.startswith(('v2:', 'collocation:')) and not any(n['channel'] == ch for n in notices):
                    notices.append(dict(channel=ch, time=when, last_success=last[ch], last_good_global=min(last.values()), source='BU'))
            else:
                fault_exposed = bool(spec['kind'] != 'none' and ordinary_count >= spec['onset_action'] and ch == spec['target_channel'] and not action['replacement'])
                equal(action['fault_applied'], fault_exposed, 'actual fault exposure')
                if action['observation']['verdict'] == 'accept':
                    deps = support[action['input_index']]
                    if action['replacement'] or not deps.get(ch, (None, False, None))[1]:
                        deps[ch] = (when, action['replacement'], fault_exposed)
                if not action['replacement']:
                    ordinary_count += 1
        offset += 2*(len(trace['actions'])+1)
    return records, evaluator, notices, last, offset, hashes, info_by_id


def projection(records, notice, memory, reviewed):
    """Produce policy-visible records and independent ranked IDs from provenance."""
    visible = copy.deepcopy([r for r in records if r['time'] < notice['time'] and r['id'] not in reviewed])
    edits = []
    for record in visible:
        eligible = [e for e in record['links'] if e['channel'] == notice['channel'] and e['time'] > notice['last_success']]
        if memory in ('Gx', 'Gxprime') and eligible:
            old = copy.deepcopy(record)
            if memory == 'Gx':
                edge = eligible[-1]
                family, band = notice['channel'].split(':')
                after = family+':'+str(int(band)+(1 if int(band)<5 else -1))
            else:
                edge = next(e for e in record['links'] if e['channel'].removeprefix('independent:') == 'normalization')
                after = ('independent:' if edge['channel'].startswith('independent:') else '')+'first_moment'
            before = edge['channel']
            edge['channel'] = after
            edits.append(dict(record_id=record['id'], edge_id=edge['edge_id'], before=before, after=after, l1=abs(CHANNELS.index(before.removeprefix('independent:'))-CHANNELS.index(after.removeprefix('independent:')))))
            equal(len(record['links']), len(old['links']), 'corruption degree')
            equal([e['time'] for e in record['links']], [e['time'] for e in old['links']], 'corruption time')
            equal(sum(a != b for a, b in zip(record['links'], old['links'])), 1, 'corruption one edge')
    # T/A/N projection carries only identity and closure, never evaluator fields.
    if memory in ('T', 'A', 'N'):
        visible = [dict(id=r['id'], time=r['time']) for r in records if r['time'] < notice['time'] and r['id'] not in reviewed]
    ranked = []
    for row in visible:
        times = [e['time'] for e in row.get('links', []) if e['channel'] == notice['channel'] and e['time'] > notice['last_success']]
        if memory == 'N' or memory == 'T' and row['time'] <= notice['last_success'] or memory in ('G', 'Gx', 'Gxprime') and not times:
            continue
        score = max(times) if memory in ('G', 'Gx', 'Gxprime') else row['time']
        ranked.append((-round(float(score), 9), -row['time'], row['id']))
    return [x[2] for x in sorted(ranked)], edits, visible


def guard_check(result, info):
    equal(result['grids'], [4096, 8192, 16384, 32768], 'replacement grids')
    values = [complex(*v) for v in result['values']]
    step = P.v2_step(values, 32768, complex(*info['outcome']['value']), info['tolerance'])
    comparison = diagnostic_comparison(result['step'], step, 'v2')
    equal(comparison['passed'], True, 'replacement step: '+str(comparison['failures']))
    verdict = step['verdict']
    margin = info['tolerance']-step['discrepancy']-step['uncertainty_estimate'] if verdict == 'accept' else step['discrepancy']-step['uncertainty_estimate']-info['tolerance'] if verdict == 'violation_detected' else None
    contraction = .5*step['previous_difference']-step['last_difference']
    threshold = 4*step['rounding']
    passed = margin is not None and margin > threshold and contraction > threshold
    equal(result['unguarded_verdict'], verdict, 'unguarded verdict')
    safety = dict(multiplier=4, decision_margin=margin, contraction_margin=contraction, threshold=threshold, passed=passed)
    comparison = diagnostic_comparison(result['safety'], safety, 'v2')
    equal(comparison['passed'], True, 'guard safety: '+str(comparison['failures']))
    equal(result['verdict'], verdict if passed else 'inconclusive', 'guarded verdict')
    return comparison


def recall_check(result, records, evaluator, notices, memory, units, price, infos, require_snapshots=False):
    reviewed, withdrawn, selected = set(), set(), set()
    expected_attempts = []
    equal(len(result['interventions']), len(notices), 'notice/intervention count')
    for notice, intervention in zip(notices, result['interventions']):
        ranked, edits, visible = projection(records, notice, memory, reviewed)
        expected_intervention = dict(notice=notice, edits=edits, changed_edges=len(edits), l1=sum(e['l1'] for e in edits))
        if require_snapshots or 'records' in intervention:
            expected_intervention['records'] = visible
        equal(intervention, expected_intervention, 'corruption/intervention')
        if memory in ('Gx', 'Gxprime'):
            other = 'Gxprime' if memory == 'Gx' else 'Gx'
            _, paired, paired_records = projection(records, notice, other, reviewed)
            equal([e['record_id'] for e in edits], [e['record_id'] for e in paired], 'corruption record parity')
            equal([e['l1'] for e in edits], [1]*len(paired), 'corruption magnitude parity')
            equal([(r['id'], r['time'], len(r['links']), [e['time'] for e in r['links']]) for r in visible], [(r['id'], r['time'], len(r['links']), [e['time'] for e in r['links']]) for r in paired_records], 'corruption degree/time parity')
        selected.update(ranked)
        for delay, rid in enumerate(ranked[:max(0, units-len(reviewed))], 1):
            expected_attempts.append((rid, notice['time'], delay))
            reviewed.add(rid)
    equal(len(result['attempts']), len(expected_attempts), 'recall attempt count')
    for attempt, expected in zip(result['attempts'], expected_attempts):
        equal((attempt['record_id'], attempt['notice_time'], attempt['service_delay']), expected, 'recall ranking/cutoff')
        guard_check(attempt['result'], infos[attempt['record_id']])
        revoke = attempt['result']['verdict'] != 'accept'
        equal(attempt['withdrawn'], revoke, 'recall retained iff accept')
        equal(attempt['cost'], price, 'recall charged cost')
        if revoke:
            withdrawn.add(attempt['record_id'])
    for key, value in [('selected', selected), ('reviewed', reviewed), ('withdrawn', withdrawn), ('unreviewed_selected', selected-reviewed)]:
        equal(result[key], sorted(value), key)
    for key, value in [('cost', len(expected_attempts)*price), ('budget', units*price), ('unused_budget', (units-len(expected_attempts))*price)]:
        equal(result[key], value, 'recall '+key)
    equal(result['record_count'], len(records), 'record count')
    equal(result['memory_schema'], ['id', 'time'] if memory in ('T', 'A', 'N') else ['id', 'time', 'links', 'last_success', 'last_good_global'], 'visible schema')
    accepted = set(evaluator)
    affected = {r['id'] for r in records if any(r['time'] < n['time'] and n['channel'] in evaluator[r['id']]['affected_channels'] for n in notices)}
    hits = affected & withdrawn
    metrics = dict(residual_false_accepts=sum(not evaluator[r]['accurate'] for r in accepted-withdrawn), unnecessary_correct_recalls=sum(evaluator[r]['accurate'] for r in withdrawn), recall_cost=len(expected_attempts)*price, attempted=len(expected_attempts), withdrawn=len(withdrawn), affected=len(affected), affected_false=sum(not evaluator[r]['accurate'] for r in affected), true_withdrawals=len(hits), precision=len(hits)/len(withdrawn) if withdrawn else None, recall=len(hits)/len(affected) if affected else None, selected_affected=len(affected & selected), unwithdrawn_affected=len(affected-withdrawn), service_delays=[a['service_delay'] for a in result['attempts'] if a['withdrawn']], accepted=len(accepted), initial_false=sum(not evaluator[r]['accurate'] for r in accepted))
    equal(result['metrics'], metrics, 'recomputed metrics')
    return metrics


def audit_series(directory, require_snapshots=False, bands=None):
    directory = Path(directory)
    intake = read(directory/'intake.json')
    intake_check(intake)
    reference = read(directory/'reference.json')
    ledger = read(directory/'ledger.json.gz')
    recalls = read(directory/'recalls.json.gz')
    summary = read(directory/'summary.json')
    model = read(Path(__import__('attention').__file__).resolve().parent/'self_model_s3.json')
    price = sum([4096, 8192, 16384, 32768])*model['action_model']['cost_weights']['fv_cell']
    cells = 0
    for fault in FAULTS:
        def package_stream():
            for p in range(4):
                archive = read(directory/f'{fault}-{p}.json.gz')
                equal(archive['fault_spec']['kind'], fault, 'registered fault family')
                if bands is not None:
                    family = 'underestimated' if fault == 'underestimated' else 'permissive'
                    channel = ('v2:' if fault == 'underestimated' else 'collocation:')+str(bands[family])
                    equal(archive['fault_spec']['target_channel'], channel, 'frozen target band')
                yield archive
        packages = package_stream()
        records, evaluator, notices, last, offset, hashes, infos = reconstruct(packages, intake, reference)
        for key, value in [('records', records), ('evaluator', evaluator), ('BU_notices', notices), ('last_success', last), ('package_hashes', hashes)]:
            equal(ledger[key][fault], value, 'ledger '+key)
        first = read(directory/f'{fault}-0.json.gz')
        target = first['fault_spec']['target_channel']
        for scenario in ([fault, 'false_notice'] if fault == 'none' else [fault]):
            external = [] if scenario == 'none' else [dict(channel=target, time=offset, last_success=last[target], last_good_global=min(last.values()), source='external')]
            for branch, ns in [('external', external), ('BU', notices)]:
                for units in (2, 8, 32):
                    for memory in MEMORIES:
                        result = recalls[scenario][branch][str(units)][memory]
                        metrics = recall_check(result, records, evaluator, ns, memory, units, price, infos, require_snapshots)
                        equal(summary[scenario][branch][str(units)][memory], metrics, 'series summary')
                        cells += 1
    return dict(passed=True, series=directory.name, cells=cells, intake_sha256=digest(intake))



def tree_close(actual, expected, label):
    if isinstance(expected, dict):
        equal(set(actual), set(expected), label+' keys')
        for key in expected:
            tree_close(actual[key], expected[key], label+'/'+key)
    elif isinstance(expected, list):
        equal(len(actual), len(expected), label+' length')
        for i, value in enumerate(expected):
            tree_close(actual[i], value, label+'/'+str(i))
    elif isinstance(expected, float):
        close(actual, expected, label)
    else:
        equal(actual, expected, label)


def aggregate_check(rows, saved):
    """Check every saved aggregate and paired series vector from audited rows."""
    keys = ('residual_false_accepts', 'unnecessary_correct_recalls', 'recall_cost', 'attempted', 'withdrawn', 'affected', 'affected_false', 'true_withdrawals', 'initial_false', 'accepted')
    def stats(values, total=True):
        mean = sum(values)/len(values)
        se = math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1)/len(values)) if len(values)>1 else None
        out = dict(mean=mean, standard_error=se, by_series=values)
        if total:
            out['total'] = sum(values)
        return out
    expected = {}
    for scenario in rows[0]:
        expected[scenario] = {}
        for branch in rows[0][scenario]:
            expected[scenario][branch] = {}
            for budget in rows[0][scenario][branch]:
                target = expected[scenario][branch][budget] = {}
                for memory in MEMORIES:
                    target[memory] = {key:stats([float(r[scenario][branch][budget][memory][key]) for r in rows]) for key in keys}
                    if memory != 'G':
                        target[memory]['paired_minus_G'] = {key:stats([float(r[scenario][branch][budget][memory][key]-r[scenario][branch][budget]['G'][key]) for r in rows], False) for key in keys[:3]}
    tree_close(saved, expected, 'aggregate')
    return expected


def freeze_check(rows, saved):
    candidates = {}
    for fault in ('permissive', 'underestimated'):
        bands = []
        for band in range(6):
            counts = [rows[i][fault]['external']['32']['N'] for i in (band, band+6)]
            bands.append(dict(band=band, affected=sum(c['affected'] for c in counts), affected_false=sum(c['affected_false'] for c in counts)))
        chosen = sorted(bands, key=lambda b:(-b['affected_false'], -b['affected'], b['band']))[0]
        candidates[fault] = dict(candidates=bands, selected=chosen['band'], exposure_gate=chosen['affected'] >= 10 and chosen['affected_false'] >= 5)
    expected = dict(experiment='B001', development_seeds=list(range(20261206,20261218)), candidates=candidates, bands={f:c['selected'] for f,c in candidates.items()}, all_exposure_gates_pass=all(c['exposure_gate'] for c in candidates.values()))
    equal(saved, expected, 'development freeze selection')
    return expected


def closure_check(summary, freeze, saved):
    cells = []
    for fault in ('permissive', 'underestimated'):
        for budget in ('2','8','32'):
            cell = summary[fault]['external'][budget]
            cells.append(dict(fault=fault, budget=budget, T_no_worse=all(cell['T'][key]['total'] <= cell['G'][key]['total'] for key in ('residual_false_accepts','unnecessary_correct_recalls'))))
    expected = dict(cells=cells, links_unnecessary_here=all(c['T_no_worse'] for c in cells), exposure_gates=freeze['candidates'])
    equal(saved, expected, 'closure')
    return expected


def source_check(manifest, root, freeze_path=None):
    """Verify recorded bytes against the recorded Git tree, not today's checkout."""
    import subprocess
    head = manifest['git_head']
    def blob(commit, path):
        return subprocess.check_output(['git','show',commit+':'+path], cwd=root, stderr=subprocess.PIPE)
    for path, expected in manifest['source_sha256'].items():
        equal(hashlib.sha256(blob(head,path)).hexdigest(), expected, 'source hash '+path)
    protocol = 'binding/protocols/B001-recall.md'
    equal(hashlib.sha256(blob(head,protocol)).hexdigest(), manifest['protocol_sha256'], 'protocol hash')
    equal(hashlib.sha256(blob(manifest['protocol_commit'],protocol)).hexdigest(), manifest['protocol_sha256'], 'registered protocol hash')
    subprocess.run(['git','merge-base','--is-ancestor',manifest['protocol_commit'],head], cwd=root, check=True)
    if freeze_path is not None:
        relative = str(Path(freeze_path).resolve().relative_to(root.resolve()))
        data = Path(freeze_path).read_bytes()
        equal(hashlib.sha256(data).hexdigest(), manifest['freeze_sha256'], 'freeze file hash')
        equal(blob(head,relative), data, 'freeze committed before run')
    # Numerical math and pricing used by this auditor must match run-time sources.
    for path in ('lab/frozen_pipeline.py','attention/self_model_s3.json'):
        equal(hashlib.sha256((root/path).read_bytes()).hexdigest(), manifest['source_sha256'][path], 'audit dependency hash '+path)
    return dict(source_files=len(manifest['source_sha256']), protocol_ancestry=True, freeze_committed=freeze_path is not None)


def disjoint_check(development, confirmation):
    def hashes(root):
        rows = [read(Path(root)/f'series-{i:02d}'/'intake.json') for i in range(12)]
        return {r['request_sha256'] for data in rows for r in data['requests']}
    a, b = hashes(development), hashes(confirmation)
    equal(len(a), 240, 'unique development requests')
    equal(len(b), 240, 'unique confirmation requests')
    equal(a & b, set(), 'development/confirmation request overlap')
    return dict(development_requests=len(a), confirmation_requests=len(b), overlap=0)


def audit(directory, development=None, freeze_path=None):
    directory = Path(directory)
    manifest = read(directory/'manifest.json')
    datasets = [read(directory/f'series-{i:02d}'/'intake.json') for i in range(12)]
    equal(manifest['intake_sha256'], digest(datasets), 'run intake digest')
    start = 20261206 if manifest['phase'] == 'development' else 20261218
    equal([x['seed'] for x in datasets], list(range(start, start+12)), 'registered seeds')
    confirmation = manifest['phase'] == 'confirmation'
    if confirmation and (development is None or freeze_path is None):
        raise AuditError('Confirmation audit requires --development and --freeze')
    chosen = read(freeze_path)['bands'] if confirmation else None
    reports = [audit_series(directory/f'series-{i:02d}', confirmation, chosen if confirmation else dict(permissive=i%6, underestimated=i%6)) for i in range(12)]
    rows = [read(directory/f'series-{i:02d}'/'summary.json') for i in range(12)]
    aggregate = aggregate_check(rows, read(directory/'summary.json'))
    root = Path(__import__('attention').__file__).resolve().parents[1]
    if confirmation:
        if development is None or freeze_path is None:
            raise AuditError('Confirmation audit requires --development and --freeze')
        dev_rows = [read(Path(development)/f'series-{i:02d}'/'summary.json') for i in range(12)]
        freeze = freeze_check(dev_rows, read(freeze_path))
        closure = closure_check(aggregate, freeze, read(directory/'closure.json'))
        disjoint = disjoint_check(development, directory)
    else:
        freeze = freeze_check(rows, read(directory/'freeze.json'))
        closure, disjoint = None, None
    source = source_check(manifest, root, freeze_path if confirmation else None)
    return dict(passed=True, series_count=len(reports), cells=sum(r['cells'] for r in reports), series=reports, aggregate_verified=True, freeze_selection_verified=True, closure=closure, disjoint_intake=disjoint, sources=source, limitations=LIMITATIONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--series', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--development')
    parser.add_argument('--freeze')
    args = parser.parse_args()
    try:
        report = audit_series(args.directory) if args.series else audit(args.directory, args.development, args.freeze)
    except (AuditError, KeyError, StopIteration, FileNotFoundError) as error:
        report = dict(passed=False, error=str(error))
    encoded = json.dumps(report, indent=2)+'\n'
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded, end='')
    raise SystemExit(0 if report['passed'] else 1)
