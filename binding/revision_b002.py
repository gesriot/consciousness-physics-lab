"""Conditional B002 confirmation on frozen, qualified exposure choices."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
from attention import channel_monitor
import numpy as np
from attention import s3
from attention.a007 import atomic_json, decode
from attention.build_a007_library import CHANNELS
from attention.channel_monitor import guarded_uniform, REPLACEMENT_GRIDS
from attention.io import read_json
from lab.experiment_audit import metadata
from lab.replay import canonical_sha256
from lab.scientist import ROOT
from binding import b002
from binding.memory import MEMORIES, BUDGETS
from binding.qualification import replacement, sha

PROTOCOL = 'binding/protocols/B002-recall.md'
SOURCES = sorted(set(b002.SOURCES + ['binding/revision_b002.py', 'binding/memory.py']))


def visible(records, notice, memory, reviewed=()):
    if memory not in MEMORIES:
        raise ValueError('Unknown memory')
    rows = copy.deepcopy([r for r in records if r['time'] < notice['time'] and r['id'] not in reviewed])
    edits = []
    if memory in ('Gx', 'Gxprime'):
        for row in rows:
            relevant = [e for e in row['links'] if e['channel'] == notice['channel'] and e['time'] > notice['last_success']]
            if not relevant:
                continue
            if memory == 'Gx':
                edge = relevant[-1]
            else:
                unrelated = [e for e in row['links'] if e['channel'].removeprefix('independent:') == 'normalization']
                if len(unrelated) != 1:
                    raise ValueError('Expected exactly one normalization dependency')
                edge = unrelated[0]
            before = edge['channel']
            edge['channel'] = before.removeprefix('independent:') if before.startswith('independent:') else 'independent:'+before
            edits.append(dict(record_id=row['id'], edge_id=edge.get('edge_id'), before=before,
                              after=edge['channel'], binary_distance=1))
    if memory in ('T', 'A', 'N'):
        rows = [dict(id=r['id'], time=r['time']) for r in rows]
    return rows, edits


def selection(rows, notice, memory, noise=False):
    ranked = []
    for row in rows:
        links = [e['time'] for e in row.get('links', []) if e['channel'] == notice['channel'] and e['time'] > notice['last_success']]
        if memory == 'N' or (memory == 'T' and row['time'] <= notice['last_success']):
            continue
        if memory in ('G', 'Gx', 'Gxprime') and not links:
            continue
        score = float(max(links)) if memory in ('G', 'Gx', 'Gxprime') else float(row['time'])
        if noise:
            score = float(np.nextafter(score, np.inf))
        ranked.append((round(score, 9), row['time'], row['id']))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [row[2] for row in ranked]


def recall(records, notices, memory, units, replace, noise=False):
    reviewed, withdrawn, considered = set(), set(), set()
    attempts, interventions = [], []
    for notice in notices:
        rows, edits = visible(records, notice, memory, reviewed)
        selected = selection(rows, notice, memory, noise)
        considered.update(selected)
        interventions.append(dict(notice=notice, records=rows, edits=edits,
                                  changed_edges=len(edits), binary_distance=len(edits), selected=selected))
        for rid in selected[:max(0, units-len(reviewed))]:
            result, price = replace(rid, notice['channel'])
            reviewed.add(rid)
            revoke = result['verdict'] != 'accept'
            if revoke:
                withdrawn.add(rid)
            attempts.append(dict(record_id=rid, channel=notice['channel'], notice_time=notice['time'],
                                 service_delay=1+sum(a['notice_time'] == notice['time'] for a in attempts),
                                 result=result, withdrawn=revoke, cost=price))
    return dict(attempts=attempts, interventions=interventions, reviewed=sorted(reviewed),
                withdrawn=sorted(withdrawn), considered=sorted(considered), budget_units=units,
                unused_units=units-len(reviewed), cost=sum(a['cost'] for a in attempts))


def metrics(result, records, truth, notices):
    accepted = {r['id'] for r in records}
    withdrawn, reviewed = set(result['withdrawn']), set(result['reviewed'])
    affected = {r['id'] for r in records if any(r['time'] < n['time'] and n['channel'] in truth[r['id']]['affected_channels'] for n in notices)}
    selected_hits = reviewed & affected
    return dict(residual_false_accepts=sum(not truth[r]['accurate'] for r in accepted-withdrawn),
                unnecessary_correct_recalls=sum(truth[r]['accurate'] for r in withdrawn),
                recall_cost=result['cost'], attempted=len(reviewed), withdrawn=len(withdrawn),
                confirmed=sum(a['result']['verdict'] == 'accept' for a in result['attempts']),
                confirmation_fraction=sum(a['result']['verdict'] == 'accept' for a in result['attempts'])/len(reviewed) if reviewed else None,
                affected=len(affected), affected_false=sum(not truth[r]['accurate'] for r in affected),
                selected_affected=len(selected_hits), selection_precision=len(selected_hits)/len(reviewed) if reviewed else None,
                selection_recall=len(selected_hits)/len(affected) if affected else None,
                true_withdrawals=len(affected & withdrawn), accepted=len(accepted),
                initial_false=sum(not truth[r]['accurate'] for r in accepted))


def revise(ordinary, selected, noise=False):
    ordinary = Path(ordinary)
    intake = read_json(ordinary/'intake.json')
    requests = [decode(r['request']) for r in intake['requests']]
    weights = s3.load_model()['action_model']['cost_weights']
    cache, candidates = {}, {}
    def make_replace(kind):
        def execute(rid, channel):
            family = channel.split(':')[0]
            if family not in ('v2', 'collocation'):
                raise ValueError('Unexpected non-numerical alarm')
            package, index = map(int, rid.split(':'))
            if package not in candidates:
                candidates[package] = read_json(ordinary/f'none-{package}.json.gz')['inputs']
            info = candidates[package][index]
            request = requests[info['task_id']]
            key = (kind, rid, family)
            if key not in cache:
                if kind == 'cross':
                    instrument = 'graded_v2' if family == 'collocation' else 'collocation'
                    with b002.numerical_noise(noise):
                        result = replacement(instrument, request, complex(*info['outcome']['value']))
                    result['instrument'] = instrument
                    result['charged_work'] = dict(grids=list(b002.P.CONFIG.grids)) if instrument == 'graded_v2' else dict(degrees=list(b002.P.CONFIG.third_degrees))
                else:
                    original_uniform = channel_monitor.uniform_response
                    def perturbed_uniform(*args, **kwargs):
                        return b002.perturb(original_uniform(*args, **kwargs), True)
                    context = patch.object(channel_monitor, 'uniform_response', perturbed_uniform) if noise else nullcontext()
                    with context:
                        result = guarded_uniform(SimpleNamespace(request=request, outcome=info['outcome']))
                    result['instrument'] = 'B001_uniform'
                cache[key] = result
            price = b002.native_price(channel, request, weights) if kind == 'cross' else sum(REPLACEMENT_GRIDS)*weights['fv_cell']
            return copy.deepcopy(cache[key]), price
        return execute
    results = {}
    for fault, duration in selected.items():
        ledger = read_json(ordinary/f'ledger-{duration}.json.gz')
        family = b002.FAMILY[fault]
        for scenario, source in ((fault, fault), ('none:'+fault, 'none'), ('false_notice:'+fault, 'none')):
            records, truth = ledger['records'][source], ledger['evaluator'][source]
            external = [] if scenario.startswith('none:') else [dict(channel=c, time=ledger['offsets'][source],
                last_success=ledger['last_success'][source][c], last_good_global=min(ledger['last_success'][source].values()),
                source='external') for c in CHANNELS if c.startswith(family+':')]
            results[scenario] = dict(fault=fault, duration=duration, source=source, branches={})
            for branch, notices in (('external', external), ('BU', ledger['BU_notices'][source])):
                cells = {}
                for budget in BUDGETS:
                    cells[str(budget)] = {}
                    for memory in MEMORIES:
                        paired = {}
                        for kind in ('cross', 'uniform'):
                            result = recall(records, notices, memory, budget, make_replace(kind), noise)
                            result['metrics'] = metrics(result, records, truth, notices)
                            paired[kind] = result
                        if [(a['record_id'],a['channel']) for a in paired['cross']['attempts']] != [(a['record_id'],a['channel']) for a in paired['uniform']['attempts']]:
                            raise AssertionError('Counterfactual selection changed')
                        paired['avoided_correct_losses'] = paired['uniform']['metrics']['unnecessary_correct_recalls']-paired['cross']['metrics']['unnecessary_correct_recalls']
                        cells[str(budget)][memory] = paired
                results[scenario]['branches'][branch] = dict(notices=notices, cells=cells)
    return results


def compact(result):
    return {scenario: {'branches': {branch: {'cells': {budget: {memory: {
        'cross': {'metrics': paired['cross']['metrics']},
        'uniform': {'metrics': paired['uniform']['metrics']},
        'avoided_correct_losses': paired['avoided_correct_losses']}
        for memory, paired in memories.items()} for budget, memories in content['cells'].items()}}
        for branch, content in value['branches'].items()}} for scenario, value in result.items()}


def summarize(rows):
    output = {}
    numeric = ('residual_false_accepts', 'unnecessary_correct_recalls', 'recall_cost', 'attempted', 'withdrawn', 'confirmed', 'affected', 'affected_false', 'selected_affected', 'true_withdrawals', 'accepted', 'initial_false')
    def stats(values):
        a = np.asarray(values, dtype=float)
        return dict(total=float(a.sum()), mean=float(a.mean()), standard_error=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None, by_series=list(values))
    for scenario in rows[0]:
        output[scenario] = {}
        for branch in ('external', 'BU'):
            output[scenario][branch] = {}
            for budget in map(str, BUDGETS):
                cell = {}
                for memory in MEMORIES:
                    cells = [r[scenario]['branches'][branch]['cells'][budget] for r in rows]
                    entry = {kind: {key: stats([c[memory][kind]['metrics'][key] for c in cells]) for key in numeric} for kind in ('cross','uniform')}
                    entry['avoided_correct_losses'] = stats([c[memory]['avoided_correct_losses'] for c in cells])
                    entry['paired_minus_G'] = {key: stats([c[memory]['cross']['metrics'][key]-c['G']['cross']['metrics'][key] for c in cells]) for key in numeric[:3]}
                    for kind in ('cross', 'uniform'):
                        s = entry[kind]
                        s['confirmation_fraction'] = s['confirmed']['total']/s['attempted']['total'] if s['attempted']['total'] else None
                        s['selection_precision'] = s['selected_affected']['total']/s['attempted']['total'] if s['attempted']['total'] else None
                        s['selection_recall'] = s['selected_affected']['total']/s['affected']['total'] if s['affected']['total'] else None
                    cell[memory] = entry
                output[scenario][branch][budget] = cell
    return output


def closure(summary, selected):
    cells = []
    for fault in selected:
        for budget in map(str, BUDGETS):
            s = summary[fault]['external'][budget]
            cells.append(dict(fault=fault, budget=int(budget), T_no_worse=all(s['T']['cross'][k]['total'] <= s['G']['cross'][k]['total'] for k in ('residual_false_accepts','unnecessary_correct_recalls'))))
    return dict(cells=cells, links_unnecessary_here=bool(cells) and all(c['T_no_worse'] for c in cells), tested=bool(cells))


def run(output, freeze_path, replay=None, noise=False, override=None):
    output, freeze_path = Path(output), Path(freeze_path)
    if output.exists():
        raise FileExistsError(str(output))
    b002.committed_file(ROOT/b002.QUALIFICATION)
    b002.committed_file(freeze_path)
    qualified, frozen = read_json(ROOT/b002.QUALIFICATION), read_json(freeze_path)
    selected = frozen['selected']
    if not selected or not frozen['confirmation_allowed'] or frozen['qualification_sha256'] != sha(ROOT/b002.QUALIFICATION):
        raise ValueError('No committed passing exposure')
    for fault, duration in selected.items():
        if (duration not in b002.DURATIONS or b002.TOOLS[fault] not in qualified['qualified']
                or not frozen['candidates'][fault]['exposure_gate']
                or duration != frozen['candidates'][fault]['selected_packages']):
            raise ValueError('Unqualified admission')
    audit = metadata(PROTOCOL, SOURCES, override)
    audit.update(exposure_freeze_sha256=sha(freeze_path), qualification_sha256=sha(ROOT/b002.QUALIFICATION))
    if replay:
        old = read_json(Path(replay)/'manifest.json')
        for key in ('source_sha256','protocol_sha256','exposure_freeze_sha256','qualification_sha256'):
            if old[key] != audit[key]:
                raise ValueError('Replay source mismatch: '+key)
        datasets = [read_json(Path(replay)/f'series-{i:02d}'/'ordinary'/'intake.json') for i in range(12)]
        if canonical_sha256(datasets) != old['intake_sha256']:
            raise ValueError('Replay intake mismatch')
    else:
        datasets = [b002.intake(seed,max(selected.values())) for seed in range(20270113,20270125)]
    development = [read_json(freeze_path.parent/f'series-{i:02d}'/'intake.json') for i in range(12)]
    qualification_intakes = [read_json(ROOT/'binding/results/B001/development-complete'/f'series-{i:02d}'/'intake.json') for i in range(12)]
    old_requests = {r['request_sha256'] for d in development+qualification_intakes for r in d['requests']}
    if [d['seed'] for d in datasets] != list(range(20270113,20270125)) or any(d['packages'] != max(selected.values()) for d in datasets):
        raise ValueError('Unregistered confirmation intake')
    new_requests = [r['request_sha256'] for d in datasets for r in d['requests']]
    if old_requests & set(new_requests) or len(new_requests) != len(set(new_requests)):
        raise ValueError('Repeated or overlapping confirmation request')
    output.mkdir(parents=True)
    atomic_json(output/'manifest.json', dict(experiment='B002-confirmation', **audit, selected=selected,
        last_bit_noise=noise, intake_sha256=canonical_sha256(datasets), independent_series=12,
        requests=len(new_requests), exact_development_overlap=0, development_intake_sha256=canonical_sha256(development), qualification_intake_sha256=canonical_sha256(qualification_intakes),
        sampling_unit='12 series; faults, controls, budgets, memories and replacement counterfactuals paired within series'))
    rows = []
    for i, data in enumerate(datasets):
        directory = output/f'series-{i:02d}'
        directory.mkdir()
        b002.series(data, directory/'ordinary', selected, noise)
        result = revise(directory/'ordinary', selected, noise)
        atomic_json(directory/'revision.json.gz', result)
        rows.append(compact(result))
        del result
        atomic_json(output/'progress.json',dict(completed_series=len(rows),summary=summarize(rows)))
    summary = summarize(rows)
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'closure.json',closure(summary,selected))
    return summary


def compare(first, second):
    first, second = Path(first), Path(second)
    stats = dict(decisions_pass=True, intake_pass=True, changed_numerics=0, max_absolute_difference=0., diagnostic_unguarded_changes=0)
    exact_numeric = {'cost','spent','budget','recall_cost','ordinary_cost','replacement_cost','sample_cost','full_cost','sample_deadline','ordinary_cost_before','reserve_before'}
    def walk(a,b,key=''):
        if key in ('package_hashes','shadow_C2_converging','antecedent_ratio','seconds','decision_seconds','hash_seconds'):
            return
        if key == 'unguarded_verdict':
            stats['diagnostic_unguarded_changes'] += a != b
            return
        if isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys(): stats['decisions_pass']=False; return
            for k in a: walk(a[k],b[k],k)
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b): stats['decisions_pass']=False; return
            for x,y in zip(a,b):walk(x,y,key)
        elif isinstance(a,float) and key not in exact_numeric:
            if isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(a) or not np.isfinite(b): stats['decisions_pass']=False
            elif a!=b:
                stats['changed_numerics']+=1
                stats['max_absolute_difference']=max(stats['max_absolute_difference'],abs(a-b))
        elif type(a) is not type(b) or a!=b: stats['decisions_pass']=False
    manifests = [read_json(p/'manifest.json') for p in (first,second)]
    for key in ('intake_sha256','selected','exposure_freeze_sha256','qualification_sha256','source_sha256','protocol_sha256','development_intake_sha256','qualification_intake_sha256'):
        if manifests[0][key]!=manifests[1][key]:stats['intake_pass']=False
    for i in range(12):
        left,right = first/f'series-{i:02d}',second/f'series-{i:02d}'
        walk(read_json(left/'revision.json.gz'),read_json(right/'revision.json.gz'))
        for path in sorted((left/'ordinary').glob('*.json.gz')):
            walk(read_json(path),read_json(right/'ordinary'/path.name))
    if read_json(first/'summary.json') != read_json(second/'summary.json') or read_json(first/'closure.json') != read_json(second/'closure.json'):
        stats['decisions_pass']=False
    return stats


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--freeze')
    parser.add_argument('--replay')
    parser.add_argument('--last-bit-noise',action='store_true')
    parser.add_argument('--runtime-override')
    parser.add_argument('--compare',nargs=2)
    args=parser.parse_args()
    if args.compare:
        atomic_json(args.output,compare(*args.compare))
    else:
        if not args.freeze:parser.error('--freeze is required for confirmation')
        run(args.output,args.freeze,args.replay,args.last_bit_noise,args.runtime_override)
