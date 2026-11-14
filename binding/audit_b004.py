"""Independent reconstruction of source chains, paid actions and B004 metrics."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import subprocess
import numpy as np
import scipy
from pathlib import Path

from binding import source_lab_b004 as physics

MEMORIES = ('G', 'G1', 'F', 'T', 'N', 'Gx', 'Gxprime')
LIMITS = (2, 8, 32)
BRANCHES = ('revision', 'conflict')
CORRUPTIONS = ('clean', 'units', 'regime', 'version')
PRICE = {1: 1, 2: 67, 3: 68}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def fail(message):
    raise ValueError(message)


def require(test, message):
    if not test:
        fail(message)


def exact(actual, expected, label):
    require(sha(actual) == sha(expected), label)


def numeric(actual, expected, label):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool), label)
    require(math.isfinite(actual), label)
    require(math.isclose(actual, expected, rel_tol=2e-11, abs_tol=1e-25), label)


def accurate(value, reference):
    return abs(value-reference) <= 1e-10 + 1e-5*abs(reference)


def reconstruct(series):
    """Derive the entire numerical laboratory from the saved common inputs."""
    require(set(series) == {'intake', 'fault', 'records', 'recipes', 'sources', 'truth',
                           'pairs', 'affected', 'damaged_sources', 'notice_sources', 'notice_versions'},
            'series schema')
    inputs, fault = series['intake'], series['fault']
    require(fault in CORRUPTIONS, 'unregistered corruption')
    require(set(inputs) == {'seed', 'groups', 'targets', 'timestamps'}, 'input schema')
    require(type(inputs['seed']) is int, 'integer seed')
    require(len(inputs['groups']) == 12, 'twelve groups required')
    require(len(inputs['targets']) == 2 and len(set(inputs['targets'])) == 2 and
            all(type(i) is int and 0 <= i < 12 for i in inputs['targets']), 'target indices')
    exact(sorted(inputs['timestamps']), list(range(48)), 'ingestion permutation')
    sources, recipes, expected_records, gold = {}, {}, [], {}
    affected, damaged_sources, pairs, notice = [], [], [], []
    seen_ids = set()
    for group_index, group in enumerate(inputs['groups']):
        require(set(group) == {'source', 'alternate', 'z', 'radius', 'radius2', 'z2',
                              'claims', 'query'}, 'group schema')
        primary, alternate = group['source'], group['alternate']
        p = physics.normalize_source(primary)
        a = physics.normalize_source(alternate)
        require(primary['version'] == alternate['version'] == 2, 'target catalogue version')
        numeric(p['radius_m'], a['radius_m'], 'independent table radius')
        numeric(p['diffusivity_m2_s'], a['diffusivity_m2_s'], 'independent table diffusivity')
        numeric(p['radius_m'], group['radius'], 'known inversion radius')
        require(1-1e-12 <= group['z']*physics.mean_time(primary) <= 10+1e-12, 'frequency range')
        require(1.3-1e-12 <= group['radius2']/group['radius'] <= 2.5+1e-12, 'prediction radius')
        require(.5-1e-12 <= group['z2']/group['z'] <= 2+1e-12, 'prediction frequency')
        ids = group['claims']
        require(len(ids) == 4 and all(isinstance(c, str) for c in ids), 'claim ids')
        all_ids = ids+[primary['id'], alternate['id'], group['query']]
        require(len(set(all_ids)) == 7 and not seen_ids.intersection(all_ids), 'unique opaque ids')
        seen_ids.update(all_ids)
        source_id, alternate_id = primary['id'], alternate['id']
        sources[source_id], sources[alternate_id] = copy.deepcopy(primary), copy.deepcopy(alternate)
        changed = copy.deepcopy(primary)
        is_target = group_index in inputs['targets']
        broken = is_target and fault != 'clean'
        mode = 'exact'
        if is_target:
            notice.append(('method:' if fault == 'regime' else '')+source_id)
        if broken:
            damaged_sources.append('method:'+source_id if fault == 'regime' else source_id)
            if fault == 'units':
                changed['radius']['unit'] = 'm'
            elif fault == 'version':
                changed['conductivity']['value'] *= 4
                changed['version'] = 1
            else:
                mode = 'asymptotic'
        first, second, third, other = ids
        recipes[first] = {'operation': 'response', 'source': source_id, 'z': group['z']}
        recipes[second] = {'operation': 'inverse', 'parent': first,
                           'radius': group['radius'], 'z': group['z']}
        recipes[third] = {'operation': 'prediction', 'parent': second,
                          'radius': group['radius2'], 'z': group['z2']}
        recipes[other] = {'operation': 'response', 'source': alternate_id, 'z': group['z']}
        values = [physics.response(changed, group['z'], mode=mode)]
        values.append(physics.infer_diffusivity(values[0], group['radius'], group['z']))
        values.append(physics.predict_response(values[1], group['radius2'], group['z2']))
        values.append(physics.response(alternate, group['z']))
        truths = [physics.response(primary, group['z'])]
        truths.append(physics.infer_diffusivity(truths[0], group['radius'], group['z']))
        truths.append(physics.predict_response(truths[1], group['radius2'], group['z2']))
        truths.append(physics.response(alternate, group['z']))
        for pos, claim_id in enumerate(ids):
            depth = (1, 2, 3, 1)[pos]
            parents = ([], [first], [second], [])[pos]
            query = group['query'] if pos in (0, 3) else claim_id
            table = changed if pos == 0 else alternate
            direct = []
            if pos in (0, 3):
                direct = [{'source': source_id if pos == 0 else alternate_id,
                           'version': table['version'],
                           'units': {k: v['unit'] for k, v in table.items() if isinstance(v, dict)},
                           'regime': 'properties', 'kind': 'table'},
                          {'source': 'method:'+(source_id if pos == 0 else alternate_id),
                           'version': 1, 'units': {}, 'regime': mode if pos == 0 else 'exact',
                           'kind': 'method'}]
            expected_records.append({'id': claim_id, 'depth': depth, 'value': values[pos],
                                     'unit': 'm2/s' if pos == 1 else '1', 'query': query,
                                     'parents': parents, 'direct': direct, 'request': 'request:'+query,
                                     'time': inputs['timestamps'][len(expected_records)],
                                     'operation': recipes[claim_id]['operation']})
            gold[claim_id] = truths[pos]
            if broken and pos < 3:
                affected.append(claim_id)
        pairs.append(sorted([first, other]))
    exact(series['sources'], sources, 'catalogue mismatch')
    exact(series['recipes'], recipes, 'recipe mismatch')
    exact(series['pairs'], sorted(pairs), 'pair mismatch')
    exact(series['affected'], sorted(affected), 'affected claims mismatch')
    exact(series['damaged_sources'], sorted(damaged_sources), 'damaged sources mismatch')
    exact(series['notice_sources'], sorted(notice), 'notice sources mismatch')
    exact(series['notice_versions'], {sid:1 if fault == 'regime' else 2 for sid in notice},
          'notice versions mismatch')
    require(len(series['records']) == 48, 'record count')
    for actual, expected in zip(series['records'], expected_records):
        exact({k:v for k,v in actual.items() if k != 'value'},
              {k:v for k,v in expected.items() if k != 'value'}, 'record structure')
        numeric(actual['value'], expected['value'], 'observed chain value')
    exact(sorted(series['truth']), sorted(gold), 'truth keys')
    for claim_id, expected in gold.items():
        numeric(series['truth'][claim_id], expected, 'hidden reference value')
    return gold


def conflict_list(records):
    pairs = []
    for index, a in enumerate(records):
        for b in records[index+1:]:
            if (a['query'], a['unit']) == (b['query'], b['unit']):
                if not accurate(a['value'], b['value']) and not accurate(b['value'], a['value']):
                    pairs.append(sorted((a['id'], b['id'])))
    return sorted(pairs)


def expected_order(records, policy, roots, source_ids):
    graph = {r['id']: list(r['parents']) for r in records}
    byid = {r['id']: r for r in records}
    edited = []
    for row in records:
        if policy in ('Gx', 'Gxprime') and any(byid[p]['depth'] == 1 for p in row['parents']):
            old = row['parents'][0] if policy == 'Gx' else row['request']
            edited.append({'id': row['id'], 'before': old, 'after': 'external:'+old})
            if policy == 'Gx':
                graph[row['id']][0] = 'external:'+old
    direct_roots = set(roots)
    if policy not in ('T', 'N'):
        direct_roots.update(r['id'] for r in records
                            if any(a['source'] in source_ids for a in r['direct']))
    marked = set(direct_roots)
    if policy == 'G1':
        marked |= {r['id'] for r in records if set(graph[r['id']]) & direct_roots}
    if policy in ('G', 'Gx', 'Gxprime'):
        # Maximum depth is three; breadth-first traversal is independent of scan order.
        frontier = set(direct_roots)
        while frontier:
            children = {c for c, parents in graph.items() if set(parents) & frontier} - marked
            marked.update(children)
            frontier = children
    if policy == 'T':
        return sorted(byid, key=lambda c: (-byid[c]['time'], c)), [], edited
    # Registered scores are integers 0..3, invariant under last-bit noise and rounding.
    order = sorted(byid, key=lambda c: (-(4-byid[c]['depth']) if c in marked else 0, c))
    return order, sorted(marked), edited


def audit_output(series, output, key, gold):
    policy, limit, branch = key.split('|')
    limit = int(limit)
    require(policy in MEMORIES and limit in LIMITS and branch in BRANCHES, 'condition key')
    require(set(output) == {'protected', 'trace', 'final', 'metrics'}, 'output schema')
    protected, trace = output['protected'], output['trace']
    records = series['records']
    byid = {r['id']: r for r in records}
    conflicts = conflict_list(records)
    spent, position = 0, 0
    roots, source_ids, diagnosed = set(), set(), set()
    expected_trace = []
    if branch == 'revision':
        source_ids.update(series['notice_sources'])
    elif policy != 'N':
        for pair in conflicts:
            chosen = pair[0]
            price = PRICE[byid[chosen]['depth']]
            if spent+price > limit*69:
                break
            require(position < len(trace), 'missing diagnosis')
            actual = trace[position]
            require('answer' in actual, 'missing diagnosis answer')
            numeric(actual['answer'], gold[chosen], 'diagnosis calculation')
            answer = actual['answer']
            bad = sorted(c for c in pair if not accurate(byid[c]['value'], answer))
            sources = set()
            for c in bad:
                for support in byid[c]['direct']:
                    if support['kind'] == 'method':
                        if support['regime'] != 'exact':
                            sources.add(support['source'])
                    else:
                        catalog = series['sources'][support['source']]
                        units = {k:v['unit'] for k,v in catalog.items() if isinstance(v, dict)}
                        if support['version'] != catalog['version'] or support['units'] != units:
                            sources.add(support['source'])
            sources = sorted(sources)
            spent += price
            expected_trace.append({'kind': 'diagnose', 'pair': pair, 'chosen': chosen,
                                   'cost': price, 'answer': answer, 'implicated': bad,
                                   'sources': sources, 'spent': spent})
            roots.update(bad)
            source_ids.update(sources)
            diagnosed.update(sources)
            position += 1
    order, implicated, edits = expected_order(records, policy, roots, source_ids)
    final = {r['id']: {'accepted': True, 'value': r['value']} for r in records}
    if policy != 'N' and (branch == 'revision' or conflicts):
        for cid in order:
            if spent >= limit*69:
                break
            spent += 1
            final[cid]['accepted'] = False
            cost = PRICE[byid[cid]['depth']]
            event = {'kind': 'withdraw', 'id': cid, 'depth': byid[cid]['depth'],
                     'cost': 1, 'spent': spent}
            if spent+cost <= limit*69:
                require(position < len(trace), 'missing repair')
                require('answer' in trace[position], 'missing repair answer')
                answer = trace[position]['answer']
                numeric(answer, gold[cid], 'repair calculation')
                spent += cost
                event.update(kind='repair', answer=answer, cost=cost+1, spent=spent)
                final[cid] = {'accepted': True, 'value': answer}
            expected_trace.append(event)
            position += 1
            if event['kind'] == 'withdraw':
                break
    exact(trace, expected_trace, 'action trace mismatch')
    expected_protected = {'policy': policy, 'budget': limit, 'branch': branch, 'order': order,
                          'implicated': implicated, 'edits': edits,
                          'events': [{k:v for k,v in e.items() if k != 'answer'} for e in expected_trace],
                          'accepted': {c:v['accepted'] for c,v in final.items()}, 'spent': spent,
                          'conflicts': conflicts, 'diagnosed': sorted(diagnosed),
                          'capacity_bytes': len(records)*2048}
    exact(protected, expected_protected, 'protected decisions mismatch')
    exact(output['final'], final, 'final state mismatch')
    bad = {c for c,r in byid.items() if not accurate(r['value'], gold[c])}
    selected = {e['id'] for e in expected_trace if e['kind'] != 'diagnose'}
    incorrect = {c for c,v in final.items() if v['accepted'] and not accurate(v['value'], gold[c])}
    lost = {c for c,v in final.items() if not v['accepted'] and c not in bad}
    depth_metrics = {}
    for depth in (1,2,3):
        subset = {c for c,r in byid.items() if r['depth'] == depth}
        depth_metrics[str(depth)] = {'initial_bad': len(subset & bad),
                                    'remaining_bad': len(subset & incorrect),
                                    'selected': len(subset & selected),
                                    'selected_bad': len(subset & selected & bad),
                                    'unnecessary': len((subset & selected)-bad),
                                    'correct_lost': len(subset & lost)}
    damaged = set(series['damaged_sources'])
    metrics = {'initial_bad': len(bad), 'remaining_bad': len(incorrect),
               'unnecessary': len(selected-bad), 'correct_lost': len(lost),
               'cost': spent,
               'evaluations': sum(e['cost'] - (e['kind'] != 'diagnose') for e in expected_trace),
               'localized_correct': len(diagnosed & damaged), 'localized_total': len(diagnosed),
               'target_sources': len(damaged), 'conflicts': len(conflicts),
               'resolved_conflicts': sum(not (final[a]['accepted'] and final[b]['accepted'] and
                                             not accurate(final[a]['value'], final[b]['value']) and
                                             not accurate(final[b]['value'], final[a]['value']))
                                         for a,b in conflicts),
               'depths': depth_metrics}
    exact(output['metrics'], metrics, 'metric mismatch')
    require(spent <= limit*69, 'budget exceeded')


def audit_case(series, outputs, conditions):
    gold = reconstruct(series)
    expected = {f'{p}|{b}|{s}' for p in MEMORIES for b in LIMITS for s in BRANCHES}
    exact(sorted(conditions), sorted(expected), 'condition coverage')
    require(set(outputs) == set(conditions.values()), 'unreferenced or absent output')
    for identifier, value in outputs.items():
        require(identifier == sha(value), 'output digest')
    for key, identifier in conditions.items():
        audit_output(series, outputs[identifier], key, gold)
    return {'pass': True, 'conditions': len(conditions), 'claims': len(gold)}


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def audit_archive(path, faults=CORRUPTIONS):
    payload = read_json(path)
    require(set(payload) == {'inputs', 'intake_sha256', 'cases'}, 'archive schema')
    require(set(faults) <= set(CORRUPTIONS) and 'clean' in faults, 'admitted corruption list')
    exact(sorted(payload['cases']), sorted(faults), 'corruption coverage')
    require(payload['intake_sha256'] == sha(payload['inputs']), 'intake digest')
    reports = {}
    for fault, case in payload['cases'].items():
        require(set(case) == {'series', 'outputs', 'conditions', 'physical_verifier_evaluations'}, 'case schema')
        require('intake' not in case['series'], 'duplicated common input')
        series = dict(case['series'], intake=payload['inputs'])
        require(series['fault'] == fault, 'case fault mismatch')
        reports[fault] = audit_case(series, case['outputs'], case['conditions'])
        visited = set()
        for result in case['outputs'].values():
            for event in result['trace']:
                if event['kind'] == 'diagnose':
                    visited.add(event['chosen'])
                elif event['kind'] == 'repair':
                    visited.add(event['id'])
        depths = {r['id']:r['depth'] for r in series['records']}
        exact(case['physical_verifier_evaluations'], sum(PRICE[depths[c]] for c in visited),
              'physical verification cache accounting')
    return {'pass': True, 'seed': payload['inputs']['seed'], 'cases': reports,
            'conditions': sum(r['conditions'] for r in reports.values())}



ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('binding/source_lab_b004.py', 'binding/reconcile_b004.py', 'binding/run_b004.py',
                'vendor/sphere-thermal-cli/sphere_thermal/response.py')
PROTOCOL_PATH = 'binding/protocols/B004-source-reconciliation.md'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_summary(archives):
    buckets = {}
    for archive in archives:
        for corruption, case in archive['cases'].items():
            for condition, address in case['conditions'].items():
                buckets.setdefault(corruption+'|'+condition, []).append(case['outputs'][address]['metrics'])
    cells = {}
    for key, rows in sorted(buckets.items()):
        totals, means, errors = {}, {}, {}
        for field in rows[0]:
            if field == 'depths':
                continue
            values = [row[field] for row in rows]
            n = len(values)
            totals[field] = sum(values)
            means[field] = math.fsum(values)/n
            errors[field] = (math.sqrt(math.fsum((value-means[field])**2 for value in values)/(n*(n-1)))
                             if n > 1 else 0.)
        depths = {d:{field:sum(row['depths'][d][field] for row in rows)
                     for field in rows[0]['depths'][d]} for d in ('1','2','3')}
        cells[key] = {'series': len(rows), 'total':totals, 'mean':means,
                      'standard_error':errors, 'depths':depths}
    failures, present = [], False
    for corruption in CORRUPTIONS[1:]:
        for limit in LIMITS:
            for signal in BRANCHES:
                g_key = f'{corruption}|G|{limit}|{signal}'
                if g_key not in cells:
                    continue
                present = True
                g = cells[g_key]['total']
                f = cells[f'{corruption}|F|{limit}|{signal}']['total']
                if f['remaining_bad'] > g['remaining_bad'] or f['unnecessary'] > g['unnecessary']:
                    failures.append(f'{corruption}|F-v-G|{limit}|{signal}')
    return {'cells': cells, 'independent_series':len(archives),
            'closure_F_no_worse_in_every_primary_cell':present and not failures,
            'closure_failure_cells':failures}


def summary_equal(actual, expected):
    require(set(actual) == set(expected), 'summary schema')
    for key in ('independent_series', 'closure_F_no_worse_in_every_primary_cell', 'closure_failure_cells'):
        exact(actual[key], expected[key], 'summary closure or replication count')
    exact(sorted(actual['cells']), sorted(expected['cells']), 'summary cells')
    for key, expected_cell in expected['cells'].items():
        cell = actual['cells'][key]
        require(set(cell) == set(expected_cell), 'summary cell schema')
        for field in ('series','total','depths'):
            exact(cell[field], expected_cell[field], 'summary integer count')
        for field in ('mean','standard_error'):
            exact(sorted(cell[field]), sorted(expected_cell[field]), 'summary numeric keys')
            for metric in cell[field]:
                numeric(cell[field][metric], expected_cell[field][metric], 'summary numeric value')



def independent_gate(archives):
    exposure = {f:{str(d):{'affected':0, 'inaccurate':0} for d in (1,2,3)}
                for f in CORRUPTIONS[1:]}
    for archive in archives:
        for fault in CORRUPTIONS[1:]:
            series = archive['cases'][fault]['series']
            records = {r['id']:r for r in series['records']}
            for cid in series['affected']:
                bucket = exposure[fault][str(records[cid]['depth'])]
                bucket['affected'] += 1
                bucket['inaccurate'] += int(not accurate(records[cid]['value'], series['truth'][cid]))
    admitted = [f for f in CORRUPTIONS[1:] if all(v['affected'] >= 10 and v['inaccurate'] >= 5
                                                  for v in exposure[f].values())]
    counters = dict.fromkeys(('G_F_deep','G_G1_depth3','G1_F_depth2','G_Gx','F_T','active_N'), 0)
    placebos, conflicts = 0, 0
    for archive in archives:
        for fault in admitted:
            case = archive['cases'][fault]
            for budget in LIMITS:
                for branch in BRANCHES:
                    traces = {p:case['outputs'][case['conditions'][f'{p}|{budget}|{branch}']]['trace']
                              for p in MEMORIES}
                    def choices(p, depths):
                        return [event['id'] for event in traces[p]
                                if event['kind'] != 'diagnose' and event['depth'] in depths]
                    for key, a, b, depths in (
                        ('G_F_deep','G','F',(2,3)), ('G_G1_depth3','G','G1',(3,)),
                        ('G1_F_depth2','G1','F',(2,)),
                        ('G_Gx','G','Gx',(1,2,3)), ('F_T','F','T',(1,2,3))):
                        counters[key] += choices(a,depths) != choices(b,depths)
                    counters['active_N'] += bool(choices('G',(1,2,3)))
                    placebos += choices('G',(1,2,3)) != choices('Gxprime',(1,2,3))
                    if budget == 2 and branch == 'conflict':
                        conflicts += len(conflict_list(case['series']['records']))
    return {'exposure':exposure, 'admitted_faults':admitted, 'discernibility_counts':counters,
            'placebo_mismatches':placebos, 'conflicts':conflicts,
            'pass_gate':bool(admitted) and all(counters.values()) and not placebos and conflicts > 0}

def audit_run(directory, runtime_override=False):
    directory = Path(directory).resolve()
    manifest = read_json(directory/'manifest.json')
    require(manifest.get('schema') == 1, 'manifest schema')
    phase = manifest.get('phase')
    require(phase in ('development','confirmation'), 'phase')
    seeds = list(range(20270301,20270313)) if phase == 'development' else list(range(20270313,20270325))
    exact(manifest['seeds'], seeds, 'registered seed cohort')
    faults = manifest['faults']
    require(len(faults) == len(set(faults)) and set(faults) <= set(CORRUPTIONS) and
            faults[0] == 'clean', 'admitted faults')
    if phase == 'development':
        exact(faults, list(CORRUPTIONS), 'development corruption coverage')
    exact(sorted(manifest['source_sha256']), sorted(SOURCE_FILES), 'source inventory')
    for source in SOURCE_FILES:
        require(manifest['source_sha256'][source] == file_sha(ROOT/source), 'current source identity')
        committed = subprocess.check_output(['git','show',manifest['git_head']+':'+source],cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == manifest['source_sha256'][source],
                'committed source identity')
    require(manifest['protocol_sha256'] == file_sha(ROOT/PROTOCOL_PATH), 'protocol identity')
    committed_protocol = subprocess.check_output(['git','show',manifest['git_head']+':'+PROTOCOL_PATH],cwd=ROOT)
    require(hashlib.sha256(committed_protocol).hexdigest() == manifest['protocol_sha256'],
            'committed protocol identity')
    runtime = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
               'platform':platform.system(),'machine':platform.machine()}
    if not runtime_override:
        exact(manifest['runtime'], runtime, 'runtime differs; use --runtime-override')
    expected_names = {f'series-{seed}.json.gz' for seed in seeds}
    exact(sorted(manifest['files']), sorted(expected_names), 'archive coverage')
    exact(sorted(manifest['intakes']), sorted(str(seed) for seed in seeds), 'intake coverage')
    archives, reports = [], []
    for seed in seeds:
        name = f'series-{seed}.json.gz'
        require(file_sha(directory/name) == manifest['files'][name], 'archive file digest')
        archive = read_json(directory/name)
        require(archive['inputs']['seed'] == seed, 'archive seed')
        require(archive['intake_sha256'] == manifest['intakes'][str(seed)], 'manifest intake digest')
        reports.append(audit_archive(directory/name, faults))
        archives.append(archive)
    if phase == 'confirmation':
        freeze_path = ROOT/'binding/results/B004/freeze.json'
        freeze = read_json(freeze_path)
        require(file_sha(freeze_path) == manifest.get('freeze_sha256'), 'freeze digest')
        rel = str(freeze_path.relative_to(ROOT))
        committed_freeze = subprocess.check_output(['git','show',manifest['git_head']+':'+rel],cwd=ROOT)
        require(hashlib.sha256(committed_freeze).hexdigest() == manifest['freeze_sha256'],
                'freeze must precede confirmation')
        exact(freeze['source_sha256'], manifest['source_sha256'], 'frozen source inventory')
        exact(freeze['protocol_sha256'], manifest['protocol_sha256'], 'frozen protocol')
        exact(freeze['development_directory'], 'binding/results/B004/development-first',
              'development proof location')
        development = ROOT/freeze['development_directory']
        dev_manifest = read_json(development/'manifest.json')
        exact(freeze['development_manifest'], dev_manifest, 'frozen development manifest')
        require(file_sha(development/'manifest.json') == freeze['development_manifest_sha256'],
                'frozen development manifest digest')
        dev_archives = []
        for name, checksum in dev_manifest['files'].items():
            require(file_sha(development/name) == checksum, 'frozen development archive digest')
            dev_archives.append(read_json(development/name))
        exact(freeze['gate'], independent_gate(dev_archives), 'frozen gate mismatch')
        require(freeze['gate']['pass_gate'] is True, 'failed gate cannot confirm')
        exact(faults, ['clean']+freeze['gate']['admitted_faults'], 'confirmation admitted corruptions')
        require(freeze['frozen_after_repeats'] is True, 'repeat freeze order')
        require(len(freeze['repeat_comparisons']) == 2, 'two development repeats required')
        exact(sorted(freeze['repeat_comparison_sha256']), sorted(freeze['repeat_comparisons']),
              'repeat comparison hash coverage')
        for relative in freeze['repeat_comparisons']:
            comparison_path = (ROOT/relative).resolve()
            require(comparison_path.is_relative_to(ROOT/'binding/results/B004'), 'comparison proof location')
            require(file_sha(comparison_path) == freeze['repeat_comparison_sha256'][relative],
                    'comparison proof digest')
            comparison = read_json(comparison_path)
            require(comparison['decisions_pass'] is True and not comparison['differing_decisions'],
                    'development repeat failed')
        require(len(freeze['repeat_audit_sha256']) == 2, 'two development repeat audits required')
        for relative, checksum in freeze['repeat_audit_sha256'].items():
            proof_path = (ROOT/relative).resolve()
            require(proof_path.is_relative_to(ROOT/'binding/results/B004'), 'audit proof location')
            require(file_sha(proof_path) == checksum, 'audit proof digest')
            require(read_json(proof_path)['pass'] is True, 'development audit failed')
    summary_equal(read_json(directory/'summary.json'), independent_summary(archives))
    if phase == 'development':
        exact(read_json(directory/'gate.json'), independent_gate(archives), 'development gate mismatch')
    return {'pass':True,'phase':phase,'series':len(reports),
            'conditions':sum(r['conditions'] for r in reports),
            'manifest_sha256':file_sha(directory/'manifest.json'),
            'auditor_sha256':file_sha(__file__), 'runtime_override':bool(runtime_override),
            'runtime':runtime,'archives':reports,'summary_pass':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', nargs='?')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--runtime-override', action='store_true')
    args = parser.parse_args()
    require(bool(args.archive) != bool(args.input), 'choose archive or --input')
    result = audit_run(args.input, args.runtime_override) if args.input else audit_archive(args.archive)
    text = json.dumps(result, sort_keys=True)+'\n'
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
