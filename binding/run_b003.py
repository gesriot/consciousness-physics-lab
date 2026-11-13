"""Prospective B003 intake, admission, paired execution and sequential replay."""
import argparse
import copy
from contextlib import contextmanager, ExitStack
from functools import lru_cache
import shutil
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from attention import agent, s2, s3
from attention.a001 import reference_audit
from attention.a007 import atomic_json, decode
from attention.build_a007_library import CHANNELS
from attention.io import read_json
from binding import b002
from binding.qualification import sha
from lab import frozen_pipeline as P, radial_collocation, executor as executor_module
from lab.executor import execute
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.replay import canonical_sha256
from lab.scientist import ROOT

PROTOCOL = 'binding/protocols/B003-self-checks.md'
QUALIFICATION = b002.QUALIFICATION
EXPOSURE = 'binding/results/B002/exposure-first/freeze.json'
DEVELOPMENT = tuple(range(20270201, 20270213))
CONFIRMATION = tuple(range(20270213, 20270225))
DURATIONS = (4, 8, 16)
POLICIES = ('S', 'Sx', 'Sxprime', 'RR', 'BU')
QUOTAS = (1, 2, 4)
COLLOCATION = tuple(c for c in CHANNELS if c.startswith('collocation:'))
STREAMS = ('requests', 'condition_order', 'sample_execution', 'single_target', 'rr_phase')


@contextmanager
def numerical_memoization(maxsize=32768):
    """Cache pure solver values locally; callers still execute and charge every action."""
    caches = {}
    with ExitStack() as stack:
        for module, name in ((P, 'model_response'), (P, 'probe_response'),
                             (radial_collocation, 'response'),
                             (executor_module, 'modal_partial_sums')):
            original = getattr(module, name)
            cached = lru_cache(maxsize=maxsize)(original)
            def wrapped(*args, _cached=cached, **kwargs):
                # Modal mode lists are value arguments, not mutable solver state.
                args = tuple(tuple(value) if isinstance(value, list) else value for value in args)
                kwargs = {key: tuple(value) if isinstance(value, list) else value
                          for key, value in kwargs.items()}
                return copy.deepcopy(_cached(*args, **kwargs))
            caches[module.__name__+'.'+name] = cached
            stack.enter_context(patch.object(module, name, wrapped))
        try:
            yield caches
        finally:
            for cached in caches.values():
                cached.cache_clear()


def intake(seed, packages=16):
    """Preserve the B002 input distribution and separate all five random streams."""
    data = b002.intake(seed, packages)
    children = np.random.SeedSequence(seed).spawn(5)
    target = np.random.default_rng(children[3])
    phase = np.random.default_rng(children[4])
    data['single_target'] = COLLOCATION[int(target.integers(len(COLLOCATION)))]
    data['rr_phase'] = int(phase.integers(len(CHANNELS)))
    data['streams'] = {name: dict(entropy=seed, spawn_key=list(child.spawn_key))
                       for name, child in zip(STREAMS, children)}
    return data


def validate_intake(data):
    # Archived request bytes are authoritative across supported runtimes. Check
    # their hashes, not a local regeneration through platform floating arithmetic.
    b002.validate_intake(data)
    seed = data['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('Invalid intake seed')
    children = np.random.SeedSequence(seed).spawn(5)
    streams = {name: dict(entropy=seed, spawn_key=list(child.spawn_key))
               for name, child in zip(STREAMS, children)}
    order = np.random.default_rng(children[1])
    samples = np.random.default_rng(children[2])
    target = np.random.default_rng(children[3])
    phase = np.random.default_rng(children[4])
    if data['streams'] != streams:
        raise ValueError('Independent stream provenance differs from seed')
    if data['single_target'] != COLLOCATION[int(target.integers(len(COLLOCATION)))]:
        raise ValueError('Target stream differs from seed')
    if data['rr_phase'] != int(phase.integers(len(CHANNELS))):
        raise ValueError('RR phase stream differs from seed')
    expected_seeds = [int(samples.integers(0, 2**31)) for _ in range(data['packages'])]
    if data['sample_seeds'] != expected_seeds:
        raise ValueError('Sample execution stream differs from seed')
    for row in data['requests']:
        if row['condition_order'] != [str(c) for c in order.permutation(b002.CONDITIONS)]:
            raise ValueError('Condition order stream differs from seed')


def source_files():
    tracked = subprocess.check_output(['git', 'ls-files', 'binding', 'attention', 'lab'],
                                      cwd=ROOT, text=True).splitlines()
    return sorted(p for p in tracked if '/results/' not in p
                  and (p.endswith('.py') or p.endswith('.json')))


def preflight(override):
    for filename in (PROTOCOL, QUALIFICATION, EXPOSURE,
                     'binding/run_b003.py', 'binding/self_checks_b003.py',
                     'binding/audit_b003.py'):
        b002.committed_file(ROOT / filename)
    sources = source_files()
    for filename in sources:
        b002.committed_file(ROOT / filename)
    info = metadata(PROTOCOL, sources, override)
    info['protocol_latest_commit'] = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', PROTOCOL], cwd=ROOT, text=True).strip()
    q, e = read_json(ROOT / QUALIFICATION), read_json(ROOT / EXPOSURE)
    if 'graded_v2' not in q['qualified'] or e['selected'].get('permissive') != 4:
        raise ValueError('Inherited B002 admission does not match registration')
    info.update(qualification_sha256=sha(ROOT / QUALIFICATION),
                exposure_sha256=sha(ROOT / EXPOSURE), command=list(sys.argv))
    return info


def check_overlap(datasets, development=None):
    """Check canonical exact request identities, ignoring copies of old archives."""
    old = set()
    sources = {}
    for experiment in ('B001', 'B002'):
        for path in sorted((ROOT / 'binding/results' / experiment).rglob('intake.json')):
            data = read_json(path)
            old.update(row['request_sha256'] for row in data['requests'])
            sources[str(path.relative_to(ROOT))] = sha(path)
    if not sources:
        raise ValueError('Historical overlap audit has no inputs')
    if development:
        for path in sorted(Path(development).glob('series-*/intake.json')):
            old.update(row['request_sha256'] for row in read_json(path)['requests'])
            sources[str(path.resolve().relative_to(ROOT.resolve()))] = sha(path)
        if len(list(Path(development).glob('series-*/intake.json'))) != 12:
            raise ValueError('Development overlap audit needs twelve series')
    seen = set()
    for data in datasets:
        validate_intake(data)
        for row in data['requests']:
            identity = row['request_sha256']
            if identity in old or identity in seen:
                raise ValueError('Exact request overlap: ' + identity)
            seen.add(identity)
    return dict(passed=True, unique_new_requests=len(seen),
                historical_unique_requests=len(old), sources=sources)


def prepare_package(data, package_id, previous, static):
    """Construct the fault-free common allowance without physical evaluator truth."""
    specs, infos = [], []
    for row in data['requests'][package_id*5:(package_id+1)*5]:
        request = decode(row['request'])
        for condition in row['condition_order']:
            executor = agent.Executor(request, row['request_sha256'], condition, previous)
            full = agent.run_policy(executor, 'C1', float('inf'), static['action_model'],
                                    list(agent.ORDER), full=True)
            original = P.evaluate(request, row['request_sha256'], executor.outcome, executor.model)
            if full['verdict'] != original['policies']['strict']['verdict']:
                raise AssertionError('Clean full pipeline parity')
            infos.append(dict(task_id=row['task_id'], condition=condition,
                              full_cost=full['spent'], full_actions=copy.deepcopy(full['actions']),
                              clean_full_verdict=full['verdict'], outcome=copy.deepcopy(executor.outcome)))
            specs.append((request, row['request_sha256'], condition, previous))
        previous = execute(request, P.plan_modes(request)).value
    return specs, infos, previous


def scenarios(stage, frozen=None):
    if stage == 'development':
        return [('single', 16)]
    single = frozen['single_channel']['selected_packages']
    cells = [('whole', 4)]
    if single is not None:
        cells.append(('single', single))
    cells.extend(('clean', duration) for duration in sorted({d for _, d in cells}))
    return cells


def policy_cells(stage):
    if stage == 'development':
        return [('N0', 0)]
    return [('N0', 0)] + [(policy, quota) for quota in QUOTAS for policy in POLICIES]


def cell_name(environment, duration, policy, quota):
    return f'{environment}-d{duration}-{policy}-q{quota}'


def evaluate_records(result, truth, infos, references, package_id):
    """Attach physical truth only after policy execution has returned."""
    for record in result['records']:
        rid = record['id']
        if rid in truth:
            continue
        package, index = map(int, rid.split(':'))
        if package != package_id:
            raise ValueError('Missing historical evaluator identity')
        info = infos[index]
        value = complex(*info['outcome']['value'])
        ref = references[info['task_id']]
        affected = sorted({a['channel'] for a in result['actions']
                           if a['kind'] == 'ordinary' and a.get('input_index') == index
                           and a.get('fault_applied', False)
                           and any(edge['kind'] == 'channel' and edge['channel'] == a['channel']
                                   and edge['time'] == a['tick']
                                   for edge in record['original_links'])})
        truth[rid] = dict(accurate=bool(abs(value-ref) <= info['tolerance']),
                          absolute_error=abs(value-ref), reference=[ref.real, ref.imag],
                          affected_channels=affected, package_id=package, input_index=index)


def summarize(result, truth, actions, costs, duration, environment, target):
    records = result['records']
    accepted = {r['id'] for r in records}
    standing = {r['id'] for r in records if r.get('standing', True)}
    accurate = {rid for rid, row in truth.items() if row['accurate']}
    samples = [a for a in actions if a['kind'] == 'sample' and a['package_id'] >= 1]
    corrupted = set(COLLOCATION if environment == 'whole' else
                    (target,) if environment == 'single' else ())
    alarms = [a for a in actions if a['kind'] == 'sample' and a.get('mismatch', False)]
    detections = [a['package_id'] for a in alarms if a['channel'] in corrupted
                  and a['package_id'] >= 1 and a.get('fault_applied', False)]
    affected = {rid for rid, row in truth.items() if row['affected_channels']}
    return dict(initial_acceptances=len(accepted), initial_false_acceptances=len(accepted-accurate),
                false_remaining=len(standing-accurate), accurate_withdrawn=len((accepted-standing)&accurate),
                correct_final=len(standing&accurate), correct_final_ids=sorted(standing&accurate),
                initially_accurate_ids=sorted(accepted&accurate), standing_ids=sorted(standing),
                affected=len(affected), affected_false=len(affected-accurate),
                actual_cost=sum(costs.values()), costs=costs,
                detection_delay=min(detections)-1 if detections else duration,
                detection_censored=not bool(detections),
                post_onset_samples=len(samples),
                corrupted_sample_fraction=(sum(a['channel'] in corrupted for a in samples)/len(samples)
                                           if samples else None),
                clean_false_alarms=len(alarms) if environment == 'clean' else None,
                clean_sample_expenditure=costs.get('sample', 0.) if environment == 'clean' else None,
                clean_false_alarm_cost=(sum(a['cost'] for a in alarms)+
                    sum(a['cost'] for a in actions if a['kind'] in ('recall', 'future'))
                    if environment == 'clean' else None))


def series(data, directory, stage, frozen=None, noise=False):
    from binding.self_checks_b003 import State, run_package
    validate_intake(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'intake.json', data)
    model, static = s3.load_model(), s2.load_model()
    library = read_json(ROOT/'attention/sample_library_a007.json')
    start = time.monotonic()
    references, reference = reference_audit([decode(row['request']) for row in data['requests']])
    reference['all_tasks_audited'] = len(data['requests'])
    reference['elapsed_seconds'] = time.monotonic()-start
    reference['operational_cost'] = 0
    atomic_json(directory/'reference.json', reference)
    cells = [(env, duration, policy, quota) for env, duration in scenarios(stage, frozen)
             for policy, quota in policy_cells(stage)]
    states = {cell_name(*cell): State(cell[2], cell[3], data['rr_phase']) for cell in cells}
    truths = {key: {} for key in states}
    actions = {key: [] for key in states}
    costs = {key: {} for key in states}
    summaries = {}
    previous_request = Request(1, 1, .01)
    previous = execute(previous_request, P.plan_modes(previous_request)).value
    with numerical_memoization(), b002.cross_channels(), b002.numerical_noise(noise) as noise_counts:
        for package_id in range(data['packages']):
            specs, infos, previous = prepare_package(data, package_id, previous, static)
            budget = sum(info['full_cost'] for info in infos)
            for info, spec in zip(infos, specs):
                info['tolerance'] = spec[0].tolerance
            for cell in cells:
                environment, duration, policy, quota = cell
                if package_id >= duration:
                    continue
                key = cell_name(*cell)
                fault_channels = (() if environment == 'clean' or package_id == 0 else
                                  COLLOCATION if environment == 'whole' else (data['single_target'],))
                result = run_package([agent.Executor(*spec) for spec in specs], infos,
                                     data['requests'], package_id, budget, model, library,
                                     states[key], fault_channels, noise=noise,
                                     sample_seed=data['sample_seeds'][package_id])
                for action in result['actions']:
                    if action['kind'] == 'ordinary' and action.get('action') == 'modal':
                        actual = action['original_observation']['outcome']['value']
                        if actual != infos[action['input_index']]['outcome']['value']:
                            raise AssertionError('Candidate differs from shared clean construction')
                evaluate_records(result, truths[key], infos, references, package_id)
                actions[key].extend(dict(a, package_id=package_id) for a in result['actions'])
                for category, amount in result['costs'].items():
                    costs[key][category] = costs[key].get(category, 0.)+amount
                archive = dict(package_id=package_id, cell=key, environment=environment,
                               duration=duration, policy=policy, quota=quota, budget=budget,
                               sample_seed=data['sample_seeds'][package_id],
                               fault_channels=list(fault_channels), inputs=infos,
                               result=result, evaluator=copy.deepcopy(truths[key]))
                path = directory/key/f'package-{package_id:02d}.json.gz'
                atomic_json(path, archive)
                if path.stat().st_size > 50*1024*1024:
                    raise ValueError('Archive exceeds 50 MiB')
                if package_id+1 == duration or (stage == 'development' and package_id+1 in DURATIONS):
                    summary = summarize(result, truths[key], actions[key], costs[key],
                                        package_id+1, environment, data['single_target'])
                    name = cell_name(environment, package_id+1, policy, quota)
                    summaries[name] = copy.deepcopy(summary)
                    atomic_json(directory/(name+'-summary.json'), summary)
            atomic_json(directory/'progress.json', dict(completed_packages=package_id+1,
                                                       solver_noise_calls=noise_counts))
            print('Saved', directory.name, 'package', package_id+1, flush=True)
    if stage == 'confirmation':
        for cell in cells:
            environment, duration, policy, quota = cell
            summary = summaries[cell_name(*cell)]
            baseline = summaries[cell_name(environment, duration, 'N0', 0)]
            clean_baseline = summaries[cell_name('clean', duration, 'N0', 0)]
            current, clean = set(summary['correct_final_ids']), set(baseline['correct_final_ids'])
            summary['opportunity_cost'] = dict(gross_lost=len(clean-current),
                                               gross_gained=len(current-clean), net=len(clean)-len(current))
            clean_ids = set(clean_baseline['correct_final_ids'])
            summary['contrast_against_clean_N0'] = dict(gross_lost=len(clean_ids-current),
                gross_gained=len(current-clean_ids), net=len(clean_ids)-len(current))
            summary['accurate_ordinary_never_obtained'] = len(set(baseline['initially_accurate_ids'])-
                                                            set(summary['initially_accurate_ids']))
    atomic_json(directory/'summary.json', summaries)
    return summaries


def aggregate(rows, frozen):
    """Describe twelve paired series without counting variants as replicates."""
    def stats(values):
        return dict(per_series=values, total=sum(values), mean=float(np.mean(values)),
                    standard_error=float(np.std(values, ddof=1)/np.sqrt(len(values)))
                    if len(values) > 1 else None)
    measures = ('false_remaining', 'accurate_withdrawn', 'actual_cost',
                'initial_acceptances', 'correct_final', 'accurate_ordinary_never_obtained',
                'detection_delay', 'post_onset_samples')
    cells = {}
    paired = {}
    dominance = {}
    closure_checks = []
    for environment, duration in scenarios('confirmation', frozen):
        for policy, quota in policy_cells('confirmation'):
            key = cell_name(environment, duration, policy, quota)
            cells[key] = {name: stats([row[key][name] for row in rows]) for name in measures}
            cells[key]['detection_censored'] = [row[key]['detection_censored'] for row in rows]
            cells[key]['corrupted_sample_fraction'] = [row[key]['corrupted_sample_fraction'] for row in rows]
            cells[key]['opportunity_cost'] = {name: stats([row[key]['opportunity_cost'][name] for row in rows])
                                             for name in ('gross_lost', 'gross_gained', 'net')}
            if policy == 'N0':
                continue
            baseline = cell_name(environment, duration, 'S', quota)
            paired[key] = {name: stats([row[key][name]-row[baseline][name] for row in rows])
                           for name in measures}
        for quota in QUOTAS:
            skey = cell_name(environment, duration, 'S', quota)
            rkey = cell_name(environment, duration, 'RR', quota)
            st = {name: cells[skey][name]['total'] for name in measures}
            rt = {name: cells[rkey][name]['total'] for name in measures}
            rr_no_worse = all(rt[name] <= st[name] for name in ('false_remaining', 'accurate_withdrawn'))
            s_no_worse = all(st[name] <= rt[name] for name in ('false_remaining', 'accurate_withdrawn'))
            dominance[f'{environment}-d{duration}-q{quota}'] = dict(
                S_minus_RR_actual_cost=st['actual_cost']-rt['actual_cost'],
                RR_no_worse_on_outcomes=rr_no_worse, S_no_worse_on_outcomes=s_no_worse,
                S_outcome_advantage_at_no_higher_cost=(s_no_worse and not rr_no_worse
                                                       and st['actual_cost'] <= rt['actual_cost']),
                S_outcome_cost_dominance=(s_no_worse and st['actual_cost'] <= rt['actual_cost']
                    and (not rr_no_worse or st['actual_cost'] < rt['actual_cost'])))
            if environment != 'clean':
                closure_checks.append(rr_no_worse)
    graph_unnecessary = bool(closure_checks) and all(closure_checks)
    predictions = {}
    predictions['S_whole_first_faulty_majority'] = {
        str(q): sum(row[cell_name('whole', 4, 'S', q)]['detection_delay'] == 0
                    and not row[cell_name('whole', 4, 'S', q)]['detection_censored'] for row in rows) > len(rows)/2
        for q in QUOTAS}
    predictions['BU_whole_undetected_series'] = {
        str(q): sum(row[cell_name('whole', 4, 'BU', q)]['detection_censored'] for row in rows) for q in QUOTAS}
    predictions['RR_whole_detected_series'] = {
        str(q): sum(not row[cell_name('whole', 4, 'RR', q)]['detection_censored'] for row in rows) for q in QUOTAS}
    predictions['RR_later_than_S_quota1_penalized_delay'] = (
        cells[cell_name('whole', 4, 'RR', 1)]['detection_delay']['total'] >
        cells[cell_name('whole', 4, 'S', 1)]['detection_delay']['total'])
    single = frozen['single_channel']['selected_packages']
    predictions['single_channel_status'] = 'tested' if single else 'untested_not_admitted'
    if single:
        predictions['single_false_acceptance_totals'] = {
            str(q): {p: cells[cell_name('single', single, p, q)]['false_remaining']['total']
                     for p in ('S', 'Sx', 'Sxprime', 'RR')} for q in (1, 2)}
    predictions['no_clean_false_alarms'] = all(
        row[cell_name(env, d, p, q)]['clean_false_alarms'] == 0
        for row in rows for env, d in scenarios('confirmation', frozen) if env == 'clean'
        for p, q in policy_cells('confirmation'))
    return dict(independent_series=len(rows), cells=cells, paired_schedule_minus_S=paired,
                dominance=dominance, predictions=predictions,
                closure=dict(primary_cells=len(closure_checks), graph_stakes_unnecessary=graph_unnecessary,
                    statement=('Graph stakes are unnecessary for allocating self-checks in this laboratory.'
                               if graph_unnecessary else 'The registered closure criterion was not met.')))


def freeze(rows, audit):
    candidates = []
    for duration in DURATIONS:
        key = cell_name('single', duration, 'N0', 0)
        totals = {name: sum(row[key][name] for row in rows)
                  for name in ('initial_acceptances', 'affected', 'affected_false')}
        candidates.append(dict(packages=duration, **totals,
                               passed=totals['affected'] >= 10 and totals['affected_false'] >= 5))
    passing = [row['packages'] for row in candidates if row['passed']]
    return dict(experiment='B003-development', development_seeds=list(DEVELOPMENT),
                completed_series=len(rows), qualification_sha256=audit['qualification_sha256'],
                exposure_sha256=audit['exposure_sha256'], source_sha256=audit['source_sha256'],
                protocol_sha256=audit['protocol_sha256'], whole_family_packages=4,
                single_channel=dict(candidates=candidates,
                                    selected_packages=min(passing) if passing else None),
                status='freeze_before_confirmation' if len(rows) == 12 else 'incomplete')


def run(output, stage='development', development=None, replay=None, noise=False, override=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    audit = preflight(override)
    frozen = None
    if stage == 'confirmation':
        if not development:
            raise ValueError('Confirmation requires committed development freeze')
        development = (ROOT / Path(development)).resolve()
        development_relative = development.relative_to(ROOT.resolve())
        freeze_path = development/'freeze.json'
        b002.committed_file(freeze_path)
        frozen = read_json(freeze_path)
        if frozen['status'] != 'freeze_before_confirmation' or frozen['completed_series'] != 12:
            raise ValueError('Development is incomplete')
        from binding.audit_b003 import verify_sources
        development_manifest = read_json(development/'manifest.json')
        if frozen['source_sha256'] != development_manifest['source_sha256']:
            raise ValueError('Development source manifest differs from freeze')
        audit.update(verify_sources(development_manifest))
        for key in ('qualification_sha256', 'exposure_sha256', 'protocol_sha256'):
            if frozen[key] != audit[key]:
                raise ValueError('Development freeze mismatch: '+key)
        for relative, digest in frozen['artifact_sha256'].items():
            # Progress is rewritten on final completion; scientific artifacts are immutable.
            if relative != 'progress.json' and sha(Path(development)/relative) != digest:
                raise ValueError('Development artifact differs from freeze: '+relative)
        audit['development_directory'] = str(development_relative)
        audit['freeze_sha256'] = sha(freeze_path)
        audit['freeze_commit'] = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--', str(freeze_path.resolve().relative_to(ROOT))],
            cwd=ROOT, text=True).strip()
    seeds = DEVELOPMENT if stage == 'development' else CONFIRMATION
    packages = 16 if stage == 'development' else max(d for _, d in scenarios(stage, frozen))
    if replay:
        original = read_json(Path(replay)/'manifest.json')
        from binding.audit_b003 import verify_sources
        compatibility = verify_sources(original)
        source_transition = original['source_sha256'] != audit['source_sha256']
        portability_keys = ('source_compatibility_sha256', 'source_compatibility_commit')
        correction_keys = ('verifier_correction_sha256', 'verifier_correction_commit')
        development_transition = (stage == original.get('stage') == 'development'
                                  and all(compatibility.get(key) for key in portability_keys))
        correction_transition = (stage == original.get('stage')
                                 and all(compatibility.get(key) for key in correction_keys))
        if source_transition and not (development_transition or correction_transition):
            raise ValueError('Replay source transition lacks exact verified provenance')
        for key, value in compatibility.items():
            if key in audit and audit[key] != value:
                raise ValueError('Replay compatibility differs from confirmation freeze')
            audit[key] = value
        for key in ('source_sha256', 'protocol_sha256', 'qualification_sha256', 'exposure_sha256',
                    'freeze_sha256', 'source_compatibility_sha256', 'source_compatibility_commit',
                    'verifier_correction_sha256', 'verifier_correction_commit'):
            introduced_key = (key in correction_keys and correction_transition or
                              key in portability_keys and development_transition)
            if source_transition and (key == 'source_sha256' or
                                      (introduced_key and key in compatibility and key not in original)):
                continue
            if original.get(key) != audit.get(key):
                raise ValueError('Replay metadata mismatch: '+key)
        datasets = [read_json(Path(replay)/f'series-{i:02d}'/'intake.json') for i in range(12)]
        if canonical_sha256(datasets) != original['intake_sha256']:
            raise ValueError('Replay intake digest mismatch')
    else:
        datasets = [intake(seed, packages) for seed in seeds]
    if [data['seed'] for data in datasets] != list(seeds) or any(data['packages'] != packages for data in datasets):
        raise ValueError('Incorrect registered population or duration')
    overlap = check_overlap(datasets, development if stage == 'confirmation' else None)
    output.mkdir(parents=True)
    atomic_json(output/'manifest.json', dict(experiment='B003', stage=stage, **audit,
        last_bit_noise=noise, intake_sha256=canonical_sha256(datasets), overlap=overlap,
        independent_series=12, scenarios=scenarios(stage, frozen), policies=policy_cells(stage),
        N0_quota_references={str(q): 'N0-q0' for q in QUOTAS},
        sampling_unit='Independent series; policies, quotas and durations are paired'))
    rows = []
    for i, data in enumerate(datasets):
        rows.append(series(data, output/f'series-{i:02d}', stage, frozen, noise))
        atomic_json(output/'progress.json', dict(completed_series=len(rows), status='running'))
        print('Completed', stage, 'series', len(rows), 'of', len(datasets), flush=True)
    atomic_json(output/'series-summaries.json', rows)
    if stage == 'development':
        frozen_result = freeze(rows, audit)
        frozen_result['intake_sha256'] = canonical_sha256(datasets)
        frozen_result['artifact_sha256'] = file_manifest(output)
        atomic_json(output/'freeze.json', frozen_result)
    else:
        report = aggregate(rows, frozen)
        report['target_strata'] = {target: [i for i, data in enumerate(datasets) if data['single_target'] == target]
                                   for target in COLLOCATION}
        atomic_json(output/'aggregate.json', report)
    atomic_json(output/'progress.json', dict(completed_series=12, status='complete'))
    return rows


def file_manifest(directory):
    return {str(path.relative_to(directory)): sha(path) for path in sorted(Path(directory).rglob('*'))
            if path.is_file()}


def sequential_repeats(first, stage, development=None, override=None):
    """Repeat in strict sequence and retain only hashes, audits and comparisons."""
    from binding.audit_b003 import audit_directory, compare_directories
    first = Path(first)
    atomic_json(first/'first-file-manifest.json', file_manifest(first))
    atomic_json(first/'independent-audit.json', audit_directory(first))
    for label, noise in (('ordinary-repeat', False), ('last-bit-noise-repeat', True)):
        temporary = Path(tempfile.mkdtemp(prefix='b003-'+label+'-', dir=first.parent))
        output = temporary/'archive'
        start = time.monotonic()
        log = dict(stage=stage, replay=str(first), last_bit_noise=noise,
                   runtime_override=override, execution='sequential', status='incomplete')
        try:
            run(output, stage, development, first, noise, override)
            audit = audit_directory(output)
            atomic_json(first/(label+'-audit.json'), audit)
            comparison = compare_directories(first, output)
            atomic_json(first/(label+'-comparison.json'), comparison)
            if not comparison['decisions_pass']:
                raise AssertionError(label+' changed protected decisions')
            log['status'] = 'passed'
        except BaseException as error:
            log.update(status='failed', error_type=type(error).__name__, error=str(error),
                       retained_archive=str(output))
            raise
        finally:
            atomic_json(first/(label+'-file-manifest.json'), file_manifest(output))
            log['elapsed_seconds'] = time.monotonic()-start
            atomic_json(first/(label+'-log.json'), log)
            if log['status'] == 'passed':
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('development', 'confirmation'), default='development')
    parser.add_argument('--development')
    parser.add_argument('--replay')
    parser.add_argument('--last-bit-noise', action='store_true')
    parser.add_argument('--runtime-override')
    parser.add_argument('--repeats', action='store_true')
    args = parser.parse_args()
    run(args.output, args.stage, args.development, args.replay,
        args.last_bit_noise, args.runtime_override)
    if args.repeats:
        sequential_repeats(args.output, args.stage, args.development, args.runtime_override)


if __name__ == '__main__':
    main()
