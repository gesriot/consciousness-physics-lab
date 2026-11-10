"""B002 whole-family exposure with qualified clean cross replacements."""
import argparse
from contextlib import contextmanager, ExitStack
import copy
import hashlib
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from attention import agent, s2, s3, channel_monitor as monitor
from attention.a001 import reference_audit
from attention.a007 import atomic_json, decode
from attention.build_a007_library import CHANNELS
from attention.io import read_json
from lab import frozen_pipeline as P, radial_collocation, executor as executor_module
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES
from lab.replay import canonical_sha256
from lab.scientist import ROOT
from binding.b001 import append_ledger
from binding.qualification import replacement, perturb, sha

PROTOCOL = 'binding/protocols/B002-qualification.md'
QUALIFICATION = 'binding/results/B002/qualification-freeze.json'
DURATIONS = (4, 8, 16)
FAULTS = ('none', 'permissive', 'underestimated')
TOOLS = {'permissive': 'graded_v2', 'underestimated': 'collocation'}
FAMILY = {'permissive': 'collocation', 'underestimated': 'v2'}
SOURCES = ['binding/b002.py', 'binding/b001.py', 'binding/qualification.py',
           'attention/channel_monitor.py', 'attention/agent.py',
           'attention/calibration_samples.py', 'attention/sample_library_a007.json',
           'attention/self_model_s2.json', 'attention/self_model_s3.json',
           'lab/frozen_pipeline.py', 'lab/radial_collocation.py']


class FamilyTarget(str):
    """Injection-only equality selector; real channel identities never change."""
    def __new__(cls, family):
        if family not in ('v2', 'collocation'):
            raise ValueError('Unregistered numerical fault family')
        return super().__new__(cls, family + ':*')

    def __eq__(self, channel):
        return isinstance(channel, str) and channel in [str(self).split(':')[0]+':'+str(i) for i in range(6)]

    __hash__ = None


def native_price(which, request, weights):
    family = which.split(':')[0]
    if family == 'collocation':
        return sum(P.CONFIG.grids)*weights['fv_cell']
    if family == 'v2':
        return sum(n**3 for n in P.CONFIG.third_degrees)*weights['collocation_degree_cubed']
    return agent.cost(family, request, {'cost_weights': weights})


@contextmanager
def cross_channels():
    original_result = monitor.replacement_result
    def replace(which, executor, weights):
        family = which.split(':')[0]
        if family not in ('v2', 'collocation'):
            return original_result(which, executor, weights)
        instrument = 'graded_v2' if family == 'collocation' else 'collocation'
        result = replacement(instrument, executor.request, complex(*executor.outcome['value']))
        result['instrument'] = instrument
        result['charged_work'] = dict(grids=list(P.CONFIG.grids)) if instrument == 'graded_v2' else dict(degrees=list(P.CONFIG.third_degrees))
        return result, native_price(which, executor.request, weights)
    with patch.object(monitor, 'replacement_price', native_price), patch.object(monitor, 'replacement_result', replace):
        yield


@contextmanager
def numerical_noise(enabled):
    """Perturb numerical output boundaries once, before diagnostics or scheduling."""
    counts = {}
    with ExitStack() as stack:
        if enabled:
            for module, name in ((P, 'model_response'), (P, 'probe_response'),
                                 (radial_collocation, 'response')):
                original = getattr(module, name)
                label = module.__name__ + '.' + name
                def wrapped(*args, _original=original, _label=label, **kwargs):
                    counts[_label] = counts.get(_label, 0)+1
                    return perturb(_original(*args, **kwargs), True)
                stack.enter_context(patch.object(module, name, wrapped))
            original_modal = executor_module.modal_partial_sums
            def modal(*args, **kwargs):
                counts['modal_partial_sums'] = counts.get('modal_partial_sums', 0)+1
                return np.array([perturb(v, True) for v in original_modal(*args, **kwargs)])
            stack.enter_context(patch.object(executor_module, 'modal_partial_sums', modal))
        yield counts


def intake(seed, packages=16):
    if packages not in DURATIONS:
        raise ValueError('Unregistered series duration')
    rng, order, samples = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    rows = []
    for i in range(packages*5):
        cohort = COHORTS[i % 5]
        lo, hi = LOG_RANGES[cohort]
        log_time = float(rng.uniform(-2, 2))
        u, v = float(rng.random()), float(rng.random())
        log_tol = float(rng.uniform(-6 if cohort == 'S' else -4, -4 if cohort == 'S' else -1))
        angle = ((-np.pi/2+.1*v if v < .5 else np.pi/2-.1*(1-v)) if cohort == 'O' else -np.pi/2+np.pi*v)
        time = 10**log_time
        r = Request(10**(lo+(hi-lo)*u)*np.exp(1j*angle)/time, time, 10**log_tol)
        rows.append(dict(task_id=i, package_id=i//5, cohort=cohort, request=encode(r),
                         request_sha256=request_hash(r), condition_order=[str(c) for c in order.permutation(CONDITIONS)]))
    return dict(seed=seed, packages=packages, requests=rows,
                sample_seeds=[int(samples.integers(0, 2**31)) for _ in range(packages)])


def validate_intake(data):
    packages = data['packages']
    if packages not in DURATIONS or len(data['requests']) != 5*packages or len(data['sample_seeds']) != packages:
        raise ValueError('Series dimensions')
    for i, row in enumerate(data['requests']):
        if (row['task_id'], row['package_id'], row['cohort']) != (i, i//5, COHORTS[i % 5]):
            raise ValueError('Series order')
        r = decode(row['request'])
        if request_hash(r) != row['request_sha256'] or sorted(row['condition_order']) != sorted(CONDITIONS):
            raise ValueError('Replay intake mismatch')


def exposure_counts(records, evaluator):
    affected = [r['id'] for r in records if evaluator[r['id']]['affected_channels']]
    return dict(accepted=len(records), initial_false=sum(not evaluator[r['id']]['accurate'] for r in records),
                affected=len(affected), affected_false=sum(not evaluator[r]['accurate'] for r in affected))


def series(data, directory, admitted, noise=False):
    validate_intake(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'intake.json', data)
    model, static = s3.load_model(), s2.load_model()
    library = read_json(ROOT/'attention/sample_library_a007.json')
    requests = [decode(row['request']) for row in data['requests']]
    references, audit = reference_audit(requests)
    audit['all_tasks_audited'] = len(requests)
    atomic_json(directory/'reference.json', audit)
    previous_request = Request(1, 1, .01)
    previous = execute(previous_request, P.plan_modes(previous_request)).value
    faults = ('none',) + tuple(admitted)
    records, evaluators, notices = ({f: [] for f in faults}, {f: {} for f in faults}, {f: [] for f in faults})
    lasts = {f: {c: -1 for c in CHANNELS} for f in faults}
    offsets = {f: 0 for f in faults}
    package_hashes = {f: [] for f in faults}
    snapshots = {}
    with cross_channels(), numerical_noise(noise) as noise_counts:
        for package_id in range(data['packages']):
            specs, infos = [], []
            for index in range(package_id*5, (package_id+1)*5):
                r, row = requests[index], data['requests'][index]
                for condition in row['condition_order']:
                    e = agent.Executor(r, row['request_sha256'], condition, previous)
                    full = agent.run_policy(e, 'C1', float('inf'), static['action_model'], list(agent.ORDER), full=True)
                    original = P.evaluate(r, row['request_sha256'], e.outcome, e.model)
                    if full['verdict'] != original['policies']['strict']['verdict']:
                        raise AssertionError('Full pipeline parity')
                    value = complex(*e.outcome['value'])
                    infos.append(dict(task_id=index, condition=condition, full_cost=full['spent'],
                        outcome=copy.deepcopy(e.outcome), reference=[references[index].real, references[index].imag],
                        absolute_error=abs(value-references[index]), tolerance=r.tolerance,
                        accurate=bool(abs(value-references[index]) <= r.tolerance)))
                    specs.append((r, row['request_sha256'], condition, previous))
                previous = execute(r, P.plan_modes(r)).value
            cost = sum(i['full_cost'] for i in infos)
            for fault in faults:
                family = FAMILY.get(fault, 'collocation')
                selector = FamilyTarget(family)
                spec = dict(kind=fault, target_channel=selector, onset_action=0 if package_id else 10**9,
                            circle_phase=0, random_seed=data['sample_seeds'][package_id])
                result = monitor.run_batch([agent.Executor(*s) for s in specs], 'BU', cost, model, library,
                                           package_id, spec, 1., noise)
                if result['spent'] > cost+1e-8:
                    raise AssertionError('Package budget')
                # Outcomes must be the actual candidate produced in the ordinary trace.
                for action in result['actions']:
                    if not action['sample'] and action['action'] == 'modal':
                        actual = action['original_observation']['outcome']['value']
                        if actual != infos[action['input_index']]['outcome']['value']:
                            raise AssertionError('Ordinary candidate differs from shared construction')
                ns = append_ledger(result, offsets[fault], lasts[fault], records[fault], evaluators[fault], infos, package_id)
                seen = {n['channel'] for n in notices[fault]}
                notices[fault].extend(n for n in ns if n['channel'] not in seen)
                offsets[fault] += 2*(len(result['actions'])+1)
                archive = dict(package_id=package_id, inputs=infos,
                               fault_spec={**spec, 'target_channel': str(selector), 'target_family': family}, result=result)
                atomic_json(directory/f'{fault}-{package_id}.json.gz', archive)
                package_hashes[fault].append(canonical_sha256(archive))
            if package_id+1 in DURATIONS:
                duration = package_id+1
                snapshot = dict(records=copy.deepcopy(records), evaluator=copy.deepcopy(evaluators),
                    BU_notices=copy.deepcopy(notices), last_success=copy.deepcopy(lasts),
                    offsets=copy.deepcopy(offsets), package_hashes=copy.deepcopy(package_hashes))
                atomic_json(directory/f'ledger-{duration}.json.gz', snapshot)
                counts = {f: exposure_counts(records[f], evaluators[f]) for f in faults}
                atomic_json(directory/f'exposure-{duration}.json', counts)
                snapshots[str(duration)] = counts
            atomic_json(directory/'progress.json', dict(completed_packages=package_id+1, solver_noise_calls=noise_counts))
            print('Saved', directory.name, 'package', package_id+1, flush=True)
    return snapshots


def freeze(rows, admitted):
    candidates = {}
    for fault in admitted:
        scores = []
        for duration in DURATIONS:
            totals = {key: sum(row[str(duration)][fault][key] for row in rows)
                      for key in ('accepted', 'initial_false', 'affected', 'affected_false')}
            scores.append(dict(packages=duration, **totals,
                               exposure_gate=totals['affected'] >= 10 and totals['affected_false'] >= 5))
        passed = [s for s in scores if s['exposure_gate']]
        chosen = passed[0]['packages'] if passed else None
        candidates[fault] = dict(instrument=TOOLS[fault], candidates=scores, selected_packages=chosen,
                                 exposure_gate=bool(passed))
    selected = {f: c['selected_packages'] for f, c in candidates.items() if c['exposure_gate']}
    return dict(experiment='B002-exposure', development_seeds=list(range(20270101, 20270113)),
                qualification_sha256=sha(ROOT/QUALIFICATION), candidates=candidates,
                selected=selected, confirmation_allowed=bool(selected),
                status='freeze_before_confirmation' if selected else 'stopped_no_passing_exposure')


def committed_file(path):
    relative = str(Path(path).resolve().relative_to(ROOT))
    saved = subprocess.check_output(['git', 'show', 'HEAD:'+relative], cwd=ROOT)
    if saved != Path(path).read_bytes():
        raise ValueError('Required frozen artifact differs from committed bytes')


def run(output, replay=None, noise=False, override=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    committed_file(ROOT/QUALIFICATION)
    qualification = read_json(ROOT/QUALIFICATION)
    admitted = [fault for fault, tool in TOOLS.items() if tool in qualification['qualified']]
    if not admitted or not qualification['exposure_allowed']:
        raise ValueError('No qualified replacement; exposure not permitted')
    audit = metadata(PROTOCOL, SOURCES, override)
    audit['qualification_sha256'] = sha(ROOT/QUALIFICATION)
    if replay:
        saved = read_json(Path(replay)/'manifest.json')
        for key in ('protocol_sha256', 'source_sha256', 'qualification_sha256'):
            if saved[key] != audit[key]:
                raise ValueError('Replay mismatch: '+key)
        datasets = [read_json(Path(replay)/f'series-{i:02d}'/'intake.json') for i in range(12)]
        if canonical_sha256(datasets) != saved['intake_sha256']:
            raise ValueError('Replay intake mismatch')
    else:
        datasets = [intake(seed) for seed in range(20270101, 20270113)]
    output.mkdir(parents=True)
    atomic_json(output/'manifest.json', dict(experiment='B002-exposure', **audit,
        last_bit_noise=noise, admitted=admitted, durations=list(DURATIONS),
        intake_sha256=canonical_sha256(datasets), independent_series=12,
        sampling_unit='Series; durations are paired prefixes, not independent samples'))
    rows = []
    for i, data in enumerate(datasets):
        rows.append(series(data, output/f'series-{i:02d}', admitted, noise))
        atomic_json(output/'progress.json', dict(completed_series=len(rows), partial=freeze(rows, admitted)))
    frozen = freeze(rows, admitted)
    atomic_json(output/'freeze.json', frozen)
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay')
    parser.add_argument('--last-bit-noise', action='store_true')
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    run(args.output, args.replay, args.last_bit_noise, args.runtime_override)
