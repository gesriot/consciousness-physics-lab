"""E016 held-out confirmation of the unchanged frozen pipeline."""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import mpmath as mp
import numpy as np
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.extrapolation import high_precision
from lab.fault_holdout import FAULTS, inject
from lab import frozen_pipeline as pipeline
from lab.guard import Request
from lab.matched_mixture import components
from lab.replay import canonical_sha256, load_replay, protect_output
from lab.scientist import ROOT, analytic

PROTOCOL = 'experiments/E016-frozen-pipeline.md'
CONDITIONS = FAULTS + ('model_scale', 'slab', 'mixture')
POLICIES = ('strict', 'soft')
COHORTS = 'WHMOS'
LOG_RANGES = dict(W=(-6, 6), H=(6, 12), M=(-1, 3), O=(-6, 12), S=(-6, 6))
SOURCES = ['lab/pipeline_holdout.py', 'lab/frozen_pipeline.py',
    'lab/convergence_candidates.py', 'lab/radial_collocation.py',
    'lab/matched_mixture.py', 'lab/extrapolation.py', 'lab/three_way.py',
    'lab/surface_order.py', 'lab/fault_holdout.py', 'lab/geometry_invariants.py',
    'lab/common_mode.py', 'lab/allocation.py', 'lab/executor.py',
    'lab/experiment_audit.py', 'lab/guard.py', 'lab/finite_volume.py',
    'lab/replay.py', 'lab/scientist.py',
    'vendor/sphere-thermal-cli/sphere_thermal/response.py']
ARCHIVES = {'E005': 'tolerance-allocation', 'E006': 'fault-heldout',
    'E007': 'verifier-stress', 'E008': 'common-mode', 'E009': 'surface-order',
    'E010': 'E010-geometry-invariants', 'E011': 'E011-extrapolation',
    'E013': 'E013-matched-mixture'}
PROVENANCE = {'environment', 'git_head', 'source_sha256', 'runtime_override', 'replay'}


def pair(value):
    return [value.real, value.imag]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def exact_log_coordinate(record):
    lo, hi = record['log_magnitude_interval']
    numerator, denominator = record['magnitude_u_ratio']
    return Fraction(lo) + (hi - lo) * Fraction(numerator, denominator)


def audit_selected(record):
    return exact_log_coordinate(record) > 8 or record['task_id'] % 10 == 0


def validate_intake(records):
    for index, record in enumerate(records):
        cohort = COHORTS[index // 48]
        if record['cohort'] != cohort or record['log_magnitude_interval'] != list(LOG_RANGES[cohort]):
            raise ValueError('Replay cohort or coordinate range mismatch')
        numerator, denominator = record['magnitude_u_ratio']
        if (type(numerator) is not int or type(denominator) is not int or
                denominator <= 0 or not 0 <= numerator < denominator):
            raise ValueError('Invalid exact magnitude coordinate')
        if sorted(record['condition_order']) != sorted(CONDITIONS):
            raise ValueError('Replay condition order is not the registered permutation')
        if record['request']['budget'] != 4096:
            raise ValueError('Replay budget differs from registration')


def make_intake(replay=None):
    if replay is not None:
        saved, requests, provenance = load_replay(replay, 'E016', 240)
        records = copy.deepcopy(saved['intake_channel'])
        validate_intake(records)
        return records, requests, provenance
    rng = np.random.default_rng(20261127)
    order = np.random.default_rng(20261128)
    records, requests = [], []
    for task_id in range(240):
        cohort = COHORTS[task_id // 48]
        lo, hi = LOG_RANGES[cohort]
        log_time = float(rng.uniform(-2, 2))
        u, v = float(rng.random()), float(rng.random())
        log_tolerance = float(rng.uniform(-6 if cohort == 'S' else -4,
                                          -4 if cohort == 'S' else -1))
        log_magnitude = lo + (hi - lo) * u
        magnitude = 10 ** log_magnitude
        phase = ((-np.pi / 2 + .1 * v if v < .5 else np.pi / 2 - .1 * (1 - v))
                 if cohort == 'O' else -np.pi / 2 + np.pi * v)
        mean_time = 10 ** log_time
        request = Request(magnitude * np.exp(1j * phase) / mean_time,
                          mean_time, 10 ** log_tolerance)
        requests.append(request)
        records.append(dict(task_id=task_id, cohort=cohort, request=encode(request),
            request_sha256=request_hash(request), sampled_log_time=log_time,
            sampled_log_tolerance=log_tolerance, sampled_log_magnitude=log_magnitude,
            sampled_magnitude=magnitude, sampled_phase=float(phase), phase_u=v,
            log_magnitude_interval=[lo, hi], magnitude_u_ratio=list(u.as_integer_ratio()),
            condition_order=[str(c) for c in order.permutation(CONDITIONS)]))
    validate_intake(records)
    return records, requests, None


def reference_audit(records, requests):
    values = [complex(analytic(r.z, r.mean_time)) for r in requests]
    precise_values, audits = {}, []
    for record, request, value in zip(records, requests, values):
        if not audit_selected(record):
            continue
        exact = high_precision(request)
        precise_values[record['task_id']] = complex(*map(float, exact))
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
        with mp.workdps(60):
            error = float(abs(mp.mpc(value.real, value.imag) - mp.mpc(*exact))) if finite else None
        allowance = 1e-13 + 1e-12 * abs(precise_values[record['task_id']])
        audits.append(dict(task_id=record['task_id'], cohort=record['cohort'],
            selection='exact_coordinate_above_8' if exact_log_coordinate(record) > 8 else 'task_id_mod_10',
            analytic=pair(value) if finite else None, high_precision=exact,
            absolute_error=error, allowance=allowance, passed=bool(finite and error <= allowance)))
    fallback = not all(a['passed'] for a in audits)
    if fallback:
        for index, request in enumerate(requests):
            if index not in precise_values:
                exact = high_precision(request)
                precise_values[index] = complex(*map(float, exact))
                value = values[index]
                finite = math.isfinite(value.real) and math.isfinite(value.imag)
                with mp.workdps(60):
                    error = float(abs(mp.mpc(value.real, value.imag) - mp.mpc(*exact))) if finite else None
                allowance = 1e-13 + 1e-12 * abs(precise_values[index])
                audits.append(dict(task_id=index, cohort=records[index]['cohort'],
                    selection='whole_sample_fallback', analytic=pair(value) if finite else None,
                    high_precision=exact, absolute_error=error, allowance=allowance,
                    passed=bool(finite and error <= allowance)))
        values = [precise_values[index] for index in range(len(requests))]
    audit = dict(precision_decimal_digits=60, mpmath_version=mp.__version__,
        initial_audited_tasks=sum(a['selection'] != 'whole_sample_fallback' for a in audits),
        audited_tasks=len(audits), failed_tasks=sum(not a['passed'] for a in audits),
        fallback_all=fallback, absolute_allowance=1e-13, relative_allowance=1e-12,
        max_absolute_error=max((a['absolute_error'] for a in audits if a['absolute_error'] is not None), default=None),
        records=sorted(audits, key=lambda a: a['task_id']))
    return values, audit


def request_key(record):
    r = record['request']
    return tuple(float(v).hex() for v in (*r['z'], r['mean_time'], r['tolerance']))


def overlap_audit(intake):
    current = {}
    for r in intake:
        current.setdefault(request_key(r), []).append(r['task_id'])
    comparisons = {}
    for experiment, directory in ARCHIVES.items():
        path = ROOT / 'results' / directory / 'results.json'
        if not path.exists():
            comparisons[experiment] = dict(status='unavailable', source=str(path.relative_to(ROOT)))
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        records = data.get('intake_channel', data.get('tasks', data.get('runs', [])))
        unique = {}
        for record in records:
            if 'request' in record:
                unique.setdefault(request_key(record), record['task_id'])
        matches = [dict(current_task_ids=current[k], earlier_task_id=v)
                   for k, v in unique.items() if k in current]
        comparisons[experiment] = dict(status='checked', source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(raw).hexdigest(), earlier_unique_requests=len(unique),
            exact_request_matches=len(matches), matches=matches)
    for experiment in ('E012', 'E014', 'E015'):
        comparisons[experiment] = dict(status='not_task_intake',
            explanation='Registered development coordinate map, not an independent task sample')
    return dict(definition='Exact binary64 real/imaginary z, mean_time and tolerance',
        within_E016_duplicate_requests=len(intake) - len(current), comparisons=comparisons)


def intrinsic_reference(request, model):
    """Diagnostic model truth only; never passed to a component decision."""
    if model != 'mixture':
        exact = high_precision(request, model)
    else:
        with mp.workdps(60):
            value = mp.mpc(0)
            for weight, scale, _ in components('mixture'):
                component = Request(request.z, request.mean_time * scale, request.tolerance)
                value += mp.mpf(weight) * mp.mpc(*high_precision(component, 'slab'))
            exact = [mp.nstr(value.real, 60), mp.nstr(value.imag, 60)]
    return dict(value=pair(complex(*map(float, exact))), high_precision=exact)


def coverage_evidence(decision, reference, model, intrinsic=None):
    intrinsic = reference if intrinsic is None else intrinsic
    numerical = decision['independent']
    values = dict(zip(numerical['grids'], numerical['values']))
    v2 = []
    for step in numerical['steps']:
        error = abs(complex(*values[step['last_grid']]) - reference)
        intrinsic_error = abs(complex(*values[step['last_grid']]) - intrinsic)
        v2.append(dict(grid=step['last_grid'], reference_error=error,
            intrinsic_numerical_error=intrinsic_error,
            intrinsic_numerical_covered=bool(intrinsic_error <= step['uncertainty_estimate']),
            uncertainty=step['uncertainty_estimate'], covered=bool(error <= step['uncertainty_estimate']),
            decisive=step['verdict'] != 'inconclusive', model=model,
            interpretation='sphere_numerical_error' if model == 'sphere' else 'model_discrepancy_against_original_sphere',
            both_differences_within_rounding=step['both_differences_within_rounding'],
            C1_converging=step['converging'], C2_converging=step['shadow_C2_converging'],
            C1_C2_disagreement=step['converging'] != step['shadow_C2_converging']))
    third = decision['third']
    error = abs(complex(*third['values'][-1]) - reference)
    return dict(V2=v2, third=dict(reference_error=error, uncertainty=third['uncertainty'],
        covered=bool(error <= third['uncertainty']), converging=third['converging'],
        decisive=third['verdict'] != 'inconclusive', verdict=third['verdict']))


def component_work(decision, model):
    count = 2 if model == 'mixture' else 1
    return dict(verification_fv_cells=decision['independent']['cells_solved'] * count,
        probe_modal_terms=sum(p['modal_terms'] for p in decision['probes'].values()) * count,
        probe_fv_cells=sum(p['finite_volume_cells'] for p in decision['probes'].values()) * count,
        third_matrix_solves=decision['third']['matrix_solves'],
        third_sum_degrees=sum(decision['third']['degrees']))


def sum_work(records):
    if not records:
        return {}
    return {key: sum(r[key] for r in records) for key in records[0]}


def summarize(rows):
    clean = [r for r in rows if r['condition'] == 'clean']
    steps = [s for r in rows for s in r['coverage']['V2']]
    decisive = [s for s in steps if s['decisive']]
    third = [r['coverage']['third'] for r in rows if r['coverage']['third']['decisive']]
    coverage = dict(V2_decisive_steps=len(decisive),
        V2_decisive_misses=sum(not s['covered'] for s in decisive),
        V2_sphere_decisive_steps=sum(s['model'] == 'sphere' for s in decisive),
        V2_sphere_decisive_misses=sum(s['model'] == 'sphere' and not s['covered'] for s in decisive),
        V2_intrinsic_numerical_misses=sum(not s['intrinsic_numerical_covered'] for s in decisive),
        V2_model_discrepancy_misses=sum(s['model'] != 'sphere' and not s['covered'] for s in decisive),
        third_decisive_steps=len(third), third_decisive_misses=sum(not s['covered'] for s in third))
    unique_v2 = {(r['task_id'], r['model'], s['grid']): s
                 for r in rows for s in r['coverage']['V2'] if s['decisive']}
    unique_third = {r['task_id']: r['coverage']['third'] for r in rows
                    if r['coverage']['third']['decisive']}
    coverage['unique_task_model_grid_decisive_steps'] = len(unique_v2)
    coverage['unique_task_model_grid_decisive_misses'] = sum(not s['covered'] for s in unique_v2.values())
    coverage['unique_intrinsic_numerical_misses'] = sum(not s['intrinsic_numerical_covered'] for s in unique_v2.values())
    coverage['unique_sphere_decisive_misses'] = sum(s['model'] == 'sphere' and not s['covered'] for s in unique_v2.values())
    coverage['unique_third_decisive_tasks'] = len(unique_third)
    coverage['unique_third_decisive_misses'] = sum(not s['covered'] for s in unique_third.values())
    policies = {}
    for policy in POLICIES:
        accepted = lambda r: r['decision']['policies'][policy]['verdict'] == 'accept'
        violation = lambda r: r['decision']['policies'][policy]['verdict'] == 'violation_detected'
        abstained = [r for r in rows if r['decision']['policies'][policy]['verdict'] == 'inconclusive']
        policies[policy] = dict(
            **{v: sum(r['decision']['policies'][policy]['verdict'] == v for r in rows)
               for v in ('accept', 'inconclusive', 'violation_detected')},
            false_acceptances=sum(accepted(r) and not r['accurate'] for r in rows),
            accurate_violations=sum(r['accurate'] and violation(r) for r in rows),
            accurate_numerical_violations=sum(r['accurate'] and violation(r) and
                (r['decision']['independent']['verdict'] == 'violation_detected' or
                 r['decision']['third']['verdict'] == 'violation_detected') for r in rows),
            accurate_model_or_request_violations=sum(r['accurate'] and violation(r) and
                not all(r['decision']['checks'].values()) for r in rows),
            clean_accurate_violations=sum(r['accurate'] and violation(r) for r in clean),
            clean_accurate_accepted=sum(r['accurate'] and accepted(r) for r in clean),
            abstention_reasons=dict(
                V2=sum(r['decision']['independent']['verdict'] == 'inconclusive' for r in abstained),
                resolution=sum(not r['decision']['resolution']['accepted'] for r in abstained),
                model_checks=sum(not all(r['decision']['checks'].values()) for r in abstained),
                third=sum(r['decision']['third']['verdict'] == 'inconclusive' for r in abstained)),
            coverage=coverage.copy())
    return dict(tasks=len({r['task_id'] for r in rows}), paired_rows=len(rows),
        accurate=sum(r['accurate'] for r in rows),
        clean=dict(count=len(clean), accurate=sum(r['accurate'] for r in clean),
                   inaccurate=sum(not r['accurate'] for r in clean)),
        policies=policies, coverage=coverage,
        strictness_cost=policies['soft']['clean_accurate_accepted']-policies['strict']['clean_accurate_accepted'],
        noise=dict(operational_steps=len(steps), decisive_steps=len(decisive),
            decisive_both_differences_within_rounding=sum(s['both_differences_within_rounding'] for s in decisive),
            operational_C1_C2_disagreements=sum(s['C1_C2_disagreement'] for s in steps),
            decisive_C1_C2_disagreements=sum(s['C1_C2_disagreement'] for s in decisive)),
        component_work=sum_work([r['work'] for r in rows]),
        gate_failures={k: sum(not r['decision']['checks'][k] for r in rows)
                       for k in rows[0]['decision']['checks']} if rows else {})


def scientific_view(value, top=True):
    if isinstance(value, dict):
        return {k: scientific_view(v, False) for k, v in value.items()
                if not k.endswith('seconds') and not (top and k in PROVENANCE)}
    if isinstance(value, list):
        return [scientific_view(v, False) for v in value]
    return value


def run(output, replay=None, runtime_override=None):
    output = Path(output)
    intake_path = output.with_name(output.stem + '-intake.json')
    audit_path = output.with_name(output.stem + '-reference-audit.json')
    for path in (output, intake_path, audit_path):
        protect_output(path, replay)
    audit = metadata(PROTOCOL, SOURCES, runtime_override)
    records, requests, replay_provenance = make_intake(replay)
    if replay_provenance is not None:
        audit['replay'] = replay_provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(intake_path, dict(experiment='E016', **audit, intake_channel=records))
    references, reference = reference_audit(records, requests)
    write_json(audit_path, reference)
    initial_request = Request(1, 1, .01)
    initial_receipt = execute(initial_request, pipeline.plan_modes(initial_request))
    previous = initial_receipt.value
    tasks, rows = [], []
    for record, request, truth in zip(records, requests, references):
        modes = pipeline.plan_modes(request)
        receipt = execute(request, modes)
        generation = dict(base_modal_terms=receipt.modes, model_modal_terms=0,
                          injection_modal_terms=0, base_execution_seconds=receipt.kernel_seconds)
        local = []
        for condition in record['condition_order']:
            model = condition if condition in ('model_scale', 'slab', 'mixture') else 'sphere'
            started = time.perf_counter()
            if condition in FAULTS:
                outcome, extra = inject(request, receipt, condition, previous)
                generation['injection_modal_terms'] += extra
            else:
                value = pipeline.model_response('modal', request.z, request.mean_time, modes, model)
                outcome = dict(value=pair(value), calls=1, modes_spent=modes,
                               trace=[dict(call=1, modes=modes)])
                extra = modes * (2 if model == 'mixture' else 1)
                generation['model_modal_terms'] += extra
            artifact_seconds = time.perf_counter() - started
            decision = pipeline.evaluate(request, record['request_sha256'], outcome, model)
            error = abs(complex(*outcome['value']) - truth)
            local.append(dict(task_id=record['task_id'], cohort=record['cohort'],
                condition=condition, model=model, outcome=outcome, decision=decision,
                reference=pair(truth), tolerance=request.tolerance, absolute_error=error,
                accurate=bool(error <= request.tolerance), artifact_additional_modal_terms=extra,
                artifact_seconds=artifact_seconds, work=component_work(decision, model)))
        model_references = {model: intrinsic_reference(request, model)
                            for model in ('sphere', 'model_scale', 'slab', 'mixture')}
        for row in local:
            intrinsic = complex(*model_references[row['model']]['value'])
            row['coverage'] = coverage_evidence(row['decision'], truth, row['model'], intrinsic)
        rows.extend(local)
        tasks.append(dict(task_id=record['task_id'], cohort=record['cohort'],
            base_modes=modes, generation=generation, reference=pair(truth), model_references=model_references))
        previous = receipt.value
    generation = sum_work([t['generation'] for t in tasks])
    components = sum_work([r['work'] for r in rows])
    total_modal_terms = (initial_receipt.modes + generation['base_modal_terms'] +
        generation['model_modal_terms'] + generation['injection_modal_terms'] + components['probe_modal_terms'])
    work = dict(generation=generation, components=components, initial_stale_modes=initial_receipt.modes,
        total_modal_terms=total_modal_terms,
        total_fv_cells=components['verification_fv_cells'] + components['probe_fv_cells'],
        policy_accounting='Both policies reuse all computed components; incremental policy work is zero. Do not sum their full costs.',
        full_cost_per_policy={p: dict(modal_terms=total_modal_terms,
            fv_cells=components['verification_fv_cells'] + components['probe_fv_cells'],
            third_matrix_solves=components['third_matrix_solves'], third_sum_degrees=components['third_sum_degrees'])
            for p in POLICIES})
    cohort_work = {}
    for cohort in COHORTS:
        generated = sum_work([t['generation'] for t in tasks if t['cohort'] == cohort])
        checked = sum_work([r['work'] for r in rows if r['cohort'] == cohort])
        modal = sum(generated[k] for k in ('base_modal_terms', 'model_modal_terms', 'injection_modal_terms')) + checked['probe_modal_terms']
        cells = checked['verification_fv_cells'] + checked['probe_fv_cells']
        cohort_work[cohort] = dict(generation=generated, components=checked,
            accounting='Initial stale-value generation excluded from cohorts; included once in overall work.',
            full_cost_per_policy={p: dict(modal_terms=modal, fv_cells=cells,
                third_matrix_solves=checked['third_matrix_solves'], third_sum_degrees=checked['third_sum_degrees'])
                for p in POLICIES})
    result = dict(experiment='E016', **audit, task_seed=20261127, order_seed=20261128,
        paired_tasks=240, sampling_unit='Request; nine conditions and two policies are paired',
        intake_channel=records, intake_sha256=canonical_sha256(records),
        reference_audit=reference, reference_audit_sha256=canonical_sha256(reference),
        request_overlap=overlap_audit(records), tasks=tasks, runs=rows,
        summary=summarize(rows), cohort_summary={c:summarize([r for r in rows if r['cohort'] == c]) for c in COHORTS},
        condition_summary={c:summarize([r for r in rows if r['condition'] == c]) for c in CONDITIONS},
        cohort_condition_summary={cohort: {condition: summarize([r for r in rows
            if r['cohort'] == cohort and r['condition'] == condition]) for condition in CONDITIONS}
            for cohort in COHORTS}, work=work, cohort_work=cohort_work)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    result = run(args.output, args.replay, args.runtime_override)
    print(json.dumps(result['summary'], indent=2))
