"""E011 frozen graded verification and held-out extrapolation experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import mpmath as mp
import numpy as np
from scipy.linalg import solve_banded
from lab.allocation import propose_modes
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.fault_holdout import FAULTS as LOCAL_FAULTS, inject
from lab.geometry_invariants import modal_test_hook, pair, probes
from lab.guard import Request, final_guard
from lab.replay import load_replay, protect_output
from lab.scientist import analytic
from lab.surface_order import faces, surface_response
from lab.three_way import GRIDS, check_three_way, step_evidence, shadow_evidence
from lab.verifier_stress import stratum

PROTOCOL = 'experiments/E011-extrapolation.md'
CONDITIONS = LOCAL_FAULTS + ('model_scale', 'slab')
MODELS = ('sphere', 'model_scale', 'slab')
SOURCES = ['lab/extrapolation.py', 'lab/allocation.py', 'lab/common_mode.py',
    'lab/executor.py', 'lab/experiment_audit.py', 'lab/fault_holdout.py',
    'lab/geometry_invariants.py', 'lab/guard.py', 'lab/replay.py', 'lab/scientist.py',
    'lab/surface_order.py', 'lab/three_way.py', 'lab/verifier_stress.py',
    'lab/finite_volume.py', 'vendor/sphere-thermal-cli/sphere_thermal/response.py']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def model_parameters(model):
    if model not in MODELS:
        raise ValueError('Unregistered model')
    return (1 if model == 'slab' else 3), (1000 if model == 'model_scale' else 1)


def modal_model(z, mean_time, modes, model):
    dimension, scale = model_parameters(model)
    return modal_test_hook(z, mean_time * scale, modes, dimension)


def graded_model(z, mean_time, cells, model):
    dimension, scale = model_parameters(model)
    if dimension == 3:
        return surface_response(z, mean_time * scale, cells)
    edges = faces(cells)
    centres = (edges[:-1] + edges[1:]) / 2
    volumes = np.diff(edges)
    kappa = 1 / (3 * mean_time)
    conductance = kappa / np.diff(centres)
    boundary = kappa / (1 - centres[-1])
    diagonal = np.asarray(z * volumes, dtype=complex)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    diagonal[-1] += boundary
    band = np.zeros((3, cells), dtype=complex)
    band[0, 1:] = -conductance
    band[1] = diagonal
    band[2, :-1] = -conductance
    rhs = np.zeros(cells, dtype=complex)
    rhs[-1] = boundary
    return complex(np.dot(volumes, solve_banded((1, 1), band, rhs)) / sum(volumes))


def invariant_probes(request, model):
    # Reuse the exact E010 formulas, sizes and allowances, changing only solvers.
    with patch('lab.geometry_invariants.modal_test_hook',
               side_effect=lambda z, t, n, d: modal_model(z, t, n, model)), \
         patch('lab.geometry_invariants.fv_test_hook',
               side_effect=lambda z, t, n, d: graded_model(z, t, n, model)):
        return probes(request, request.mean_time, 3)


def combine_verdict(numerical, resolution, checks):
    if not all(checks.values()):
        return 'violation_detected', 'model_or_request_violation'
    if numerical == 'violation_detected':
        return numerical, 'numerical_violation'
    if numerical == 'accept' and resolution:
        return 'accept', 'all_gates_pass'
    return 'inconclusive', 'insufficient_resolution' if not resolution else 'numerical_inconclusive'


def evaluate(request, original_digest, outcome, model):
    started = time.perf_counter()
    resolution = final_guard(request, outcome)
    numerical = check_three_way(request, complex(*outcome['value']),
        solver=lambda z, t, n: graded_model(z, t, n, model))
    decision_seconds = time.perf_counter() - started
    started = time.perf_counter()
    digest = request_hash(request)
    hash_seconds = time.perf_counter() - started
    evidence = invariant_probes(request, model)
    old = evidence['old']['probes'].values()
    checks = dict(hash=digest == original_digest,
        normalization=all(p['normalization_accepted'] for p in old),
        first_moment=all(p['first_moment_accepted'] for p in old),
        second_moment=all(p['accepted'] for p in evidence['second_moment']['probes'].values()),
        high_frequency=all(p['accepted'] for p in evidence['high_frequency']['probes'].values()))
    verdict, reason = combine_verdict(numerical['verdict'], resolution['accepted'], checks)
    return dict(outcome=outcome, model=model, resolution=resolution, independent=numerical,
        original_digest=original_digest, operational_digest=digest, checks=checks,
        probes=evidence, verdict=verdict, verdict_reason=reason, accepted=verdict == 'accept',
        decision_seconds=decision_seconds, hash_seconds=hash_seconds)


def high_precision(request, model='sphere'):
    dimension, scale = model_parameters(model)
    with mp.workdps(60):
        z = mp.mpc(request.z.real, request.z.imag)
        t = mp.mpf(request.mean_time) * scale
        if not z:
            value = mp.mpc(1)
        else:
            x = mp.sqrt(dimension * (dimension + 2) * z * t)
            value = mp.tanh(x) / x if dimension == 1 else 3 * (mp.coth(x) / x - 1 / x**2)
        return [mp.nstr(value.real, 60), mp.nstr(value.imag, 60)]


def make_intake(replay=None):
    if replay is not None:
        saved, requests, provenance = load_replay(replay, 'E011', 192)
        return saved['intake_channel'], requests, provenance
    rng = np.random.default_rng(20261122)
    intake, requests = [], []
    for task_id in range(192):
        cohort = 'ABCK'[task_id // 48]
        t = 10 ** rng.uniform(-2, 2)
        magnitude = 10 ** rng.uniform(6 if cohort == 'A' else -6, 8 if cohort == 'A' else 6)
        u = rng.uniform()
        phase = (-np.pi/2 + .1*u if u < .5 else np.pi/2 - .1*(1-u)) if cohort == 'B' else -np.pi/2 + np.pi*u
        tolerance = 10 ** rng.uniform(-6 if cohort == 'C' else -4, -4 if cohort == 'C' else -1)
        request = Request(magnitude * np.exp(1j * phase) / t, t, tolerance)
        requests.append(request)
        intake.append(dict(task_id=task_id, cohort=cohort, request=encode(request),
            request_sha256=request_hash(request), sampled_magnitude=magnitude, sampled_phase=phase))
    return intake, requests, None


def reference_audit(intake, requests):
    records = []
    for record, request in zip(intake, requests):
        if record['cohort'] not in ('A', 'B'):
            continue
        exact = high_precision(request)
        reference = complex(*map(float, exact))
        value = complex(analytic(request.z, request.mean_time))
        # Retain the high-precision discrepancy, not a subtraction of rounded references.
        with mp.workdps(60):
            discrepancy = float(abs(mp.mpc(value.real, value.imag) - mp.mpc(*exact)))
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
        records.append(dict(task_id=record['task_id'], cohort=record['cohort'],
            high_precision=exact, analytic=pair(value) if finite else None,
            absolute_error=discrepancy if finite else None,
            passed=bool(finite and discrepancy <= 1e-13 + 1e-12 * abs(reference))))
    fallback = not all(r['passed'] for r in records)
    return dict(precision_decimal_digits=60, mpmath_version=mp.__version__,
        absolute_allowance=1e-13, relative_allowance=1e-12, audited_tasks=len(records),
        failed_tasks=sum(not r['passed'] for r in records), fallback_all_AB=fallback,
        max_absolute_error=max((r['absolute_error'] for r in records if r['absolute_error'] is not None), default=None),
        records=records)


def summarize(rows, coverage, tasks):
    result = {}
    for cohort in 'ABCK':
        group = [r for r in rows if r['cohort'] == cohort]
        steps = [s for s in coverage if s['cohort'] == cohort]
        decisive_ids = {(r['task_id'], r['model'], s['last_grid']) for r in group
                        for s in r['independent']['steps'] if s['verdict'] != 'inconclusive'}
        decisive = [s for s in steps if (s['task_id'], s['model'], s['grid']) in decisive_ids]
        clean = [r for r in group if r['fault'] == 'clean']
        strata = {}
        for model in MODELS:
            for name in ('[0,.5)', '[.5,1)', '[1,2)', '[2,4)', '[4,infinity)'):
                selected = [s for s in steps if s['model'] == model and s['stratum'] == name]
                strata[model + ':' + name] = dict(steps=len(selected), misses=sum(not s['covered'] for s in selected),
                    max_ratio=max((s['ratio'] for s in selected), default=None))
        result[cohort] = dict(tasks=len(clean), paired_rows=len(group),
            false_acceptances=sum(r['accepted'] and not r['accurate'] for r in group),
            accurate_violation=sum(r['accurate'] and r['verdict'] == 'violation_detected' for r in group),
            accurate_numerical_violation=sum(r['accurate'] and r['independent']['verdict'] == 'violation_detected' for r in group),
            accurate_model_violation=sum(r['accurate'] and r['verdict_reason'] == 'model_or_request_violation' for r in group),
            clean_accurate_violation=sum(r['accurate'] and r['verdict'] == 'violation_detected' for r in clean),
            coverage=dict(steps=len(steps), misses=sum(not s['covered'] for s in steps),
                max_ratio=max(s['ratio'] for s in steps), decisive_steps=len(decisive),
                decisive_misses=sum(not s['covered'] for s in decisive),
                sphere_decisive_steps=sum(s['model'] == 'sphere' for s in decisive),
                sphere_decisive_misses=sum(s['model'] == 'sphere' and not s['covered'] for s in decisive), strata=strata),
            clean=dict(accurate=sum(r['accurate'] for r in clean),
                accurate_accepted=sum(r['accurate'] and r['accepted'] for r in clean),
                inaccurate=sum(not r['accurate'] for r in clean),
                inconclusive=sum(r['verdict'] == 'inconclusive' for r in clean),
                invariant_false_rejections={k: sum(not r['checks'][k] for r in clean)
                    for k in ('normalization', 'first_moment', 'second_moment', 'high_frequency')}),
            inconclusive=sum(r['verdict'] == 'inconclusive' for r in group),
            numerical_inconclusive=sum(r['independent']['verdict'] == 'inconclusive' for r in group),
            futility_stops=sum(r['independent']['stop_reason'] == 'futility' for r in group),
            conditions={fault: dict(rows=len([r for r in group if r['fault'] == fault]),
                accepted=sum(r['accepted'] for r in group if r['fault'] == fault),
                accurate=sum(r['accurate'] for r in group if r['fault'] == fault),
                false_acceptances=sum(r['accepted'] and not r['accurate'] for r in group if r['fault'] == fault))
                for fault in CONDITIONS})
    return result


def run(output, replay=None, runtime_override=None):
    output = Path(output)
    protect_output(output, replay)
    audit_path = output.with_name(output.stem + '-reference-audit.json')
    intake_path = output.with_name(output.stem + '-intake.json')
    for path in (audit_path, intake_path):
        protect_output(path, replay)
    audit = metadata(PROTOCOL, SOURCES, runtime_override)
    intake, requests, replay_metadata = make_intake(replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(intake_path, dict(experiment='E011', intake_channel=intake, **audit))
    reference_check = reference_audit(intake, requests)
    write_json(audit_path, reference_check)
    # Both file handles are closed before ANY operational evaluation starts.
    reference_by_id = {r['task_id']: r for r in reference_check['records']}
    audit['reference_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    audit['intake_sha256'] = hashlib.sha256(intake_path.read_bytes()).hexdigest()
    if replay_metadata is not None:
        audit['replay'] = replay_metadata
    order = np.random.default_rng(20261123)
    initial = Request(1, 1, .01)
    stale_receipt = execute(initial, propose_modes(initial, .5))
    previous = stale_receipt.value
    rows, tasks, coverage = [], [], []
    started_all = time.perf_counter()
    for record, request in zip(intake, requests):
        task_id, cohort = record['task_id'], record['cohort']
        receipt = execute(request, propose_modes(request, .5))
        local = []
        for fault in order.permutation(CONDITIONS):
            fault = str(fault)
            model = fault if fault in ('model_scale', 'slab') else 'sphere'
            started = time.perf_counter()
            if fault in LOCAL_FAULTS:
                outcome, extra = inject(request, receipt, fault, previous)
            else:
                value = modal_model(request.z, request.mean_time, receipt.modes, model)
                outcome = dict(value=pair(value), calls=1, modes_spent=receipt.modes,
                    trace=[dict(call=1, modes=receipt.modes)])
                extra = receipt.modes
            injection_seconds = time.perf_counter() - started
            local.append(dict(task_id=task_id, cohort=cohort, fault=fault,
                additional_artifact_modes=extra, injection_seconds=injection_seconds,
                **evaluate(request, record['request_sha256'], outcome, model)))
        # Operational gates for all eight conditions are now complete.
        audit_row = reference_by_id.get(task_id)
        reference = (complex(*map(float, audit_row['high_precision'])) if audit_row and reference_check['fallback_all_AB']
                     else complex(analytic(request.z, request.mean_time)))
        reference_kind = 'high_precision' if audit_row and reference_check['fallback_all_AB'] else 'analytic'
        paths, shadow_seconds = {}, 0.
        model_references = {'sphere': reference}
        for model in MODELS:
            if model != 'sphere':
                model_references[model] = complex(*map(float, high_precision(request, model)))
            started = time.perf_counter()
            values = [graded_model(request.z, request.mean_time, n, model) for n in GRIDS]
            shadow_seconds += time.perf_counter() - started
            paths[model] = values
            dimension, scale = model_parameters(model)
            kappa = 1 / (dimension * (dimension + 2) * request.mean_time * scale)
            for i, n in enumerate(GRIDS):
                if n < 512:
                    continue
                evidence = step_evidence(values[:i+1], n, receipt.value, request.tolerance)
                uncertainty = evidence['uncertainty_estimate']
                error = abs(values[i] - model_references[model])
                hs = 1 - faces(n)[-2]
                dh = math.sqrt(kappa / abs(request.z)) / hs
                coverage.append(dict(task_id=task_id, cohort=cohort, model=model, grid=n,
                    surface_width=hs, delta_over_h=dh, stratum=stratum(dh), error=error,
                    uncertainty=uncertainty, ratio=error/uncertainty, covered=error <= uncertainty,
                    converging=evidence['converging'],
                    original_sphere_discrepancy=abs(values[i] - reference),
                    exact_model_bias=abs(model_references[model] - reference)))
        for row in local:
            value = complex(*row['outcome']['value'])
            error = abs(value - reference)
            row.update(absolute_error=error, accurate=error <= request.tolerance,
                shadow=shadow_evidence(request, value, paths[row['model']]))
        rows.extend(local)
        tasks.append(dict(**record, reference=pair(reference), reference_kind=reference_kind,
            model_references={k: pair(v) for k, v in model_references.items()},
            base_modes=receipt.modes, base_execution_seconds=receipt.kernel_seconds,
            shadow_values={k: [pair(v) for v in values] for k, values in paths.items()},
            shadow_cells=3 * sum(GRIDS), shadow_seconds=shadow_seconds))
        previous = receipt.value
    work = dict(initial_stale_modes=stale_receipt.modes,
        initial_stale_seconds=stale_receipt.kernel_seconds,
        base_execution_modes=sum(t['base_modes'] for t in tasks),
        base_execution_seconds=sum(t['base_execution_seconds'] for t in tasks),
        additional_artifact_modes=sum(r['additional_artifact_modes'] for r in rows),
        injection_seconds=sum(r['injection_seconds'] for r in rows),
        decision_cells=sum(r['independent']['cells_solved'] for r in rows),
        decision_seconds=sum(r['decision_seconds'] for r in rows),
        hash_seconds=sum(r['hash_seconds'] for r in rows),
        shadow_cells=sum(t['shadow_cells'] for t in tasks),
        shadow_seconds=sum(t['shadow_seconds'] for t in tasks),
        operational_and_diagnostic_wall_seconds=time.perf_counter() - started_all)
    for group in ('old', 'second_moment', 'high_frequency'):
        work[group] = {k: sum(r['probes'][group][k] for r in rows)
                      for k in ('modal_terms', 'finite_volume_cells', 'seconds')}
    result = dict(experiment='E011', **audit, task_seed=20261122, order_seed=20261123,
        paired_tasks=192, sampling_unit='task; eight conditions and repeated grids are dependent',
        reference_audit=reference_check, intake_channel=intake, tasks=tasks,
        coverage_steps=coverage, runs=rows, summary=summarize(rows, coverage, tasks), work=work)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    result = run(args.output, args.replay, args.runtime_override)
    print(json.dumps(dict(summary=result['summary'], reference_audit={k: v for k, v in result['reference_audit'].items() if k != 'records'}, work=result['work']), indent=2))
