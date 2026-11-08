"""Frozen P: compose the selected numerical, request and model checks.

No reference solution enters this module. Legacy arithmetic is delegated rather
than reimplemented; CONFIG records its registered constants in one manifest.
The manifest is not an independent set of runtime tuning knobs. Fingerprints
retain E008/E010 uniform finite volumes; only V2 uses E009 graded volumes.
E011 used graded fingerprint arithmetic: archive regression checks that its
component decisions survive this explicit return to the original probes.
"""
from dataclasses import asdict, dataclass
import time
from unittest.mock import patch
import numpy as np
from lab import extrapolation, geometry_invariants, matched_mixture
from lab.allocation import propose_modes
from lab.convergence_candidates import evidence, flags
from lab.executor import request_hash
from lab.guard import final_guard
from lab.radial_collocation import check as third_check
from lab.three_way import GRIDS


@dataclass(frozen=True)
class Parameters:
    grids: tuple = GRIDS
    first_decisive_grid: int = 512
    candidate: str = 'C1'
    shadow_candidate: str = 'C2'
    planner_fraction: float = .5
    uncertainty_difference_factor: int = 2
    rounding_factor: int = 32
    contraction: float = .5
    futility_grids: tuple = (1024, 2048)
    futility_tolerance_factor: int = 4
    normalization_allowance: float = 2e-4
    first_moment_allowance: float = 2e-3
    second_moment_allowance: float = .02
    high_frequency_allowance: float = .03
    old_probe_step: float = 1e-6
    second_probe_step: float = .01
    high_frequency_coordinate: float = 1e4
    probe_modes: int = 4096
    old_probe_cells: int = 512
    other_probe_cells: int = 4096
    third_degrees: tuple = (64, 128)
    third_contraction_divisor: int = 8
    third_rounding_power: int = 4


CONFIG = Parameters()
MODELS = ('sphere', 'model_scale', 'slab', 'mixture')
POLICIES = ('strict', 'soft')


def parameters():
    return asdict(CONFIG)


def plan_modes(request):
    return propose_modes(request, CONFIG.planner_fraction)


def model_response(kind, z, mean_time, size, model='sphere'):
    """Shared-model test hooks; third_check always retains the sphere model."""
    if kind not in ('modal', 'fv') or model not in MODELS:
        raise ValueError('Unregistered solver or model')
    solver = extrapolation.modal_model if kind == 'modal' else extrapolation.graded_model
    if model != 'mixture':
        return solver(z, mean_time, size, model)
    return sum(w * solver(z, mean_time * scale, size, 'slab')
               for w, scale, _ in matched_mixture.components('mixture'))


def probe_response(kind, z, mean_time, size, model='sphere'):
    if kind not in ('modal', 'fv') or model not in MODELS:
        raise ValueError('Unregistered solver or model')
    if model == 'model_scale':
        return matched_mixture.model_response(kind,z,1000*mean_time,size,'clean')
    condition = 'clean' if model == 'sphere' else model
    return matched_mixture.model_response(kind,z,mean_time,size,condition)


def model_probes(request, model='sphere'):
    with patch('lab.geometry_invariants.modal_test_hook',
               side_effect=lambda z,t,n,d: probe_response('modal',z,t,n,model)), \
         patch('lab.geometry_invariants.fv_test_hook',
               side_effect=lambda z,t,n,d: probe_response('fv',z,t,n,model)):
        return geometry_invariants.probes(request, request.mean_time, 3)


def checks_from_probes(probes, operational_digest, original_digest):
    old = probes['old']['probes'].values()
    return dict(hash=operational_digest == original_digest,
        normalization=all(p['normalization_accepted'] for p in old),
        first_moment=all(p['first_moment_accepted'] for p in old),
        second_moment=all(p['accepted'] for p in probes['second_moment']['probes'].values()),
        high_frequency=all(p['accepted'] for p in probes['high_frequency']['probes'].values()))


def v2_step(values, grid, value, tolerance):
    # The layer argument is unused by C1 and C2. Do not introduce a layer gate.
    result = evidence(values, grid, value, tolerance, 0., CONFIG.candidate)
    differences = [abs(b-a) for a,b in zip(values, values[1:])]
    rounding = CONFIG.rounding_factor*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
    result.update(rounding=float(rounding),
        last_difference=float(differences[-1]), previous_difference=float(differences[-2]),
        both_differences_within_rounding=bool(max(differences[-2:]) <= rounding),
        shadow_C2_converging=flags(values,grid,0.)[CONFIG.shadow_candidate])
    return result


def check_v2(request, value, model='sphere', solver=None):
    """Evaluate only operational grids; optional solver supports archive replay."""
    if solver is None:
        solver = lambda z,t,n: model_response('fv',z,t,n,model)
    grids, values, steps = [], [], []
    for grid in CONFIG.grids:
        grids.append(grid)
        values.append(solver(request.z,request.mean_time,grid))
        if grid >= CONFIG.first_decisive_grid:
            step = v2_step(values,grid,value,request.tolerance)
            steps.append(step)
            if step['verdict'] != 'inconclusive' or step['futile']:
                break
    last = steps[-1]
    reason = ('decisive' if last['verdict'] != 'inconclusive' else 'futility' if last['futile']
              else 'rounding-limit' if last['blocked_rounding'] else 'grid_limit')
    return dict(accepted=last['verdict']=='accept', verdict=last['verdict'],
        stop_reason=reason, grids=grids, values=[geometry_invariants.pair(v) for v in values],
        steps=steps, cells_solved=sum(grids))


def combine(numerical, resolution, checks, third):
    """Both policies differ only in treatment of third-party abstention."""
    base, base_reason = extrapolation.combine_verdict(numerical, resolution, checks)
    result = {}
    for policy in POLICIES:
        if base == 'violation_detected':
            verdict, reason = base, base_reason
        elif third == 'violation_detected':
            verdict, reason = third, 'third_numerical_violation'
        elif base != 'accept':
            verdict, reason = base, base_reason
        elif policy == 'strict' and third != 'accept':
            verdict, reason = 'inconclusive', 'third_inconclusive'
        else:
            verdict, reason = 'accept', 'all_gates_pass'
        result[policy] = dict(verdict=verdict, accepted=verdict=='accept', reason=reason)
    return result


def evaluate(request, original_digest, outcome, model='sphere'):
    started = time.perf_counter()
    resolution = final_guard(request,outcome)
    numerical = check_v2(request,complex(*outcome['value']),model)
    decision_seconds = time.perf_counter()-started
    started = time.perf_counter()
    digest = request_hash(request)
    hash_seconds = time.perf_counter()-started
    probes = model_probes(request,model)
    checks = checks_from_probes(probes,digest,original_digest)
    third = third_check(request.z,request.mean_time,complex(*outcome['value']),request.tolerance)
    return dict(outcome=outcome, model=model, resolution=resolution, independent=numerical,
        original_digest=original_digest, operational_digest=digest, checks=checks,
        probes=probes, third=third,
        policies=combine(numerical['verdict'],resolution['accepted'],checks,third['verdict']),
        decision_seconds=decision_seconds, hash_seconds=hash_seconds)
