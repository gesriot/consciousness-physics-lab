"""E010 experiment-only TEST HOOKS; shared, mean-preserving geometry errors."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import jn_zeros
from lab.allocation import propose_modes
from lab.common_mode import encode
from lab.executor import request_hash
from lab.experiment_audit import metadata
from lab.fault_holdout import verify
from lab.finite_volume import response
from lab.guard import Request
from lab.replay import load_replay, protect_output
from lab.scientist import analytic, modal_partial_sums

GEOMETRIES = {'clean': 3, 'slab': 1, 'cylinder': 2}
DEFENSES = ('hash', 'normalization', 'first_moment', 'old_invariants',
            'second_moment', 'high_frequency', 'all')
POLICIES = ('D',) + DEFENSES + tuple('D_' + key for key in DEFENSES)


@lru_cache(maxsize=None)
def roots(dimension, size):
    if dimension == 1:
        values = (np.arange(1, size + 1) - .5) * np.pi
    elif dimension == 2:
        values = jn_zeros(0, size)
    elif dimension == 3:
        values = np.arange(1, size + 1) * np.pi
    else:
        raise ValueError('unregistered geometry')
    values.setflags(write=False)
    return values


def modal_test_hook(z, mean_time, modes, dimension):
    if dimension == 3:
        return complex(modal_partial_sums(z, mean_time, [modes])[0])
    eigen = roots(dimension, modes) ** 2
    rates = eigen / (dimension * (dimension + 2) * mean_time)
    return complex(np.sum((2 * dimension / eigen) * rates / (rates + z)))


def fv_test_hook(z, mean_time, cells, dimension):
    if dimension == 3:
        return response(z, mean_time, cells)
    if dimension not in (1, 2):
        raise ValueError('unregistered geometry')
    faces = np.linspace(0, 1, cells + 1)
    volumes = np.diff(faces ** dimension) / dimension
    kappa = 1 / (dimension * (dimension + 2) * mean_time)
    conductance = kappa * faces[1:-1] ** (dimension - 1) * cells
    boundary = 2 * kappa * cells
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
    temperature = solve_banded((1, 1), band, rhs)
    return complex(np.dot(volumes, temperature) / np.sum(volumes))


def pair(value):
    return [value.real, value.imag]


def probes(request, original_time, dimension):
    """All calls, including both zero evaluations, are fresh and timed."""
    result = {}
    for group in ('old', 'second_moment', 'high_frequency'):
        started = time.perf_counter()
        records = {}
        for name, solver in (('modal', modal_test_hook), ('finite_volume', fv_test_hook)):
            size = 512 if group == 'old' and name == 'finite_volume' else 4096
            if group == 'high_frequency':
                value = solver(1e4 / original_time, request.mean_time, size, dimension)
                scaled = 100 * value
                relative = abs(scaled / np.sqrt(3 / 5) - 1)
                records[name] = dict(value=pair(value), scaled=pair(scaled),
                    relative_error=float(relative), accepted=bool(relative <= .03))
            else:
                h = (1e-6 if group == 'old' else .01) / original_time
                zero = solver(0, request.mean_time, size, dimension)
                shifted = solver(1j * h, request.mean_time, size, dimension)
                record = dict(zero=pair(zero), shifted=pair(shifted), h=h)
                if group == 'old':
                    mean = -shifted.imag / h
                    norm_error = abs(zero - 1)
                    relative = abs(mean / original_time - 1)
                    record.update(estimated_mean=mean, normalization_error=norm_error,
                        mean_relative_error=relative, normalization_accepted=bool(norm_error <= 2e-4),
                        first_moment_accepted=bool(relative <= 2e-3))
                else:
                    moment = 2 * (zero.real - shifted.real) / (h * original_time) ** 2
                    relative = abs(moment / (20 / 7) - 1)
                    record.update(normalized_second_moment=moment,
                                  relative_error=relative, accepted=bool(relative <= .02))
                records[name] = record
        result[group] = dict(probes=records, seconds=time.perf_counter() - started,
            modal_terms=4096 if group == 'high_frequency' else 8192,
            finite_volume_cells={'old': 1024, 'second_moment': 8192, 'high_frequency': 4096}[group])
    return result


def evaluate(original, original_digest, geometry):
    dimension = GEOMETRIES[geometry]
    modes = propose_modes(original, .5)
    started = time.perf_counter()
    value = modal_test_hook(original.z, original.mean_time, modes, dimension)
    execution_seconds = time.perf_counter() - started
    outcome = dict(value=pair(value), calls=1, modes_spent=modes, trace=[dict(call=1, modes=modes)])
    started = time.perf_counter()
    with patch('lab.allocation.response', side_effect=lambda z, t, n: fv_test_hook(z, t, n, dimension)):
        decision = verify(original, outcome, 'D')
    verification_seconds = time.perf_counter() - started
    started = time.perf_counter()
    operational_digest = request_hash(original)
    hash_accepted = original_digest == operational_digest
    hash_seconds = time.perf_counter() - started
    evidence = probes(original, original.mean_time, dimension)
    old = evidence['old']['probes'].values()
    checks = dict(hash=hash_accepted,
        normalization=all(p['normalization_accepted'] for p in old),
        first_moment=all(p['first_moment_accepted'] for p in old))
    checks['old_invariants'] = checks['normalization'] and checks['first_moment']
    for key in ('second_moment', 'high_frequency'):
        checks[key] = all(p['accepted'] for p in evidence[key]['probes'].values())
    checks['all'] = all(checks.values())
    accepted = {'D': decision['accepted'], **checks,
                **{'D_' + key: decision['accepted'] and verdict for key, verdict in checks.items()}}
    # Oracle is strictly diagnostic, after ALL standalone and combined decisions.
    reference = complex(analytic(original.z, original.mean_time))
    error = abs(value - reference)
    return dict(geometry=geometry, dimension=dimension, original_request=encode(original),
        operational_request=encode(original), original_digest=original_digest,
        operational_digest=operational_digest, outcome=outcome,
        artifact_status='explicit_experimental_untrusted_outcome', decision=decision,
        probes=evidence, accepted=accepted, reference=pair(reference), absolute_error=error,
        accurate=bool(error <= original.tolerance), execution_seconds=execution_seconds,
        verification_seconds=verification_seconds, hash_seconds=hash_seconds)


def summarize(rows):
    summary = {}
    for geometry in GEOMETRIES:
        group = [r for r in rows if r['geometry'] == geometry]
        policies = {}
        for policy in POLICIES:
            policies[policy] = dict(accepted=sum(r['accepted'][policy] for r in group),
                rejected=sum(not r['accepted'][policy] for r in group),
                false_acceptances=sum(r['accepted'][policy] and not r['accurate'] for r in group),
                accurate_rejected=sum(not r['accepted'][policy] and r['accurate'] for r in group),
                inaccurate_rejected=sum(not r['accepted'][policy] and not r['accurate'] for r in group),
                clean_false_rejected=sum(not r['accepted'][policy] for r in group) if geometry == 'clean' else None)
        summary[geometry] = dict(tasks=len(group), accurate=sum(r['accurate'] for r in group), policies=policies)
    return summary


def run(replay=None, runtime_override=None):
    sources = ['lab/geometry_invariants.py', 'lab/common_mode.py', 'lab/replay.py',
        'lab/experiment_audit.py', 'lab/fault_holdout.py', 'lab/allocation.py',
        'lab/executor.py', 'lab/guard.py', 'lab/finite_volume.py', 'lab/scientist.py',
        'vendor/sphere-thermal-cli/sphere_thermal/response.py']
    audit = metadata('experiments/E010-geometry-invariants.md', sources, runtime_override)
    saved = None
    if replay is not None:
        saved, requests, audit['replay'] = load_replay(replay, 'E010', 96)
    rng = np.random.default_rng(20261120)
    order = np.random.default_rng(20261121)
    rows, intake = [], []
    for task_id in range(96):
        if saved is None:
            t = 10 ** rng.uniform(-2, 2)
            magnitude = 10 ** rng.uniform(-6, 5)
            phase = rng.uniform(-np.pi / 2, np.pi / 2)
            tolerance = 10 ** rng.uniform(-4, -1)
            original = Request(magnitude * np.exp(1j * phase) / t, t, tolerance)
        else:
            original = requests[task_id]
        digest = request_hash(original)
        intake.append(dict(task_id=task_id, request=encode(original), request_sha256=digest))
        for geometry in order.permutation(tuple(GEOMETRIES)):
            rows.append(dict(task_id=task_id, **evaluate(original, digest, str(geometry))))
    work = dict(execution_modal_terms=sum(r['outcome']['modes_spent'] for r in rows),
                decision_fv_cells=sum(r['decision']['independent']['cells_solved'] for r in rows))
    for group in ('old', 'second_moment', 'high_frequency'):
        work[group] = {key: sum(r['probes'][group][key] for r in rows)
                      for key in ('modal_terms', 'finite_volume_cells', 'seconds')}
    for key in ('execution_seconds', 'verification_seconds', 'hash_seconds'):
        work[key] = sum(r[key] for r in rows)
    return dict(experiment='E010', **audit, task_seed=20261120, order_seed=20261121,
        paired_tasks=96, sampling_unit='task; three paired geometries and policies are dependent',
        accounting='One artifact and D run per geometry; no response cache or discarded probes. Only eigenvalue constants are cached; construction included in solver timing.',
        intake_channel=intake, runs=rows, summary=summarize(rows), work=work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    protect_output(args.output, args.replay)
    result = run(args.replay, args.runtime_override)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
