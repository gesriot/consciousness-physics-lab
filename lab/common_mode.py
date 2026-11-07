"""E008 explicit TEST HOOKS for shared request/model faults; not production API."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from lab.scientist import analytic, modal_partial_sums
from lab.guard import Request
from lab.executor import request_hash
from lab.allocation import propose_modes
from lab.fault_holdout import verify
from lab.finite_volume import response
from lab.experiment_audit import metadata
from lab.replay import load_replay, protect_output

FAULTS = ('clean', 'request_time', 'request_conjugate', 'model_scale')
POLICIES = ('D', 'D_hash', 'D_invariants', 'D_both')


def encode(request):
    result = asdict(request)
    result['z'] = [complex(request.z).real, complex(request.z).imag]
    return result


def corrupt(original, fault):
    if fault == 'request_time':
        return replace(original, mean_time=1000 * original.mean_time), 1
    if fault == 'request_conjugate':
        return replace(original, z=original.z.conjugate()), 1
    if fault == 'model_scale':
        return original, 1000
    if fault == 'clean':
        return original, 1
    raise ValueError('unregistered common-mode fault')


def modal_test_hook(z, mean_time, modes, scale):
    return complex(modal_partial_sums(z, mean_time * scale, [modes])[0])


def fv_test_hook(z, mean_time, cells, scale):
    return response(z, mean_time * scale, cells)


def invariant_defense(request, original_time, scale):
    h = 1e-6 / original_time
    probes = {}
    for name, solver, size in (('modal', modal_test_hook, 4096),
                               ('finite_volume', fv_test_hook, 512)):
        zero = solver(0, request.mean_time, size, scale)
        shifted = solver(1j * h, request.mean_time, size, scale)
        estimated = -shifted.imag / h
        normalization_error = abs(zero - 1)
        mean_relative_error = abs(estimated / original_time - 1)
        probes[name] = dict(zero=[zero.real, zero.imag],
            shifted=[shifted.real, shifted.imag], estimated_mean=estimated,
            normalization_error=normalization_error,
            mean_relative_error=mean_relative_error,
            accepted=bool(normalization_error <= 2e-4 and mean_relative_error <= 2e-3))
    return dict(h=h, normalization_allowance=2e-4, mean_relative_allowance=2e-3,
                probes=probes, accepted=all(p['accepted'] for p in probes.values()),
                modal_terms=8192, finite_volume_cells=1024)


def evaluate(original, original_digest, fault):
    request, scale = corrupt(original, fault)
    modes = propose_modes(request, .5)
    started = time.perf_counter()
    value = modal_test_hook(request.z, request.mean_time, modes, scale)
    execution_seconds = time.perf_counter() - started
    outcome = dict(value=[value.real, value.imag], calls=1, modes_spent=modes,
                   trace=[dict(call=1, modes=modes)])
    started = time.perf_counter()
    # Scoped experiment-only dependency substitution. The production algorithm,
    # U, grids, guard and caller's Request are unchanged.
    with patch('lab.allocation.response',
               side_effect=lambda z, t, n: fv_test_hook(z, t, n, scale)):
        decision = verify(request, outcome, 'D')
    verification_seconds = time.perf_counter() - started
    started = time.perf_counter()
    operational_digest = request_hash(request)
    hash_accepted = operational_digest == original_digest
    invariants = invariant_defense(request, original.mean_time, scale)
    defense_seconds = time.perf_counter() - started
    accepted = {'D': decision['accepted'],
                'D_hash': decision['accepted'] and hash_accepted,
                'D_invariants': decision['accepted'] and invariants['accepted'],
                'D_both': decision['accepted'] and hash_accepted and invariants['accepted']}
    # The only oracle call is after all decisions and probes.
    reference = complex(analytic(original.z, original.mean_time))
    error = abs(value - reference)
    return dict(fault=fault, original_request=encode(original),
        operational_request=encode(request), original_digest=original_digest,
        operational_digest=operational_digest, model_scale_test_hook=scale,
        artifact_status='explicit_experimental_untrusted_outcome', outcome=outcome,
        decision=decision, hash_accepted=hash_accepted, invariants=invariants,
        accepted=accepted, reference=[reference.real, reference.imag],
        absolute_error=error, accurate=bool(error <= original.tolerance),
        execution_seconds=execution_seconds, verification_seconds=verification_seconds,
        defense_seconds=defense_seconds)


def summarize(rows):
    summary = {}
    for fault in FAULTS:
        group = [r for r in rows if r['fault'] == fault]
        summary[fault] = dict(tasks=len(group), accurate=sum(r['accurate'] for r in group),
            hash_rejected=sum(not r['hash_accepted'] for r in group),
            invariants_rejected=sum(not r['invariants']['accepted'] for r in group),
            policies={})
        for policy in POLICIES:
            summary[fault]['policies'][policy] = dict(
                accepted=sum(r['accepted'][policy] for r in group),
                false_acceptances=sum(r['accepted'][policy] and not r['accurate'] for r in group),
                accurate_rejected=sum(not r['accepted'][policy] and r['accurate'] for r in group),
                inaccurate_rejected=sum(not r['accepted'][policy] and not r['accurate'] for r in group))
    return summary


def run(replay=None, runtime_override=None):
    sources = ['lab/common_mode.py', 'lab/replay.py', 'lab/experiment_audit.py', 'lab/fault_holdout.py',
        'lab/allocation.py', 'lab/executor.py', 'lab/guard.py', 'lab/finite_volume.py',
        'lab/scientist.py', 'vendor/sphere-thermal-cli/sphere_thermal/response.py']
    audit = metadata('experiments/E008-common-mode.md', sources, runtime_override)
    saved = None
    if replay is not None:
        saved, requests, audit['replay'] = load_replay(replay, 'E008', 96)
    rng = np.random.default_rng(20261116)
    order = np.random.default_rng(20261117)
    rows = []
    intake = []
    for task_id in range(96):
        if saved is None:
            t = 10 ** rng.uniform(-2, 2)
            magnitude = 10 ** rng.uniform(-6, 5)
            phase = rng.uniform(-np.pi / 2, np.pi / 2)
            tolerance = 10 ** rng.uniform(-4, -1)
            original = Request(magnitude * np.exp(1j * phase) / t, t, tolerance)
        else:
            original = requests[task_id]
        # Independent intake record is populated before any corruption occurs.
        digest = request_hash(original)
        intake.append(dict(task_id=task_id, request=encode(original), request_sha256=digest))
        for fault in order.permutation(FAULTS):
            row = evaluate(original, digest, str(fault))
            rows.append(dict(task_id=task_id, **row))
    return dict(experiment='E008', **audit, task_seed=20261116, order_seed=20261117,
        paired_tasks=96, sampling_unit='task; four paired conditions and policies are dependent',
        accounting='One artifact and D run per condition shared by four policy comparisons; no probe cache; independent intake saved before corruption.',
        intake_channel=intake, runs=rows, summary=summarize(rows),
        work=dict(execution_modal_terms=sum(r['outcome']['modes_spent'] for r in rows),
                  decision_fv_cells=sum(r['decision']['independent']['cells_solved'] for r in rows),
                  defense_modal_terms=sum(r['invariants']['modal_terms'] for r in rows),
                  defense_fv_cells=sum(r['invariants']['finite_volume_cells'] for r in rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='Read exact requests from archived results.json')
    parser.add_argument('--runtime-override', help='Explicit audit reason for using an unpinned runtime')
    args = parser.parse_args()
    protect_output(args.output, args.replay)
    result = run(args.replay, args.runtime_override)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
