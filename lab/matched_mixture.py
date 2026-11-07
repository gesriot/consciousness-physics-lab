"""E013 experiment-only common slab-mixture faults and independent sphere defense."""
import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import time
from unittest.mock import patch
import mpmath as mp
import numpy as np
from lab.allocation import propose_modes
from lab.common_mode import encode
from lab.executor import request_hash
from lab.experiment_audit import metadata
from lab.fault_holdout import verify
from lab.geometry_invariants import modal_test_hook, fv_test_hook, probes, pair
from lab.guard import Request
from lab.radial_collocation import check as third_check
from lab.replay import load_replay, protect_output, canonical_sha256
from lab.scientist import analytic

CONDITIONS = ('clean', 'mixture', 'slab')
POLICIES = ('D', 'D_gates', 'D_gates_third', 'third')
PROTOCOL = 'experiments/E013-matched-mixture.md'
SOURCES = ['lab/matched_mixture.py', 'lab/radial_collocation.py',
           'lab/allocation.py', 'lab/fault_holdout.py', 'lab/geometry_invariants.py',
           'lab/common_mode.py', 'lab/executor.py', 'lab/experiment_audit.py',
           'lab/guard.py', 'lab/finite_volume.py', 'lab/replay.py', 'lab/scientist.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py']


@lru_cache(maxsize=1)
def derive_parameters():
    with mp.workdps(60):
        def equations(a, x, y):
            return (a*x+(1-a)*y-1, a*x*x+(1-a)*y*y-mp.mpf(25)/21,
                    a/mp.sqrt(x)+(1-a)/mp.sqrt(y)-3/mp.sqrt(5))
        result = mp.findroot(equations, (.19, .098, 1.21), tol=mp.mpf('1e-55'))
        a, x, y = result
        return dict(weights=[mp.nstr(a, 60), mp.nstr(1-a, 60)],
                    times=[mp.nstr(x, 60), mp.nstr(y, 60)],
                    residuals=[mp.nstr(v, 60) for v in equations(a, x, y)])


def derive_three_slabs():
    with mp.workdps(60):
        x = mp.mpf('.03')
        def equations(a, b, c, y, z):
            return (a+b+c-1, a*x+b*y+c*z-1,
                    a*x*x+b*y*y+c*z*z-mp.mpf(25)/21,
                    a/mp.sqrt(x)+b/mp.sqrt(y)+c/mp.sqrt(z)-3/mp.sqrt(5),
                    a*x**3+b*y**3+c*z**3-mp.mpf(25)/17)
        try:
            roots = mp.findroot(equations, (.06,.25,.69,.6,1.3),
                                tol=mp.mpf('1e-50'), maxsteps=200)
            a, b, c, y, z = roots
            valid = all(mp.im(v) == 0 and mp.re(v) > 0 for v in (a,b,c,x,y,z))
            return dict(positive=bool(valid), weights=[mp.nstr(v,60) for v in (a,b,c)],
                        times=[mp.nstr(v,60) for v in (x,y,z)],
                        residuals=[mp.nstr(v,60) for v in equations(*roots)])
        except (ValueError, ZeroDivisionError) as error:
            return dict(positive=False, failure=str(error))


def components(condition):
    if condition == 'clean':
        return [(1., 1., 3)]
    if condition == 'slab':
        return [(1., 1., 1)]
    if condition == 'mixture':
        p = derive_parameters()
        return [(float(w), float(t), 1) for w, t in zip(p['weights'], p['times'])]
    raise ValueError('Unregistered condition')


def model_response(kind, z, t, size, condition):
    solver = modal_test_hook if kind == 'modal' else fv_test_hook
    return sum(w * solver(z, t * x, size, dimension)
               for w, x, dimension in components(condition))


def evaluate(request, digest, condition):
    count = len(components(condition))
    modes = propose_modes(request, .5)
    started = time.perf_counter()
    value = model_response('modal', request.z, request.mean_time, modes, condition)
    execution_seconds = time.perf_counter() - started
    outcome = dict(value=pair(value), calls=1, modes_spent=modes,
                   trace=[dict(call=1, modes=modes)])
    started = time.perf_counter()
    with patch('lab.allocation.response', side_effect=lambda z,t,n:
               model_response('fv', z,t,n,condition)):
        decision = verify(request, outcome, 'D')
    verification_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual_digest = request_hash(request)
    hash_seconds = time.perf_counter() - started
    with patch('lab.geometry_invariants.modal_test_hook', side_effect=lambda z,t,n,d:
               model_response('modal', z,t,n,condition)), \
         patch('lab.geometry_invariants.fv_test_hook', side_effect=lambda z,t,n,d:
               model_response('fv', z,t,n,condition)):
        evidence = probes(request, request.mean_time, 3)
    old = evidence['old']['probes'].values()
    checks = dict(hash=actual_digest == digest,
                  normalization=all(p['normalization_accepted'] for p in old),
                  first_moment=all(p['first_moment_accepted'] for p in old),
                  second_moment=all(p['accepted'] for p in evidence['second_moment']['probes'].values()),
                  high_frequency=all(p['accepted'] for p in evidence['high_frequency']['probes'].values()))
    third = third_check(request.z, request.mean_time, value, request.tolerance)
    base = 'accept' if decision['accepted'] else 'reject'
    gated = base if all(checks.values()) else 'reject'
    verdicts = dict(D=base, D_gates=gated,
                    D_gates_third=third['verdict'] if gated == 'accept' else 'reject',
                    third=third['verdict'])
    work = dict(artifact_modal_terms=modes*count,
                decision_fv_cells=decision['independent']['cells_solved']*count,
                probes_modal_terms=sum(p['modal_terms'] for p in evidence.values())*count,
                probes_fv_cells=sum(p['finite_volume_cells'] for p in evidence.values())*count,
                third_matrix_solves=third['matrix_solves'], third_sum_degrees=192)
    return dict(condition=condition, outcome=outcome, original_digest=digest,
                operational_digest=actual_digest, components=count, decision=decision,
                probes=evidence, checks=checks, third=third, verdicts=verdicts,
                accepted={p: v == 'accept' for p,v in verdicts.items()}, work=work,
                execution_seconds=execution_seconds, verification_seconds=verification_seconds,
                hash_seconds=hash_seconds)


def policy_costs(rows):
    costs = {}
    for policy in POLICIES:
        use_d = policy != 'third'
        use_gates = policy in ('D_gates', 'D_gates_third')
        use_third = policy in ('third', 'D_gates_third')
        costs[policy] = dict(
            modal_terms=sum(r['work']['artifact_modal_terms'] +
                            (r['work']['probes_modal_terms'] if use_gates else 0) for r in rows),
            fv_cells=sum((r['work']['decision_fv_cells'] if use_d else 0) +
                         (r['work']['probes_fv_cells'] if use_gates else 0) for r in rows),
            third_matrix_solves=sum(r['work']['third_matrix_solves'] for r in rows) if use_third else 0,
            seconds=sum(r['execution_seconds'] +
                        (r['verification_seconds'] if use_d else 0) +
                        (r['hash_seconds']+sum(p['seconds'] for p in r['probes'].values()) if use_gates else 0) +
                        (r['third']['seconds'] if use_third else 0) for r in rows))
    return costs


def summarize(rows):
    summary = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition'] == condition]
        policies = {}
        for p in POLICIES:
            policies[p] = dict(
                **{v: sum(r['verdicts'][p] == v for r in group)
                   for v in ('accept','reject','violation_detected','inconclusive')},
                nonaccepted=sum(not r['accepted'][p] for r in group),
                false_acceptances=sum(r['accepted'][p] and not r['accurate'] for r in group),
                accurate_rejected=sum(r['accurate'] and r['verdicts'][p] in ('reject','violation_detected') for r in group),
                accurate_nonaccepted=sum(r['accurate'] and not r['accepted'][p] for r in group),
                inaccurate_rejected=sum(not r['accurate'] and r['verdicts'][p] in ('reject','violation_detected') for r in group),
                clean_false_rejections=sum(r['accurate'] and r['verdicts'][p] in ('reject','violation_detected') for r in group) if condition == 'clean' else None)
        summary[condition] = dict(tasks=len(group), accurate=sum(r['accurate'] for r in group),
                                 third_inconclusive=sum(r['third']['verdict']=='inconclusive' for r in group),
                                 third_inconclusive_fraction=sum(r['third']['verdict']=='inconclusive' for r in group)/len(group),
                                 gate_failures={k:sum(not r['checks'][k] for r in group) for k in group[0]['checks']},
                                 policies=policies, policy_costs=policy_costs(group))
    return summary


def overlap_audit(intake):
    def key(record):
        q = record['request']
        return tuple(float(v).hex() for v in (*q['z'], q['mean_time'], q['tolerance']))
    current = {}
    for record in intake:
        current.setdefault(key(record), []).append(record['task_id'])
    comparisons = {}
    for experiment, path, field in (
        ('E009', 'results/surface-order/results.json', 'tasks'),
        ('E010', 'results/E010-geometry-invariants/results.json', 'intake_channel'),
        ('E011', 'results/E011-extrapolation/results.json', 'intake_channel')):
        raw = Path(path).read_bytes()
        earlier = json.loads(raw)[field]
        matches = [dict(current_task_ids=current[key(record)], earlier_task_id=record['task_id'])
                   for record in earlier if key(record) in current]
        comparisons[experiment] = dict(source=path, source_sha256=hashlib.sha256(raw).hexdigest(),
                                       earlier_tasks=len(earlier), exact_request_matches=len(matches), matches=matches)
    return dict(definition='Exact binary64 real/imaginary z, mean_time and tolerance; budget4096 throughout',
                within_E013_duplicate_requests=len(intake)-len(current), comparisons=comparisons)


def run(replay=None, runtime_override=None, intake_path=None):
    audit = metadata(PROTOCOL, SOURCES, runtime_override)
    parameters = derive_parameters()
    three = derive_three_slabs()
    rng = np.random.default_rng(20261125)
    order = np.random.default_rng(20261126)
    saved = None
    if replay is not None:
        saved, requests, audit['replay'] = load_replay(replay, 'E013', 96)
    intake, rows, generated = [], [], []
    for task_id in range(96):
        cohort = 'broad' if task_id < 48 else 'intermediate'
        if saved is None:
            t = 10 ** rng.uniform(-2,2)
            magnitude = 10 ** rng.uniform(-6 if task_id < 48 else -1, 6 if task_id < 48 else 3)
            phase = rng.uniform(-np.pi/2, np.pi/2)
            tol = 10 ** rng.uniform(-4,-1)
            request = Request(magnitude*np.exp(1j*phase)/t, t, tol)
        else:
            request = requests[task_id]
        digest = request_hash(request)
        intake.append(dict(task_id=task_id, cohort=cohort, request=encode(request), request_sha256=digest))
        generated.append(request)
    if intake_path is not None:
        Path(intake_path).parent.mkdir(parents=True, exist_ok=True)
        Path(intake_path).write_text(json.dumps(dict(experiment='E013', intake_channel=intake,
            intake_sha256=canonical_sha256(intake), **audit), indent=2, allow_nan=False)+'\n')
    for record, request in zip(intake, generated):
        task_id, cohort, digest = record['task_id'], record['cohort'], record['request_sha256']
        local = [dict(task_id=task_id, cohort=cohort, **evaluate(request,digest,str(c)))
                 for c in order.permutation(CONDITIONS)]
        reference = complex(analytic(request.z, request.mean_time))
        for row in local:
            error = abs(complex(*row['outcome']['value']) - reference)
            row.update(reference=pair(reference), absolute_error=error,
                       tolerance=request.tolerance, accurate=bool(error <= request.tolerance))
        rows.extend(local)
    work = {k:sum(r['work'][k] for r in rows) for k in rows[0]['work']}
    for key in ('execution_seconds','verification_seconds','hash_seconds'):
        work[key] = sum(r[key] for r in rows)
    work['probe_seconds'] = sum(g['seconds'] for r in rows for g in r['probes'].values())
    work['third_seconds'] = sum(r['third']['seconds'] for r in rows)
    return dict(experiment='E013', **audit, task_seed=20261125, order_seed=20261126,
                paired_tasks=96, sampling_unit='task; three conditions and four policies are paired',
                two_slab_parameters=parameters, three_slab_construction=three,
                request_overlap=overlap_audit(intake),
                intake_channel=intake, intake_sha256=canonical_sha256(intake), runs=rows,
                summary=summarize(rows), cohort_summary={c:summarize([r for r in rows if r['cohort']==c])
                                                        for c in ('broad','intermediate')}, work=work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    protect_output(args.output, args.replay)
    intake_path = args.output.with_name(args.output.stem+'-intake.json')
    protect_output(intake_path, args.replay)
    result = run(args.replay, args.runtime_override, intake_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))
