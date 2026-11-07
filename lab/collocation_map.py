"""E015 fixed tolerance planes sharing fresh collocation evaluations."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
from lab import coverage_map as base
from lab import radial_collocation as solver
from lab.experiment_audit import metadata

PROTOCOL = 'experiments/E015-collocation-map.md'
SOURCES = ['lab/collocation_map.py', 'lab/radial_collocation.py',
           'lab/coverage_map.py', 'lab/scientist.py', 'lab/experiment_audit.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py']
TOLERANCES = tuple(10.**k for k in range(-7, 0))


def classify(values, reference):
    difference = abs(values[1] - values[0])
    rounding = 32 * np.finfo(float).eps * 128**4 * max(1, abs(values[1]))
    uncertainty = 2 * difference + rounding
    error = abs(values[1] - reference)
    finite = all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values)
    steps = []
    for tolerance in TOLERANCES:
        converging = bool(finite and difference <= tolerance / 8)
        verdict = 'inconclusive'
        if converging and error + uncertainty <= tolerance:
            verdict = 'accept'
        elif converging and error - uncertainty > tolerance:
            verdict = 'violation_detected'
        steps.append(dict(tolerance=tolerance, converging=converging,
                          covered=bool(error <= uncertainty), oracle_verdict=verdict,
                          floor_blocks_acceptance=bool(uncertainty > tolerance)))
    return dict(difference=difference, rounding=rounding, uncertainty=uncertainty,
                error=error, rounding_only=bool(2*difference < error <= uncertainty),
                steps=steps)


def evaluate(key, level, reference):
    logq, phase, q = base.coordinate(key)
    values = [solver.response(q, 1., n) for n in (64, 128)]
    return dict(coordinate=list(key), level=level, log10_magnitude=logq,
                phase=phase, q=base.pair(q), reference=base.pair(reference),
                values=[base.pair(v) for v in values], **classify(values, reference))


def controls():
    rows = []
    for mag in (.01, 1., 1e4):
        for phase in (0., .7):
            q = mag * np.exp(1j*phase)
            for degree in (64, 128):
                a = solver.response(q, 1., degree)
                for t in (.125, 3., 8.):
                    b = solver.response(q/t, t, degree)
                    allowance = 32*np.finfo(float).eps*degree**4*max(1, abs(a))
                    rows.append(dict(kind='dimensionless', q=base.pair(q), degree=degree,
                        time=t, error=abs(a-b), allowance=allowance,
                        passed=bool(abs(a-b) <= allowance)))
                solver.operators.cache_clear()
                b = solver.response(q, 1., degree)
                rows.append(dict(kind='fresh_operator', q=base.pair(q), degree=degree,
                                 passed=a == b, error=abs(a-b)))
    return rows


def region(rows):
    return dict(count=len(rows),
        log10_min=min((r['log10_magnitude'] for r in rows), default=None),
        log10_max=max((r['log10_magnitude'] for r in rows), default=None),
        phase_min=min((r['phase'] for r in rows), default=None),
        phase_max=max((r['phase'] for r in rows), default=None),
        levels={str(k):sum(r['level'] == k for r in rows) for k in range(4)})


def summarize(records):
    planes = {}
    for i, tolerance in enumerate(TOLERANCES):
        planes[str(tolerance)] = {
            'dangerous_misses': region([r for r in records if r['steps'][i]['converging'] and not r['steps'][i]['covered']]),
            'nonconverging': region([r for r in records if not r['steps'][i]['converging']]),
            'covered_nonconverging': region([r for r in records if not r['steps'][i]['converging'] and r['steps'][i]['covered']]),
            'rounding_only': region([r for r in records if r['rounding_only']]),
            'floor_blocks_acceptance': region([r for r in records if r['steps'][i]['floor_blocks_acceptance']]),
            'oracle_verdicts': {v:sum(r['steps'][i]['oracle_verdict'] == v for r in records)
                                for v in ('accept', 'inconclusive', 'violation_detected')}}
    return dict(points=len(records), planes=planes,
                uncovered=region([r for r in records if not r['steps'][0]['covered']]),
                max_error_over_U=max(r['error']/r['uncertainty'] for r in records))


def boundaries(records):
    columns = {}
    for r in records:
        columns.setdefault(r['coordinate'][1], []).append(r)
    result = []
    for y, rows in sorted(columns.items()):
        rows.sort(key=lambda r:r['coordinate'][0])
        for i, tolerance in enumerate(TOLERANCES):
            states = [(r['steps'][i]['converging'], r['steps'][i]['covered'],
                       r['steps'][i]['oracle_verdict'], r['rounding_only']) for r in rows]
            transitions = [dict(left=rows[j-1]['coordinate'], right=rows[j]['coordinate'],
                                left_state=states[j-1], right_state=states[j])
                           for j in range(1, len(rows)) if states[j] != states[j-1]]
            result.append(dict(y=y, tolerance=tolerance, first=rows[0]['coordinate'],
                               last=rows[-1]['coordinate'], transitions=transitions))
    return result


def run(directory, runtime_override=None):
    output = Path(directory)
    if output.exists():
        raise ValueError('Output directory already exists')
    meta = metadata(PROTOCOL, SOURCES, runtime_override)
    output.mkdir(parents=True)
    base.write(output/'metadata.json', meta)
    checks = controls()
    base.write(output/'controls.json', checks)
    if not all(r['passed'] for r in checks):
        raise RuntimeError('Controls failed; map not evaluated')
    rng = np.random.default_rng(20261128)
    active = base.base_cells()
    records, refinements, batches = {}, [], []
    started = time.perf_counter()
    for level in range(4):
        keys = sorted({k for c in active for k in base.corners(c)} - records.keys())
        refs, audit = base.reference_batch(keys, rng)
        base.write(output/f'reference-level-{level}.json.gz', audit)
        for k in keys:
            records[k] = evaluate(k, level, refs[k])
        batches.append(dict(level=level, new_points=len(keys), audit_count=audit['audited'],
            reference_failures=audit['failed'], fallback=audit['fallback'], max_reference_error=audit['max_error']))
        print(batches[-1], flush=True)
        flagged = [c for c in active if base.needs_refinement(c, records)] if level < 3 else []
        refinements.append(dict(level=level, cells_tested=len(active), refined_cells=flagged))
        active = [child for c in flagged for child in base.children(c)]
    ordered = [records[k] for k in sorted(records)]
    summary = summarize(ordered)
    base.write(output/'map.json.gz', dict(experiment='E015', **meta, points=ordered,
        batches=batches, refinements=refinements, summary=summary, seconds=time.perf_counter()-started))
    base.write(output/'boundaries.json.gz', boundaries(ordered))
    base.write(output/'summary.json', dict(**summary, batches=batches))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--runtime-override')
    args = parser.parse_args()
    print(run(args.directory, args.runtime_override))
