"""E009 nested mesh and factorial verifier; no oracle in decision functions."""
import math
import numpy as np
from scipy.linalg import solve_banded
from lab.finite_volume import response as uniform_response
from lab.three_way import GRIDS, classify, ratio, step_evidence as frozen_step

CELLS = ('uniform/frozen', 'surface/frozen', 'uniform/order', 'surface/order')


def faces(cells):
    if cells not in GRIDS:
        raise ValueError('Registered dyadic grid required')
    base = -np.expm1(-4 * np.arange(65) / 64) / -np.expm1(-4.)
    subdivisions = cells // 64
    return np.concatenate([np.linspace(a, b, subdivisions + 1)[:-1]
                           for a, b in zip(base[:-1], base[1:])] + [np.array([1.])])


def surface_response(z, mean_time, cells):
    z = complex(z)
    if not (math.isfinite(z.real) and math.isfinite(z.imag) and z.real >= 0
            and math.isfinite(mean_time) and mean_time > 0):
        raise ValueError('Finite right-half-plane request and positive time required')
    edges = faces(cells)
    centres = (edges[:-1] + edges[1:]) / 2
    volumes = np.diff(edges**3) / 3
    kappa = 1 / (15 * mean_time)
    conductance = kappa * edges[1:-1]**2 / np.diff(centres)
    boundary = kappa / (1 - centres[-1])
    diagonal = (z * volumes).astype(complex)
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


def solver(mesh):
    if mesh == 'uniform':
        return uniform_response
    if mesh == 'surface':
        return surface_response
    raise ValueError('Unknown mesh')


def evidence(values, grid, value, tolerance, estimator):
    step = frozen_step(values, grid, value, tolerance)
    if estimator == 'frozen':
        return dict(step, observed_order=None, uncertainty_fallback=False)
    if estimator != 'order':
        raise ValueError('Unknown estimator')
    diffs = [abs(b-a) for a, b in zip(values, values[1:])]
    p = math.log2(diffs[-2] / diffs[-1]) if min(diffs[-2:]) > 0 else None
    fallback = p is None or not math.isfinite(p) or p <= 0
    if fallback:
        p = p if p is not None and math.isfinite(p) else None
        uncertainty = step['uncertainty_estimate']
    else:
        rounding = 32 * np.finfo(float).eps * grid**2 * max(1, abs(values[-1]))
        uncertainty = 2 * diffs[-1] / math.expm1(p * math.log(2)) + rounding
    converging = p is not None and 1.5 <= p <= 2.5
    verdict = classify(converging, step['discrepancy'], uncertainty, tolerance)
    futile = (grid in (1024, 2048) and verdict == 'inconclusive' and uncertainty >= 4*tolerance
              and ratio(diffs[-1], diffs[-2]) >= .5 and ratio(diffs[-2], diffs[-3]) >= .5)
    return dict(last_grid=grid, converging=converging, uncertainty_estimate=float(uncertainty),
                discrepancy=step['discrepancy'], verdict=verdict, futile=bool(futile),
                observed_order=p, uncertainty_fallback=fallback)


def check(request, value, mesh, estimator):
    values, steps, grids = [], [], []
    response = solver(mesh)
    for grid in GRIDS:
        grids.append(grid)
        values.append(response(request.z, request.mean_time, grid))
        if grid >= 512:
            step = evidence(values, grid, value, request.tolerance, estimator)
            steps.append(step)
            if step['verdict'] != 'inconclusive' or step['futile']:
                break
    verdict = steps[-1]['verdict']
    return dict(accepted=verdict == 'accept', verdict=verdict,
                stop_reason='decisive' if verdict != 'inconclusive' else
                ('futility' if steps[-1]['futile'] else 'grid_limit'),
                grids=grids, values=[[v.real, v.imag] for v in values], steps=steps,
                cells_solved=sum(grids))


def shadow(request, value, values, estimator):
    steps = [evidence(values[:i+1], n, value, request.tolerance, estimator)
             for i, n in enumerate(GRIDS) if n >= 512]
    first = next((s for s in steps if s['verdict'] != 'inconclusive'), steps[-1])
    return dict(steps=steps, any_accept=any(s['verdict'] == 'accept' for s in steps),
                any_violation=any(s['verdict'] == 'violation_detected' for s in steps),
                first_decisive=first['verdict'], first_decisive_grid=first['last_grid'],
                terminal_verdict=steps[-1]['verdict'],
                decisive_reversal=any(s['verdict'] not in ('inconclusive', first['verdict']) for s in steps))
