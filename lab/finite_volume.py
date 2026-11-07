"""Independent radial finite-volume Laplace solver, prescribed surface value 1."""
import math
import numpy as np
from scipy.linalg import solve_banded


def response(z, mean_time, cells):
    if type(cells) is not int or not 4 <= cells <= 65536:
        raise ValueError('cells must be an integer in [4,65536]')
    z = complex(z)
    if not (math.isfinite(z.real) and math.isfinite(z.imag) and z.real >= 0):
        raise ValueError('finite z with Re z >= 0 required')
    if not math.isfinite(mean_time) or mean_time <= 0:
        raise ValueError('positive finite mean_time required')
    h = 1 / cells
    faces = np.linspace(0, 1, cells + 1)
    volumes = np.diff(faces**3) / 3
    kappa = 1 / (15 * mean_time)  # nondimensional sphere radius = 1
    conductance = kappa * faces[1:-1]**2 / h
    boundary = 2 * kappa / h  # surface is half a cell from final centre
    diagonal = z * volumes
    diagonal = diagonal.astype(complex)
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


def convergence(z, mean_time):
    grids = (64, 128, 256, 512)
    values = [response(z, mean_time, cells) for cells in grids]
    differences = [abs(b-a) for a,b in zip(values, values[1:])]
    # Diffusion matrix conditioning grows roughly quadratically with grid size.
    # This scale-aware floor is heuristic, not a certified backward-error bound.
    rounding = 32 * np.finfo(float).eps * grids[-1]**2 * max(1, abs(values[-1]))
    converging = differences[-1] <= max(.5 * differences[-2], rounding)
    return {'grids': list(grids), 'values': [[v.real,v.imag] for v in values],
            'differences': differences, 'converging': bool(converging),
            'uncertainty_estimate': 2 * differences[-1] + rounding}


def check_value(z, mean_time, tolerance, value):
    value = complex(value)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive finite tolerance required')
    evidence = convergence(z, mean_time)
    discrepancy = abs(value - complex(*evidence['values'][-1]))
    accepted = evidence['converging'] and math.isfinite(discrepancy) and discrepancy + evidence['uncertainty_estimate'] <= tolerance
    return {'accepted': bool(accepted), 'discrepancy': discrepancy, **evidence}
