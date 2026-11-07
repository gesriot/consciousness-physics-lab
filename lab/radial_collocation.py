"""Independent sphere boundary-value solver from the transformed radial equation."""
from functools import lru_cache
import math
import time
import numpy as np
from scipy.linalg import solve


@lru_cache(maxsize=2)
def operators(degree):
    if degree not in (64, 128):
        raise ValueError('Registered degrees are 64 and 128')
    n = degree
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    r = (1 + x) / 2
    c = np.ones(n + 1)
    c[[0, -1]] = 2
    c *= (-1.) ** np.arange(n + 1)
    dx = x[:, None] - x[None, :]
    derivative = (c[:, None] / c[None, :]) / (dx + np.eye(n + 1))
    derivative -= np.diag(derivative.sum(axis=1))
    second = 4 * derivative @ derivative
    weights = np.zeros(n + 1)
    weights[[0, -1]] = 1 / (n*n - 1)
    v = np.ones(n - 1)
    for k in range(1, n // 2):
        v -= 2 * np.cos(2 * k * theta[1:n]) / (4*k*k - 1)
    v -= np.cos(n * theta[1:n]) / (n*n - 1)
    weights[1:n] = 2 * v / n
    weights /= 2
    for value in (r, second, weights):
        value.setflags(write=False)
    return r, second, weights


def response(z, mean_time, degree):
    z = complex(z)
    if not (math.isfinite(z.real) and math.isfinite(z.imag) and z.real >= 0
            and math.isfinite(mean_time) and mean_time > 0):
        raise ValueError('Finite right-half-plane input and positive time required')
    r, second, weights = operators(degree)
    matrix = np.asarray(second, dtype=complex) - 15 * z * mean_time * np.eye(degree + 1)
    matrix[0, :] = 0
    matrix[-1, :] = 0
    matrix[0, 0] = matrix[-1, -1] = 1
    rhs = np.zeros(degree + 1, dtype=complex)
    rhs[0] = 1
    w = solve(matrix, rhs, assume_a='gen', check_finite=True)
    return complex(3 * np.dot(weights, r * w))


def check(z, mean_time, value, tolerance):
    started = time.perf_counter()
    values = [response(z, mean_time, degree) for degree in (64, 128)]
    difference = abs(values[1] - values[0])
    rounding = 32 * np.finfo(float).eps * 128**4 * max(1, abs(values[1]))
    uncertainty = 2 * difference + rounding
    converging = bool(all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values)
                      and difference <= tolerance / 8)
    discrepancy = abs(value - values[1])
    verdict = 'inconclusive'
    if converging and discrepancy + uncertainty <= tolerance:
        verdict = 'accept'
    elif converging and discrepancy - uncertainty > tolerance:
        verdict = 'violation_detected'
    return dict(values=[[v.real, v.imag] for v in values], degrees=[64, 128],
                difference=float(difference), rounding=float(rounding),
                uncertainty=float(uncertainty), discrepancy=float(discrepancy),
                converging=converging, verdict=verdict, matrix_solves=2,
                seconds=time.perf_counter() - started)
