"""Passive heat conduction in an inert sphere with prescribed surface temperature.

The transfer function g(z) maps the surface temperature to the volume-mean
temperature of the sphere for signals proportional to exp(z t), i.e. the
ratio of Laplace transforms with zero initial temperature. Notation of the
dissertation, section 3.2: T = R**2 / (15 kappa) is the mean relaxation time,
q = sqrt(15 T z), and

    g(z; T) = 3 (q coth q - 1) / q**2                         (analytic)
            = sum_{n>=1} w_n / (1 + z tau_n)                   (modal)
    w_n = 6 / (pi**2 n**2),   tau_n = 15 T / (pi**2 n**2).

z is measured in 1/[time] and T in [time]; only x = 15 T z enters.
Only Re z >= 0 is supported. The two representations are evaluated by
separate code paths and never correct each other.
"""
from fractions import Fraction
import math

import numpy as np

PI2 = math.pi**2
EPS = float(np.finfo(float).eps)
# |x| at or below which the Taylor series replaces the cancelling closed form.
SERIES_MAX_ABS_X = 1.0
# |c_k| = 6 zeta(2k+2) / pi**(2k+2); for |x| <= 1 the terms k >= 20 sum to < 1e-20.
SERIES_TERMS = 20
# Larger |x| is rejected: 15 T z must stay far from floating-point overflow.
MAX_ABS_X = 1e300
# exp(-2q) underflows to zero beyond this real part of 2q.
EXP_CUTOFF = 750.0
# Integral of (1 + s**4)**(-1/2) over (0, inf) equals Gamma(1/4)**2 / (4 sqrt(pi)).
QUARTIC_INTEGRAL = math.gamma(0.25)**2 / (4 * math.sqrt(math.pi))
# Heuristic multiple of eps for rounding of g values whose magnitude is <= 1.
ROUNDING_FACTOR = 128
# Modes summed per vectorized chunk: bounds memory for large N.
CHUNK = 1 << 18


def _taylor_coefficients(count):
    """c_k of g = sum_k c_k x**k, from q coth q = sum_k 4**k B_2k q**2k / (2k)!.

    Bernoulli numbers are generated exactly; this does not use the modal series.
    """
    bernoulli = [Fraction(1)]
    for m in range(1, 2 * count + 3):
        bernoulli.append(-sum(math.comb(m + 1, k) * bernoulli[k] for k in range(m)) / (m + 1))
    return np.array([float(3 * 4**(k + 1) * bernoulli[2 * k + 2] / math.factorial(2 * k + 2))
                     for k in range(count)])


TAYLOR = _taylor_coefficients(SERIES_TERMS)


def scaled_argument(z, mean_time):
    """Return x = 15 T z after checking the supported domain."""
    if not (math.isfinite(mean_time) and mean_time > 0):
        raise ValueError("mean_time must be finite and positive")
    z = np.asarray(z, dtype=complex)
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    if np.any(z.real < 0):
        raise ValueError("only Re z >= 0 is supported")
    if np.any(np.abs(z) * (15.0 * mean_time) > MAX_ABS_X):
        raise ValueError(f"|15 T z| must not exceed {MAX_ABS_X:g}")
    return 15.0 * mean_time * z


def _series(x):
    """Taylor branch, accurate for |x| <= SERIES_MAX_ABS_X."""
    acc = np.zeros_like(x)
    for c in TAYLOR[::-1]:
        acc = acc * x + c
    return acc


def _closed_form(x):
    """3 (q coth q - 1) / q**2 written as 3 r (coth q - r), r = 1/q.

    The principal root has Re q >= |q|/sqrt(2) for Re x >= 0, so
    |exp(-2q)| < 1 and coth q = (1 + e)/(1 - e) cannot overflow.
    For |q| >= 1 the denominator satisfies |1 - e| >= 1 - exp(-sqrt(2)).
    """
    q = np.sqrt(x)
    r = 1 / q
    e = np.zeros_like(q)
    keep = 2 * q.real < EXP_CUTOFF
    e[keep] = np.exp(-2 * q[keep])
    return 3 * r * ((1 + e) / (1 - e) - r)


def analytic(z, mean_time):
    """Exact g(z; T) for an array of z with Re z >= 0 (section 3.2 formula)."""
    x = scaled_argument(z, mean_time)
    g = np.empty_like(x)
    small = np.abs(x) <= SERIES_MAX_ABS_X
    g[small] = _series(x[small])
    g[~small] = _closed_form(x[~small])
    return g


def modal_partial_sums(z, mean_time, modes):
    """Partial sums g_N(z) = sum_{n<=N} w_n / (1 + z tau_n) for each N in modes.

    z is a scalar; modes is strictly increasing. The weights are not
    renormalized: the missing tail is part of the truncation error.
    Each term is computed as 6 / (pi**2 n**2 + x), which is algebraically
    identical to w_n / (1 + z tau_n). Every term lies in the same closed
    quadrant, so the summation has no cancellation.
    """
    x = complex(scaled_argument(z, mean_time))
    if any(n < 1 for n in modes) or any(b <= a for a, b in zip(modes, modes[1:])):
        raise ValueError("modes must be positive and strictly increasing")
    chunks, sums, start = [], [], 1
    for stop in modes:
        for lo in range(start, stop + 1, CHUNK):
            n = np.arange(lo, min(lo + CHUNK, stop + 1), dtype=float)
            chunks.append(complex(np.sum(6.0 / (PI2 * n * n + x))))
        start = stop + 1
        sums.append(complex(math.fsum(c.real for c in chunks),
                            math.fsum(c.imag for c in chunks)))
    return np.array(sums)


def tail_bound(z, mean_time, modes):
    """Upper bound of |sum_{n>N} w_n / (1 + z tau_n)| in exact arithmetic.

    For Re x >= 0, |pi**2 n**2 + x| >= sqrt(pi**4 n**4 + X**2), X = |x|.
    A: sum_{n>N} 6/(pi**2 n**2) <= 6/(pi**2 (N + 1/2)), since 1/t**2 is
       convex and the midpoint rule underestimates its integral.
    B: f(t) = 6/sqrt(pi**4 t**4 + X**2) decreases, so the tail is at most
       int_N^inf f = 6/(pi sqrt X) * int_{s0}^inf ds/sqrt(1 + s**4),
       s0 = pi N / sqrt X. With s1 = min(s0, 1), int_0^{s0} >= s1 f(s1),
       hence the last integral is <= Q - s1/sqrt(1 + s1**4).
    Both bounds are non-increasing in N. Rounding is not included.
    """
    x_abs = np.abs(scaled_argument(z, mean_time))
    bound_a = 6.0 / (PI2 * (modes + 0.5))
    with np.errstate(divide="ignore"):
        root = np.sqrt(x_abs)
        s1 = np.minimum(math.pi * modes / root, 1.0)
        bound_b = 6.0 / (math.pi * root) * (QUARTIC_INTEGRAL - s1 / np.sqrt(1 + s1**4))
    return np.minimum(bound_a, bound_b)


def rounding_allowance(*values):
    """Heuristic floating-point allowance, not a proven bound.

    The analytic branches lose at most a few bits to cancellation and the
    modal terms do not cancel; 128 eps per unit magnitude leaves margin.
    """
    return ROUNDING_FACTOR * EPS * sum(abs(v) for v in values)
