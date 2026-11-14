"""Dimensioned property tables and three-step passive-sphere derivations.

All public calculations are real Laplace-frequency calculations. A source row
contains no hidden truth, fault label or evaluator answer. Unit conversion does
not repair an incorrect declaration: a radius labelled m is interpreted as m.
The deliberately selectable high-frequency formula is not a general solver.
"""
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = str(ROOT / 'vendor/sphere-thermal-cli')
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)
from sphere_thermal.response import analytic


# SI dimensions are mass, length, time and temperature exponents, respectively.
DIMENSIONS = {
    'radius': (0, 1, 0, 0),
    'diffusivity': (0, 2, -1, 0),
    'conductivity': (1, 1, -3, -1),
    'density': (1, -3, 0, 0),
    'heat_capacity': (0, 2, -2, -1),
    'mean_time': (0, 0, 1, 0),
    'frequency': (0, 0, -1, 0),
    'response': (0, 0, 0, 0),
}
UNITS = {
    'radius': {'m': 1., 'mm': 1e-3, 'cm': 1e-2},
    'diffusivity': {'m2/s': 1., 'mm2/s': 1e-6, 'cm2/s': 1e-4},
    'conductivity': {'W/(m K)': 1., 'W/(cm K)': 100.},
    'density': {'kg/m3': 1., 'g/cm3': 1000.},
    'heat_capacity': {'J/(kg K)': 1., 'kJ/(kg K)': 1000.},
}
INVERSE_ITERATIONS = 64
INVERSE_EVALS = INVERSE_ITERATIONS + 2
INVERSE_KAPPA_BRACKET = (1e-18, 1e6)

SI_UNITS = {'radius': 'm', 'diffusivity': 'm2/s', 'conductivity': 'W/(m K)',
            'density': 'kg/m3', 'heat_capacity': 'J/(kg K)'}


def _positive(value, label, allow_zero=False):
    if isinstance(value, (bool, str, complex)):
        raise ValueError(f'{label} must be a real finite number')
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{label} must be a real finite number') from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{label} must be finite and positive')
    return value


def quantity_si(quantity, kind):
    """Convert one explicit quantity; units from other dimensions are rejected."""
    if kind not in UNITS or not isinstance(quantity, dict):
        raise ValueError('a supported dimension and explicit quantity are required')
    if set(quantity) != {'value', 'unit'}:
        raise ValueError('quantity must contain exactly value and unit')
    unit = quantity['unit']
    if not isinstance(unit, str) or unit not in UNITS[kind]:
        raise ValueError(f'invalid {kind} unit: {unit!r}')
    return _positive(_positive(quantity['value'], kind) * UNITS[kind][unit], kind)


def normalize_source(source):
    """Return SI properties with identity, version and declared-unit provenance.

    Schema: id, version, radius and either diffusivity or all of conductivity,
    density and heat_capacity. Properties are {value: number, unit: string}.
    The two alternative diffusivity representations must not coexist.
    """
    if not isinstance(source, dict):
        raise ValueError('source must be a mapping')
    source_id, version = source.get('id'), source.get('version')
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('nonempty source id required')
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError('positive integer source version required')
    property_names = {'conductivity', 'density', 'heat_capacity'}
    direct = 'diffusivity' in source
    required = {'id', 'version', 'radius'} | ({'diffusivity'} if direct else property_names)
    if set(source) != required:
        raise ValueError('source needs radius and exactly one diffusivity representation')
    radius = quantity_si(source['radius'], 'radius')
    quantities = ['radius', 'diffusivity'] if direct else ['radius', *sorted(property_names)]
    if direct:
        diffusivity = quantity_si(source['diffusivity'], 'diffusivity')
    else:
        conductivity = quantity_si(source['conductivity'], 'conductivity')
        density = quantity_si(source['density'], 'density')
        heat_capacity = quantity_si(source['heat_capacity'], 'heat_capacity')
        # Divide successively to avoid overflow of rho*c for otherwise valid SI inputs.
        diffusivity = _positive(conductivity / density / heat_capacity, 'diffusivity')
    return {'source_id': source_id, 'version': version, 'radius_m': radius,
            'diffusivity_m2_s': diffusivity,
            'declared_units': {name: source[name]['unit'] for name in quantities},
            'si_units': {'radius': 'm', 'diffusivity': 'm2/s'},
            'dimensions': {'radius': list(DIMENSIONS['radius']),
                           'diffusivity': list(DIMENSIONS['diffusivity'])}}


def mean_time(source):
    """R squared / (15*kappa), in seconds, from an explicit source table."""
    properties = normalize_source(source)
    return _mean_time_si(properties['radius_m'], properties['diffusivity_m2_s'])


def _mean_time_si(radius, kappa):
    radius, kappa = _positive(radius, 'radius'), _positive(kappa, 'diffusivity')
    # Log arithmetic avoids overflowing R squared when the final ratio is finite.
    try:
        result = math.exp(2 * math.log(radius) - math.log(15.) - math.log(kappa))
    except OverflowError as exc:
        raise ValueError('mean time is outside finite positive range') from exc
    return _positive(result, 'mean time')


def _perturb(value, noise):
    return float(np.nextafter(value, math.inf)) if noise and value != 0 else float(value)


def _count(stats, name):
    if stats is not None:
        stats[name] = stats.get(name, 0) + 1


def _analytic_response(z, T, noise=False, stats=None):
    _count(stats, 'analytic_evaluations')
    value = float(np.real(analytic(z, T)))
    return _perturb(_positive(value, 'response'), noise and z != 0)


def response(source, z, mode='exact', noise=False, stats=None):
    """Return the dimensionless response; z is declared in inverse seconds."""
    z = _positive(z, 'frequency', allow_zero=True)
    T = mean_time(source)
    if mode == 'exact':
        return _analytic_response(z, T, noise=noise, stats=stats)
    elif mode in ('asymptotic', 'high_frequency'):
        if z == 0:
            raise ValueError('high-frequency approximation is undefined at zero')
        _count(stats, 'asymptotic_evaluations')
        value = math.sqrt(3. / (5. * T * z))
    else:
        raise ValueError('unknown response mode')
    value = _positive(value, 'response')
    # Preserve the exact DC normalization while perturbing nonzero-frequency outputs.
    return _perturb(value, noise and z != 0)


def infer_diffusivity(value, radius, z, noise=False, stats=None):
    """Invert the exact sphere response with a fixed paid evaluation count.

    The SI kappa bracket is [1e-18, 1e6] m2/s. Two endpoint evaluations and
    exactly 64 log-space bisection evaluations are paid even after floating-point
    interval stagnation. Values outside the bracket or outside (0, 1) are
    explicitly unresolved. No final unpaid numerical check is performed.
    """
    value = _positive(value, 'response')
    radius, z = _positive(radius, 'radius'), _positive(z, 'frequency')
    if value >= 1:
        raise ValueError('inverse response must lie strictly between zero and one')
    lower, upper = map(math.log, INVERSE_KAPPA_BRACKET)

    def evaluate(log_kappa):
        return predict_response(math.exp(log_kappa), radius, z,
                                noise=noise, stats=stats)

    at_lower, at_upper = evaluate(lower), evaluate(upper)
    if not at_lower <= value <= at_upper:
        raise ValueError('inverse response is outside the supported diffusivity bracket')
    for _ in range(INVERSE_ITERATIONS):
        middle = (lower + upper) / 2
        observed = evaluate(middle)
        _count(stats, 'inverse_iterations')
        if observed < value:
            lower = middle
        else:
            upper = middle
    return _positive(math.exp((lower + upper) / 2), 'inferred diffusivity')


def predict_response(kappa, radius, z, noise=False, stats=None):
    """Predict at a new radius using an inferred SI diffusivity (chain depth 3)."""
    z = _positive(z, 'frequency', allow_zero=True)
    return _analytic_response(z, _mean_time_si(radius, kappa), noise=noise, stats=stats)
