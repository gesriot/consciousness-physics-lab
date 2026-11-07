"""Final tolerance gate independent of mutable planner state.

This verifies resolution and accounting, not arbitrary corruption of values.
"""
from dataclasses import dataclass
import math
import numpy as np

from lab.scientist import tail_bound, rounding_allowance


@dataclass(frozen=True)
class Request:
    z: complex
    mean_time: float
    tolerance: float
    budget: int = 4096
    max_calls: int = 3


def final_guard(request, outcome):
    """Recompute the resolution requirement from the original request.

    Fail closed for malformed outcomes. No analytic reference is consulted.
    Request must originate outside the mutable planning state.
    """
    try:
        if not isinstance(request, Request):
            raise ValueError('request must be an immutable Request')
        if not math.isfinite(request.tolerance) or request.tolerance <= 0:
            raise ValueError('invalid requested tolerance')
        for count in (request.budget, request.max_calls):
            if type(count) is not int or count < 1:
                raise ValueError('invalid request budget')
        trace = outcome['trace']
        if not isinstance(trace, list) or not 1 <= len(trace) <= request.max_calls:
            raise ValueError('invalid call count')
        modes = [row['modes'] for row in trace]
        if any(type(n) is not int or n < 1 for n in modes):
            raise ValueError('invalid mode count')
        if sum(modes) > request.budget:
            raise ValueError('mode budget exceeded')
        if outcome['modes_spent'] != sum(modes) or outcome['calls'] != len(trace):
            raise ValueError('inconsistent accounting')
        if [row['call'] for row in trace] != list(range(1, len(trace) + 1)):
            raise ValueError('inconsistent call sequence')
        value = complex(*outcome['value'])
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError('nonfinite result')
        estimate = float(tail_bound(request.z, request.mean_time, np.array([modes[-1]]))[0]) + rounding_allowance(value)
        accepted = math.isfinite(estimate) and estimate <= request.tolerance
        return {'accepted': accepted, 'reason': 'resolution_estimate_met' if accepted else 'insufficient_resolution',
                'recomputed_error_estimate': estimate, 'requested_tolerance': request.tolerance}
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        return {'accepted': False, 'reason': 'invalid_request_or_outcome', 'detail': str(error)}
