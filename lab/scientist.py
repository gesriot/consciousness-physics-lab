"""Bounded numerical scientist pilot; explicit state is operational, not phenomenal."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/sphere-thermal-cli'))
from sphere_thermal.response import analytic, modal_partial_sums, tail_bound, rounding_allowance

POLICIES = ('adaptive', 'fixed', 'budget_once', 'stale', 'irrelevant')


@dataclass
class AttentionState:
    target_tolerance: float
    modes_spent: int = 0
    last_tail: float | None = None
    label: str = 'resolution'


def solve(z, mean_time, tolerance, policy='adaptive', budget=4096, max_calls=3):
    """No analytic oracle is available to this decision loop.

    Repeated sums count again against budget. Tail status is a numerical
    estimate with heuristic rounding allowance, not a rigorous certificate.
    """
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    for value in (budget, max_calls):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('budgets must be positive integers')
    state = AttentionState(tolerance * (10 if policy == 'stale' else 1),
                           label='unrelated label' if policy == 'irrelevant' else 'resolution')
    trace = []
    # A geometric schedule independent of answers, with total work <= budget.
    weights = [2**i for i in range(max_calls)]
    schedule = [budget * w // sum(weights) for w in weights]
    schedule[-1] += budget - sum(schedule)
    result = None
    for call in range(max_calls):
        remaining = budget - state.modes_spent
        if not remaining:
            break
        if policy == 'fixed':
            n = max(1, min(remaining, schedule[call]))
            reason = 'predefined geometric schedule'
        elif policy == 'budget_once':
            n, reason = remaining, 'single full-budget sum'
        else:
            ratio = 6 / math.pi**2 / state.target_tolerance
            n = remaining if not math.isfinite(ratio) or ratio >= remaining else max(1, math.ceil(ratio - .5))
            reason = 'invert conservative tail bound using current target'
        start = time.perf_counter()
        result = complex(modal_partial_sums(z, mean_time, [n])[0])
        bound = float(tail_bound(z, mean_time, np.array([n]))[0])
        estimate = bound + rounding_allowance(result)
        elapsed = time.perf_counter() - start
        state.modes_spent += n
        state.last_tail = bound
        claimed = estimate <= state.target_tolerance
        trace.append({'call': call + 1, 'modes': n, 'reason': reason,
                      'bound_plus_heuristic_rounding': estimate,
                      'kernel_seconds': elapsed, 'state': asdict(state),
                      'claims_target_met': claimed})
        if claimed or policy == 'budget_once':
            break
    return {'policy': policy, 'value': [result.real, result.imag],
            'claimed_target_met': trace[-1]['claims_target_met'],
            'modes_spent': state.modes_spent, 'calls': len(trace), 'trace': trace}


def run_pilot(seed=20261107):
    rng = random.Random(seed)
    rows = []
    for mean_time in (.1, 1., 10.):
        for zt in (0, 1e-8, .1, 1, 10j, 100):
            z = zt / mean_time
            for tolerance in (.01, .001, .0001):
                policies = list(POLICIES)
                rng.shuffle(policies)
                for policy in policies:
                    outcome = solve(z, mean_time, tolerance, policy)
                    # Evaluation happens only after decisions are complete.
                    reference = complex(analytic(z, mean_time))
                    error = abs(complex(*outcome['value']) - reference)
                    outcome.update(task={'mean_time': mean_time, 'z': [complex(z).real, complex(z).imag],
                                         'tolerance': tolerance}, absolute_error=error,
                                   accurate=error <= tolerance,
                                   false_accuracy_claim=outcome['claimed_target_met'] and error > tolerance)
                    rows.append(outcome)
    summary = {}
    for policy in POLICIES:
        subset = [r for r in rows if r['policy'] == policy]
        summary[policy] = {'tasks': len(subset), 'accurate': sum(r['accurate'] for r in subset),
                           'false_accuracy_claims': sum(r['false_accuracy_claim'] for r in subset),
                           'modes_spent': sum(r['modes_spent'] for r in subset),
                           'calls': sum(r['calls'] for r in subset),
                           'kernel_seconds': sum(t['kernel_seconds'] for r in subset for t in r['trace'])}
    return {'experiment': 'E002', 'seed': seed, 'split': 'development',
            'limits': {'mode_terms_per_task': 4096, 'calls_per_task': 3},
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__).resolve(), ROOT / 'vendor/sphere-thermal-cli/sphere_thermal/response.py']},
            'summary': summary, 'runs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
