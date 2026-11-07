"""Frozen E003 holdout: independent final tolerance guard, no planner tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from lab.guard import Request, final_guard
from lab.scientist import ROOT, POLICIES, solve, analytic


def run():
    rng = np.random.default_rng(20261108)
    order = np.random.default_rng(20261109)
    rows = []
    for task_id in range(96):
        mean_time = 10 ** rng.uniform(-2, 2)
        magnitude = 10 ** rng.uniform(-7, 3)
        phase = rng.uniform(-np.pi / 2, np.pi / 2)
        tolerance = 10 ** rng.uniform(-4, -1)
        z = magnitude * np.exp(1j * phase) / mean_time
        request = Request(z, mean_time, tolerance)
        for policy in order.permutation(POLICIES):
            outcome = solve(z, mean_time, tolerance, str(policy))
            started = time.perf_counter()
            decision = final_guard(request, outcome)
            guard_seconds = time.perf_counter() - started
            error = abs(complex(*outcome['value']) - complex(analytic(z, mean_time)))
            accurate = bool(error <= tolerance)
            rows.append({'task_id': task_id, 'request': {'z': [z.real, z.imag], 'mean_time': mean_time, 'tolerance': tolerance},
                         'outcome': outcome, 'guard': decision, 'guard_seconds': guard_seconds,
                         'absolute_error': error, 'accurate': accurate})
    summary = {}
    for policy in POLICIES:
        subset = [r for r in rows if r['outcome']['policy'] == policy]
        summary[policy] = {'tasks': len(subset), 'accurate': sum(r['accurate'] for r in subset),
                           'raw_false_claims': sum(r['outcome']['claimed_target_met'] and not r['accurate'] for r in subset),
                           'guard_accepted': sum(r['guard']['accepted'] for r in subset),
                           'false_acceptances': sum(r['guard']['accepted'] and not r['accurate'] for r in subset),
                           'accurate_rejected': sum(not r['guard']['accepted'] and r['accurate'] for r in subset),
                           'modes_spent': sum(r['outcome']['modes_spent'] for r in subset),
                           'guard_seconds': sum(r['guard_seconds'] for r in subset)}
    return {'experiment': 'E003', 'split': 'held_out_from_E002', 'task_seed': 20261108, 'order_seed': 20261109,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/'lab/scientist.py', ROOT/'lab/guard.py', Path(__file__).resolve(),
                                        ROOT/'vendor/sphere-thermal-cli/sphere_thermal/response.py']},
            'summary': summary, 'runs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
