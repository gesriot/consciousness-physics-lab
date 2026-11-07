"""Warm kernel comparison at equal modal count, excluding import and CLI startup."""
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/sphere-thermal-cli'))
from sphere_thermal.response import modal_partial_sums


def scalar(n):
    values = [6 / (math.pi**2 * k * k + 15j) for k in range(1, n + 1)]
    return complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))


def run():
    rows = []
    for n in (1024, 65536):
        kernels = {'numpy': lambda: modal_partial_sums(1j, 1, [n])[0],
                   'python_fsum': lambda: scalar(n)}
        samples = {key: [] for key in kernels}
        values = {key: fun() for key, fun in kernels.items()}  # warm both
        for i in range(9):
            order = list(kernels) if i % 2 == 0 else list(reversed(kernels))
            for key in order:
                start = time.perf_counter()
                for _ in range(3):
                    kernels[key]()
                samples[key].append((time.perf_counter() - start) / 3)
        difference = abs(values['numpy'] - values['python_fsum'])
        if difference > 1e-13:
            raise AssertionError('kernels disagree')
        rows.append({'modes': n, 'absolute_difference': float(difference),
                     'seconds': samples,
                     'median_seconds': {k: statistics.median(v) for k, v in samples.items()}})
    return {'scope': 'warm kernel including allocations; 9 alternating samples, 3 evaluations/sample',
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
            'runs': rows}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
