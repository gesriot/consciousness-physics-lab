#!/usr/bin/env python3
"""Run every test with deterministic one-bit noise in numerical solver results.

Patch before test discovery so from-import aliases use perturbed dense, banded
and modal-sum solvers. Analytic references and exact zero components stay intact.
This probes arithmetic portability, unlike noise in rounded scheduler scores.
"""
import argparse
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'vendor/sphere-thermal-cli'))
from sphere_thermal import response as modal_response
COUNTS = {}


def perturb(value):
    out = np.array(value, copy=True)
    views = (out.real, out.imag) if np.iscomplexobj(out) else (out,)
    for view in views:
        mask = np.isfinite(view) & (view != 0)
        view[mask] = np.nextafter(view[mask], np.inf)
    return out


def noisy(name, solver):
    @wraps(solver)
    def call(*args, **kwargs):
        result = solver(*args, **kwargs)
        COUNTS[name] = COUNTS.get(name, 0) + 1
        return perturb(result)
    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='tests')
    args = parser.parse_args()
    with ExitStack() as stack:
        for module, name in ((scipy.linalg, 'solve_banded'), (scipy.linalg, 'solve'), (np.linalg, 'solve'),
                             (modal_response, 'modal_partial_sums')):
            stack.enter_context(patch.object(module, name, noisy(module.__name__+'.'+name, getattr(module,name))))
        suite = unittest.defaultTestLoader.discover(args.directory)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('One-bit solver perturbation calls:', dict(sorted(COUNTS.items())), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
