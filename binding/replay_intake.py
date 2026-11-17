"""Validate regenerated input structure without replacing authenticated inputs."""
import math


def compare_regenerated(saved, regenerated, max_ulps=8):
    """Return last-bit diagnostics; non-floating fields and structure are exact.

    Callers must authenticate saved bytes against their manifest before using
    them. This comparison never changes the saved input or its canonical hash.
    Eight local ULPs accommodate a few rounded arithmetic operations, without
    an absolute tolerance that would hide a change in a very small input.
    """
    differences = {}

    def visit(a, b, path):
        if type(a) is not type(b):
            raise ValueError('Regenerated intake type mismatch: '+path)
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError('Regenerated intake keys mismatch: '+path)
            for key in a:
                visit(a[key], b[key], path+'.'+str(key))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError('Regenerated intake length mismatch: '+path)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path+'.'+str(i))
        elif isinstance(a, float):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('Nonfinite regenerated intake: '+path)
            absolute = abs(a-b)
            ulps = absolute/max(math.ulp(a), math.ulp(b))
            if ulps > max_ulps:
                raise ValueError('Regenerated intake numeric mismatch: '+path)
            if a != b:
                differences[path] = dict(saved=a, regenerated=b, abs=absolute,
                    rel=absolute/max(abs(a), abs(b)), ulps=ulps)
        elif a != b:
            raise ValueError('Regenerated intake value mismatch: '+path)
    visit(saved, regenerated, 'inputs')
    return differences
