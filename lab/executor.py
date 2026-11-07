"""Trusted-process execution boundary: planner proposes N, never returned value.

Frozen records prevent accidental mutation, not malicious Python code. They are
not signatures or remote attestation. The caller must trust this module/solver.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time
from lab.guard import Request
from lab.scientist import modal_partial_sums


@dataclass(frozen=True)
class Receipt:
    request_sha256: str
    modes: int
    value: complex
    kernel_seconds: float


def request_hash(request):
    return hashlib.sha256(json.dumps({'z': [complex(request.z).real,complex(request.z).imag],
        'mean_time': request.mean_time,'tolerance':request.tolerance,
        'budget':request.budget,'max_calls':request.max_calls},sort_keys=True,allow_nan=False).encode()).hexdigest()


def execute(request, modes):
    if not isinstance(request, Request):
        raise ValueError('immutable original Request required')
    if not math.isfinite(request.tolerance) or request.tolerance <= 0:
        raise ValueError('positive finite tolerance required')
    if type(request.budget) is not int or request.budget < 1 or type(request.max_calls) is not int or request.max_calls < 1:
        raise ValueError('positive integer budgets required')
    if type(modes) is not int or not 1 <= modes <= request.budget:
        raise ValueError('plan violates execution budget')
    started = time.perf_counter()
    value = complex(modal_partial_sums(request.z,request.mean_time,[modes])[0])
    return Receipt(request_hash(request),modes,value,time.perf_counter()-started)


def as_outcome(request, receipt):
    if not isinstance(receipt, Receipt) or receipt.request_sha256 != request_hash(request):
        raise ValueError('receipt/request mismatch')
    return {'value':[receipt.value.real,receipt.value.imag], 'calls':1,
            'modes_spent':receipt.modes,'trace':[{'call':1,'modes':receipt.modes}]}
