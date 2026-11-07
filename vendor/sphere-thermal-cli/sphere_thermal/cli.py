"""JSON command-line interface.

Usage: python -m sphere_thermal [REQUEST.json | -]

The request is read from the file or, with "-" or no argument, from stdin.
stdout receives exactly one JSON document; diagnostics go to stderr.
Exit codes: 0 ok, 1 internal error, 2 invalid input or usage,
3 tolerance not met, 4 methods inconsistent.
"""
from dataclasses import dataclass
import json
import math
import sys
import time
import traceback

import numpy as np

from . import __version__, response

SCHEMA_VERSION = "1.0"
MODES = ("analytic", "modal", "compare", "convergence")
MAX_MODES = 10_000_000
MAX_POINTS = 10_000
MAX_SPHERES = 1_000
MAX_TERMS = 1_000_000_000
WEIGHT_SUM_TOL = 1e-9
MAX_UNIT_LABEL = 32
REQUIRED_MODES_SEARCH_LIMIT = 10**15
EXIT_CODES = {"ok": 0, "tolerance_not_met": 3, "inconsistent": 4}

NOTES = [
    "g(z) is the volume-mean sphere temperature divided by the surface temperature "
    "for signals exp(z t) (Laplace transform, zero initial state); dimensionless.",
    "tail_bound is a rigorous upper bound of the modal truncation error "
    "|sum_{n>N} w_n/(1+z tau_n)| in exact arithmetic for Re z >= 0; "
    "it does not include floating-point rounding.",
    "rounding_allowance = 128*eps*(|analytic| + |modal|) is a heuristic "
    "estimate of floating-point error, not a proven bound.",
    "abs_discrepancy = |analytic - modal| is observed; it contains truncation and "
    "rounding of both methods. consistent means abs_discrepancy <= tail_bound + rounding_allowance.",
    "tolerance_met means tail_bound + rounding_allowance <= tolerance for the modal value.",
    "The analytic and modal values are two representations of one model computed "
    "independently; their agreement verifies the implementation, not the physics.",
]

USAGE = """usage: python -m sphere_thermal [REQUEST.json | -]

Reads a JSON request from REQUEST.json or stdin and writes a JSON response
to stdout. See README.md for the schema. Exit codes: 0 ok, 1 internal error,
2 invalid input, 3 tolerance not met, 4 methods inconsistent.
"""


class InputError(Exception):
    def __init__(self, path, message, kind="invalid_input"):
        super().__init__(message)
        self.path, self.message, self.kind = path, message, kind


@dataclass
class Sphere:
    weight: float
    mean_time: float


@dataclass
class Request:
    raw: dict
    mode: str
    units: dict
    spheres: list
    points: np.ndarray
    levels: list
    tolerance: float | None


def _type_name(value):
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            type(None): "null"}.get(type(value), "number")


def _object(value, path, required, optional=()):
    if not isinstance(value, dict):
        raise InputError(path, f"must be an object, got {_type_name(value)}")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        allowed = ", ".join(list(required) + list(optional))
        raise InputError(f"{path}.{unknown[0]}", f"unknown key; allowed keys: {allowed}")
    for key in required:
        if key not in value:
            raise InputError(f"{path}.{key}", "required key is missing")
    return value


def _array(value, path, max_len):
    if not isinstance(value, list):
        raise InputError(path, f"must be an array, got {_type_name(value)}")
    if not value:
        raise InputError(path, "must not be empty")
    if len(value) > max_len:
        raise InputError(path, f"has {len(value)} entries; at most {max_len} are allowed")
    return value


def _number(value, path, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(path, f"must be a number, got {_type_name(value)}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise InputError(path, "must be finite")
    if positive and not number > 0:
        raise InputError(path, f"must be positive, got {value}")
    return number


def _integer(value, path, low, high):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(path, f"must be an integer, got {_type_name(value)} {value!r}")
    if not low <= value <= high:
        raise InputError(path, f"must be between {low} and {high}, got {value}")
    return value


def _unit_label(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_UNIT_LABEL \
            or not value.isprintable():
        raise InputError(path, f"must be a non-empty printable string of at most "
                               f"{MAX_UNIT_LABEL} characters, e.g. \"s\" or \"mm\"")
    return value


def _forbid(data, keys, mode):
    for key in keys:
        if key in data:
            raise InputError(f"$.{key}", f"is not used in mode \"{mode}\"; remove it")


def _sphere(value, path, single, units):
    _object(value, path, (), ("weight", "mean_time", "radius", "diffusivity"))
    if "weight" in value:
        weight = _number(value["weight"], f"{path}.weight", positive=True)
    elif single:
        weight = 1.0
    else:
        raise InputError(f"{path}.weight", "required for every sphere of a mixture")
    geometric = "radius" in value or "diffusivity" in value
    if ("mean_time" in value) == geometric:
        raise InputError(path, "give either mean_time or both radius and diffusivity")
    if "mean_time" in value:
        return Sphere(weight, _number(value["mean_time"], f"{path}.mean_time", positive=True))
    for key in ("radius", "diffusivity"):
        if key not in value:
            raise InputError(f"{path}.{key}", "required together with "
                                              f"{'diffusivity' if key == 'radius' else 'radius'}")
    if "length" not in units:
        raise InputError("$.units.length", "required when a sphere is given by radius and diffusivity")
    radius = _number(value["radius"], f"{path}.radius", positive=True)
    kappa = _number(value["diffusivity"], f"{path}.diffusivity", positive=True)
    mean_time = radius * radius / (15.0 * kappa)
    if not (math.isfinite(mean_time) and mean_time > 0):
        raise InputError(path, f"derived mean_time = radius^2/(15 diffusivity) = {mean_time} "
                               "is not a finite positive number")
    return Sphere(weight, mean_time)


def parse_request(data):
    """Validate a decoded JSON request; raise InputError with a JSON path."""
    _object(data, "$", ("schema_version", "mode", "units", "spheres", "points"),
            ("modes", "modes_list", "tolerance"))
    if data["schema_version"] != SCHEMA_VERSION:
        raise InputError("$.schema_version", f"must be \"{SCHEMA_VERSION}\", "
                                             f"got {data['schema_version']!r}")
    mode = data["mode"]
    if mode not in MODES:
        raise InputError("$.mode", f"must be one of {', '.join(MODES)}; got {mode!r}")

    units = _object(data["units"], "$.units", ("time",), ("length",))
    for key in units:
        _unit_label(units[key], f"$.units.{key}")

    raw_spheres = _array(data["spheres"], "$.spheres", MAX_SPHERES)
    spheres = [_sphere(s, f"$.spheres[{j}]", len(raw_spheres) == 1, units)
               for j, s in enumerate(raw_spheres)]
    weight_sum = math.fsum(s.weight for s in spheres)
    if abs(weight_sum - 1) > WEIGHT_SUM_TOL:
        raise InputError("$.spheres", f"weights must sum to 1 within {WEIGHT_SUM_TOL:g}; "
                                      f"their sum is {weight_sum!r}")

    raw_points = _array(data["points"], "$.points", MAX_POINTS)
    points = []
    for i, p in enumerate(raw_points):
        path = f"$.points[{i}]"
        _object(p, path, ("re", "im"))
        re = _number(p["re"], f"{path}.re")
        im = _number(p["im"], f"{path}.im")
        if re < 0:
            raise InputError(f"{path}.re", f"Re z must be >= 0 (right half-plane or "
                                           f"imaginary axis), got {re}")
        points.append(complex(re + 0.0, im))
    points = np.array(points)

    if mode == "analytic":
        _forbid(data, ("modes", "modes_list", "tolerance"), mode)
        levels = []
    elif mode in ("modal", "compare"):
        _forbid(data, ("modes_list",), mode)
        if "modes" not in data:
            raise InputError("$.modes", f"required in mode \"{mode}\"")
        levels = [_integer(data["modes"], "$.modes", 1, MAX_MODES)]
    else:
        _forbid(data, ("modes",), mode)
        if "modes_list" not in data:
            raise InputError("$.modes_list", "required in mode \"convergence\"")
        raw = data["modes_list"]
        if not isinstance(raw, list) or len(raw) < 2:
            raise InputError("$.modes_list", "must be an array of at least two integers")
        levels = [_integer(n, f"$.modes_list[{k}]", 1, MAX_MODES) for k, n in enumerate(raw)]
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise InputError("$.modes_list", "must be strictly increasing")
    tolerance = (_number(data["tolerance"], "$.tolerance", positive=True)
                 if "tolerance" in data else None)

    largest_time = max(s.mean_time for s in spheres)
    for i, z in enumerate(points):
        if 15.0 * largest_time * abs(z) > response.MAX_ABS_X:
            raise InputError(f"$.points[{i}]", f"|15 T z| exceeds {response.MAX_ABS_X:g} "
                                               "for the largest mean_time")
    terms = len(points) * len(spheres) * (levels[-1] if levels else 0)
    if terms > MAX_TERMS:
        raise InputError("$", f"points x spheres x modes = {terms} exceeds the work limit "
                              f"{MAX_TERMS}; reduce one of them")
    return Request(data, mode, units, spheres, points, levels, tolerance)


def _complex(value):
    return {"re": float(value.real), "im": float(value.imag)}


def _modes_required(req):
    """Smallest N with mixture tail_bound <= tolerance at every point, or None."""
    def enough(n):
        bound = sum(s.weight * response.tail_bound(req.points, s.mean_time, n)
                    for s in req.spheres)
        return bool(np.all(bound <= req.tolerance))
    if not enough(REQUIRED_MODES_SEARCH_LIMIT):
        return None
    lo, hi = 0, REQUIRED_MODES_SEARCH_LIMIT
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if enough(mid) else (mid, hi)
    return hi


def evaluate(req):
    """Compute the requested values and build the response document."""
    z, levels, tol = req.points, req.levels, req.tolerance
    analytic = None
    if req.mode != "modal":
        analytic = sum(s.weight * response.analytic(z, s.mean_time) for s in req.spheres)
    modal = np.zeros((len(z), len(levels)), dtype=complex)
    tail = np.zeros((len(z), len(levels)))
    for s in req.spheres:
        for k, n in enumerate(levels):
            tail[:, k] += s.weight * response.tail_bound(z, s.mean_time, n)
        for i, zi in enumerate(z):
            modal[i] += s.weight * response.modal_partial_sums(zi, s.mean_time, levels)

    results, rows = [], []
    for i, zi in enumerate(z):
        entries = []
        for k, n in enumerate(levels):
            value = modal[i, k]
            entry = {"modes": n, "value": _complex(value), "abs_discrepancy": None,
                     "tail_bound": float(tail[i, k]), "rounding_allowance": None,
                     "consistent": None, "tolerance_met": None}
            if analytic is None:
                allowance = response.rounding_allowance(value)
            else:
                allowance = response.rounding_allowance(analytic[i], value)
                entry["abs_discrepancy"] = float(abs(analytic[i] - value))
                entry["consistent"] = bool(entry["abs_discrepancy"] <= tail[i, k] + allowance)
            entry["rounding_allowance"] = float(allowance)
            if tol is not None:
                entry["tolerance_met"] = bool(tail[i, k] + allowance <= tol)
            entries.append(entry)
        rows.append(entries)
        results.append({"z": _complex(zi),
                        "analytic": None if analytic is None else _complex(analytic[i]),
                        "modal": entries})

    summary_levels = []
    for k, n in enumerate(levels):
        column = [row[k] for row in rows]
        compared = analytic is not None
        summary_levels.append({
            "modes": n,
            "max_abs_discrepancy": max(e["abs_discrepancy"] for e in column) if compared else None,
            "max_tail_bound": max(e["tail_bound"] for e in column),
            "all_consistent": all(e["consistent"] for e in column) if compared else None,
            "tolerance_met": all(e["tolerance_met"] for e in column) if tol is not None else None,
        })

    messages = []
    all_consistent = None
    if levels and analytic is not None:
        all_consistent = all(level["all_consistent"] for level in summary_levels)
        for i, row in enumerate(rows):
            for e in row:
                if not e["consistent"]:
                    messages.append(
                        f"Inconsistent: at point {i}, N={e['modes']}, abs_discrepancy="
                        f"{e['abs_discrepancy']:.3e} exceeds tail_bound + rounding_allowance="
                        f"{e['tail_bound'] + e['rounding_allowance']:.3e}. This indicates a "
                        "defect or rounding beyond the heuristic allowance.")
    tolerance_met = smallest = required = None
    if tol is not None:
        tolerance_met = summary_levels[-1]["tolerance_met"]
        smallest = next((lv["modes"] for lv in summary_levels if lv["tolerance_met"]), None)
        required = _modes_required(req)
        floor = max(e["rounding_allowance"] for row in rows for e in row)
        if not tolerance_met:
            worst = max(range(len(z)), key=lambda i: rows[i][-1]["tail_bound"]
                        + rows[i][-1]["rounding_allowance"])
            last = rows[worst][-1]
            need = ("no N up to 1e15 suffices" if required is None else
                    f"the truncation bound alone requires N >= {required}" +
                    (f", above the hard limit {MAX_MODES}" if required > MAX_MODES else ""))
            messages.append(
                f"Requested tolerance {tol:g} is NOT met at the mode limit N={levels[-1]}: "
                f"tail_bound + rounding_allowance = "
                f"{last['tail_bound'] + last['rounding_allowance']:.3e} at point {worst}; {need}.")
        if tol <= floor:
            messages.append(
                f"Requested tolerance {tol:g} is at or below the estimated rounding level "
                f"{floor:.3e}; double precision cannot certify it.")

    status = ("inconsistent" if all_consistent is False else
              "tolerance_not_met" if tolerance_met is False else "ok")
    mean_time = math.fsum(s.weight * s.mean_time for s in req.spheres)
    time_unit = req.units["time"]
    units = {"time": time_unit, "z": f"1/{time_unit}", "mean_time": time_unit,
             "g": "1 (dimensionless)", "abs_discrepancy": "1", "tail_bound": "1"}
    if "length" in req.units:
        length = req.units["length"]
        units.update(radius=length, diffusivity=f"{length}^2/{time_unit}")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "sphere-thermal-cli", "version": __version__},
        "status": status,
        "mode": req.mode,
        "input": req.raw,
        "model": {
            "quantity": "g(z) = mean sphere temperature / surface temperature, signals exp(z t)",
            "assumptions": "passive conduction in a homogeneous inert sphere, ideal thermal "
                           "contact, prescribed surface temperature, zero initial state",
            "analytic": "g(z;T) = 3 (q coth q - 1)/q^2, q = sqrt(15 T z), T = R^2/(15 kappa); "
                        "Taylor series in 15 T z for |15 T z| <= 1",
            "modal": "g_N(z;T) = sum_{n=1..N} w_n/(1 + z tau_n), w_n = 6/(pi^2 n^2), "
                     "tau_n = 15 T/(pi^2 n^2); weights not renormalized",
            "mixture": "g(z) = sum_j a_j g(z;T_j), a_j > 0, sum_j a_j = 1; "
                       "the same N modes are used for every sphere",
            "domain": "Re z >= 0",
            "source": "dissertation manuscript, section 3.2",
        },
        "units": units,
        "medium": {
            "spheres": [{"weight": s.weight, "mean_time": s.mean_time} for s in req.spheres],
            "weight_sum": math.fsum(s.weight for s in req.spheres),
            "mean_relaxation_time": mean_time,
        },
        "modes": levels,
        "tolerance": tol,
        "results": results,
        "summary": {
            "levels": summary_levels,
            "all_consistent": all_consistent,
            "tolerance_met": tolerance_met,
            "smallest_modes_meeting_tolerance": smallest,
            "modes_required_by_bound": required,
        },
        "messages": messages,
        "notes": NOTES,
    }


def _reject_constant(name):
    raise InputError("$", f"non-finite literal {name} is not allowed", "invalid_json")


def _unique_pairs(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise InputError("$", f"duplicate key {key!r}", "invalid_json")
        seen.add(key)
    return dict(pairs)


def _read_request(argv):
    if len(argv) > 1 or (argv and argv[0].startswith("-") and argv[0] != "-"):
        raise InputError("argv", f"expected at most one argument (a path or -), got {argv}",
                         "usage")
    if argv and argv[0] != "-":
        try:
            with open(argv[0], "rb") as f:
                raw = f.read()
        except OSError as exc:
            raise InputError("argv", f"cannot read {argv[0]}: {exc.strerror}", "io") from exc
    else:
        if sys.stdin is None or sys.stdin.isatty():
            raise InputError("argv", "no input: pass a request file or pipe JSON to stdin",
                             "usage")
        raw = sys.stdin.buffer.read()
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_unique_pairs)
    except UnicodeDecodeError as exc:
        raise InputError("$", f"input is not UTF-8: {exc.reason}", "invalid_json") from exc
    except json.JSONDecodeError as exc:
        raise InputError("$", f"malformed JSON: {exc.msg} at line {exc.lineno} column "
                              f"{exc.colno}", "invalid_json") from exc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["-h"], ["--help"]):
        sys.stderr.write(USAGE)
        return 0
    start = time.perf_counter()
    try:
        out = evaluate(parse_request(_read_request(argv)))
        text = _dumps(out)
        code = EXIT_CODES[out["status"]]
        sys.stderr.write(f"sphere-thermal: mode={out['mode']} points={len(out['results'])} "
                         f"spheres={len(out['medium']['spheres'])} modes={out['modes']} "
                         f"status={out['status']} time={time.perf_counter() - start:.3f}s\n")
        for message in out["messages"]:
            sys.stderr.write(f"sphere-thermal: {message}\n")
    except InputError as exc:
        text = _dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                       "error": {"kind": exc.kind, "path": exc.path, "message": exc.message}})
        code = 2
        sys.stderr.write(f"sphere-thermal: error ({exc.kind}) at {exc.path}: {exc.message}\n")
    except Exception as exc:  # report unexpected failures as JSON, details on stderr
        traceback.print_exc(file=sys.stderr)
        text = _dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                       "error": {"kind": "internal", "path": None, "message": repr(exc)}})
        code = 1
    sys.stdout.write(text)
    return code


def _dumps(document):
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
