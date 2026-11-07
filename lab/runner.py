"""Bounded, deterministic tool-use baseline. No LLM, network or subjective claims."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "vendor" / "sphere-thermal-cli"


def run_experiment(request, repeats=3):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 10:
        raise ValueError("repeats must be an integer in [1, 10]")
    payload = json.dumps(request, sort_keys=True, allow_nan=False).encode()
    durations, outputs = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "sphere_thermal"], cwd=SOLVER,
                              input=payload, capture_output=True, timeout=60)
        durations.append(time.perf_counter() - start)
        outputs.append({"exit_code": proc.returncode, "response": json.loads(proc.stdout),
                        "diagnostics": proc.stderr.decode()})
    import numpy as np
    return {"schema_version": "0.1", "kind": "deterministic_tool_baseline",
            "consciousness_inference": None,
            "request_sha256": hashlib.sha256(payload).hexdigest(),
            "request": request, "repeats": repeats,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "machine": platform.machine()},
            "timing": {"scope": "CLI subprocess wall time, including Python/NumPy startup",
                       "seconds": durations, "median_seconds": statistics.median(durations)},
            "runs": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run_experiment(json.loads(args.request.read_text()), args.repeats)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return max(r["exit_code"] for r in result["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
