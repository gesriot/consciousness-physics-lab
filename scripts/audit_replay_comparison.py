#!/usr/bin/env python3
"""Exercise archived repeats and explicitly synthetic last-bit perturbations."""
import copy
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_experiment_runs import compare_details, numeric_summary, input_digest


def perturb(value):
    if isinstance(value, dict):
        return {key: (item if key in ('request', 'intake_channel', 'sampled_magnitude',
                                      'sampled_phase') else perturb(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [perturb(item) for item in value]
    return math.nextafter(value, math.inf) if isinstance(value, float) else value


def run():
    root = Path(__file__).resolve().parents[1] / 'results'
    reports = []
    for original_path, repeat_path in [
        ('surface-order/results.json', 'platform-replay/e009-replay.json'),
        ('E011-extrapolation/results.json', 'E011-extrapolation/repeat.json')]:
        original = json.loads((root / original_path).read_text())
        repeat = json.loads((root / repeat_path).read_text())
        changed = perturb(repeat)
        failures, provenance, diagnostic = compare_details(original, changed, 'decisions')
        control = copy.deepcopy(changed)
        control['runs'][0]['accepted'] = not control['runs'][0]['accepted']
        control_failures = compare_details(original, control, 'decisions')[0]
        summary = numeric_summary(repeat, changed)
        reports.append(dict(experiment=original['experiment'],
            original=original_path, repeat=repeat_path, repeat_environment=repeat['environment'],
            archived_failures=len(compare_details(original, repeat, 'decisions')[0]),
            synthetic_failures=len(failures), changed_numeric_fields=sum(r['changed'] for r in summary.values()),
            exact_synthetic_failures=len(compare_details(repeat, changed, 'exact')[0]),
            synthetic_diagnostic_differences=len(diagnostic),
            changed_decision_detected=bool(control_failures),
            canonical_inputs_equal=input_digest(original) == input_digest(changed),
            synthetic_numeric_summary=summary))
        assert not failures and control_failures
    return dict(description='Bundled archival repeats and synthetic nextafter perturbations; not a macOS rerun',
                reports=reports)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
