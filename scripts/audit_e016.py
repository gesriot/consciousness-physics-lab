#!/usr/bin/env python3
"""Recompute E016 endpoints and compare all non-timing scientific fields."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.pipeline_holdout import COHORTS, CONDITIONS, scientific_view, summarize
from scripts.compare_experiment_runs import differences


def audit(original, repeat):
    result = {}
    for label, data in (('original', original), ('repeat', repeat)):
        failures = differences(scientific_view(data['summary']), scientific_view(summarize(data['runs'])))
        for cohort in COHORTS:
            failures += differences(scientific_view(data['cohort_summary'][cohort]),
                scientific_view(summarize([r for r in data['runs'] if r['cohort'] == cohort])), path=cohort)
            for condition in CONDITIONS:
                failures += differences(scientific_view(data['cohort_condition_summary'][cohort][condition]),
                    scientific_view(summarize([r for r in data['runs']
                        if r['cohort'] == cohort and r['condition'] == condition])), path=cohort + '.' + condition)
        for condition in CONDITIONS:
            failures += differences(scientific_view(data['condition_summary'][condition]),
                scientific_view(summarize([r for r in data['runs'] if r['condition'] == condition])), path=condition)
        result[label + '_summary_mismatches'] = failures
    result['scientific_mismatches'] = differences(scientific_view(original), scientific_view(repeat))
    result['excluded_provenance'] = ['environment', 'git_head', 'source_sha256', 'runtime_override', 'replay']
    result['excluded_measures'] = 'Keys ending in seconds only; all other scientific fields compared exactly'
    result['passed'] = not any(result[k] for k in ('original_summary_mismatches', 'repeat_summary_mismatches', 'scientific_mismatches'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit output exists; preserve it and choose a new path')
    original, repeat = [json.loads(p.read_bytes()) for p in (args.original, args.repeat)]
    result = audit(original, repeat)
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.original, args.repeat)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
