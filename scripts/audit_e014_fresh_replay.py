"""Supplemental fresh FV replay from unchanged E009 and E011 saved requests."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from lab import convergence_candidates as candidates
from lab.coverage_map import pair, write
from lab.experiment_audit import metadata
from lab.extrapolation import graded_model, combine_verdict
from lab.surface_order import surface_response, faces
from lab.three_way import GRIDS

SCRIPT = 'scripts/audit_e014_fresh_replay.py'


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def run(directory):
    output = Path(directory)
    if output.exists():
        raise ValueError('Output directory already exists')
    provenance = metadata(candidates.PROTOCOL, candidates.SOURCES + [SCRIPT])
    output.mkdir(parents=True)
    started = time.perf_counter()
    archived = read_gzip('results/E014-convergence-candidates/first/cost-replay.json.gz')
    datasets, paths, checks = {}, {}, []
    hashes = {}
    # Finish and persist every fresh-path comparison BEFORE candidate replay.
    for filename in candidates.ARCHIVES:
        raw = Path(filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        experiment = data['experiment']
        datasets[experiment] = data
        for task in data['tasks']:
            req = task['request']
            z, t = complex(*req['z']), req['mean_time']
            models = ('surface',) if experiment == 'E009' else ('sphere', 'model_scale', 'slab')
            for model in models:
                values = [surface_response(z, t, n) if experiment == 'E009'
                          else graded_model(z, t, n, model) for n in GRIDS]
                paths[experiment, task['task_id'], model] = values
                expected = [complex(*v) for v in task['shadow_values'][model]]
                checks.append(dict(experiment=experiment, task_id=task['task_id'], model=model,
                    request=req, values=[pair(v) for v in values],
                    exact_by_grid={str(n): a == b for n, a, b in zip(GRIDS, values, expected)},
                    max_absolute_difference=max(abs(a-b) for a, b in zip(values, expected))))
    path_passed = all(all(row['exact_by_grid'].values()) for row in checks)
    hash_passed = hashes == archived['source_sha256']
    path_report = dict(**provenance,
        archive_sha256=hashes, archive_hashes_match_original=hash_passed,
        exact_paths=path_passed, path_count=len(checks), fv_solves=len(checks)*len(GRIDS),
        cells_solved=len(checks)*sum(GRIDS), paths=checks)
    write(output/'fresh-paths.json.gz', path_report)
    if not path_passed or not hash_passed:
        result = dict(passed=False, exact_paths=path_passed,
                      archive_hashes_match_original=hash_passed, candidate_replay_performed=False)
        write(output/'summary.json', result)
        return result
    rows = []
    for experiment, data in datasets.items():
        tasks = {task['task_id']: task for task in data['tasks']}
        for old in data['runs']:
            if experiment == 'E009' and old['cell'] != 'surface/frozen':
                continue
            task = tasks[old['task_id']]
            req = task['request']
            z, t = complex(*req['z']), req['mean_time']
            model = old.get('model', 'sphere')
            path_key = 'surface' if experiment == 'E009' else model
            values = paths[experiment, old['task_id'], path_key]
            kappa = 1/((3 if model == 'slab' else 15000 if model == 'model_scale' else 15)*t)
            layers = {n: math.sqrt(kappa/abs(z))/(1-faces(n)[-2]) for n in GRIDS}
            outcomes = {}
            for candidate in candidates.CANDIDATES:
                replay = candidates.replay_path(values, complex(*old['outcome']['value']),
                                               req['tolerance'], layers, candidate)
                if experiment == 'E011':
                    verdict, reason = combine_verdict(replay['verdict'],
                        old['resolution']['accepted'], old['checks'])
                    accepted = verdict == 'accept'
                else:
                    verdict, reason = replay['verdict'], 'numerical'
                    accepted = verdict == 'accept' and old['resolution']['accepted']
                outcomes[candidate] = dict(replay, combined_verdict=verdict,
                    accepted=accepted, combined_reason=reason)
            rows.append(dict(experiment=experiment, task_id=old['task_id'], cohort=old['cohort'],
                fault=old['fault'], accurate=old['accurate'], candidates=outcomes))
    row_comparison = [a == b for a, b in zip(rows, archived['rows'])]
    exact_rows = len(rows) == len(archived['rows']) and all(row_comparison)
    # Per-row equality covers every decision step, verdict, stop reason and cost.
    # Also recompute aggregate operational cells and endpoint changes independently.
    totals = {}
    for experiment in ('E009', 'E011', 'combined'):
        subset = [r for r in rows if experiment == 'combined' or r['experiment'] == experiment]
        totals[experiment] = {}
        for candidate in candidates.CANDIDATES:
            out = dict(rows=len(subset), accurate_accepts=0, lost_accurate_accepts=0,
                gained_accurate_accepts=0, false_accepts=0, new_false_accepts=0,
                violation_verdict_changes=0, numerical_violation_changes=0, decision_cells=0)
            for row in subset:
                a, b = row['candidates'][candidate], row['candidates']['C0']
                accurate = row['accurate']
                out['accurate_accepts'] += accurate and a['accepted']
                out['lost_accurate_accepts'] += accurate and b['accepted'] and not a['accepted']
                out['gained_accurate_accepts'] += accurate and not b['accepted'] and a['accepted']
                out['false_accepts'] += not accurate and a['accepted']
                out['new_false_accepts'] += not accurate and not b['accepted'] and a['accepted']
                out['violation_verdict_changes'] += ((a['combined_verdict']=='violation_detected')
                                                     != (b['combined_verdict']=='violation_detected'))
                out['numerical_violation_changes'] += ((a['verdict']=='violation_detected')
                                                       != (b['verdict']=='violation_detected'))
                out['decision_cells'] += a['cells_solved']
            totals[experiment][candidate] = out
    exact_totals = totals == archived['summary']
    write(output/'fresh-cost-replay.json.gz', dict(rows=rows, summary=totals))
    result = dict(passed=bool(exact_rows and exact_totals), exact_paths=path_passed,
        archive_hashes_match_original=hash_passed, candidate_replay_performed=True,
        fresh_paths=len(checks), fresh_fv_solves=len(checks)*len(GRIDS),
        fresh_fv_cells=len(checks)*sum(GRIDS), rows=len(rows),
        exact_candidate_rows=exact_rows, exact_summary=exact_totals,
        differing_row_indices=[i for i, same in enumerate(row_comparison) if not same],
        seconds=time.perf_counter()-started)
    write(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    result = run(args.directory)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
