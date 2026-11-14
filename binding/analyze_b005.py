"""Describe saved B005 trajectories without generating or retuning an experiment."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

PRIMARY = ('remaining_bad', 'withdrawn_good', 'false_accusations', 'cost')
CORRUPTIONS = ('units', 'version', 'regime', 'intermediate')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def aggregate(rows):
    """Rows are independent series, never individual dependent conditions."""
    keys = rows[0].keys()
    n = len(rows)
    totals = {key: math.fsum(row[key] for row in rows) for key in keys}
    means = {key: totals[key] / n for key in keys}
    errors = {key: (math.sqrt(math.fsum((row[key]-means[key])**2 for row in rows)
                              / (n*(n-1))) if n > 1 else 0.0) for key in keys}
    return dict(independent_series=n, total=totals, mean=means, standard_error=errors)


def localization(rows):
    result = aggregate(rows)
    total = result['total']
    opportunities = total['localization_opportunities']
    correct = total['correctly_localized']
    result['correct_localization_rate'] = correct / opportunities if opportunities else None
    result['mean_checks_for_correct_localization'] = (
        total['checks_for_correct_localization'] / correct if correct else None)
    result['opportunity_note'] = 'No signal or no true culprit is not a localization opportunity.'
    return result


def action_sequence(output):
    """Selected operations and targets, excluding scores and numerical answers."""
    return [{key: row[key] for key in ('kind', 'id', 'candidate') if key in row}
            for row in output['trace']]


def analyze(directory):
    directory = Path(directory)
    manifest = read(directory / 'manifest.json')
    scope_rows, primary_rows, depth_rows, false_rows = {}, {}, {}, {}
    pair_cells, size_samples, fingerprints = {}, {}, {}
    sequences_per_scope = {}
    consumed = {}
    series_count = 0
    for filename, hashes in sorted(manifest['files'].items()):
        path = directory / filename
        archive = read(path)
        canonical_sha = hashlib.sha256(canonical(archive)).hexdigest()
        if canonical_sha != hashes['canonical_json_sha256']:
            raise ValueError('Canonical series hash mismatch: ' + filename)
        consumed[filename] = dict(canonical_json_sha256=canonical_sha,
                                  compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        series_count += 1
        series_primary, series_false, series_patterns = {}, {}, {}
        for case_name, case in sorted(archive['cases'].items()):
            fault, scope = case_name.split('|')
            series = case['series']
            if (series['fault'], series['scope']) != (fault, scope):
                raise ValueError('Case metadata mismatch')
            for label, ref in sorted(case['conditions'].items()):
                policy, budget, signal = label.split('|')
                output = case['outputs'][ref]
                metric = output['metrics']
                # Accumulate scopes before across-series statistics.
                key = f'{fault}|{budget}|{signal}|{policy}'
                primary = {name: metric[name] for name in PRIMARY}
                primary.update(unused_budget=metric['unused_budget'], suspected=metric['suspected'],
                               initial_bad=metric['initial_bad'], resolved=metric['resolved'])
                scope_rows.setdefault(f'{fault}|{budget}|{signal}|{scope}|{policy}', []).append(primary)
                combined = series_primary.setdefault(key, {name: 0 for name in primary})
                for name, value in primary.items():
                    combined[name] += value
                signal_depth = metric['signal_depth']
                loc = {name: metric[name] for name in (
                    'localization_opportunities', 'correctly_localized', 'false_accusations',
                    'checks', 'checks_to_localization', 'cost', 'resolved', 'suspected')}
                loc['checks_for_correct_localization'] = (
                    metric['checks_to_localization'] if metric['correctly_localized'] else 0)
                depth_key = f'{fault}|{budget}|{signal}|{scope}|{signal_depth}|{policy}'
                depth_rows.setdefault(depth_key, []).append(loc)
                if fault in ('false_measurement', 'clean'):
                    row = {name: metric[name] for name in (
                        'remaining_bad', 'withdrawn_good', 'false_accusations', 'cost',
                        'unused_budget', 'resolved', 'suspected', 'repeated_measurement', 'checks')}
                    row['signals'] = int(bool(output['protected']['signal_ids']))
                    combined_false = series_false.setdefault(key, {name: 0 for name in row})
                    for name, value in row.items():
                        combined_false[name] += value
                sizes = size_samples.setdefault(policy, dict(record_bytes=[], memory_bytes=[], capacity_bytes=[]))
                sizes['record_bytes'].extend(output['diagnostics']['record_bytes'])
                sizes['memory_bytes'].append(metric['memory_bytes'])
                sizes['capacity_bytes'].append(metric['capacity_bytes'])
                if fault in CORRUPTIONS:
                    series_patterns.setdefault((scope, label), {})[fault] = action_sequence(output)
                if policy == 'G':
                    other = case['outputs'][case['conditions'][f'L|{budget}|{signal}']]
                    pair = pair_cells.setdefault(f'{fault}|{budget}|{signal}|{scope}',
                        dict(conditions=0, selected_action_differences=0,
                             protected_action_differences=0, primary_metric_differences=0))
                    pair['conditions'] += 1
                    pair['selected_action_differences'] += action_sequence(output) != action_sequence(other)
                    pair['protected_action_differences'] += (
                        output['protected']['actions'] != other['protected']['actions'])
                    pair['primary_metric_differences'] += any(
                        metric[name] != other['metrics'][name] for name in PRIMARY)
        for key, row in series_primary.items():
            primary_rows.setdefault(key, []).append(row)
        for key, row in series_false.items():
            false_rows.setdefault(key, []).append(row)
        for (scope, label), patterns in series_patterns.items():
            faults = sorted(patterns)
            for index, left in enumerate(faults):
                for right in faults[index+1:]:
                    key = f'{scope}|{label}|{left}={right}'
                    row = fingerprints.setdefault(key, dict(paired_series=0, identical_action_sequences=0))
                    row['paired_series'] += 1
                    row['identical_action_sequences'] += patterns[left] == patterns[right]
            groups = {}
            for fault, pattern in patterns.items():
                groups.setdefault(hashlib.sha256(canonical(pattern)).hexdigest(), []).append(fault)
            sequences_per_scope.setdefault(scope+'|'+label, []).append(sorted(sorted(g) for g in groups.values()))
    if not series_count:
        raise ValueError('No complete series in manifest')
    summaries = {key: aggregate(rows) for key, rows in sorted(primary_rows.items())}
    size_report = {}
    for policy, samples in sorted(size_samples.items()):
        size_report[policy] = {name: dict(observations=len(values), minimum=min(values), maximum=max(values),
                                         mean=math.fsum(values)/len(values))
                               for name, values in samples.items()}
    closure_l, closure_f = [], []
    for key, cell in summaries.items():
        if not key.endswith('|G'):
            continue
        prefix = key[:-1]
        for measure in PRIMARY:
            g = cell['total'][measure]
            l = summaries[prefix+'L']['total'][measure]
            f = summaries[prefix+'F']['total'][measure]
            if l > g:
                closure_l.append(dict(cell=prefix[:-1], measure=measure, G=g, L=l))
            if f > g or f > l:
                closure_f.append(dict(cell=prefix[:-1], measure=measure, G=g, L=l, F=f))
    return dict(
        scope='Descriptive analysis of saved series; no new experiment or policy changes.',
        phase=manifest['phase'], independent_series=series_count,
        primary_cell_key='fault|budget|signal|policy; scopes summed within series',
        primary_cells=summaries,
        scope_cell_key='fault|budget|signal|scope|policy',
        scope_cells={key: aggregate(rows) for key, rows in sorted(scope_rows.items())},
        localization_cell_key='fault|budget|signal|scope|signal_depth|policy',
        localization_by_signal_depth={key: localization(rows) for key, rows in sorted(depth_rows.items())},
        false_measurement_and_clean={key: aggregate(rows) for key, rows in sorted(false_rows.items())},
        G_L_comparison=pair_cells,
        G_L_total_selected_action_differences=sum(r['selected_action_differences'] for r in pair_cells.values()),
        G_L_total_protected_action_differences=sum(r['protected_action_differences'] for r in pair_cells.values()),
        G_L_compared_conditions=sum(r['conditions'] for r in pair_cells.values()),
        storage_bytes=size_report,
        storage_note='Observed serialized record and memory bytes, not Python heap size; repeated conditions are dependent.',
        dependency_patterns=fingerprints,
        dependency_note='Equal selected operations/targets within paired cases, excluding answers and scores; not independent replications.',
        fault_action_equivalence_by_series=sequences_per_scope,
        closure_L_no_worse_than_G=not closure_l,
        closure_F_no_worse_than_G_and_L=not closure_f,
        closure_L_failures=closure_l, closure_F_failures=closure_f,
        input_hashes=consumed,
        uncertainty_note='Standard errors use independent series; scopes and claims do not add independent samples.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Saved complete B005 run directory')
    parser.add_argument('--output', help='JSON destination; stdout if omitted')
    args = parser.parse_args()
    result = canonical(analyze(args.input)) + b'\n'
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result)
    else:
        print(result.decode(), end='')


if __name__ == '__main__':
    main()
