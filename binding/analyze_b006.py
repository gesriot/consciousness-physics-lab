"""Summarize saved fallible-diagnosis trajectories without generating outcomes."""
import argparse
import math
from pathlib import Path
from binding import run_b006 as saved

PRIMARY = ('false_accusations', 'remaining_bad', 'lost_good', 'cost')


def _compare(primary, variant, comparator):
    cells = {}
    for key, row in sorted(primary.items()):
        env, name, budget, branch = key.split('|')
        if name != variant:
            continue
        other = primary[f'{env}|{comparator}|{budget}|{branch}']['total']
        values = {m: row['total'][m] for m in PRIMARY}
        controls = {m: other[m] for m in PRIMARY}
        delta = {m: values[m] - controls[m] for m in PRIMARY}
        no_worse = all(x <= 0 for x in delta.values())
        no_better = all(x >= 0 for x in delta.values())
        relation = ('equal' if all(x == 0 for x in delta.values()) else
                    'P_dominates' if no_worse else
                    'control_dominates' if no_better else 'tradeoff')
        cells[f'{env}|{budget}|{branch}'] = {
            'P': values, 'control': controls, 'P_minus_control': delta,
            'relation': relation, 'P_no_worse': no_worse,
            'contributing_series': row['series']}
    return {'comparator': comparator, 'cells': cells,
            'relations': {r: sum(c['relation'] == r for c in cells.values())
                          for r in ('equal', 'P_dominates', 'control_dominates', 'tradeoff')}}


def _empty_secondary():
    return {k: 0 for k in ('episodes', 'signals', 'false_objections', 'true_faults',
                          'rejected_false', 'corrected_counterexample', 'checks',
                          'repeated_checks', 'observation_repeats', 'false_accusations',
                          'attempted_false_accusations', 'remaining_bad', 'lost_good',
                          'cost', 'suspected', 'diagnosis_made', 'diagnosis_correct')}


def _add_secondary(container, key, seed, episode, out):
    row = container.setdefault(key, {'totals': _empty_secondary(), 'seeds': set()})
    row['seeds'].add(seed)
    totals = row['totals']
    totals['episodes'] += 1
    totals['signals'] += bool(episode['signal_ids'])
    totals['false_objections'] += episode['evaluator']['kind'] == 'false_objection'
    totals['true_faults'] += episode['evaluator']['kind'] == 'true_fault'
    for name in totals:
        if name not in ('episodes', 'signals', 'false_objections', 'true_faults'):
            totals[name] += out['metrics'].get(name, 0)


def _finish_secondary(container):
    answer = {}
    for key, row in sorted(container.items()):
        t = row['totals']
        answer[key] = {'dependent_episode_conditions': t['episodes'],
                       'independent_series': len(row['seeds']), 'total': t,
                       'false_objections_rejected_fraction':
                       t['rejected_false'] / t['false_objections'] if t['false_objections'] else None,
                       'real_counterexamples_corrected_fraction':
                       t['corrected_counterexample'] / t['true_faults'] if t['true_faults'] else None}
    return answer


def _empty_calibration():
    return {'samples': [], 'abstentions': 0, 'no_signal': 0, 'missing_probability': 0,
            'abstention_probabilities': [], 'seeds': set()}


def _add_calibration(row, seed, episode, out, fixed_confidences, variant):
    row['seeds'].add(seed)
    if not episode['signal_ids']:
        row['no_signal'] += 1
        return
    probability = out['terminal']['probability']
    if out['metrics']['abstained']:
        row['abstentions'] += 1
        if probability is not None:
            row['abstention_probabilities'].append(probability)
        return
    if not out['metrics']['diagnosis_made']:
        raise ValueError('Signaled episode has neither diagnosis nor abstention')
    if probability is None:
        probability = fixed_confidences.get(variant)
    if probability is None:
        row['missing_probability'] += 1
        return
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid reported probability')
    row['samples'].append((seed, probability, out['metrics']['diagnosis_correct']))


def _finish_calibration(container, phase):
    result = {}
    for key, row in sorted(container.items()):
        samples = row['samples']
        series = {}
        for seed, probability, target in samples:
            series.setdefault(seed, []).append((probability-target)**2)
        series_means = [math.fsum(values) / len(values) for values in series.values()]
        mean = math.fsum(series_means) / len(series_means) if series_means else None
        n = len(series_means)
        se = (math.sqrt(math.fsum((x-mean)**2 for x in series_means)/(n*(n-1)))
              if n > 1 else None)
        bins = []
        for index in range(10):
            subset = [sample for sample in samples if min(9, int(sample[1]*10)) == index]
            bins.append({'lower': index/10, 'upper': (index+1)/10,
                         'dependent_decisions': len(subset),
                         'independent_series': len({x[0] for x in subset}),
                         'mean_probability': math.fsum(x[1] for x in subset)/len(subset) if subset else None,
                         'observed_accuracy': math.fsum(x[2] for x in subset)/len(subset) if subset else None})
        result[key] = {
            'independent_series': len(row['seeds']),
            'dependent_decisions': len(samples), 'abstentions': row['abstentions'],
            'no_signal': row['no_signal'], 'missing_probability': row['missing_probability'],
            'abstention_mean_reported_probability':
                math.fsum(row['abstention_probabilities']) / len(row['abstention_probabilities'])
                if row['abstention_probabilities'] else None,
            'brier_pooled_descriptive': math.fsum((p-y)**2 for _, p, y in samples)/len(samples) if samples else None,
            'brier_series_mean': mean, 'brier_series_standard_error': se,
            'series_with_decisions': n, 'bins': bins,
            'fixed_confidence_origin': 'development_in_sample' if phase == 'development' else 'frozen_development',
        }
    return result


def analyze_archives(archives, phase, selection):
    """Pure saved-data aggregation. No case generation or policy simulation."""
    if phase not in ('development', 'confirmation') or not archives:
        raise ValueError('Expected nonempty development or confirmation archives')
    seeds = [a['seed'] for a in archives]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate independent series')
    if phase == 'confirmation' and not selection.get('pass_gate'):
        raise ValueError('Confirmation requires a successful frozen gate')
    confidences = selection['fixed_confidences']
    summary = saved.summarize(archives, confidences)
    primary = summary['primary_cells']
    variants = sorted({key.split('|')[1] for key in primary})
    comparisons = {}
    for variant in variants:
        if variant.startswith('P_'):
            comparisons[variant] = {
                control: _compare(primary, variant, control)
                for control in dict.fromkeys(('D', 'R', selection['selected_K']))}
    signal_groups, depths, scopes, calibration = {}, {}, {}, {}
    dependent_conditions = 0
    for archive in archives:
        for environment, body in archive['environments'].items():
            episodes = body['series']['episodes']
            for condition, outputs in body['outputs'].items():
                if len(outputs) != len(episodes):
                    raise ValueError('Truncated episode-condition sequence')
                variant, budget = condition.split('|')
                for episode, out in zip(episodes, outputs):
                    dependent_conditions += 1
                    prefix = f"{environment}|{variant}|{budget}|{episode['signal_branch']}"
                    _add_secondary(signal_groups, prefix+'|'+episode['evaluator']['kind'], archive['seed'], episode, out)
                    _add_secondary(depths, prefix+'|'+str(out['metrics']['signal_depth']), archive['seed'], episode, out)
                    _add_secondary(scopes, prefix+'|'+episode['evaluator']['scope'], archive['seed'], episode, out)
                    _add_calibration(calibration.setdefault(prefix, _empty_calibration()), archive['seed'],
                                     episode, out, confidences, variant)
    closure = (saved.closure(summary, selection) if phase == 'confirmation' else
               {'evaluated': False, 'reason': 'Development is not confirmation; no closure claim.'})
    if phase == 'confirmation':
        closure['evaluated'] = True
    return {
        'phase': phase, 'independent_series': len(archives),
        'dependent_episode_conditions': dependent_conditions,
        'independence_note': 'Episodes, environments, policies and budgets within a series are paired and dependent.',
        'gate_status': {k: selection[k] for k in ('pass_gate', 'common_gate', 'selected_K', 'selected_P',
                       'best_constant', 'configurations', 'exposure', 'false_objections',
                       'D_false_accusations_moderate') if k in selection},
        'P_comparisons': comparisons,
        'secondary_by_signal_kind': _finish_secondary(signal_groups),
        'secondary_by_depth': _finish_secondary(depths),
        'secondary_by_scope': _finish_secondary(scopes),
        'calibration_by_primary_cell': _finish_calibration(calibration, phase),
        'closure': closure,
        'limitations': [
            'Certified repairs are not fallible; false accusation and wrong final value are distinct.',
            'Fractional posterior self-training can reinforce an incorrect diagnosis.',
            'Brier scores exclude abstentions from scoring but report their count and confidence separately.',
            'Pooled rates are descriptive dependent observations; uncertainty is computed across series only.',
            'A tradeoff or failure of dominance is not general superiority.',
        ],
    }


def analyze(directory, freeze=None):
    directory = Path(directory)
    manifest = saved.read(directory/'manifest.json')
    if manifest['phase'] == 'development':
        selection = saved.read(directory/'gate.json')
    else:
        if freeze is None:
            raise ValueError('Confirmation analysis requires frozen selection')
        if saved.sha_file(freeze) != manifest['freeze_sha256']:
            raise ValueError('Freeze hash mismatch')
        selection = saved.read(freeze)['gate']
    archives, consumed = [], {}
    for name, hashes in sorted(manifest['files'].items()):
        saved.verify_file(directory/name, hashes)
        archive = saved.read(directory/name)
        if archive['intake_sha256'] != manifest['intakes'][str(archive['seed'])]:
            raise ValueError('Intake identity mismatch')
        archives.append(archive)
        consumed[name] = hashes
    if sorted(a['seed'] for a in archives) != sorted(manifest['seeds']):
        raise ValueError('Series list mismatch')
    result = analyze_archives(archives, manifest['phase'], selection)
    result['consumed_files'] = consumed
    result['manifest_sha256'] = saved.sha_file(directory/'manifest.json')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--freeze')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    saved.write(args.output, analyze(args.directory, args.freeze))


if __name__ == '__main__':
    main()
