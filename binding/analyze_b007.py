"""Summarize saved fallible-diagnosis trajectories without generating outcomes."""
import argparse
import math
from pathlib import Path
from binding import run_b007 as saved
from binding import analyze_b006 as common_analysis

PRIMARY = ('false_accusations', 'remaining_bad', 'lost_good', 'cost')


_compare = common_analysis._compare
_empty_secondary = common_analysis._empty_secondary
_add_secondary = common_analysis._add_secondary
_finish_secondary = common_analysis._finish_secondary
_empty_calibration = common_analysis._empty_calibration
_add_calibration = common_analysis._add_calibration
_finish_calibration = common_analysis._finish_calibration


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
                for control in dict.fromkeys(('D', 'R', selection['selected_K'], 'P0_'+variant[2:], 'Pstar_'+variant[2:]))}
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
    names, seeds, consumed = [], [], {}
    for name, hashes in sorted(manifest['files'].items()):
        saved.verify_file(directory/name, hashes)
        archive = saved.read(directory/name)
        if (type(archive['seed']) is not int or
                archive['intake_sha256'] != saved.common.B.digest(archive['inputs']) or
                archive['intake_sha256'] != manifest['intakes'][str(archive['seed'])]):
            raise ValueError('Intake identity mismatch')
        names.append(name)
        seeds.append(archive['seed'])
        consumed[name] = hashes
    if sorted(seeds) != sorted(manifest['seeds']):
        raise ValueError('Series list mismatch')
    archives = saved.ArchiveCollection(directory,names)
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
