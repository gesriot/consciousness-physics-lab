"""Descriptive reporting from frozen confirmation summaries, without new trials."""
import argparse
import hashlib
import json
import math
from pathlib import Path

METRICS = ('false_remaining', 'accurate_withdrawn', 'actual_cost', 'correct_final',
           'accurate_ordinary_never_obtained', 'detection_delay', 'detection_censored',
           'post_onset_samples', 'corrupted_sample_fraction', 'clean_false_alarms',
           'clean_sample_expenditure', 'clean_false_alarm_cost')
CLEAN_INAPPLICABLE = ('detection_delay', 'detection_censored', 'corrupted_sample_fraction')


def stats(values):
    if any(value is None for value in values):
        return dict(per_series=values, total=None, mean=None, standard_error=None,
                    status='undefined_or_inapplicable_registered_measure')
    mean = sum(values)/len(values)
    return dict(per_series=values, total=sum(values), mean=mean,
                standard_error=math.sqrt(sum((value-mean)**2 for value in values)
                                         /(len(values)-1)/len(values)))


def summarize(archive):
    archive = Path(archive)
    rows = json.loads((archive/'series-summaries.json').read_text())
    if len(rows) != 12 or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('Expected twelve complete paired confirmation series')
    def values(key, metric):
        if key.startswith('clean-') and metric in CLEAN_INAPPLICABLE:
            return [None]*len(rows)
        return [row[key][metric] for row in rows]
    output = dict(status='post-hoc arithmetic reporting; no new experiment or policy retuning',
        source_sha256={name: hashlib.sha256((archive/name).read_bytes()).hexdigest()
                       for name in ('manifest.json', 'series-summaries.json')},
        independent_unit='series; schedules, quotas, environments and candidate variants are paired',
        N0_reuse='One trajectory per environment is reused in three quota contrasts, not three independent observations.',
        clean_inapplicable_measures=list(CLEAN_INAPPLICABLE),
        standard_error_formula='sqrt(sum((x-mean)^2)/(n-1)/n)', cells={}, paired_schedule_minus_S={})
    for key in rows[0]:
        output['cells'][key] = {metric: stats(values(key, metric)) for metric in METRICS}
        output['cells'][key]['cost_categories'] = {
            category: stats([row[key]['costs'][category] for row in rows])
            for category in rows[0][key]['costs']}
    scenarios = sorted({'-'.join(key.split('-')[:2]) for key in rows[0]})
    for scenario in scenarios:
        for quota in (1, 2, 4):
            baseline = f'{scenario}-S-q{quota}'
            for policy in ('S', 'Sx', 'Sxprime', 'RR', 'BU', 'N0'):
                key = f'{scenario}-{policy}-q{quota if policy != "N0" else 0}'
                output['paired_schedule_minus_S'][f'{scenario}-{policy}-matched-q{quota}'] = {
                    metric: stats([None if x is None or y is None else x-y
                                   for x, y in zip(values(key, metric), values(baseline, metric))])
                    for metric in METRICS}
    return output


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=here/'confirmation-first')
    parser.add_argument('--output', type=Path, default=here/'descriptive-statistics.json')
    args = parser.parse_args()
    result = summarize(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
