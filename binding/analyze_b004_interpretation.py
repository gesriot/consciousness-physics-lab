"""Describe priority selection and stopping losses in the saved B004 archive."""
import argparse
import gzip
import json
from pathlib import Path


def analyze(directory):
    cells = 0
    lost_histogram = {0: 0, 1: 0}
    deep_records = 0
    mechanism = {}
    for path in sorted(Path(directory).glob('series-*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            archive = json.load(stream)
        for fault, case in archive['cases'].items():
            series = case['series']
            records = {r['id']: r for r in series['records']}
            bad = {cid for cid, value in series['truth'].items()
                   if abs(records[cid]['value'] - value) > 1e-10 + 1e-5 * abs(value)}
            for record in records.values():
                if record['depth'] in (2, 3):
                    assert record['direct'] == []
                    deep_records += 1
            for label, ref in case['conditions'].items():
                output = case['outputs'][ref]
                cells += 1
                lost = {cid for cid, state in output['final'].items()
                        if not state['accepted'] and cid not in bad}
                assert len(lost) == output['metrics']['correct_lost']
                assert len(lost) in (0, 1)
                lost_histogram[len(lost)] += 1
                if lost:
                    assert output['trace'][-1]['kind'] == 'withdraw'
                    assert output['trace'][-1]['id'] in lost
                policy, budget, branch = label.split('|')
                if fault == 'clean' or policy not in ('G', 'Gx') or budget != '8':
                    continue
                row = mechanism.setdefault(f'{fault}|{branch}|{policy}',
                    dict(priority_selected=0, priority_bad=0, initial_bad=0,
                         unnecessary=0, unnecessary_after_candidates=0))
                implicated = set(output['protected']['implicated'])
                actions = [e for e in output['trace'] if e['kind'] in ('repair', 'withdraw')]
                seen = set()
                blind_seen = False
                row['initial_bad'] += len(bad)
                for event in actions:
                    cid = event['id']
                    if cid in implicated:
                        assert not blind_seen
                        row['priority_selected'] += 1
                        row['priority_bad'] += cid in bad
                    else:
                        blind_seen = True
                    if cid not in bad:
                        row['unnecessary'] += 1
                        row['unnecessary_after_candidates'] += implicated <= seen
                    seen.add(cid)
    return dict(conditions=cells, final_correct_loss_histogram=lost_histogram,
                deep_records_without_direct_attributes=deep_records,
                mechanism=mechanism,
                scope='Descriptive reanalysis of stored confirmation data; no new trials.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='binding/results/B004/confirmation-first')
    parser.add_argument('--output')
    args = parser.parse_args()
    payload = json.dumps(analyze(args.input), sort_keys=True, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(payload)
    else:
        print(payload, end='')
