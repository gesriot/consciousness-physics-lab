"""Build S2 exclusively from explicitly enumerated development archives."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from attention.s2 import VERDICTS, features, _probs

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'results/E016-frozen-pipeline/results.json',
    'attention/results/A001/results.json',
    'results/E014-convergence-candidates/first/map.json.gz',
    'results/E015-collocation-map/map.json.gz',
    'results/common-mode/summary.md',
    'results/E010-geometry-invariants/summary.md',
    'results/E013-matched-mixture/summary.md',
    'attention/self_model.json',
]


def read(name):
    path = ROOT/name
    with gzip.open(path, 'rt') if name.endswith('.gz') else path.open() as stream:
        return json.load(stream)


def build():
    action_model = read('attention/self_model.json')
    names = ['modal', 'binding', 'normalization', 'first_moment', 'second_moment', 'high_frequency', 'collocation'] + [f'v2_{n}' for n in (64,128,256,512,1024,2048,4096)]
    maps = {}
    for name, path in [('v2', SOURCES[2]), ('collocation', SOURCES[3])]:
        points = read(path)['points']
        maps[name] = []
        for p in points:
            row = [p['log10_magnitude']/6, p['phase']/(math.pi/2)]
            if name == 'v2':
                row += [s['uncertainty'] if s['candidates']['C1'] else None for s in p['steps']]
            else:
                row += [p['difference'], p['uncertainty']]
            maps[name].append(row)
    model = dict(version=2, neighbors=16, action_model=action_model, action_names=names, maps=maps,
                 development_sources_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
                 feature_names=['log10_abs_zT/6','signed_phase/(pi/2)','log10_tolerance/3','modal_attainable_4096','v2_map_accept_feasible','collocation_map_accept_feasible'],
                 smoothing='One pseudocount per verdict; beta(1,1) for inaccurate given violation.',
                 map_rule='Nearest archived point in the first two feature coordinates; archive-order ties. Empirical prediction, not a proof. V2 any C1-converging U<=tolerance; collocation difference<=tolerance/8 and U<=tolerance.',
                 modal_rule='tail_bound(request,4096)+128eps<=tolerance; request-only resolution heuristic, not a value computation.',
                 corruption='D swaps accept/inconclusive counts in records; Dprime same swap in unused_records.',
                 allocation_prediction='D redirects full-completion allocation away from predicted acceptance toward predicted inconclusive requests.',
                 evidence_notes={'E008':'Binding catches request changes; present conditions do not contain request swaps.', 'E010':'Geometry faults evade normalization/mean but second moment and high frequency can detect them.', 'E013':'Matched mixture escapes all four feature checks.', 'E014':'V2 uses ordinary C1 contraction; roundoff exception is unsafe.', 'E015':'Collocation floor blocks acceptance below 1.9073486328125e-6, not necessarily detection.'})
    records = []
    for path in SOURCES[:2]:
        data = read(path)
        intake = {r['task_id']:r['request'] for r in data['intake_channel']}
        grouped = {}
        for r in data['runs']:
            grouped.setdefault(r['task_id'], []).append(r)
        for task_id, runs in sorted(grouped.items()):
            row = dict(id=f"{data['experiment']}:{task_id}", features=features(intake[task_id], model)[0], counts=[0,0,0], inaccurate_violations=0, actions={a:[0,0,0] for a in names})
            for r in runs:
                if data['experiment']=='A001':
                    verdict = r['full']['verdict']
                    actions = {a['action']:a['observation']['verdict'] for a in r['full']['actions'] if a['action']!='declare'}
                else:
                    d = r['decision']
                    verdict = d['policies']['strict']['verdict']
                    actions = {'modal':'accept' if d['resolution']['accepted'] else 'inconclusive', 'collocation':d['third']['verdict']}
                    for a in names[1:6]:
                        actions[a] = 'accept' if d['checks']['hash' if a=='binding' else a] else 'violation_detected'
                    for grid in d['independent']['grids']:
                        actions[f'v2_{grid}'] = 'inconclusive'
                    for step in d['independent']['steps']:
                        actions[f"v2_{step['last_grid']}"] = step['verdict']
                row['counts'][VERDICTS.index(verdict)] += 1
                row['inaccurate_violations'] += int(verdict=='violation_detected' and not r['accurate'])
                for a, v in actions.items():
                    row['actions'][a][VERDICTS.index(v)] += 1
            records.append(row)
    model['records'] = records
    model['unused_records'] = copy.deepcopy(records)
    counts = [sum(r['counts'][j] for r in records) for j in range(3)]
    model['base_probabilities'] = _probs(counts)
    model['development_counts'] = counts
    model['development_requests'] = len(records)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite frozen model')
    args.output.write_text(json.dumps(build(), separators=(',', ':'), allow_nan=False)+'\n')
