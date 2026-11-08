"""Freeze joint categorical traces from A001 and A002 development only."""
import argparse
import hashlib
from pathlib import Path
from attention import agent, s2, s3
from attention.io import read_json, write_json, source_bytes

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['attention/results/A001/results.json', 'attention/results/A002/results.json.gz',
           'attention/self_model_s2.json', 'attention/self_model.json']


def build():
    model2 = s2.load_model()
    groups, trajectories = [], []
    for path in SOURCES[:2]:
        data = read_json(ROOT/path)
        requests = {r['task_id']: r['request'] for r in data['intake_channel']}
        grouped = {}
        rows = data['runs'] if data['experiment'] == 'A001' else [r for p in data['packages'] for r in p['inputs']]
        for row in rows: grouped.setdefault(row['task_id'], []).append(row)
        for task_id, runs in sorted(grouped.items()):
            group = len(groups)
            groups.append(dict(id=f"{data['experiment']}:{task_id}", features=s2.features(requests[task_id], model2)[0]))
            if len(runs) != 9: raise ValueError('Development request requires nine traces')
            for row in runs:
                actions = {a['action']: a['observation'] for a in row['full']['actions'] if a['action'] != 'declare'}
                trajectories.append(dict(group_index=group, accurate=row['accurate'], full_verdict=row['full']['verdict'],
                    outcomes=[s3.token(a, actions[a]) if a in actions else -1 for a in agent.ORDER]))
    identity = {a: [0, 1, 2, 3] for a in agent.ORDER}
    return dict(version=3, neighbors=16, groups=groups, trajectories=trajectories,
        action_names=list(agent.ORDER), action_model=model2['action_model'],
        observation_mapping=identity, unused_observation_mapping={a:list(v) for a,v in identity.items()},
        development_sources_sha256={s:hashlib.sha256(source_bytes(ROOT/s)).hexdigest() for s in SOURCES},
        likelihood='Deterministic categorical equality; uniform joint trajectories; widen once to all development if local support is empty; then package C2 fallback if all support is empty.',
        missing='Missing V2 outcomes remain -1, never imputed as observations.',
        feature_definition='Unchanged frozen S2 maps and request feature function.',
        sampling_unit='120 request groups and 1080 dependent condition traces, not independent observations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Refuse to overwrite frozen model')
    write_json(args.output, build())
