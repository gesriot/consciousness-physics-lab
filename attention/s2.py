"""Frozen request-only development predictions, without numerical solves."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from lab.scientist import tail_bound

VERDICTS = ('accept', 'violation_detected', 'inconclusive')
_COORDINATES = {}


def load_model(path=None):
    return json.loads(Path(path or Path(__file__).with_name('self_model_s2.json')).read_text())


def corrupt(model, policy):
    result = copy.deepcopy(model)
    if policy in ('D', 'Dprime'):
        for row in result['records' if policy == 'D' else 'unused_records']:
            row['counts'][0], row['counts'][2] = row['counts'][2], row['counts'][0]
    return result


def _request(request):
    if isinstance(request, dict):
        return complex(*request['z']), request['mean_time'], request['tolerance']
    return request.z, request.mean_time, request.tolerance


def features(request, model):
    z, time, tolerance = _request(request)
    q = z*time
    x = [math.log10(max(abs(q), 1e-300))/6, math.atan2(q.imag, q.real)/(math.pi/2)]
    modal_floor = float(tail_bound(z, time, np.array([4096]))[0]) + 128*np.finfo(float).eps
    flags = []
    distances = []
    for name in ('v2', 'collocation'):
        rows = model['maps'][name]
        # Archive order is the deterministic tie breaker. No interpolation of values.
        key = id(rows)
        if key not in _COORDINATES:
            _COORDINATES[key] = (rows, np.asarray([r[:2] for r in rows]))
        coords = _COORDINATES[key][1]
        ds = np.sum((coords-np.asarray(x))**2, axis=1)
        index = int(np.argmin(ds))
        row = rows[index]
        distances.append(float(math.sqrt(ds[index])))
        if name == 'v2':
            flags.append(any(u is not None and u <= tolerance for u in row[2:]))
        else:
            flags.append(row[2] <= tolerance/8 and row[3] <= tolerance)
    vector = x + [math.log10(tolerance)/3, float(modal_floor <= tolerance)] + list(map(float, flags))
    return vector, dict(modal_floor=modal_floor, modal_attainable=bool(vector[3]),
                        v2_map_accept_feasible=flags[0], collocation_map_accept_feasible=flags[1],
                        collocation_accept_forbidden=bool(tolerance < 1.9073486328125e-6),
                        map_distances=distances)


def _probs(counts):
    total = sum(counts)+3
    return {v: (n+1)/total for v,n in zip(VERDICTS, counts)}


def predict(request, observed=None, model=None):
    model = load_model() if model is None else model
    vector, details = features(request, model)
    records = model['records']
    nearest = sorted(range(len(records)), key=lambda i: (
        sum((a-b)**2 for a,b in zip(vector, records[i]['features'])), i))[:model['neighbors']]
    rows = [records[i] for i in nearest]
    counts = [sum(r['counts'][j] for r in rows) for j in range(3)]
    probabilities = _probs(counts)
    correctness = (sum(r['inaccurate_violations'] for r in rows)+1)/(counts[1]+2)
    actions = {}
    for action in model['action_names']:
        p = _probs([sum(r['actions'][action][j] for r in rows) for j in range(3)])
        if action.startswith('v2_') and int(action[3:]) < 512:
            p = dict(accept=0., violation_detected=0., inconclusive=1.)
        actions[action] = dict(accept_probability=p['accept'], detection_probability=p['violation_detected'], inconclusive_probability=p['inconclusive'])
    return dict(full_probabilities=probabilities,
                solvable_probability=probabilities['accept']+probabilities['violation_detected']*correctness,
                detection_correct_probability=correctness,
                decisive_probability=1-probabilities['inconclusive'], actions=actions,
                feature_vector=vector, feature_details=details, neighbor_ids=[records[i]['id'] for i in nearest],
                base_probabilities=model['base_probabilities'])
