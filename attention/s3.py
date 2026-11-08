"""Finite joint-trace Bayes model restricted to requests and opened evidence."""
import copy
from functools import lru_cache
from pathlib import Path
import numpy as np
from attention import agent, s2
from attention.io import read_json

VERDICTS = s2.VERDICTS
ACTIONS = agent.ORDER
INDEX = {a: i for i, a in enumerate(ACTIONS)}
FEATURE_ACTIONS = ('normalization', 'high_frequency')


def token(action, observation):
    value = observation['verdict']
    if action.startswith('v2_') and value in ('accept', 'violation_detected') and not observation.get('terminal', False):
        raise ValueError('A decisive V2 result must be terminal')
    if value == 'accept': return 0
    if value == 'violation_detected': return 1
    if value != 'inconclusive': raise ValueError('Unknown observation verdict')
    return 3 if action.startswith('v2_') and observation.get('terminal', False) else 2


def observation(action, value):
    if value not in (0, 1, 2, 3): raise ValueError('Missing or invalid categorical evidence')
    result = dict(verdict=VERDICTS[value] if value < 2 else 'inconclusive')
    if action.startswith('v2_'):
        result['terminal'] = value != 2
    return result


@lru_cache(maxsize=1)
def frozen_s2():
    return s2.load_model()


def load_model(path=None):
    return read_json(path or Path(__file__).with_name('self_model_s3.json'))


def corrupt(model, policy):
    if policy not in ('B2', 'D', 'Dprime'):
        raise ValueError('Unknown updated-model policy')
    result = copy.deepcopy(model)
    if policy != 'B2':
        mapping = result['observation_mapping' if policy == 'D' else 'unused_observation_mapping']
        for action in FEATURE_ACTIONS:
            mapping[action][0], mapping[action][1] = mapping[action][1], mapping[action][0]
    return result


class Runtime:
    """Compiled development data; no current executor or hidden condition access."""
    def __init__(self, model):
        self.model = model
        self.raw = np.asarray([r['outcomes'] for r in model['trajectories']], dtype=np.int8)
        self.mapped = self.raw.copy()
        for a, index in INDEX.items():
            mapping = np.asarray(model['observation_mapping'][a], dtype=np.int8)
            present = self.raw[:, index] >= 0
            self.mapped[present, index] = mapping[self.raw[present, index]]
        self.accurate = np.asarray([r['accurate'] for r in model['trajectories']], dtype=bool)
        self.groups = np.asarray([r['group_index'] for r in model['trajectories']], dtype=int)
        self.full = np.asarray([VERDICTS.index(agent.verdict({a: observation(a, int(v))
                    for a, v in zip(ACTIONS, row) if v >= 0})) for row in self.mapped], dtype=int)
        self.correct = ((self.full == 0) & self.accurate) | ((self.full == 1) & ~self.accurate)

    def neighbors(self, request):
        vector = s2.features(request, frozen_s2())[0]
        nearest = sorted(range(len(self.model['groups'])), key=lambda i: (
            sum((a-b)**2 for a, b in zip(vector, self.model['groups'][i]['features'])), i))[:self.model['neighbors']]
        return nearest, vector


class Posterior:
    def __init__(self, request, model=None, runtime=None):
        self.runtime = runtime or Runtime(load_model() if model is None else model)
        self.neighbors, self.features = self.runtime.neighbors(request)
        self.local = np.flatnonzero(np.isin(self.runtime.groups, self.neighbors))
        self.indices = self.local.copy()
        self.expanded = False
        self.failed = False
        self.evidence = {}
        self.step = 0

    def snapshot(self):
        ids = self.indices
        probabilities = None if not len(ids) else {
            v: float(np.count_nonzero(self.runtime.full[ids] == i)/len(ids))
            for i, v in enumerate(VERDICTS)}
        masses = None if not len(ids) else {
            a: float(np.count_nonzero(self.runtime.raw[ids, INDEX[a]] == 1)/len(ids))
            for a in FEATURE_ACTIONS}
        return dict(step=self.step, full_probabilities=probabilities, support_count=len(ids),
                    expanded=self.expanded, failed=self.failed,
                    raw_feature_violation_mass=masses)

    def update(self, action, observed):
        if action in self.evidence:
            raise ValueError('Repeated evidence cannot be counted twice')
        before = self.snapshot()['raw_feature_violation_mass']
        value = token(action, observed)
        self.evidence[action] = value
        self.step += 1
        self.indices = self.indices[self.runtime.mapped[self.indices, INDEX[action]] == value]
        expanded_now = False
        if not len(self.indices) and not self.expanded:
            self.expanded = expanded_now = True
            keep = np.ones(len(self.runtime.raw), dtype=bool)
            for a, v in self.evidence.items():
                keep &= self.runtime.mapped[:, INDEX[a]] == v
            self.indices = np.flatnonzero(keep)
        self.failed = not len(self.indices)
        result = self.snapshot()
        result.update(action=action, category=value, expanded_now=expanded_now,
                      raw_feature_violation_mass_before=before)
        return result


def predict(request, observed=None, model=None):
    posterior = Posterior(request, model)
    for a, obs in (observed or {}).items(): posterior.update(a, obs)
    result = posterior.snapshot()
    result.update(neighbor_ids=[posterior.runtime.model['groups'][i]['id'] for i in posterior.neighbors],
                  feature_vector=posterior.features)
    return result


def corruption_audit(original, relevant, irrelevant):
    def differences(left, right, key):
        return [abs(a-b) for action in ACTIONS for a,b in zip(left[key][action], right[key][action]) if a != b]
    d = differences(original, relevant, 'observation_mapping')
    dp = differences(original, irrelevant, 'unused_observation_mapping')
    return dict(D_changed_numbers=len(d), Dprime_changed_numbers=len(dp), D_L1=sum(d), Dprime_L1=sum(dp),
                equal_changes=d == dp,
                Dprime_active_unchanged=original['observation_mapping'] == irrelevant['observation_mapping'],
                D_unused_unchanged=original['unused_observation_mapping'] == relevant['unused_observation_mapping'])
