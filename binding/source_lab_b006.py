"""Sequential diagnosis episodes with fallible, independently repeated checks.

Physical state resets between episodes on one fixed recipe graph. Only an
inferred diagnosis may authorize a certified catalog replacement; the catalog
cannot be read by the diagnostic policy. This separates finding from repair.
"""
import copy
import hashlib
import json
import random
from binding import source_lab_b005 as physics
from binding import diagnose_b005 as previous

ENVIRONMENTS = ('low_zero', 'high_zero', 'low_moderate', 'high_moderate')
FAULTS = ('units', 'version', 'regime', 'intermediate')
EPISODES = 32
CLEAN_RATE = .125
FALSE_RATE = {'low': .15, 'high': .65}
# Rates are conditional on a genuinely bad/good tested proposition respectively.
ERRORS = {
    'zero': {kind: {'false_good': 0., 'false_bad': 0.}
             for kind in ('check_source', 'check_claim', 'repeat_observation')},
    'moderate': {
        'check_source': {'false_good': .18, 'false_bad': .12},
        'check_claim': {'false_good': .12, 'false_bad': .18},
        'repeat_observation': {'false_good': .10, 'false_bad': .15},
    },
}


def _uniform(*key):
    """Stateless paired stream: attempts have distinct pseudorandom draws."""
    raw = json.dumps(key, separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64


def noisy_answer(truth_bad, rates, seed, kind, identity, attempt):
    if attempt < 0:
        raise ValueError('negative attempt')
    rate = rates['false_good' if truth_bad else 'false_bad']
    if not 0 <= rate <= 1:
        raise ValueError('probability outside unit interval')
    error = _uniform('probe', int(seed), kind, str(identity), int(attempt)) < rate
    return bool(truth_bad) != error


def _false_signal(case, branch, ordinal):
    rows = physics.record_map(case)
    depth = 1 + ordinal % 4
    if branch == 'measurement':
        choices = sorted(r['id'] for r in case['records'] if r['depth'] == depth)
        return [choices[(ordinal // 4) % len(choices)]]
    pairs = [p for p in case['pairs'] if rows[p[0]]['depth'] == depth]
    return sorted(pairs[(ordinal // 4) % len(pairs)])


def make_series(seed, environment, episodes=EPISODES, noise=False, inputs=None):
    """Evaluator-side factory; environment/rates never enter public episodes.

    A series has one intake and fixed identities/edges. Episode faults are reset,
    not accumulated. Error streams are independent across episodes and attempts.
    """
    if environment not in ENVIRONMENTS or episodes < 1:
        raise ValueError('unknown environment or invalid episode count')
    frequency, reliability = environment.split('_')
    inputs = physics.intake(seed) if inputs is None else copy.deepcopy(inputs)
    cache = {}
    out = []
    for index in range(episodes):
        eseed = seed * 1000 + index
        rng = random.Random(int(_uniform('episode', eseed) * 2**64))
        draw, false_draw = rng.random(), rng.random()
        fault_choice, scope_choice = rng.randrange(4), rng.randrange(2)
        kind = ('clean' if draw < CLEAN_RATE else
                'false_objection' if false_draw < FALSE_RATE[frequency] else 'true_fault')
        fault = FAULTS[fault_choice] if kind == 'true_fault' else 'clean'
        scope = ('shared', 'private')[scope_choice]
        branch = ('measurement', 'conflict')[index % 2]
        key = (fault, scope)
        if key not in cache:
            cache[key] = physics.build_case(inputs, fault, scope, noise)
        case = copy.deepcopy(cache[key])
        case['seed'] = eseed
        ids = previous.signal(case, branch) if kind == 'true_fault' else []
        if kind == 'false_objection':
            ids = _false_signal(case, branch, index // 2)
            if branch == 'measurement':
                for cid in ids:
                    case['measurements'][cid]['value'] += .1 * max(abs(case['truth'][cid]), 1e-3)
                case['false_measurement_ids'] = list(ids)
        out.append({'index': index, 'seed': eseed, 'case': case,
                    'signal_ids': ids, 'signal_branch': branch,
                    'evaluator': {'kind': kind, 'fault': fault, 'scope': scope,
                                  'environment': environment,
                                  'rates': copy.deepcopy(ERRORS[reliability]),
                                  'true_conflict': kind == 'true_fault'}})
    return {'seed': int(seed), 'environment': environment, 'intake': inputs, 'episodes': out}


def public_episode(episode):
    """An immutable-by-copy signal and provenance view without hidden truth."""
    case = episode['case']
    public = {k: copy.deepcopy(case[k]) for k in
              ('records', 'sources', 'source_order', 'pairs', 'branches', 'seed', 'budget_unit')}
    public['measurements'] = {
        cid: {k: copy.deepcopy(v) for k, v in observation.items() if k != 'repeat_value'}
        for cid, observation in case['measurements'].items()}
    return {'index': episode['index'], 'seed': episode['seed'], 'case': public,
            'signal_ids': list(episode['signal_ids']), 'signal_branch': episode['signal_branch']}


class NoisyService:
    """Paid-service capability: a probe returns a binary report, never a label.

    Private fields belong to the trusted executor, not to policy memory. Returned
    reports do not include the actual error event, rate, ground truth or catalog.
    Repairs are certified source replacements after an accusation; this is not
    a model of fallible repair. Every descendant rebuild retains its full cost.
    """
    def __init__(self, episode, noise=False):
        self._base = previous.Service(episode['case'], noise)
        self._episode = copy.deepcopy(episode)
        self._attempts = {}
        self._events = []
        self._authorized = set()

    def source_cost(self, sid):
        return self._base.source_cost(sid)

    def claim_cost(self, cid):
        return self._base.claim_cost(cid)

    def repair_cost(self, cid):
        return self._base.repair_cost(cid)

    def repeat_cost(self):
        return 2 * self._episode['case']['budget_unit']

    def _report(self, kind, identity, truth_bad):
        key = (kind, identity)
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        rates = self._episode['evaluator']['rates'][kind]
        bad = noisy_answer(truth_bad, rates, self._episode['seed'], kind, identity, attempt)
        self._events.append({'kind': kind, 'id': identity, 'attempt': attempt,
                             'truth_bad': bool(truth_bad), 'reported_bad': bad,
                             'error': bool(bad != truth_bad)})
        return bad

    def probe(self, test):
        kind, identity = test['kind'], test['id']
        if kind == 'check_source':
            truth_bad = self._base.check_source(identity)[0]
        elif kind == 'check_claim':
            truth_bad = self._base.check_claim(identity)[0]
        else:
            raise ValueError('unknown probe kind')
        return self._report(kind, identity, truth_bad)

    def repeat_observation(self, ids):
        """True means a repeated report says the discrepancy still persists."""
        case = self._episode['case']
        if self._episode['signal_branch'] == 'measurement':
            truth_bad = any(not physics.close(self._base.values[c], case['truth'][c]) for c in ids)
        else:
            truth_bad = any(not physics.close(self._base.values[a], self._base.values[b])
                           for a, b in case['pairs'] if a in ids or b in ids)
        return self._report('repeat_observation', '|'.join(sorted(ids)), truth_bad)

    def commit_repair_source(self, sid):
        """Install the certified record after caller has committed an accusation."""
        if sid not in self._base.sources:
            raise ValueError('unknown source')
        self._base.sources[sid] = copy.deepcopy(self._episode['case']['catalog'][sid])
        self._authorized.add(sid)

    def repair(self, cid):
        if not self._authorized:
            raise ValueError('repair requires a committed source accusation')
        return self._base.repair(cid)

    def evaluate(self):
        """Evaluator-only snapshot, called after policy decisions have finished."""
        case = self._episode['case']
        bad = sorted(c for c, v in self._base.values.items() if not physics.close(v, case['truth'][c]))
        return {'remaining_bad_ids': bad, 'values': copy.deepcopy(self._base.values),
                'events': copy.deepcopy(self._events), 'culprit': case['culprit'],
                'initial_bad_ids': list(case['inaccurate']),
                'kind': self._episode['evaluator']['kind']}
