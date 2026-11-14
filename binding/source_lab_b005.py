"""Dimensioned shared-ancestor laboratory for local source-error diagnosis.

The evaluator fields in a case are not part of a memory view. Re-derivation
uses only explicit direct parents and supplied source records, never truth.
"""
import copy
import math
import numpy as np
import mpmath as mp
from binding import source_lab_b004 as physics

FAULTS = ('clean', 'units', 'version', 'regime', 'intermediate', 'false_measurement')
SCOPES = ('shared', 'private')
ATOL, RTOL = 1e-10, 1e-5
OPERATION_COST = {'diffusivity': 1, 'response': 1, 'inverse': 66,
                  'prediction': 1, 'comparison': 1}


def close(a, b):
    return abs(float(a)-float(b)) <= ATOL + RTOL*abs(float(b))


def _perturb(value, noise):
    return float(np.nextafter(value, math.inf)) if noise else float(value)


def intake(seed):
    """Generate physical inputs without any corruption or memory-specific choice."""
    rng = np.random.default_rng(seed)
    materials = []
    for _ in range(3):
        kappa = float(10**rng.uniform(-7, -5))
        rho = float(rng.uniform(800, 6000))
        cp = float(rng.uniform(400, 1800))
        materials.append({'conductivity': {'value': kappa*rho*cp, 'unit': 'W/(m K)'},
                          'density': {'value': rho, 'unit': 'kg/m3'},
                          'heat_capacity': {'value': cp, 'unit': 'J/(kg K)'}})
    radii = [float(10**rng.uniform(-3, -2)) for _ in range(3)]
    branches = []
    for i in range(4):
        group = 0 if i < 2 else i-1
        m, r = materials[group], radii[group]
        kappa = m['conductivity']['value']/m['density']['value']/m['heat_capacity']['value']
        z = float(rng.uniform(3, 8))/(r*r/(15*kappa))
        branches.append({'group': group, 'z': z, 'radius2': r*float(rng.uniform(1.2, 2)),
                         'z2': z*float(rng.uniform(.7, 1.3))})
    names = rng.permutation(np.arange(100000, 100600)).tolist()
    return {'seed': int(seed), 'materials': materials, 'radii': radii,
            'branches': branches, 'names': names, 'order': rng.permutation(44).tolist()}


def _material(record):
    return (physics.quantity_si(record['conductivity'], 'conductivity') /
            physics.quantity_si(record['density'], 'density') /
            physics.quantity_si(record['heat_capacity'], 'heat_capacity'))


def _radius(record):
    return physics.quantity_si(record['quantity'], 'radius')


def record_map(case):
    return {r['id']: r for r in case['records']}


def ancestry(record_id, case):
    """All source and step-method identities, not numerical truth or diagnosis."""
    rows = record_map(case)
    memo = {}
    def visit(cid):
        if cid not in memo:
            row = rows[cid]
            found = set(row['source_ids'])
            for pid in row['parents']:
                found.update(visit(pid))
            memo[cid] = found
        return memo[cid]
    return set(visit(record_id))


def descendants(source_id, case):
    return [r['id'] for r in case['records'] if source_id in ancestry(r['id'], case)]


def rederive(record_id, case, values, sources, noise=False):
    """Recompute one local step from supplied direct parents; return full step cost.

    A caller validating a step against a catalog must pay for that catalog read
    separately and replace only that step record in the supplied source mapping.
    Agreement does not validate the numerical values of the parents.
    """
    row = record_map(case)[record_id]
    op = row['operation']
    step = sources[row['step_id']]
    if step['kind'] != 'method' or step['operation'] != op:
        raise ValueError('step method and operation disagree')
    rows = record_map(case)
    expected = {'inverse': ['1'], 'prediction': ['m2/s'], 'comparison': ['1', '1']}
    if op in expected and [rows[x]['unit'] for x in row['parents']] != expected[op]:
        raise ValueError('parent units do not match operation')
    p = [float(values[x]) for x in row['parents']]
    if not all(math.isfinite(x) for x in p):
        raise ValueError('parent values must be finite')
    if op == 'diffusivity':
        value = _material(sources[row['material_id']])
    elif op in ('response', 'prediction'):
        kappa = _material(sources[row['material_id']]) if op == 'response' else p[0]
        if kappa <= 0:
            raise ValueError('diffusivity must be positive')
        radius = _radius(sources[row['radius_id']])
        method = sources[row['method_id']]
        if method['kind'] != 'method' or method['operation'] != 'forward':
            raise ValueError('forward method required')
        if method['mode'] == 'exact':
            value = physics.predict_response(kappa, radius, row['z'], noise=noise)
        elif method['mode'] == 'high_frequency':
            value = math.sqrt(9*kappa/(radius*radius*row['z']))
            value = _perturb(value, noise)
        else:
            raise ValueError('unknown forward method')
        return value, OPERATION_COST[op]
    elif op == 'inverse':
        radius = _radius(sources[row['radius_id']])
        if step['mode'] == 'exact':
            value = physics.infer_diffusivity(p[0], radius, row['z'], noise=noise)
        elif step['mode'] == 'high_frequency':
            value = p[0]*p[0]*radius*radius*row['z']/9
            value = _perturb(value, noise)
        else:
            raise ValueError('unknown inverse method')
        # The fixed exact-inverse work allowance is charged for either mode.
        return value, OPERATION_COST[op]
    elif op == 'comparison':
        value = p[0] - p[1]
    else:
        raise ValueError('unknown operation')
    return _perturb(value, noise), OPERATION_COST[op]


def reference_values(case, noise=False):
    """Evaluator-only healthy reconstruction from catalog, without reading truth."""
    values = {}
    for row in case['records']:
        values[row['id']] = rederive(row['id'], case, values, case['catalog'], noise)[0]
    return values


def full_cost(record_id, case):
    """Cost of all unique operations and source reads in a dependency subgraph."""
    rows, claims, sources = record_map(case), set(), set()
    def visit(cid):
        if cid in claims:
            return
        claims.add(cid)
        row = rows[cid]
        sources.update(row['source_ids'])
        for pid in row['parents']:
            visit(pid)
    visit(record_id)
    return len(sources) + sum(OPERATION_COST[rows[c]['operation']] for c in claims)


def build_case(inputs, fault, scope='shared', noise=False):
    if fault not in FAULTS or scope not in SCOPES:
        raise ValueError('unknown fault or scope')
    names = iter(inputs['names'])
    def ident(prefix):
        return prefix+str(next(names))
    catalog, rows, pairs, branches, independent = {}, [], [], [], {}
    def source(kind, **fields):
        sid = ident('s')
        catalog[sid] = {'id': sid, 'kind': kind, 'version': 2, **fields}
        return sid
    def method(operation, mode='exact'):
        return source('method', operation=operation, mode=mode,
                      input_units={'inverse': ['1', 'm', '1/s'],
                                   'forward': ['m2/s', 'm', '1/s']}.get(operation, []),
                      domain={'zT_min': 0., 'zT_max': None})
    materials, radii, forwards, inverses = [], [], [], []
    for material, radius in zip(inputs['materials'], inputs['radii']):
        materials.append(source('material', **copy.deepcopy(material)))
        radii.append(source('radius', quantity={'value': radius*1000, 'unit': 'mm'}))
        forwards.append(method('forward'))
        inverses.append(method('inverse'))
    def claim(operation, parents, source_ids, quantity_id, unit, step_id=None, **fields):
        step_id = step_id or method(operation)
        cid = ident('c')
        depth = 1+max((next(r['depth'] for r in rows if r['id']==p) for p in parents), default=0)
        row = {'id': cid, 'operation': operation, 'parents': list(parents),
               'source_ids': sorted(set(source_ids+[step_id])), 'step_id': step_id,
               'quantity_id': quantity_id, 'unit': unit, 'depth': depth,
               'dimensions': [0, 2, -1, 0] if unit == 'm2/s' else [0, 0, 0, 0],
               'timestamp': inputs['order'][len(rows)], **fields}
        rows.append(row)
        return cid
    for bi, b in enumerate(inputs['branches']):
        g = b['group']
        mat, rad, forward = materials[g], radii[g], forwards[g]
        altmat = source('material', **copy.deepcopy(inputs['materials'][g]))
        altrad = source('radius', quantity={'value': inputs['radii'][g], 'unit': 'm'})
        r2 = source('radius', quantity={'value': b['radius2'], 'unit': 'm'})
        altforward = method('forward')
        q = [ident('q') for _ in range(5)]
        kappa = claim('diffusivity', [], [mat], q[0], 'm2/s', material_id=mat)
        response = claim('response', [], [mat, rad, forward], q[1], '1',
                         material_id=mat, radius_id=rad, method_id=forward, z=b['z'])
        altresponse = claim('response', [], [altmat, altrad, altforward], q[1], '1',
                            material_id=altmat, radius_id=altrad, method_id=altforward, z=b['z'])
        inverse = claim('inverse', [response], [altrad], q[0], 'm2/s', step_id=inverses[g],
                        radius_id=altrad, z=b['z'])
        altinverse = claim('inverse', [altresponse], [altrad], q[0], 'm2/s',
                           radius_id=altrad, z=b['z'])
        prediction = claim('prediction', [inverse], [r2, altforward], q[2], '1',
                           radius_id=r2, method_id=altforward, z=b['z2'])
        altprediction = claim('prediction', [altinverse], [r2, altforward], q[2], '1',
                              radius_id=r2, method_id=altforward, z=b['z2'])
        target = claim('response', [], [altmat, r2, altforward], q[2], '1',
                       material_id=altmat, radius_id=r2, method_id=altforward, z=b['z2'])
        comparison = claim('comparison', [prediction, target], [], q[3], '1')
        altcomparison = claim('comparison', [altprediction, target], [], q[3], '1')
        # A second forward path explicitly consumes derived material diffusivity.
        from_kappa = claim('prediction', [kappa], [rad, forward], q[1], '1',
                           radius_id=rad, method_id=forward, z=b['z'])
        pairs.extend([[response, altresponse], [inverse, altinverse],
                      [prediction, altprediction], [comparison, altcomparison],
                      [from_kappa, altresponse]])
        # Independent synthetic observations use a high-precision closed form
        # and the generating physical specification, not stored accepted values.
        with mp.workdps(60):
            material = inputs['materials'][g]
            kap = (mp.mpf(material['conductivity']['value']) /
                   mp.mpf(material['density']['value']) / mp.mpf(material['heat_capacity']['value']))
            def observation(radius, z):
                qv = mp.sqrt(mp.mpf(z)*mp.mpf(radius)**2/kap)
                return float(3*(qv/mp.tanh(qv)-1)/(qv*qv))
            resp = observation(inputs['radii'][g], b['z'])
            pred = observation(b['radius2'], b['z2'])
            for cid in (kappa, inverse, altinverse):
                independent[cid] = float(kap)
            for cid in (response, altresponse, from_kappa):
                independent[cid] = resp
            for cid in (prediction, altprediction, target):
                independent[cid] = pred
            for cid in (comparison, altcomparison):
                independent[cid] = 0.
        branches.append({'primary': [response, inverse, prediction, comparison],
                         'alternate': [altresponse, altinverse, altprediction, altcomparison],
                         'kappa': kappa, 'from_kappa': from_kappa})
    sources = copy.deepcopy(catalog)
    group = 0 if scope == 'shared' else 1
    culprit = None
    if fault == 'units':
        culprit = radii[group]
        sources[culprit]['quantity']['unit'] = 'm'
    elif fault == 'version':
        culprit = materials[group]
        sources[culprit]['version'] = 1
        sources[culprit]['conductivity']['value'] *= .25
    elif fault == 'regime':
        culprit = forwards[group]
        sources[culprit]['mode'] = 'high_frequency'
        sources[culprit]['domain']['zT_min'] = 10000.
    elif fault == 'intermediate':
        culprit = inverses[group]
        sources[culprit]['mode'] = 'high_frequency'
        sources[culprit]['domain']['zT_min'] = 10000.
    case = {'records': rows, 'sources': sources, 'catalog': catalog, 'pairs': pairs,
            'branches': branches, 'fault': fault, 'scope': scope, 'culprit': culprit,
            'seed': inputs['seed']}
    truth = independent
    values = {}
    for row in rows:
        values[row['id']] = rederive(row['id'], case, values, sources, noise)[0]
        row['value'] = values[row['id']]
    case['truth'] = truth
    case['affected'] = descendants(culprit, case) if culprit else []
    case['inaccurate'] = [r['id'] for r in rows if not close(r['value'], truth[r['id']])]
    # Independent synthetic observations are generated from the clean physical
    # specification, separately from stored accepted values and source corruption.
    measurements = {}
    false_ids = set(branches[0]['primary'][1:]) if fault == 'false_measurement' else set()
    for row in rows:
        cid = row['id']
        value = truth[cid]
        if cid in false_ids:
            value += .1*max(abs(value), 1e-3)
        measurements[cid] = {'claim_id': cid, 'value': float(value), 'unit': row['unit'],
                             'atol': ATOL, 'rtol': RTOL,
                             'repeat_value': truth[cid]}
    case['measurements'] = measurements
    case['false_measurement_ids'] = sorted(false_ids)
    case['budget_unit'] = max(full_cost(r['id'], case) for r in rows)
    return case
