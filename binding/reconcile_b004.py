"""Paid reconciliation with isolated memory views and immutable source recipes."""
import copy
import hashlib
import json
import math
import numpy as np
from binding import source_lab_b004 as L

POLICIES = ('G', 'G1', 'F', 'T', 'N', 'Gx', 'Gxprime')
FAULTS = ('clean', 'units', 'regime', 'version')
BUDGETS = (2, 8, 32)
SIGNALS = ('revision', 'conflict')
TICKS_PER_UNIT = 69
SLOT_BYTES = 2048
ATOL, RTOL = 1e-10, 1e-5


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def close(a, b):
    return abs(a-b) <= ATOL + RTOL*abs(b)


def rounded(value, noise=False):
    if noise:
        value = np.nextafter(float(value), np.inf)
    return round(float(value), 9)


def intake(seed):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(100000 + np.arange(200)).tolist()
    def ident(prefix):
        return prefix + str(ids.pop())
    groups = []
    for _ in range(12):
        radius = float(10**rng.uniform(-3, -1))
        kappa = float(10**rng.uniform(-7, -4))
        rho = float(rng.uniform(500, 8000))
        cp = float(rng.uniform(300, 2000))
        x = float(10**rng.uniform(0, 1))
        z = x/(radius*radius/(15*kappa))
        sid, aid = ident('s'), ident('s')
        source = dict(id=sid, version=2, radius=dict(value=radius*1000, unit='mm'),
                      conductivity=dict(value=kappa*rho*cp, unit='W/(m K)'),
                      density=dict(value=rho, unit='kg/m3'),
                      heat_capacity=dict(value=cp, unit='J/(kg K)'))
        alt = dict(id=aid, version=2, radius=dict(value=radius, unit='m'),
                   diffusivity=dict(value=kappa, unit='m2/s'))
        groups.append(dict(source=source, alternate=alt, z=z, radius=radius,
                           radius2=radius*float(rng.uniform(1.3, 2.5)),
                           z2=z*float(rng.uniform(.5, 2)),
                           claims=[ident('c') for _ in range(4)], query=ident('q')))
    targets = sorted(rng.choice(12, 2, replace=False).tolist())
    timestamps = rng.permutation(48).tolist()
    return dict(seed=int(seed), groups=groups, targets=targets, timestamps=timestamps)


def derive(recipe, sources, recipes, noise=False, stats=None):
    kind = recipe['operation']
    if kind == 'response':
        return L.response(sources[recipe['source']], recipe['z'], noise=noise, stats=stats)
    parent = derive(recipes[recipe['parent']], sources, recipes, noise, stats)
    if kind == 'inverse':
        return L.infer_diffusivity(parent, recipe['radius'], recipe['z'], noise=noise, stats=stats)
    if kind == 'prediction':
        return L.predict_response(parent, recipe['radius'], recipe['z'], noise=noise, stats=stats)
    raise ValueError('Unknown derivation operation')


def build_series(inputs, fault, noise=False):
    if fault not in FAULTS:
        raise ValueError('Unknown corruption')
    sources, recipes, records, truth, pairs = {}, {}, [], {}, []
    affected, damaged_sources = [], []
    for gi, g in enumerate(inputs['groups']):
        sid, aid = g['source']['id'], g['alternate']['id']
        sources[sid], sources[aid] = copy.deepcopy(g['source']), copy.deepcopy(g['alternate'])
        observed = copy.deepcopy(g['source'])
        damaged = gi in inputs['targets'] and fault != 'clean'
        mode = 'exact'
        if damaged:
            damaged_sources.append('method:'+sid if fault == 'regime' else sid)
            if fault == 'units':
                observed['radius']['unit'] = 'm'
            elif fault == 'version':
                observed['conductivity']['value'] *= 4
                observed['version'] = 1
            else:
                mode = 'asymptotic'
        c1, c2, c3, ca = g['claims']
        recipes[c1] = dict(operation='response', source=sid, z=g['z'])
        recipes[ca] = dict(operation='response', source=aid, z=g['z'])
        recipes[c2] = dict(operation='inverse', parent=c1, radius=g['radius'], z=g['z'])
        recipes[c3] = dict(operation='prediction', parent=c2, radius=g['radius2'], z=g['z2'])
        v1 = L.response(observed, g['z'], mode=mode, noise=noise)
        v2 = L.infer_diffusivity(v1, g['radius'], g['z'], noise=noise)
        v3 = L.predict_response(v2, g['radius2'], g['z2'], noise=noise)
        va = L.response(g['alternate'], g['z'], noise=noise)
        for cid, depth, value, parents, src, query, unit in (
            (c1, 1, v1, [], sid, g['query'], '1'),
            (c2, 2, v2, [c1], None, c2, 'm2/s'),
            (c3, 3, v3, [c2], None, c3, '1'),
            (ca, 1, va, [], aid, g['query'], '1')):
            table = observed if cid == c1 else g['alternate']
            attrs = ([] if src is None else [dict(source=src, version=table['version'],
                      units={k: v['unit'] for k, v in table.items() if isinstance(v, dict)},
                      regime='properties', kind='table')])
            if src is not None:
                attrs.append(dict(source='method:'+src, version=1, units={},
                                  regime=mode if cid == c1 else 'exact', kind='method'))
            record = dict(id=cid, depth=depth, value=value, unit=unit, query=query,
                          parents=parents, direct=attrs, request='request:'+query,
                          time=inputs['timestamps'][len(records)], operation=recipes[cid]['operation'])
            records.append(record)
            truth[cid] = derive(recipes[cid], sources, recipes)
            if damaged and cid != ca:
                affected.append(cid)
        pairs.append(sorted([c1, ca]))
    return dict(intake=inputs, fault=fault, records=records, recipes=recipes,
                sources=sources, truth=truth, pairs=sorted(pairs), affected=sorted(affected),
                damaged_sources=sorted(damaged_sources),
                notice_sources=sorted(('method:' if fault == 'regime' else '')+inputs['groups'][g]['source']['id'] for g in inputs['targets']),
                notice_versions={('method:' if fault == 'regime' else '')+inputs['groups'][g]['source']['id']: (1 if fault == 'regime' else 2) for g in inputs['targets']})


def memory_view(records, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown memory')
    out, edits = [], []
    byid = {r['id']: r for r in records}
    for raw in records:
        r = copy.deepcopy(raw)
        if policy in ('Gx', 'Gxprime') and any(byid[p]['depth'] == 1 for p in r['parents']):
            if policy == 'Gx':
                old = r['parents'][0]
                r['parents'][0] = 'external:'+old
            else:
                old = r['request']
                r['request'] = 'external:'+old
            edits.append(dict(id=r['id'], before=old, after='external:'+old))
        if policy in ('F', 'T', 'N'):
            r['parents'] = []
            r['request'] = None
        if policy in ('T', 'N'):
            r['direct'] = []
        size = len(canonical(r))
        if size > SLOT_BYTES:
            raise ValueError('Memory slot capacity exceeded')
        # Padding is not persisted; capacity is an equal allocated fixed-width slot.
        out.append(r)
    return out, edits


def candidate_order(view, policy, suspect_roots, suspect_sources, noise=False):
    roots = {r['id'] for r in view if r['id'] in suspect_roots or
             any(d['source'] in suspect_sources for d in r['direct'])}
    implicated = set(roots)
    if policy == 'G1':
        implicated.update(r['id'] for r in view if set(r['parents']) & roots)
    elif policy in ('G', 'Gx', 'Gxprime'):
        while True:
            expanded = implicated | {r['id'] for r in view if set(r['parents']) & implicated}
            if expanded == implicated:
                break
            implicated = expanded
    if policy == 'T':
        return [r['id'] for r in sorted(view, key=lambda r: (-r['time'], r['id']))], []
    scores = {r['id']: rounded((4-r['depth']) if r['id'] in implicated else 0, noise)
              for r in view}
    ordered = sorted(view, key=lambda r: (-scores[r['id']], r['id']))
    return [r['id'] for r in ordered], sorted(implicated)


def conflict_pairs(records):
    groups = {}
    for r in records:
        groups.setdefault((r['query'], r['unit']), []).append(r)
    found = []
    for rows in groups.values():
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                if not close(a['value'], b['value']) and not close(b['value'], a['value']):
                    found.append(sorted([a['id'], b['id']]))
    return sorted(found)


class RecheckService:
    """No evaluator truth is read; paid recipes access a current property catalogue."""
    def __init__(self, series, noise=False):
        self.sources = series['sources']
        self.recipes = series['recipes']
        self.noise = noise
        self.records = {r['id']:r for r in series['records']}
        self.cache = {}
        self.physical_evaluations = 0

    def audited_supports(self, cid):
        row = self.records[cid]
        found = []
        for d in row['direct']:
            if d['kind'] == 'method':
                if d['regime'] != 'exact': found.append(d['source'])
            else:
                source = self.sources[d['source']]
                units = {k:v['unit'] for k,v in source.items() if isinstance(v,dict)}
                if d['version'] != source['version'] or d['units'] != units:
                    found.append(d['source'])
        return sorted(found)

    def get(self, cid):
        if cid not in self.cache:
            stats = {}
            value = derive(self.recipes[cid], self.sources, self.recipes, self.noise, stats)
            count = stats.get('analytic_evaluations', 0) + stats.get('asymptotic_evaluations', 0)
            self.cache[cid] = (value, count)
            self.physical_evaluations += count
        return self.cache[cid]


def simulate(series, policy, budget, branch, noise=False, service=None):
    if budget not in BUDGETS or branch not in SIGNALS:
        raise ValueError('Unregistered condition')
    records = series['records']
    byid = {r['id']: r for r in records}
    view, edits = memory_view(records, policy)
    limit, spent = budget*TICKS_PER_UNIT, 0
    service = service or RecheckService(series, noise)
    trace, diagnosed, suspect_roots, suspect_sources = [], [], set(), set()
    conflicts = conflict_pairs(records)
    if branch == 'revision':
        if set(series['notice_versions']) != set(series['notice_sources']):
            raise ValueError('Versioned revision notice is incomplete')
        suspect_sources.update(series['notice_sources'])
    elif policy != 'N':
        for pair in conflicts:
            chosen = pair[0]
            price = {1:1, 2:67, 3:68}[byid[chosen]['depth']]
            if spent + price > limit:
                break
            value, actual = service.get(chosen)
            if actual != price:
                raise AssertionError('Unregistered service price')
            spent += actual
            bad = sorted(cid for cid in pair if not close(byid[cid]['value'], value))
            suspect_roots.update(bad)
            # Signal metadata is shared across memories, not inferred from hidden truth.
            found_sources = sorted({s for cid in bad for s in service.audited_supports(cid)})
            suspect_sources.update(found_sources)
            diagnosed.extend(found_sources)
            trace.append(dict(kind='diagnose', pair=pair, chosen=chosen, cost=actual,
                              answer=value, implicated=bad, sources=found_sources, spent=spent))
    order, implicated = candidate_order(view, policy, suspect_roots, suspect_sources, noise)
    final = {r['id']: dict(accepted=True, value=r['value']) for r in records}
    if policy != 'N' and (branch == 'revision' or conflicts):
        for cid in order:
            if spent >= limit:
                break
            spent += 1
            final[cid]['accepted'] = False
            cost = {1:1, 2:67, 3:68}[byid[cid]['depth']]
            event = dict(kind='withdraw', id=cid, depth=byid[cid]['depth'], cost=1, spent=spent)
            if spent + cost <= limit:
                value, actual = service.get(cid)
                if actual != cost:
                    raise AssertionError('Unregistered rederivation price')
                spent += cost
                final[cid] = dict(accepted=True, value=value)
                event.update(kind='repair', answer=value, cost=1+cost, spent=spent)
            trace.append(event)
            if event['kind'] == 'withdraw':
                break
    protected = dict(policy=policy, budget=budget, branch=branch, order=order,
                     implicated=implicated, edits=edits,
                     events=[{k:v for k,v in e.items() if k != 'answer'} for e in trace],
                     accepted={c:v['accepted'] for c,v in final.items()}, spent=spent,
                     conflicts=conflicts, diagnosed=sorted(set(diagnosed)),
                     capacity_bytes=len(records)*SLOT_BYTES)
    return dict(protected=protected, trace=trace, final=final,
                metrics=evaluate(series, final, trace, protected))


def evaluate(series, final, trace, protected):
    original = {r['id']: r for r in series['records']}
    bad = {c for c,v in series['truth'].items() if not close(original[c]['value'], v)}
    actions = [e for e in trace if e['kind'] in ('withdraw', 'repair')]
    selected = {e['id'] for e in actions}
    incorrect = {c for c,v in final.items() if v['accepted'] and not close(v['value'], series['truth'][c])}
    lost = {c for c,v in final.items() if not v['accepted'] and c not in bad}
    unnecessary = selected-bad
    depths = {}
    for d in (1,2,3):
        ids = {c for c,r in original.items() if r['depth'] == d}
        depths[str(d)] = dict(initial_bad=len(ids&bad), remaining_bad=len(ids&incorrect),
                             selected=len(ids&selected), selected_bad=len(ids&selected&bad),
                             unnecessary=len(ids&unnecessary), correct_lost=len(ids&lost))
    truth_sources = set(series['damaged_sources'])
    identified = set(protected['diagnosed'])
    return dict(initial_bad=len(bad), remaining_bad=len(incorrect),
                unnecessary=len(unnecessary), correct_lost=len(lost),
                cost=protected['spent'], evaluations=sum(e['cost']-(e['kind'] in ('repair','withdraw')) for e in trace),
                localized_correct=len(identified&truth_sources), localized_total=len(identified),
                target_sources=len(truth_sources), conflicts=len(protected['conflicts']),
                resolved_conflicts=sum(not (final[p[0]]['accepted'] and final[p[1]]['accepted'] and
                    not close(final[p[0]]['value'],final[p[1]]['value']) and
                    not close(final[p[1]]['value'],final[p[0]]['value'])) for p in protected['conflicts']), depths=depths)
