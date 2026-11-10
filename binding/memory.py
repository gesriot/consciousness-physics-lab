"""Policy-visible recall selection, without fault onset or accuracy labels."""
import copy
import numpy as np
from attention.build_a007_library import CHANNELS

MEMORIES = ('G', 'T', 'A', 'N', 'Gx', 'Gxprime')
BUDGETS = (2, 8, 32)


def corrupt(records, notified, irrelevant=False, last_success=float('-inf')):
    result = copy.deepcopy(records)
    edits = []
    j = CHANNELS.index(notified)
    neighbor = j + 1 if j % 6 < 5 else j - 1
    for record in result:
        relevant = [e for e in record['links'] if e['channel'] == notified and e['time'] > last_success]
        if not relevant:
            continue
        edge = next(e for e in record['links'] if e['channel'].removeprefix('independent:') == 'normalization') if irrelevant else relevant[-1]
        before = edge['channel']
        edge['channel'] = (('independent:' if before.startswith('independent:') else '')+'first_moment') if irrelevant else CHANNELS[neighbor]
        edits.append(dict(record_id=record['id'], edge_id=edge.get('edge_id'), before=before, after=edge['channel'],
                          l1=abs(CHANNELS.index(before.removeprefix('independent:'))-CHANNELS.index(edge['channel'].removeprefix('independent:')))))
    return result, edits


def visible_memory(records, notice, memory, reviewed=()):
    if memory not in MEMORIES:
        raise ValueError('Unknown memory')
    visible = [r for r in records if r['time'] < notice['time'] and r['id'] not in reviewed]
    edits = []
    if memory in ('Gx', 'Gxprime'):
        visible, edits = corrupt(visible, notice['channel'], memory == 'Gxprime', notice['last_success'])
    if memory in ('T', 'A', 'N'):
        visible = [dict(id=r['id'], time=r['time']) for r in visible]
    return visible, edits


def select(records, notice, memory, reviewed=(), noise=False):
    visible, edits = visible_memory(records, notice, memory, reviewed)
    selected = []
    for record in visible:
        linked = [e['time'] for e in record.get('links', [])
                  if e['channel'] == notice['channel'] and e['time'] > notice['last_success']]
        if memory == 'N' or (memory == 'T' and record['time'] <= notice['last_success']):
            continue
        if memory in ('G', 'Gx', 'Gxprime') and not linked:
            continue
        score = float(max(linked)) if memory in ('G', 'Gx', 'Gxprime') else float(record['time'])
        if noise:
            score = float(np.nextafter(score, np.inf))
        selected.append((round(score, 9), record['time'], record['id']))
    selected.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[2] for x in selected], edits


def recall(records, notices, memory, units, replace, price, noise=False):
    reviewed, withdrawn, selected_all = set(), set(), set()
    attempts, interventions = [], []
    for notice in notices:
        selected, edits = select(records, notice, memory, reviewed, noise)
        selected_all.update(selected)
        snapshot, snapshot_edits = visible_memory(records, notice, memory, reviewed)
        if edits != snapshot_edits:
            raise AssertionError('Memory snapshot differs from selection')
        interventions.append(dict(notice=notice, records=snapshot, edits=edits, changed_edges=len(edits),
                                  l1=sum(e['l1'] for e in edits)))
        for record_id in selected[:max(0, units-len(reviewed))]:
            result = replace(record_id)
            reviewed.add(record_id)
            revoke = result['verdict'] != 'accept'
            if revoke:
                withdrawn.add(record_id)
            attempts.append(dict(record_id=record_id, notice_time=notice['time'],
                                 service_delay=1+sum(a['notice_time'] == notice['time'] for a in attempts),
                                 result=result, withdrawn=revoke, cost=price))
    return dict(attempts=attempts, withdrawn=sorted(withdrawn), reviewed=sorted(reviewed),
                selected=sorted(selected_all), interventions=interventions,
                cost=len(attempts)*price, budget=units*price,
                unused_budget=(units-len(attempts))*price,
                unreviewed_selected=sorted(selected_all-reviewed))
