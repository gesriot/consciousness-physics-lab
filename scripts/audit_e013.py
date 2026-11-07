"""Check E013 endpoint arithmetic and exact replay, separating explicit provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def non_timing(value):
    if isinstance(value, dict):
        return {k: non_timing(v) for k,v in value.items() if not k.endswith('seconds')}
    if isinstance(value, list):
        return [non_timing(v) for v in value]
    return value


def audit(path):
    data = json.loads(path.read_text())
    rows, intake = data['runs'], data['intake_channel']
    assert len(intake) == 96 and len(rows) == 288
    assert len({(r['task_id'],r['condition']) for r in rows}) == 288
    for r in rows:
        q = intake[r['task_id']]['request']
        assert r['tolerance'] == q['tolerance']
        error = abs(complex(*r['outcome']['value']) - complex(*r['reference']))
        assert error == r['absolute_error']
        assert r['accurate'] == (error <= q['tolerance'])
        independent = r['decision']['independent']
        assert independent['cells_solved'] == sum(independent['grids'])
        assert r['work']['decision_fv_cells'] == r['components']*independent['cells_solved']
        assert r['work']['artifact_modal_terms'] == r['components']*r['outcome']['modes_spent']
        for step in independent['steps']:
            assert step['accepted'] == (step['converging'] and
                step['discrepancy']+step['uncertainty_estimate'] <= q['tolerance'])
        expected_d = 'accept' if independent['accepted'] and r['decision']['resolution']['accepted'] else 'reject'
        assert r['verdicts']['D'] == expected_d
        expected_gates = expected_d if all(r['checks'].values()) else 'reject'
        assert r['verdicts']['D_gates'] == expected_gates
        third = r['third']
        values = [complex(*v) for v in third['values']]
        assert abs(values[1]-values[0]) == third['difference']
        assert third['converging'] == (third['difference'] <= q['tolerance']/8)
        assert third['uncertainty'] == 2*third['difference']+third['rounding']
        expected = ('accept' if third['converging'] and third['discrepancy']+third['uncertainty'] <= q['tolerance']
                    else 'violation_detected' if third['converging'] and third['discrepancy']-third['uncertainty'] > q['tolerance']
                    else 'inconclusive')
        assert r['verdicts']['third'] == third['verdict'] == expected
        assert r['verdicts']['D_gates_third'] == (expected if expected_gates == 'accept' else 'reject')
        assert r['accepted'] == {p:v=='accept' for p,v in r['verdicts'].items()}
    for condition, summary in data['summary'].items():
        group = [r for r in rows if r['condition']==condition]
        assert summary['tasks'] == len(group) == 96
        assert summary['accurate'] == sum(r['accurate'] for r in group)
        for policy, counts in summary['policies'].items():
            for verdict in ('accept','reject','violation_detected','inconclusive'):
                assert counts[verdict] == sum(r['verdicts'][policy]==verdict for r in group)
            assert counts['false_acceptances'] == sum(r['accepted'][policy] and not r['accurate'] for r in group)
            assert counts['accurate_rejected'] == sum(r['accurate'] and r['verdicts'][policy] in ('reject','violation_detected') for r in group)
    for k in rows[0]['work']:
        assert data['work'][k] == sum(r['work'][k] for r in rows)
    saved_intake = json.loads(path.with_name(path.stem+'-intake.json').read_text())
    assert saved_intake['intake_channel'] == intake
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path)
    parser.add_argument('repeat',type=Path)
    args = parser.parse_args()
    left,right = audit(args.original),audit(args.repeat)
    assert right['replay']['source_sha256'] == hashlib.sha256(args.original.read_bytes()).hexdigest()
    a,b = non_timing(left),non_timing(right)
    replay = b.pop('replay')
    assert 'replay' not in a
    assert a == b, 'Non-timing exact comparison differs beyond explicit replay provenance'
    print('PASS: 96 tasks and 288 rows in each run; accuracy, verdicts and component work audited.')
    print('PASS: every non-timing field exact, including environment, protocol/source hashes, input overlap and diagnostics.')
    print('Excluded: keys ending seconds; repeat-only replay provenance (source digest independently verified).')
    print('Replay provenance:',json.dumps(replay,sort_keys=True))
