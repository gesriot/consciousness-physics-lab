"""Mutation tests for independent archive reconstruction, not runner replay."""
import copy
from pathlib import Path
import unittest

from binding import audit as A


def make_case():
    record = dict(id='0:0', time=8, links=[dict(edge_id='e0', channel='collocation:0', time=5, region='0', last_success=-1), dict(edge_id='e1', channel='normalization', time=7, region='normalization', last_success=-1)], last_good_global=-1, last_success={})
    notice = dict(channel='collocation:0', time=10, last_success=-1, last_good_global=-1, source='external')
    info = dict(outcome={'value':[0., 0.]}, tolerance=.1)
    values = [1., .5, .25, .125]
    step = A.P.v2_step([complex(v) for v in values], 32768, 0j, .1)
    v = step['verdict']
    margin = .1-step['discrepancy']-step['uncertainty_estimate'] if v == 'accept' else step['discrepancy']-step['uncertainty_estimate']-.1 if v == 'violation_detected' else None
    contraction = .5*step['previous_difference']-step['last_difference']
    threshold = 4*step['rounding']
    passed = margin is not None and margin > threshold and contraction > threshold
    guarded = dict(grids=[4096,8192,16384,32768], values=[[v,0.] for v in values], step=step, unguarded_verdict=v, verdict=v if passed else 'inconclusive', safety=dict(multiplier=4, decision_margin=margin, contraction_margin=contraction, threshold=threshold, passed=passed))
    assert guarded['verdict'] != 'accept'
    metrics = dict(residual_false_accepts=0, unnecessary_correct_recalls=0, recall_cost=15, attempted=1, withdrawn=1, affected=1, affected_false=1, true_withdrawals=1, precision=1., recall=1., selected_affected=1, unwithdrawn_affected=0, service_delays=[1], accepted=1, initial_false=1)
    result = dict(attempts=[dict(record_id='0:0', notice_time=10, service_delay=1, result=guarded, withdrawn=True, cost=15)], withdrawn=['0:0'], reviewed=['0:0'], selected=['0:0'], interventions=[dict(notice=notice, edits=[], changed_edges=0, l1=0)], cost=15, budget=30, unused_budget=15, unreviewed_selected=[], memory_schema=['id','time','links','last_success','last_good_global'], record_count=1, metrics=metrics)
    evaluator = {'0:0':dict(accurate=False, affected_channels=['collocation:0'])}
    return result, [record], evaluator, [notice], 'G', 2, 15, {'0:0':info}


def test_valid_recall(case):
    A.recall_check(*case)


def test_recall_mutations_detected(case, mutation):
    result, records, *_ = case
    if mutation == 'false_verdict': result['attempts'][0]['result']['verdict'] = 'accept'
    if mutation == 'count': result['attempts'] = []
    if mutation == 'cost': result['cost'] += 1
    if mutation == 'time': records[0]['links'][0]['time'] = -2
    if mutation == 'metric': result['metrics']['residual_false_accepts'] = 1
    if mutation == 'retention': result['attempts'][0]['withdrawn'] = False
    with unittest.TestCase().assertRaises(A.AuditError): A.recall_check(*case)


def test_corruption_parity_and_projection(case):
    _, records, _, notices, *_ = case
    selected, edits, gx = A.projection(records, notices[0], 'Gx', ())
    kept, other, gp = A.projection(records, notices[0], 'Gxprime', ())
    assert selected == [] and kept == ['0:0']
    assert [e['record_id'] for e in edits] == [e['record_id'] for e in other] == ['0:0']
    assert edits[0]['l1'] == other[0]['l1'] == 1
    assert len(gx[0]['links']) == len(gp[0]['links']) == 2
    assert [e['time'] for e in gx[0]['links']] == [e['time'] for e in gp[0]['links']]
    _, _, visible = A.projection(records, notices[0], 'T', ())
    assert visible == [{'id':'0:0', 'time':8}]


def test_corrupt_log_mutation(case):
    case = list(case)
    result = case[0]
    case[4] = 'Gxprime'
    _, edits, _ = A.projection(case[1], case[3][0], 'Gxprime', ())
    result['interventions'][0].update(edits=edits, changed_edges=1, l1=1)
    A.recall_check(*case)
    result['interventions'][0]['edits'][0]['record_id'] = 'wrong'
    with unittest.TestCase().assertRaises(A.AuditError): A.recall_check(*case)


def test_boundaries_and_oracle_independence(case):
    _, records, _, notices, *_ = case
    notices[0]['last_success'] = 5
    assert A.projection(records, notices[0], 'G', ())[0] == []
    assert A.projection(records, notices[0], 'T', ())[0] == ['0:0']
    notices[0]['time'] = 8
    assert A.projection(records, notices[0], 'A', ())[0] == []
    notices[0]['time'] = 10
    records[0].update(accurate=True, affected_channels=['oracle'])
    assert A.projection(records, notices[0], 'A', ())[0] == ['0:0']
    assert A.projection(records, notices[0], 'A', ['0:0'])[0] == []


def test_archive_reconstruction_if_present():
    root = Path('binding/results/B001/development-complete/series-00')
    if not (root/'ledger.json.gz').exists(): raise unittest.SkipTest('optional completed development archive absent')
    assert A.audit_series(root)['cells'] == 144
    intake, reference = A.read(root/'intake.json'), A.read(root/'reference.json')
    packages = [A.read(root/f'none-{i}.json.gz') for i in range(4)]
    baseline = A.reconstruct(packages, intake, reference)
    packages[0]['inputs'][0]['accurate'] = not packages[0]['inputs'][0]['accurate']
    with unittest.TestCase().assertRaises(A.AuditError): A.reconstruct(packages, intake, reference)
    ledger = A.read(root/'ledger.json.gz')['records']['none']
    ledger[0]['links'][0]['time'] += 1
    with unittest.TestCase().assertRaises(A.AuditError): A.equal(ledger, baseline[0], 'provenance time')


class AuditTests(unittest.TestCase):
    def test_valid(self): test_valid_recall(make_case())
    def test_negative_mutations(self):
        for mutation in ['false_verdict','count','cost','time','metric','retention']:
            with self.subTest(mutation=mutation): test_recall_mutations_detected(make_case(), mutation)
    def test_corruption(self): test_corruption_parity_and_projection(make_case())
    def test_corrupt_log(self): test_corrupt_log_mutation(make_case())
    def test_boundaries(self): test_boundaries_and_oracle_independence(make_case())
    def test_archive(self): test_archive_reconstruction_if_present()

    def test_snapshot_mutation(self):
        case = list(make_case())
        result = case[0]
        _, _, records = A.projection(case[1], case[3][0], 'G', ())
        result['interventions'][0]['records'] = records
        A.recall_check(*case, require_snapshots=True)
        result['interventions'][0]['records'][0]['links'][0]['time'] += 1
        with self.assertRaises(A.AuditError): A.recall_check(*case, require_snapshots=True)

    def test_aggregate_freeze_closure_mutations(self):
        root = Path('binding/results/B001/development-complete')
        if not (root/'summary.json').exists(): self.skipTest('optional development archive absent')
        rows = [A.read(root/f'series-{i:02d}'/'summary.json') for i in range(12)]
        saved = A.read(root/'summary.json')
        A.aggregate_check(rows, saved)
        saved['permissive']['external']['2']['T']['residual_false_accepts']['total'] += 1
        with self.assertRaises(A.AuditError): A.aggregate_check(rows, saved)
        freeze = A.read(root/'freeze.json')
        A.freeze_check(rows, freeze)
        freeze['bands']['permissive'] = (freeze['bands']['permissive']+1)%6
        with self.assertRaises(A.AuditError): A.freeze_check(rows, freeze)
        confirm = Path('binding/results/B001/confirmation')
        if (confirm/'closure.json').exists():
            summary, freeze = A.read(confirm/'summary.json'), A.read(root/'freeze.json')
            closure = A.read(confirm/'closure.json')
            A.closure_check(summary, freeze, closure)
            closure['links_unnecessary_here'] = not closure['links_unnecessary_here']
            with self.assertRaises(A.AuditError): A.closure_check(summary, freeze, closure)

    def test_source_hash_mutation(self):
        root = Path('binding/results/B001/development-complete')
        if not (root/'manifest.json').exists(): self.skipTest('optional development archive absent')
        manifest = A.read(root/'manifest.json')
        manifest['source_sha256'][next(iter(manifest['source_sha256']))] = '0'*64
        with self.assertRaises(A.AuditError): A.source_check(manifest, Path('.'))


if __name__ == '__main__': unittest.main()
