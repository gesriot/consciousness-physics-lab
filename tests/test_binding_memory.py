import copy
import unittest
from binding.memory import select, recall, corrupt
from binding.b001 import append_ledger, outcomes, intake, validate_intake
from attention.build_a007_library import CHANNELS


def record(identity='r', time=10, support=6):
    return dict(id=identity, time=time, links=[
        dict(edge_id=identity+':0', channel='collocation:0', time=support, last_success=-1),
        dict(edge_id=identity+':1', channel='normalization', time=2, last_success=-1)])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.notice = dict(channel='collocation:0', time=20, last_success=5)

    def test_support_not_closure(self):
        records = [record('old', 10, 4), record('new', 12, 7)]
        self.assertEqual(select(records, self.notice, 'G')[0], ['new'])
        self.assertEqual(select(records, self.notice, 'T')[0], ['new', 'old'])

    def test_strict_time_and_notice(self):
        self.assertEqual(select([record(support=5)], self.notice, 'G')[0], [])
        self.assertEqual(select([record(time=20)], self.notice, 'A')[0], [])

    def test_corruption_degree_volume_and_causality(self):
        records = [record()]
        left, le = corrupt(records, 'collocation:0')
        right, re = corrupt(records, 'collocation:0', True)
        self.assertEqual(len(left[0]['links']), len(right[0]['links']))
        self.assertEqual([e['l1'] for e in le], [e['l1'] for e in re])
        self.assertEqual(select(records, self.notice, 'Gx')[0], [])
        self.assertEqual(select(records, self.notice, 'Gxprime')[0], ['r'])
        self.assertEqual(records, [record()])

    def test_endpoint_neighbor(self):
        r = record(); r['links'][0]['channel'] = 'collocation:5'
        altered, edits = corrupt([r], 'collocation:5')
        self.assertEqual(altered[0]['links'][0]['channel'], 'collocation:4')
        self.assertEqual(edits[0]['l1'], 1)

    def test_budget_dedup_and_strict_recall(self):
        rs = [record('a'), record('b', 11)]
        result = recall(rs, [self.notice]*2, 'A', 1, lambda _: {'verdict':'inconclusive'}, 3.)
        self.assertEqual(result['reviewed'], ['b'])
        self.assertEqual(result['withdrawn'], ['b'])
        self.assertEqual(result['cost'], 3.)
        self.assertEqual(result['unused_budget'], 0.)

    def test_no_notice_and_N(self):
        fail = lambda _: self.fail('Replacement must not run')
        self.assertEqual(recall([record()], [], 'G', 32, fail, 1.)['cost'], 0.)
        self.assertEqual(recall([record()], [self.notice], 'N', 32, fail, 1.)['cost'], 0.)

    def test_last_bit_ties(self):
        rs = [record('b'), record('a')]
        for memory in ('G','T','A','N','Gx','Gxprime'):
            self.assertEqual(select(rs, self.notice, memory), select(rs, self.notice, memory, noise=True))

    def test_accurate_recall_not_false_positive_selection(self):
        rs = [record()]
        result = recall(rs, [self.notice], 'G', 1, lambda _: {'verdict':'accept'}, 1.)
        m = outcomes(result, rs, {'r': {'accurate': True, 'affected_channels': ['collocation:0']}}, [self.notice])
        self.assertEqual(m['unnecessary_correct_recalls'], 0)
        self.assertEqual(m['selected_affected'], 1)
        self.assertIsNone(m['precision'])
        self.assertEqual(m['recall'], 0.)

    def test_success_not_attempt_and_replacement_supersedes(self):
        last = {c:-1 for c in CHANNELS}
        def ordinary(replacement=False):
            return dict(sample=False, input_index=0, channel='collocation:0', replacement=replacement,
                        observation={'verdict':'accept'}, fault_applied=not replacement)
        actions = [ordinary(), ordinary(True), dict(sample=True, channel='collocation:0', expected='violation_detected',
                     observation={'verdict':'accept'})]
        trace = dict(actions=actions, inputs=[dict(verdict='accept', closure_action_index=2)])
        rs, ev = [], {}
        ns = append_ledger(trace, 0, last, rs, ev, [{'accurate':True}], 0)
        self.assertEqual(last['collocation:0'], -1)
        self.assertEqual(ns[0]['last_success'], -1)
        self.assertEqual(ev['0:0']['affected_channels'], [])
        self.assertEqual(rs[0]['links'][0]['channel'], 'independent:collocation:0')
        self.assertEqual(select(rs, ns[0], 'G')[0], [])
        actions[-1]['observation']['verdict'] = 'violation_detected'
        append_ledger(trace, 0, last, [], {}, [{'accurate':True}], 0)
        self.assertEqual(last['collocation:0'], 5)

    def test_seed_intake_and_tamper(self):
        x = intake(20261206)
        self.assertEqual(x, intake(20261206))
        validate_intake(x)
        x['requests'][0]['request_sha256'] = 'bad'
        with self.assertRaises(ValueError): validate_intake(x)


if __name__ == '__main__':
    unittest.main()

class IsolationTests(unittest.TestCase):
    def test_timelog_without_links(self):
        notice = dict(channel='collocation:0', time=20, last_success=5)
        r = record()
        for policy in ('T', 'A', 'N'):
            expected = select([r], notice, policy)
            self.assertEqual(expected, select([dict(id=r['id'], time=r['time'])], notice, policy))

    def test_superseded_irrelevant_edge(self):
        r = record(); r['links'][1]['channel'] = 'independent:normalization'
        result, edits = corrupt([r], 'collocation:0', True)
        self.assertEqual(edits[0]['l1'], 1)
        self.assertEqual(result[0]['links'][1]['channel'], 'independent:first_moment')

    def test_replacement_ignores_observed_diagnostics(self):
        from attention.agent import Executor
        from attention.channel_monitor import guarded_uniform
        from lab.guard import Request
        from lab.executor import request_hash
        from unittest.mock import patch
        r = Request(1., 1., .01)
        e = Executor(r, request_hash(r), 'clean', 0.)
        e.perform('modal')
        with patch('attention.channel_monitor.uniform_response', return_value=1+0j):
            first = guarded_uniform(e)
            e.observed = {'v2_2048': {'verdict':'accept', 'step': {'uncertainty_estimate':0}}}
            self.assertEqual(first, guarded_uniform(e))
