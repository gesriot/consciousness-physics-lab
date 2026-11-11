"""Contract checks for typed stakes, persistent state and paid decisions."""
import copy
import inspect
import unittest
from unittest.mock import patch
from attention.build_a007_library import CHANNELS
from lab.guard import Request
from lab.executor import request_hash
from binding import self_checks_b003 as B


def record(index=0, timestamp=4, which='collocation:2'):
    return dict(id=f'0:{index}', time=6, standing=True, task_id=0, links=[
        dict(edge_id=f'{index}:c', kind='channel', channel=which, time=timestamp),
        dict(edge_id=f'{index}:n', kind='channel', channel='normalization', time=2),
        dict(edge_id=f'{index}:r', kind='request', channel='request', time=6,
             request_id=0, request_hash=request_hash(Request(1, 1, .01)))])


class StakesTests(unittest.TestCase):
    def setUp(self):
        self.markers = {c: -1 for c in CHANNELS}
        self.prices = {c: 2. for c in CHANNELS}

    def test_only_standing_newer_raw_support(self):
        rows = [record(), record(1)]
        rows[1]['standing'] = False
        counts, scores = B.stakes(rows, self.markers, self.prices)
        self.assertEqual(counts['collocation:2'], 1)
        self.assertEqual(scores['collocation:2'], .5)
        self.markers['collocation:2'] = 4
        self.assertEqual(B.stakes(rows, self.markers, self.prices)[0]['collocation:2'], 0)

    def test_distinct_record_not_repeated_support(self):
        row = record()
        row['links'].append(copy.deepcopy(row['links'][0]))
        self.assertEqual(B.stakes([row], self.markers, self.prices)[0]['collocation:2'], 1)

    def test_price_denominator(self):
        self.prices['collocation:2'] = 4.
        self.assertEqual(B.stakes([record()], self.markers, self.prices)[1]['collocation:2'], .25)

    def test_full_relevant_and_paired_request_control(self):
        original = [record(), record(1)]
        x, xe = B.visible_graph(original, self.markers, 'Sx')
        p, pe = B.visible_graph(original, self.markers, 'Sxprime')
        self.assertEqual(B.stakes(x, self.markers, self.prices)[0]['collocation:2'], 0)
        self.assertEqual(B.stakes(p, self.markers, self.prices), B.stakes(original, self.markers, self.prices))
        self.assertEqual([e['record_id'] for e in xe], [e['record_id'] for e in pe])
        self.assertEqual(len(xe), 2)
        self.assertEqual(original[0]['links'][0]['channel'], 'collocation:2')
        self.assertEqual(p[0]['links'][-1]['request_hash'], original[0]['links'][-1]['request_hash'])

    def test_ambiguous_root_is_failure(self):
        row = record(); row['links'].append(copy.deepcopy(row['links'][-1]))
        with self.assertRaises(ValueError):
            B.visible_graph([row], self.markers, 'Sxprime')

    def test_eligible_only_after_marker(self):
        self.markers['collocation:2'] = 4
        self.assertFalse(B.visible_graph([record()], self.markers, 'Sx')[1])

    def test_suspect_excluded(self):
        chosen, _, _ = B.choose_sample('S', [record()], self.markers, {'collocation:2'}, self.prices, 0)
        self.assertEqual(chosen, 'normalization')

    def test_zero_stake_oldest_success(self):
        first, _, _ = B.choose_sample('S', [], self.markers, set(), self.prices, 0)
        self.markers[first] = 1
        second, _, _ = B.choose_sample('S', [], self.markers, set(), self.prices, 0)
        self.assertEqual((first, second), (CHANNELS[0], CHANNELS[1]))

    def test_round_before_tie(self):
        self.prices['v2:0'] = 2.+1e-12
        rows = [record(0, which='v2:0'), record(1, which='v2:1')]
        # Remove the shared normalization edges so the tied channels are maximal.
        for row in rows: row['links'] = [row['links'][0], row['links'][-1]]
        chosen, _, _ = B.choose_sample('S', rows, self.markers, set(), self.prices, 0, True)
        self.assertEqual(chosen, 'v2:0')

    def test_rr_pointer_persists_skips_suspect(self):
        which, pointer, _ = B.choose_sample('RR', [], self.markers, {'v2:5'}, self.prices, 5)
        self.assertEqual(which, 'collocation:0')
        self.assertEqual(pointer, 7)
        state = B.State('RR', 1, 5); state.rr_pointer = pointer
        self.assertEqual(state.snapshot()['rr_pointer'], 7)

    def test_selection_has_no_truth_input(self):
        params = inspect.signature(B.choose_sample).parameters
        self.assertFalse({'truth', 'target', 'onset', 'accuracy', 'fault_channels'} & set(params))
        row = record(); row['accurate'] = False
        a = B.choose_sample('S', [row], self.markers, set(), self.prices, 0)[0]
        row['accurate'] = True
        b = B.choose_sample('S', [row], self.markers, set(), self.prices, 0)[0]
        self.assertEqual(a, b)

    def test_request_root_used_and_mutation_detected(self):
        row = record(); request = Request(1, 1, .01)
        self.assertEqual(B.validate_request_link(row, {0: request}), request)
        row['links'][-1]['channel'] = 'independent:request'
        self.assertEqual(B.validate_request_link(row, {0: request}), request)
        row['links'][-1]['request_hash'] = 'changed'
        with self.assertRaises(ValueError): B.validate_request_link(row, {0: request})

    def test_alarm_queues_by_g_and_deduplicates(self):
        state = B.State('S', 1, 0)
        state.records = [record(0, 2), record(1, 4)]
        queued = state.alarm('collocation:2', 9)
        self.assertEqual([p['record_id'] for p in queued], ['0:1', '0:0'])
        self.assertFalse(state.alarm('collocation:2', 11))
        self.assertEqual(state.last_success['collocation:2'], -1)

    def test_n0_uses_zero_quota(self):
        self.assertEqual(B.State('N0', 4, 0).quota, 0)
        with self.assertRaises(ValueError): B.State('S', 8, 0)

    def test_snapshot_isolated(self):
        state = B.State('S', 1, 0)
        snap = state.snapshot(); snap['last_success']['v2:0'] = 99
        self.assertEqual(state.last_success['v2:0'], -1)

    def test_modal_reexecution_preserves_invalid_original_candidate(self):
        request = Request(1, 1, .01)
        bad = {'value': [float('nan'), 0.], 'trace': [{'call': 1, 'modes': 1}],
               'calls': 1, 'modes_spent': 1}
        with patch.object(B.b002.P, 'model_response', return_value=1+0j) as work:
            result, cost = B._cross('modal', request, bad, {'modal_term': 1.})
        self.assertEqual(work.call_count, 1)
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertEqual(result['reexecution_value'], [1., 0.])
        self.assertGreater(cost, 0.)
        self.assertFalse(result['resolution']['accepted'])

    def test_modal_invalid_fresh_result_cannot_confirm_valid_original(self):
        request = Request(1, 1, .01)
        modes = B.b002.P.plan_modes(request)
        valid = {'value': [1., 0.], 'trace': [{'call': 1, 'modes': modes}],
                 'calls': 1, 'modes_spent': modes}
        self.assertTrue(B.final_guard(request, valid)['accepted'])
        with patch.object(B.b002.P, 'model_response', return_value=complex(float('nan'), 0)):
            result, _ = B._cross('modal', request, valid, {'modal_term': 1.})
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertEqual(result['outcome'], valid)
        self.assertTrue(result['resolution']['accepted'])
        self.assertFalse(result['reexecution_resolution']['accepted'])

    def test_nonnumerical_recheck_preserves_scientific_model(self):
        with patch.object(B.agent, 'feature', return_value={'verdict': 'violation_detected'}) as work:
            result, _ = B._cross('normalization', Request(1, 1, .01), {'value': [1., 0.]},
                                 {'modal_term': 1., 'fv_cell': 1.}, 'slab')
        self.assertEqual(work.call_args.args[-1], 'slab')
        self.assertEqual(result['model'], 'slab')

    def test_native_cross_price_not_cache_discount(self):
        request = Request(1, 1, .01)
        weights = {'fv_cell': 2.}
        with patch.object(B.b002, 'replacement', return_value={'verdict': 'accept'}):
            result, cost = B._cross('collocation:0', request, {'value': [1., 0.]}, weights)
        self.assertEqual(cost, sum(B.agent.GRIDS)*2.)
        self.assertEqual(result['charged_work']['grids'], list(B.agent.GRIDS))


class ExecutionTests(unittest.TestCase):
    def execute(self, budget=4., quota=1, alarm=True, policy='S', setup=None, price_fn=None):
        from contextlib import ExitStack
        request = Request(1, 1, .01)
        outcome = {'value': [1., 0.]}
        class FakeExecutor:
            def __init__(self):
                self.request = request
                self.outcome = copy.deepcopy(outcome)
                self.observed = {}
            def perform(self, action):
                self.observed[action] = {'verdict': 'accept'}
                return {'verdict': 'accept'}
        state = B.State(policy, quota, 0)
        old = record(); old['links'] = [old['links'][0], old['links'][-1]]
        old.update(package_id=0, input_index=0, original_links=copy.deepcopy(old['links']), history=[])
        state.records = [old]
        state.requests[0] = request
        state.candidates[old['id']] = outcome
        state.tick = 8
        if setup: setup(state)
        pools = {c: [dict(sample_id=c, family=c.split(':')[0], expected='violation_detected' if alarm else 'accept')]
                 for c in CHANNELS}
        sample_result = dict(cost=1., verdict='accept', observed={'verdict': 'accept'},
                             original={'verdict': 'violation_detected' if alarm else 'accept'}, trace=[])
        with ExitStack() as stack:
            for target, name, fn in (
                (B.monitor, 'sample_pools', lambda library: pools),
                (B, 'maximum_cost', lambda sample, weights: 1.),
                (B.agent, 'cost', lambda action, request, model: 1.),
                (B.b002, 'native_price', price_fn or (lambda channel, request, weights: 1.)),
                (B.agent, 'available', lambda obs: [] if obs else ['collocation']),
                (B.agent, 'verdict', lambda obs: 'accept' if obs else 'inconclusive'),
                (B.s3, 'token', lambda action, obs: obs),
                (B.s3, 'observation', lambda action, token: token),
                (B, 'channel', lambda action, request: 'collocation:2'),
                (B.monitor, 'sample_execution', lambda *args: copy.deepcopy(sample_result)),
                (B, '_cross', lambda *args: ({'verdict': 'accept'}, 1.))):
                stack.enter_context(patch.object(target, name, fn))
            result = B.run_package([FakeExecutor()], [dict(task_id=0, outcome=outcome)],
                [dict(task_id=0, request_sha256=request_hash(request))], 1, budget,
                {'action_model': {'cost_weights': {}}}, {}, state,
                ('collocation:2',) if alarm else ())
        return state, result

    def test_alarm_recall_future_all_paid_one_allowance(self):
        state, result = self.execute()
        self.assertEqual([a['kind'] for a in result['actions']], ['sample', 'recall', 'ordinary', 'future'])
        self.assertEqual(result['spent'], 4.)
        self.assertEqual(result['costs'], dict(sample=1., recall=1., ordinary=1., replacement=0., future=1.))
        self.assertEqual(len(state.recalled), 1)
        self.assertEqual(result['inputs'][0]['verdict'], 'accept')
        self.assertEqual(state.records[-1]['links'][0]['channel'], 'independent:collocation:2')
        self.assertEqual(state.last_success['collocation:2'], -1)

    def test_future_unaffordable_stays_unresolved(self):
        state, result = self.execute(budget=3.)
        self.assertEqual(result['inputs'][0]['verdict'], 'inconclusive')
        self.assertEqual(len(state.records), 1)
        self.assertEqual(result['spent'], 3.)

    def test_clean_sample_advances_persistent_marker(self):
        state, result = self.execute(alarm=False)
        self.assertGreater(state.last_success['collocation:2'], 4)
        self.assertEqual(result['costs']['recall'], 0.)
        self.assertEqual(result['costs']['future'], 0.)

    def test_sample_deadline_excludes_retrospective_cost(self):
        _, result = self.execute(budget=6., quota=2)
        self.assertEqual([a['kind'] for a in result['actions']],
                         ['sample', 'recall', 'ordinary', 'future', 'sample'])
        samples = [a for a in result['actions'] if a['kind'] == 'sample']
        self.assertEqual(samples[-1]['ordinary_cost_before'], 2.)
        self.assertEqual(result['costs']['recall'], 1.)

    def test_unaffordable_review_head_blocks_cheaper_review(self):
        def setup(state):
            second = record(1, which='normalization')
            second['links'] = [second['links'][0], second['links'][-1]]
            second.update(history=[], original_links=copy.deepcopy(second['links']))
            state.records.append(second)
            state.pending = [dict(record_id='0:0', channel='collocation:2',
                                  notice={'last_success': -1}),
                             dict(record_id='0:1', channel='normalization',
                                  notice={'last_success': -1})]
        _, result = self.execute(budget=3., policy='N0', setup=setup,
                                 price_fn=lambda c, r, w: 5. if c == 'collocation:2' else 1.)
        self.assertFalse(any(a['kind'] == 'recall' for a in result['actions']))
        self.assertTrue(any(a['kind'] == 'ordinary' for a in result['actions']))
        self.assertEqual([p['record_id'] for p in result['state']['pending']], ['0:0', '0:1'])

    def test_quota_does_not_silently_clamp(self):
        with self.assertRaises(ValueError): self.execute(budget=1., quota=2)


if __name__ == '__main__':
    unittest.main()
