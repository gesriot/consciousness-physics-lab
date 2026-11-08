"""Fixed development checks only; no A002 task generation."""
import inspect
import unittest
from unittest.mock import patch
from attention import agent, batch, s2
from lab.executor import request_hash
from lab.guard import Request


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = s2.load_model()
        cls.request = Request(1+1j, 1, .01)

    def executors(self, count=2, condition='clean'):
        return [agent.Executor(self.request, request_hash(self.request), condition, .5)
                for _ in range(count)]

    def permutation(self, count=2):
        return [(i, a) for i in reversed(range(count)) for a in reversed(agent.ORDER)]

    def prediction(self, solvable=.8, detect=.2):
        return dict(solvable_probability=solvable,
                    full_probabilities=dict(accept=solvable, violation_detected=0., inconclusive=1-solvable),
                    actions={a: dict(detection_probability=detect, accept_probability=.5)
                             for a in agent.ORDER})

    def test_shared_zero_and_small_budget(self):
        binding = agent.cost('binding', self.request, self.model['action_model'])
        for policy in batch.POLICIES:
            with patch.object(batch, 'predict', return_value=self.prediction()):
                zero = batch.run_batch(self.executors(45), policy, 0., self.model, self.permutation(45))
                self.assertEqual(zero['spent'], 0.)
                self.assertFalse(zero['actions'])
                self.assertTrue(all(r['drop_reason'] is None for r in zero['inputs']))
                small = batch.run_batch(self.executors(45), policy, 3*binding,
                                        self.model, self.permutation(45))
                self.assertLessEqual(small['spent'], 3*binding)
                self.assertEqual(len(small['actions']), 3)
                self.assertTrue(all(r['verdict'] == 'inconclusive' for r in small['inputs']))

    def test_canonical_cheapest_and_random_selection(self):
        requests, evidence = [self.request]*2, [{}, {}]
        self.assertEqual(batch.choose('C1', requests, evidence, 100., self.model, self.permutation(), {}), (0, 'modal'))
        self.assertEqual(batch.choose('C2', requests, evidence, 100., self.model, self.permutation(), {}), (0, 'binding'))
        expected = next(pair for pair in self.permutation() if pair[1] in agent.available({}))
        for budget in (100., 200.):
            self.assertEqual(batch.choose('C3', requests, evidence, budget, self.model, self.permutation(), {}), expected)

    def test_acceptance_complete_and_early_violation(self):
        for policy in batch.POLICIES:
            result = batch.run_batch(self.executors(1), policy, 100., self.model, self.permutation(1))
            row = result['inputs'][0]
            self.assertEqual(row['verdict'], 'accept')
            self.assertEqual(agent.verdict(row['observed']), 'accept')
            self.assertTrue(set(('modal', 'binding', 'collocation')+agent.FEATURES) <= set(row['observed']))
            self.assertTrue(any(v.get('terminal') and v['verdict'] == 'accept' for v in row['observed'].values()))
        bad = self.executors(1)[0]
        bad.digest = 'wrong'
        result = batch.run_batch([bad], 'C2', 100., self.model)
        self.assertEqual(len(result['actions']), 1)
        self.assertEqual(result['inputs'][0]['verdict'], 'violation_detected')

    def test_c4_cheap_phase_and_completed_inconclusive(self):
        requests = [self.request]*2
        observed = [{}, {}]
        state = {}
        for _ in range(8):
            i, action = batch.choose('C4', requests, observed, 100., self.model, self.permutation(), state)
            self.assertIn(action, batch.CHEAP)
            observed[i][action] = dict(verdict='accept')
        self.assertEqual(batch.choose('C4', requests, observed, 100., self.model, self.permutation(), state), (0, 'modal'))
        observed[0]['modal'] = dict(verdict='inconclusive')
        self.assertEqual(batch.choose('C4', requests, observed, 100., self.model, self.permutation(), state), (1, 'modal'))
        self.assertEqual(state['dropped'][0], 'completed_inconclusive')
        preliminary = dict(modal=dict(verdict='accept'), v2_64=dict(verdict='inconclusive', terminal=False))
        self.assertFalse(batch.completed_inconclusive(preliminary))
        preliminary['v2_64']['terminal'] = True
        self.assertTrue(batch.completed_inconclusive(preliminary))
        self.assertFalse(any(a.startswith('v2_') for a in batch.remaining_actions(preliminary)))

    def test_detection_continues_after_inconclusive(self):
        evidence = [dict(modal=dict(verdict='inconclusive'))]
        p = self.prediction(.01, 0.)
        p['actions']['binding']['detection_probability'] = .5
        with patch.object(batch, 'predict', return_value=p):
            state = {}
            self.assertEqual(batch.choose('B', [self.request], evidence, 100., self.model, self.permutation(1), state), (0, 'binding'))
            self.assertEqual(state['dropped'], {})
        with patch.object(batch, 'predict', return_value=self.prediction(.01, 0.)):
            state = {}
            self.assertIsNone(batch.choose('B', [self.request], evidence, 100., self.model, self.permutation(1), state))
            self.assertEqual(state['dropped'][0], 'predicted_unresolvable')

    def test_hidden_interface_and_cached_predictions(self):
        self.assertEqual(list(inspect.signature(batch.choose).parameters),
                         ['policy', 'requests', 'observed', 'remaining', 'model', 'permutation', 'state'])
        with patch.object(batch, 'predict', return_value=self.prediction()) as prediction, \
                patch.object(agent.Executor, 'perform', side_effect=AssertionError('No execution during choice')):
            state = {}
            for _ in range(3):
                batch.choose('B', [self.request]*2, [{}, {}], 100., self.model, self.permutation(), state)
            self.assertEqual(prediction.call_count, 2)

    def test_dprime_invariance(self):
        corrupted = s2.corrupt(self.model, 'Dprime')
        for budget in (.1, 3., 20.):
            first = batch.run_batch(self.executors(1, 'slab'), 'B', budget, self.model, self.permutation(1))
            second = batch.run_batch(self.executors(1, 'slab'), 'Dprime', budget, corrupted, self.permutation(1))
            first.pop('policy')
            second.pop('policy')
            self.assertEqual(first, second)

    def test_actions_obey_dependencies_and_accounting(self):
        result = batch.run_batch(self.executors(2), 'C3', 100., self.model, self.permutation())
        evidence = [{}, {}]
        total = 0.
        for row in result['actions']:
            i, action = row['input_index'], row['action']
            self.assertIn(action, agent.available(evidence[i]))
            self.assertEqual(agent.verdict(evidence[i]), 'inconclusive')
            evidence[i][action] = row['observation']
            total += row['cost']
        self.assertEqual(result['spent'], total)
        self.assertEqual(result['spent'], sum(r['spent'] for r in result['inputs']))
        for row in result['inputs']:
            self.assertEqual(row['verdict'], agent.verdict(row['observed']))

    def test_budget_shortfall_does_not_cap_or_drop_input(self):
        binding = agent.cost('binding', self.request, self.model['action_model'])
        state = {}
        with patch.object(batch, 'predict', return_value=self.prediction()):
            self.assertIsNone(batch.choose('B', [self.request], [{}], 0., self.model,
                                            self.permutation(1), state))
            self.assertEqual(state['dropped'], {})
            self.assertEqual(batch.choose('B', [self.request], [{}], binding, self.model,
                                          self.permutation(1), state), (0, 'binding'))
        # Remaining-cost estimate includes every possible later grid, without a
        # hidden terminal-path price, and removes those grids once terminal.
        observed = dict(modal=dict(verdict='accept'))
        self.assertTrue(all(a in batch.remaining_actions(observed)
                            for a in agent.ORDER if a.startswith('v2_')))
        observed['v2_64'] = dict(verdict='inconclusive', terminal=True)
        self.assertFalse(any(a.startswith('v2_') for a in batch.remaining_actions(observed)))

    def test_invalid_budget_and_permutation(self):
        for budget in (-1., float('nan')):
            with self.assertRaises(ValueError):
                batch.run_batch(self.executors(1), 'B', budget, self.model)
        with self.assertRaises(ValueError):
            batch.run_batch(self.executors(1), 'C3', 1., self.model)
        with self.assertRaises(ValueError):
            batch.run_batch(self.executors(1), 'C3', 1., self.model, [])


if __name__ == '__main__':
    unittest.main()
