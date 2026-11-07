import unittest
from unittest.mock import patch

from lab.scientist import POLICIES, run_pilot, solve


class ScientistTests(unittest.TestCase):
    def test_all_policies_obey_actual_work_budget(self):
        for policy in POLICIES:
            for budget in (1, 2, 7, 4096):
                for calls in (1, 3, 5):
                    r = solve(1j, 1, 1e-12, policy, budget, calls)
                    self.assertLessEqual(r['modes_spent'], budget)
                    self.assertLessEqual(r['calls'], calls)
                    self.assertEqual(r['modes_spent'], sum(t['modes'] for t in r['trace']))

    def test_state_intervention_changes_action(self):
        normal = solve(0, 1, .001)
        stale = solve(0, 1, .001, 'stale')
        self.assertLess(stale['trace'][0]['modes'], normal['trace'][0]['modes'])
        self.assertEqual(normal['value'], solve(0, 1, .001, 'irrelevant')['value'])

    def test_impossible_target_not_claimed(self):
        self.assertFalse(solve(0, 1, 1e-10)['claimed_target_met'])

    def test_selection_cannot_read_evaluation_oracle(self):
        with patch("lab.scientist.analytic", side_effect=AssertionError("oracle leaked")):
            solve(1j, 1, .001)

    def test_invalid_inputs(self):
        for tol in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                solve(0, 1, tol)
        for budget in (0, True, 1.5):
            with self.assertRaises(ValueError):
                solve(0, 1, .01, budget=budget)
        with self.assertRaises(ValueError):
            solve(0, 1, .01, 'invented')
        with self.assertRaises(ValueError):
            solve(-1, 1, .01)

    def test_seed_reproduces_non_timing_results(self):
        def strip(result):
            for row in result['runs']:
                for t in row['trace']:
                    del t['kernel_seconds']
            for s in result['summary'].values():
                del s['kernel_seconds']
            return result
        self.assertEqual(strip(run_pilot()), strip(run_pilot()))


if __name__ == '__main__':
    unittest.main()
