import unittest
from unittest.mock import patch
from lab.allocation import evaluate, propose_modes, refine_value
from lab.finite_volume import check_value
from lab.guard import Request


class AllocationTests(unittest.TestCase):
    def test_fixed_refinement_matches_previous_verifier(self):
        for z in (0,1,10j,100j):
            request=Request(z,1,.001)
            a=refine_value(request,.9,max_cells=512)
            b=check_value(z,1,.001,.9)
            self.assertEqual(a['accepted'],b['accepted'])
            self.assertEqual(a['steps'][-1]['uncertainty_estimate'],b['uncertainty_estimate'])
            self.assertEqual(a['values'],b['values'])

    def test_budget_caps_and_margin(self):
        for tol in (.1,.001,1e-12):
            request=Request(1,1,tol)
            self.assertLessEqual(propose_modes(request,.5),4096)
            self.assertGreaterEqual(propose_modes(request,.5),propose_modes(request,1))
            for condition in ('A','B','C','D'):
                r=evaluate(request,condition)
                self.assertLessEqual(r['modes'],4096)
                self.assertLessEqual(r['independent']['cells_solved'],8128)

    def test_oracle_unavailable_to_decision(self):
        with patch('lab.allocation.analytic',side_effect=AssertionError('oracle leaked')):
            self.assertTrue(evaluate(Request(0,1,.001),'D')['accepted'])

    def test_registration_parameter_validation(self):
        with self.assertRaises(ValueError):propose_modes(Request(0,1,.01),.3)
        with self.assertRaises(ValueError):refine_value(Request(0,1,.01),1,1000)
