from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from lab.finite_volume import response, convergence, check_value
from lab.executor import execute, as_outcome
from lab.guard import Request
from lab.scientist import analytic


class FiniteVolumeTests(unittest.TestCase):
    def test_dc_and_complex_response(self):
        self.assertAlmostEqual(response(0,1,128).real,1,places=10)
        for z in (.01, 1, 10j, 100,100j):
            coarse=abs(response(z,1,64)-complex(analytic(z,1)))
            fine=abs(response(z,1,512)-complex(analytic(z,1)))
            self.assertLess(fine,coarse)
            self.assertLess(fine,1e-4)

    def test_independent_path_has_no_modal_or_analytic_call(self):
        with patch('lab.scientist.analytic',side_effect=AssertionError), patch('lab.scientist.modal_partial_sums',side_effect=AssertionError):
            self.assertTrue(convergence(1j,1)['converging'])

    def test_detect_value_corruption(self):
        self.assertFalse(check_value(0,1,.001,.5)['accepted'])
        self.assertTrue(check_value(0,1,.001,1)['accepted'])

    def test_validation(self):
        for args in [(-1,1,64),(1,0,64),(1,1,True),(1,1,0)]:
            with self.assertRaises(ValueError):response(*args)


class ExecutorTests(unittest.TestCase):
    def test_compute_and_bind_receipt(self):
        request=Request(0,1,.001)
        receipt=execute(request,608)
        self.assertEqual(as_outcome(request,receipt)['modes_spent'],608)
        with self.assertRaises(FrozenInstanceError):receipt.modes=1
        with self.assertRaises(ValueError):as_outcome(Request(1,1,.001),receipt)

    def test_only_plan_not_planner_value_is_accepted(self):
        with self.assertRaises(ValueError):execute(Request(0,1,.001),{'modes':608,'value':[0,0]})
        with self.assertRaises(ValueError):execute(Request(0,1,.001),4097)
