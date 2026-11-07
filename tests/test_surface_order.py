"""Pre-experiment consistency, nesting and order tests for E009."""
import unittest
from unittest.mock import patch
from lab.scientist import analytic
from lab.guard import Request
from lab.three_way import check_three_way
from lab.surface_order import faces, surface_response, check, evidence
import numpy as np

class SurfaceOrderTests(unittest.TestCase):
    def test_nested_midpoint_subdivision(self):
        for n in (64, 128, 256, 512, 1024, 2048):
            coarse, fine = faces(n), faces(2*n)
            np.testing.assert_array_equal(coarse, fine[::2])
            np.testing.assert_allclose(fine[1::2], (coarse[:-1]+coarse[1:])/2, rtol=0, atol=2e-16)
        self.assertLess(1-faces(4096)[-2], 2e-5)

    def test_second_order_against_analytic(self):
        for z in (1, 10, 100, 1+1j, 10+10j, 100+100j):
            ref = analytic(z, 1)
            errors = [abs(surface_response(z, 1, n)-ref) for n in (64,128,256)]
            for coarse, fine in zip(errors, errors[1:]):
                self.assertTrue(3.7 < coarse/fine < 4.3, (z, errors))

    def test_normalization(self):
        self.assertAlmostEqual(surface_response(0, 1, 512).real, 1, places=8)

    def test_uniform_frozen_identical_to_d3(self):
        for request in (Request(10+10j, 1, .001), Request(1e6j, 1, .0001)):
            old = check_three_way(request, .2)
            new = check(request, .2, 'uniform', 'frozen')
            for step in new['steps']:
                step.pop('observed_order'); step.pop('uncertainty_fallback')
            self.assertEqual(new, old)

    def test_order_band_and_fallback(self):
        good = evidence([1, .25, .0625, .015625],512,0,.1,'order')
        self.assertEqual(good['observed_order'],2)
        self.assertTrue(good['converging'])
        for values in ([1,1,1,1], [1,.9,.8,.7], [0,1,3,7]):
            bad = evidence(values,512,values[-1],10,'order')
            self.assertEqual(bad['verdict'],'inconclusive')

    def test_decision_no_oracle(self):
        with patch('lab.scientist.analytic', side_effect=AssertionError('oracle')):
            check(Request(10,1,.01),.1,'surface','order')

if __name__ == '__main__':
    unittest.main()
