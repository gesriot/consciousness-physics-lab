import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import ive
from lab import geometry_invariants as g
from lab.executor import request_hash
from lab.finite_volume import response
from lab.guard import Request
from lab.scientist import analytic, modal_partial_sums


class GeometryInvariantTests(unittest.TestCase):
    def test_clean_is_production_exactly(self):
        for z in (0, 1 + 2j, 1e4):
            for t in (.01, 1., 100.):
                self.assertEqual(g.fv_test_hook(z, t, 512, 3), response(z, t, 512))
                self.assertEqual(g.modal_test_hook(z, t, 4096, 3),
                                 complex(modal_partial_sums(z, t, [4096])[0]))

    def test_definition_derived_sphere_predictions_against_analytic(self):
        for t in (.01, 1., 100.):
            h = .01 / t
            second = 2 * (complex(analytic(0, t)).real - complex(analytic(1j*h, t)).real) / (h*t)**2
            self.assertLess(abs(second / (20/7) - 1), .001)
            scaled = 100 * complex(analytic(1e4/t, t))
            self.assertLess(abs(scaled / np.sqrt(3/5) - 1), .003)

    def test_wrong_geometry_solvers_against_independent_closed_forms(self):
        # Laplace radial boundary-value solutions: tanh(q)/q and 2 I1(q)/(q I0(q)).
        for d in (1, 2):
            for zt in (.1, 1, 100):
                q = np.sqrt(d*(d+2)*zt)
                exact = np.tanh(q)/q if d == 1 else 2*ive(1,q)/(q*ive(0,q))
                self.assertLess(abs(g.modal_test_hook(zt, 1., 4096, d) - exact), 1.5e-4)
                self.assertLess(abs(g.fv_test_hook(zt, 1., 4096, d) - exact), 2e-6)

    def test_old_blind_new_sensitive_and_dimension_moments(self):
        for t in (.01, 1., 100.):
            request = Request(1j/t, t, .01)
            for d in (1, 2, 3):
                result = g.probes(request, t, d)
                for record in result['old']['probes'].values():
                    self.assertTrue(record['normalization_accepted'])
                    self.assertTrue(record['first_moment_accepted'])
                for record in result['second_moment']['probes'].values():
                    self.assertLess(abs(record['normalized_second_moment'] / (4*(d+2)/(d+4)) - 1), .002)
                    self.assertEqual(record['accepted'], d == 3)
                for record in result['high_frequency']['probes'].values():
                    self.assertEqual(record['accepted'], d == 3)

    def test_probe_count_no_result_cache(self):
        request = Request(1+1j, 1., .01)
        with patch.object(g, 'modal_test_hook', wraps=g.modal_test_hook) as modal, \
             patch.object(g, 'fv_test_hook', wraps=g.fv_test_hook) as fv:
            evidence = g.probes(request, 1., 3)
        self.assertEqual(modal.call_count, 5)
        self.assertEqual(fv.call_count, 5)
        self.assertEqual(sum(x['modal_terms'] for x in evidence.values()), 20480)
        self.assertEqual(sum(x['finite_volume_cells'] for x in evidence.values()), 13312)
        self.assertTrue(all(x['seconds'] >= 0 for x in evidence.values()))

    def test_oracle_after_all_probes_and_decisions(self):
        request = Request(1+1j, 1., .01)
        digest = request_hash(request)
        for geometry in g.GEOMETRIES:
            with patch.object(g, 'probes', wraps=g.probes) as probe:
                def oracle(z, t):
                    self.assertEqual(probe.call_count, 1)
                    self.assertEqual((z, t), (request.z, request.mean_time))
                    return 100+2j
                with patch.object(g, 'analytic', side_effect=oracle) as reference:
                    altered = g.evaluate(request, digest, geometry)
                    reference.assert_called_once()
            normal = g.evaluate(request, digest, geometry)
            self.assertEqual(altered['accepted'], normal['accepted'])
            self.assertEqual(altered['decision'], normal['decision'])
            self.assertNotEqual(altered['absolute_error'], normal['absolute_error'])
            self.assertTrue(normal['accepted']['hash'])
        from lab.allocation import response as restored
        self.assertIs(restored, response)

    def test_shared_geometry_can_pass_D_but_new_checks_block(self):
        request = Request(1+1j, 1., .01)
        for geometry in ('slab', 'cylinder'):
            row = g.evaluate(request, request_hash(request), geometry)
            self.assertTrue(row['accepted']['D'])
            self.assertTrue(row['accepted']['D_old_invariants'])
            self.assertFalse(row['accurate'])
            self.assertFalse(row['accepted']['D_second_moment'])
            self.assertFalse(row['accepted']['D_high_frequency'])
