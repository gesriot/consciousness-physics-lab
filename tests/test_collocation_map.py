import unittest
from unittest.mock import patch
import numpy as np
from lab import collocation_map as m


class CollocationMapTests(unittest.TestCase):
    def test_planes_and_lattice(self):
        self.assertEqual(m.TOLERANCES, tuple(10.**k for k in range(-7, 0)))
        self.assertEqual(len({k for c in m.base.base_cells() for k in m.base.corners(c)}), 13357)
        self.assertEqual(m.base.coordinate((2880, 288))[:2], (12., np.pi/2))

    def test_false_agreement_negative_control(self):
        r = m.classify([0j, 0j], 1+0j)
        self.assertTrue(all(s['converging'] and not s['covered'] for s in r['steps']))

    def test_rounding_only_and_acceptance_floor(self):
        r = m.classify([0j, 0j], 1e-6+0j)
        self.assertTrue(r['rounding_only'])
        self.assertTrue(r['steps'][0]['floor_blocks_acceptance'])
        self.assertEqual(r['steps'][0]['oracle_verdict'], 'inconclusive')
        self.assertAlmostEqual(r['rounding'], 32*np.finfo(float).eps*128**4)

    def test_tolerance_boundary(self):
        d = m.TOLERANCES[0]/8
        self.assertTrue(m.classify([0j, complex(d)], 0j)['steps'][0]['converging'])
        self.assertFalse(m.classify([0j, complex(np.nextafter(d, np.inf))], 0j)['steps'][0]['converging'])

    def test_union_refinement(self):
        cell = (0, 0, 8, 8)
        rows = {k:m.classify([0j, 0j], 0j) for k in m.base.corners(cell)}
        self.assertFalse(m.base.needs_refinement(cell, rows))
        rows[(0, 0)]['steps'][6]['converging'] = False
        self.assertTrue(m.base.needs_refinement(cell, rows))

    def test_exact_audit_boundary_and_fallback(self):
        keys = [(2240, 144), (2241, 144)]
        _, audit = m.base.reference_batch(keys, np.random.default_rng(0))
        self.assertEqual([r['coordinate'] for r in audit['records']], [[2241, 144]])
        with patch.object(m.base, 'analytic', return_value=2+0j):
            _, audit = m.base.reference_batch(keys, np.random.default_rng(0))
        self.assertTrue(audit['fallback'])
        self.assertEqual(audit['audited'], 2)

    def test_formula_matches_frozen_checker(self):
        q = 1+2j
        reference = complex(m.base.analytic(q, 1))
        values = [m.solver.response(q, 1, n) for n in (64, 128)]
        r = m.classify(values, reference)
        for s in r['steps']:
            frozen = m.solver.check(q, 1, reference, s['tolerance'])
            self.assertEqual(frozen['converging'], s['converging'])
            self.assertEqual(frozen['uncertainty'], r['uncertainty'])
            self.assertEqual(frozen['verdict'], s['oracle_verdict'])

    def test_controls(self):
        self.assertTrue(all(r['passed'] for r in m.controls()))
