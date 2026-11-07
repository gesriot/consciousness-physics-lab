import unittest
from pathlib import Path
import tempfile
from functools import lru_cache
from unittest.mock import patch
import numpy as np
from lab import coverage_map as m
from lab import surface_order
from lab.three_way import step_evidence, GRIDS


class CoverageMapTests(unittest.TestCase):
    def test_base_and_endpoints(self):
        keys = {k for c in m.base_cells() for k in m.corners(c)}
        self.assertEqual(len(keys),361*37)
        self.assertEqual(m.coordinate((0,0))[:2],(-6,-np.pi/2))
        self.assertEqual(m.coordinate((2880,288))[:2],(12,np.pi/2))

    def test_dimensions(self):
        self.assertTrue(all(r['passed'] for r in m.dimensionless_check()))

    def test_geometry_cache_preserves_values(self):
        original = surface_order.faces
        expected = [surface_order.surface_response(1+2j,1,n) for n in GRIDS]
        with patch.object(surface_order,'faces',lru_cache(maxsize=7)(original)):
            actual = [surface_order.surface_response(1+2j,1,n) for n in GRIDS]
        self.assertEqual(expected,actual)

    def test_refinement_is_local_and_dyadic(self):
        cell = (0,0,8,8)
        records = {k:dict(steps=[dict(converging=True,covered=True)]) for k in m.corners(cell)}
        self.assertFalse(m.needs_refinement(cell,records))
        records[(0,0)]['steps'][0]['covered'] = False
        self.assertTrue(m.needs_refinement(cell,records))
        self.assertEqual(len({k for c in m.children(cell) for k in m.corners(c)}),9)

    def test_frozen_formula(self):
        key = (1000,144)
        r = m.evaluate(key,0,complex(m.analytic(m.coordinate(key)[2],1)))
        values = [complex(*v) for v in r['values']]
        for i,s in enumerate(r['steps'],3):
            expected = step_evidence(values[:i+1],GRIDS[i],complex(*r['reference']),1)
            self.assertEqual(s['converging'],expected['converging'])
            self.assertEqual(s['uncertainty'],expected['uncertainty_estimate'])

    def test_reference_fallback(self):
        keys = [(2400,144),(2408,144)]
        refs,audit = m.reference_batch(keys,np.random.default_rng(20261124))
        self.assertFalse(audit['fallback'])
        with patch.object(m,'analytic',return_value=2+0j):
            bad,audit = m.reference_batch(keys,np.random.default_rng(20261124))
        self.assertTrue(audit['fallback'])
        for k in keys:
            self.assertLess(abs(refs[k]-bad[k]),1e-13)

    def test_reference_boundary_uses_exact_coordinate(self):
        keys = [(2240, y) for y in range(0, 289, 8)] + [(2241, y) for y in range(0, 289, 8)]
        _, audit = m.reference_batch(keys, np.random.default_rng(20261124))
        self.assertEqual(audit['random_subset_size'], 3)
        self.assertEqual(audit['audited'], 40)
        self.assertEqual(sum(r['coordinate'][0] == 2241 for r in audit['records']), 37)

    def test_runtime_override_reaches_metadata_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(m, 'metadata', side_effect=RuntimeError('stop before evaluation')) as mocked:
                with self.assertRaisesRegex(RuntimeError, 'stop before evaluation'):
                    m.run(Path(directory) / 'new-map', 'Cross-platform audit')
            mocked.assert_called_once_with(m.PROTOCOL, m.SOURCES, 'Cross-platform audit')
