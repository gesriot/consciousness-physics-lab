import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from binding import qualification as q
from lab.guard import Request


class QualificationTests(unittest.TestCase):
    def test_collocation_native_convergence(self):
        values = [1+0j, 1+0j]
        step = q.protected_step('collocation', values, 1+0j, .01)
        self.assertEqual(step['verdict'], 'accept')
        self.assertGreater(step['convergence_margin'], 4*step['native_rounding'])
        self.assertEqual(step['native_rounding'], 32*np.finfo(float).eps*128**4)
        self.assertEqual(q.protected_step('collocation', values, 2+0j, .01)['verdict'], 'violation_detected')

    def test_collocation_does_not_invent_fv_contraction(self):
        result = q.protected_step('collocation', [1+0j, 1+0j], 1+0j, .01)
        self.assertTrue(result['convergence_pass'])
        self.assertEqual(result['convergence_margin'], .01/8)

    def test_native_floor_abstains(self):
        step = q.protected_step('collocation', [1+0j, 1+0j], 1+0j, 1e-7)
        self.assertEqual(step['verdict'], 'inconclusive')
        values = [1+0j]*4
        step = q.protected_step('graded_v2', values, 1+0j, .01, 512)
        self.assertEqual(step['verdict'], 'inconclusive')
        self.assertFalse(step['convergence_pass'])

    def test_graded_convergence_decisive(self):
        values = [1.064+0j, 1.016+0j, 1.004+0j, 1.001+0j]
        step = q.protected_step('graded_v2', values, 1+0j, .02, 512)
        self.assertEqual(step['verdict'], 'accept')
        self.assertGreater(step['convergence_margin'], 4*step['native_rounding'])

    def test_nonfinite_abstention(self):
        result = q.protected_step('collocation', [1+0j, complex(float('nan'), 0)], 1+0j, .01)
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertFalse(result['finite'])

    def test_solver_noise_precedes_arithmetic(self):
        request = Request(1, 1, .01)
        with patch.object(q.radial_collocation, 'response', return_value=1+1j):
            plain = q.replacement('collocation', request, 1+1j)
            noisy = q.replacement('collocation', request, 1+1j, True)
        self.assertNotEqual(plain['values'], noisy['values'])
        self.assertEqual(plain['verdict'], noisy['verdict'])

    def test_streaming_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'map.json.gz'
            data = dict(metadata='x'*70000, points=[dict(value=i, padding='y'*70000) for i in range(3)], trailing=True)
            with gzip.open(p, 'wt') as f:
                json.dump(data, f)
            self.assertEqual(list(q.map_points(p)), data['points'])

    def test_trailing_comma_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'map.json.gz'
            with gzip.open(p, 'wt') as f:
                f.write('{"points": [{"x": 1},]}')
            with self.assertRaisesRegex(ValueError, 'Trailing comma'):
                list(q.map_points(p))

    def test_empty_override_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                q.run(Path(tmp)/'result', override='  ')


if __name__ == '__main__':
    unittest.main()
