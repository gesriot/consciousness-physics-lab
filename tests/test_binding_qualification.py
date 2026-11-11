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

    def test_finite_inputs_with_overflow_abstain(self):
        result = q.protected_step('collocation', [1e308+0j]*2, -1e308+0j, 1e308)
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertFalse(result['finite'])
        json.dumps(result, allow_nan=False)

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

    def test_replay_stability_is_instrument_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = [Path(tmp)/name for name in ('first', 'repeat', 'noise')]
            for index, directory in enumerate(directories):
                directory.mkdir()
                q.atomic_json(directory/'manifest.json', dict(last_bit_noise=index == 2,
                    protocol_sha256='p', source_sha256={}, input_sha256={}))
                instruments = {name: dict(gates={'accuracy': True}, verdicts={'accept': 12},
                    coverage=dict(points=1, steps=1, protected_convergence=1,
                                  dangerous_misses=0, operational_sha256='h', source_sha256='s'))
                    for name in q.INSTRUMENTS}
                q.atomic_json(directory/'summary.json', dict(instruments=instruments))
                for series in range(12):
                    results = {name: {'verdict': 'accept'} for name in q.INSTRUMENTS}
                    if index == 2:
                        results['graded_v2']['verdict'] = 'inconclusive'
                    q.atomic_json(directory/('series-%02d.json.gz' % series),
                                  [dict(id=str(series), accurate=True, results=results)])
            result = q.finalize(*directories, Path(tmp)/'final.json')
            self.assertEqual(result['qualified'], ['collocation'])
            self.assertFalse(result['comparisons']['last_bit_noise']['decisions_pass'])
            self.assertTrue(result['instruments']['collocation']['gates']['replay_stability'])

    def test_empty_override_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                q.run(Path(tmp)/'result', override='  ')


if __name__ == '__main__':
    unittest.main()
