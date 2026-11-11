import copy
import unittest
from unittest.mock import patch
from attention import agent, s2, s3, channel_monitor as monitor
from attention.a007 import decode
from attention.io import read_json
from lab.guard import Request
from lab.executor import request_hash
from binding import b002


class ExposureTests(unittest.TestCase):
    def test_family_selector_preserves_real_channels(self):
        for family in ('v2', 'collocation'):
            selector = b002.FamilyTarget(family)
            matched = [c for c in b002.CHANNELS if c == selector]
            self.assertEqual(matched, [family+':'+str(i) for i in range(6)])
            self.assertEqual(str(selector), family+':*')
            self.assertFalse('normalization' == selector)
            self.assertFalse('independent:'+family+':0' == selector)

    def test_native_cost_charges_full_declared_work(self):
        r = Request(1, 1, .1)
        weights = s3.load_model()['action_model']['cost_weights']
        self.assertEqual(b002.native_price('collocation:2', r, weights),
                         sum(b002.P.CONFIG.grids)*weights['fv_cell'])
        self.assertEqual(b002.native_price('v2:5', r, weights),
                         (64**3+128**3)*weights['collocation_degree_cubed'])
        with b002.cross_channels():
            e = agent.Executor(r, request_hash(r), 'clean', 1)
            e.outcome = {'value': [1., 0.]}
            with patch.object(b002, 'replacement', return_value={'verdict': 'accept', 'steps': [], 'values': []}) as call:
                result, price = monitor.replacement_result('collocation:2', e, weights)
                self.assertEqual(call.call_args.args[0], 'graded_v2')
                self.assertEqual(result['verdict'], 'accept')
                self.assertEqual(result['charged_work']['grids'], list(b002.P.CONFIG.grids))
                self.assertEqual(price, b002.native_price('collocation:2', r, weights))

    def test_intake_prefixes_are_identical(self):
        long = b002.intake(123, 16)
        for duration in (4, 8):
            short = b002.intake(123, duration)
            self.assertEqual(short['requests'], long['requests'][:5*duration])
            self.assertEqual(short['sample_seeds'], long['sample_seeds'][:duration])
            b002.validate_intake(short)
        damaged = copy.deepcopy(long)
        damaged['requests'][0]['request_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'Replay intake'):
            b002.validate_intake(damaged)

    def test_freeze_requires_both_exposure_thresholds(self):
        row = {str(d): {'permissive': dict(accepted=20, initial_false=5,
            affected=9 if d == 4 else 10, affected_false=4 if d == 8 else 5)}
            for d in b002.DURATIONS}
        frozen = b002.freeze([row], ['permissive'])
        self.assertEqual(frozen['selected'], {'permissive': 16})
        row['16']['permissive']['affected_false'] = 0
        self.assertFalse(b002.freeze([row], ['permissive'])['confirmation_allowed'])

    def test_noise_is_solver_output_noise(self):
        r = Request(1, 1, .1)
        plain = b002.P.model_response('fv', r.z, r.mean_time, 64, 'sphere')
        with b002.numerical_noise(True) as counts:
            noisy = b002.P.model_response('fv', r.z, r.mean_time, 64, 'sphere')
        self.assertNotEqual(plain, noisy)
        self.assertEqual(sum(counts.values()), 1)

    def test_bu_can_keep_clean_replacement_acceptance(self):
        r = Request(1, 1, .1)
        model = s3.load_model()
        library = read_json(b002.ROOT/'attention/sample_library_a007.json')
        original_sample = monitor.sample_execution
        def mismatching(sample, weights, kind='none'):
            result = original_sample(sample, weights, 'none')
            result['verdict'] = 'violation_detected' if sample['expected'] == 'accept' else 'accept'
            result['observed']['verdict'] = result['verdict']
            return result
        spec = dict(kind='none', target_channel=b002.FamilyTarget('collocation'), onset_action=10**9,
                    circle_phase=0, random_seed=19)
        with b002.cross_channels(), patch.object(monitor, 'sample_execution', mismatching):
            result = monitor.run_batch([agent.Executor(r, request_hash(r), 'clean', 1)],
                'BU', 10000., model, library, 0, spec, 1.)
        replacements = [a for a in result['actions'] if a['replacement']]
        self.assertTrue(replacements)
        self.assertTrue(any(a['observation']['verdict'] == 'accept' for a in replacements))
        self.assertEqual(result['inputs'][0]['verdict'], 'accept')
        self.assertEqual(result['sample_allocation']['executed'], 10)


if __name__ == '__main__':
    unittest.main()
