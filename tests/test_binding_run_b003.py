"""Runner contract tests without experimental solver executions."""
import unittest
from unittest.mock import patch

from binding import run_b003 as runner


class RunnerContracts(unittest.TestCase):
    def test_synthetic_stream_prefix(self):
        short, long = runner.intake(99, 4), runner.intake(99, 8)
        self.assertEqual(short['requests'], long['requests'][:20])
        self.assertEqual(short['sample_seeds'], long['sample_seeds'][:4])
        for key in ('single_target', 'rr_phase', 'streams'):
            self.assertEqual(short[key], long[key])
        self.assertEqual(len({tuple(row['spawn_key']) for row in short['streams'].values()}), 5)
        runner.validate_intake(short)

    def test_tampered_stream_rejected(self):
        data = runner.intake(99, 4)
        data['rr_phase'] = (data['rr_phase']+1) % len(runner.CHANNELS)
        with self.assertRaises(ValueError):
            runner.validate_intake(data)

    def test_archived_float_identity_not_regenerated(self):
        data = runner.intake(99, 4)
        row = data['requests'][0]
        original = row['request']['z'][0]
        row['request']['z'][0] = float(runner.np.nextafter(original, runner.np.inf))
        self.assertNotEqual(row['request']['z'][0], original)
        row['request_sha256'] = runner.b002.request_hash(runner.decode(row['request']))
        with patch.object(runner, 'intake', side_effect=AssertionError('Must not regenerate requests')):
            runner.validate_intake(data)

    def test_archived_float_broken_hash_rejected(self):
        data = runner.intake(99, 4)
        row = data['requests'][0]
        row['request']['z'][0] = float(runner.np.nextafter(row['request']['z'][0], runner.np.inf))
        with self.assertRaises(ValueError):
            runner.validate_intake(data)

    def test_no_preonset_detection(self):
        actions = [dict(kind='sample', package_id=0, channel='collocation:0', mismatch=True,
                        fault_applied=False, cost=1.),
                   dict(kind='sample', package_id=1, channel='collocation:0', mismatch=True,
                        fault_applied=False, cost=1.),
                   dict(kind='sample', package_id=2, channel='collocation:0', mismatch=True,
                        fault_applied=True, cost=1.)]
        summary = runner.summarize(dict(records=[]), {}, actions, {'sample': 3.},
                                   4, 'single', 'collocation:0')
        self.assertEqual(summary['detection_delay'], 1)
        self.assertFalse(summary['detection_censored'])
        censored = runner.summarize(dict(records=[]), {}, actions[:2], {'sample': 2.},
                                    4, 'single', 'collocation:0')
        self.assertEqual(censored['detection_delay'], 4)
        self.assertTrue(censored['detection_censored'])

    def test_memoization_bounded_copied_reset(self):
        calls = []
        def fake(z, time, modes):
            calls.append((z, time, tuple(modes)))
            return [complex(z)]
        with patch.object(runner.executor_module, 'modal_partial_sums', fake):
            with runner.numerical_memoization(maxsize=2) as caches:
                first = runner.executor_module.modal_partial_sums(1, 1, [4])
                first[0] = 999
                second = runner.executor_module.modal_partial_sums(1, 1, [4])
                self.assertEqual(second, [1+0j])
                self.assertEqual(len(calls), 1)
                cache = caches['lab.executor.modal_partial_sums']
                self.assertEqual(cache.cache_info().maxsize, 2)
            self.assertEqual(cache.cache_info().currsize, 0)
            with runner.numerical_memoization(maxsize=2):
                runner.executor_module.modal_partial_sums(1, 1, [4])
            self.assertEqual(len(calls), 2)

    def test_N0_once_per_scenario(self):
        cells = runner.policy_cells('confirmation')
        self.assertEqual(cells.count(('N0', 0)), 1)
        self.assertEqual(len(cells), 16)
        frozen = {'single_channel': {'selected_packages': 8}}
        self.assertEqual(runner.scenarios('confirmation', frozen),
                         [('whole', 4), ('single', 8), ('clean', 4), ('clean', 8)])


if __name__ == '__main__':
    unittest.main()
