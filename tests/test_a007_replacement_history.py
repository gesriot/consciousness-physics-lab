"""Tests for the archived post-alarm replacement accounting."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.audit_a007_replacements import audit


class ReplacementHistoryTests(unittest.TestCase):
    def package(self):
        observation = dict(grids=[4096, 8192, 16384, 32768], verdict='inconclusive',
                           unguarded_verdict='accept', safety=dict(passed=False))
        replacement = dict(sample=False, replacement=True, channel='v2:2', cost=15.,
                           observation=observation, original_observation=copy.deepcopy(observation),
                           hooked_observation=copy.deepcopy(observation))
        run = dict(suspect=['v2:2'], actions=[dict(sample=True, channel='v2:2',
                    monitor=dict(mismatch=True)), replacement],
                   inputs=[dict(required_replacements=['v2:2'], verdict='inconclusive')])
        return dict(conditions=dict(underestimated={'1.0': {'BU': run}}))

    def evaluate(self, package):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive'
            path.write_bytes(b'fixture')
            with patch('binding.audit_a007_replacements.Archive') as archive:
                archive.return_value.packages.return_value = iter([package])
                return audit(path)

    def test_counts_guarded_and_raw_separately(self):
        result = self.evaluate(self.package())
        self.assertEqual(result['total']['replacements'], 1)
        self.assertEqual(result['total']['guarded_inconclusive'], 1)
        self.assertEqual(result['total']['unguarded_accept'], 1)
        self.assertEqual(result['total']['guarded_accept'], 0)
        self.assertEqual(result['total']['replacement_cost'], 15.)

    def test_missing_prior_alarm_rejected(self):
        package = self.package()
        package['conditions']['underestimated']['1.0']['BU']['actions'][0]['monitor']['mismatch'] = False
        with self.assertRaises(AssertionError):
            self.evaluate(package)

    def test_spoiled_replacement_rejected(self):
        package = self.package()
        package['conditions']['underestimated']['1.0']['BU']['actions'][1]['hooked_observation']['verdict'] = 'accept'
        with self.assertRaises(AssertionError):
            self.evaluate(package)


if __name__ == '__main__':
    unittest.main()
