"""Regression tests for the archival score reconstruction."""
import unittest
from unittest.mock import patch
from attention import channel_monitor as monitor, s3
from attention.build_a007_library import CHANNELS
from binding import b002
from binding.analyze_b002_allocation import ROOT, scores
from binding.audit import read
from lab.guard import Request
class AllocationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = s3.load_model()['action_model']['cost_weights']
        cls.pools = monitor.sample_pools(read(ROOT / 'attention/sample_library_a007.json'))
    def test_matches_reference_for_both_prices(self):
        requests = [Request(1, 1, .01), Request(2, 1, .001)]
        evidence = [{}, {'binding': {'verdict': 'accept'}}]
        last = {c: i % 3 for i, c in enumerate(CHANNELS)}
        for price in (monitor.replacement_price, b002.native_price):
            for closed in ({}, {0: 'accept'}, {0: 'accept', 1: 'accept'}):
                with patch.object(monitor, 'replacement_price', price):
                    expected = monitor.score_channels(requests, evidence, None, closed, self.weights, self.pools, last, 7, 'BU', set())
                actual = scores(requests, evidence, closed, self.weights, self.pools, last, 7, set(), price)
                self.assertEqual(actual[:2], expected[:2])
                self.assertEqual(actual[2], expected[2]['raw'])
    def test_closed_accepts_do_not_contribute(self):
        r = Request(1, 1, .01)
        args = (self.weights, self.pools, dict.fromkeys(CHANNELS, 0), 0, set(), b002.native_price)
        self.assertEqual(scores([r], [{}], set(), *args), scores([r, r], [{}, {'binding': {'verdict': 'accept'}}], {1}, *args))
    def test_price_ratio(self):
        r = Request(1, 1, .01)
        self.assertAlmostEqual(monitor.replacement_price('collocation:0', r, self.weights) / b002.native_price('collocation:0', r, self.weights), 61440 / 8128)
if __name__ == '__main__':
    unittest.main()
