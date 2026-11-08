"""Checks for the post-result oracle calculation."""
import unittest
from attention.development_a002 import oracle_input


def record(accurate, final, entries):
    return dict(accurate=accurate,full=dict(verdict=final,actions=[dict(action=a,cost=c,observation=dict(verdict=v)) for a,c,v in entries]))


class OracleTests(unittest.TestCase):
    def test_independent_detector_needs_no_modal(self):
        row=record(False,'violation_detected',[('modal',10,'accept'),('normalization',1,'violation_detected'),('collocation',2,'violation_detected')])
        self.assertEqual(oracle_input(row)['cost'],1)
        self.assertEqual(oracle_input(row)['actions'],['normalization'])

    def test_numerical_detector_includes_prerequisites(self):
        row=record(False,'violation_detected',[('modal',2,'accept'),('v2_64',1,'inconclusive'),('v2_128',2,'violation_detected'),('collocation',4,'violation_detected')])
        self.assertEqual(oracle_input(row)['cost'],5)
        self.assertEqual(oracle_input(row)['actions'],['modal','v2_64','v2_128'])

    def test_accurate_violation_is_not_correct(self):
        row=record(True,'violation_detected',[('normalization',1,'violation_detected')])
        self.assertIsNone(oracle_input(row)['cost'])

    def test_acceptance_needs_complete_certificate(self):
        row=record(True,'accept',[('modal',2,'accept'),('binding',1,'accept'),('collocation',4,'accept')])
        self.assertEqual(oracle_input(row)['cost'],7)

    def test_unresolved_inaccurate_is_unreachable(self):
        row=record(False,'inconclusive',[('modal',2,'inconclusive')])
        self.assertIsNone(oracle_input(row)['cost'])
