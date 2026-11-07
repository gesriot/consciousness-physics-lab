import unittest
from unittest.mock import patch
from lab.common_mode import corrupt, evaluate, invariant_defense, fv_test_hook
from lab.executor import request_hash
from lab.guard import Request
from lab.finite_volume import response


class CommonModeTests(unittest.TestCase):
    def setUp(self):
        self.request = Request(1 + 1j, 1., .01)
        self.digest = request_hash(self.request)

    def test_request_corruption_preserves_original_anchor(self):
        for fault in ('request_time', 'request_conjugate'):
            request, scale = corrupt(self.request, fault)
            self.assertNotEqual(request_hash(request), self.digest)
            self.assertEqual(request_hash(self.request), self.digest)
            self.assertEqual(scale, 1)

    def test_model_scale_does_not_corrupt_request(self):
        request, scale = corrupt(self.request, 'model_scale')
        self.assertIs(request, self.request)
        self.assertEqual(scale, 1000)
        self.assertEqual(fv_test_hook(1j, 1., 64, scale), response(1j, 1000., 64))

    def test_clean_invariants_and_wrong_scale(self):
        for t in (.01, 1., 100.):
            request = Request(1j / t, t, .01)
            self.assertTrue(invariant_defense(request, t, 1)['accepted'])
            self.assertFalse(invariant_defense(request, t, 1000)['accepted'])

    def test_oracle_is_after_decisions_and_reference_is_original(self):
        with patch('lab.common_mode.analytic', return_value=100 + 2j) as oracle:
            row = evaluate(self.request, self.digest, 'request_time')
            oracle.assert_called_once_with(self.request.z, self.request.mean_time)
        other = evaluate(self.request, self.digest, 'request_time')
        self.assertEqual(row['accepted'], other['accepted'])
        self.assertEqual(row['decision'], other['decision'])
        self.assertNotEqual(row['absolute_error'], other['absolute_error'])

    def test_model_fault_hash_blind_but_invariant_sensitive(self):
        row = evaluate(self.request, self.digest, 'model_scale')
        self.assertTrue(row['hash_accepted'])
        self.assertFalse(row['invariants']['accepted'])
        self.assertFalse(row['accepted']['D_both'])
        # Scoped patch must not leak into subsequent calls.
        from lab.allocation import response as restored
        self.assertIs(restored, response)

    def test_conjugation_invariant_blind_hash_sensitive(self):
        row = evaluate(self.request, self.digest, 'request_conjugate')
        self.assertTrue(row['invariants']['accepted'])
        self.assertFalse(row['hash_accepted'])
        self.assertFalse(row['accepted']['D_both'])
