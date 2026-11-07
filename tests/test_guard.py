import copy
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from lab.guard import Request, final_guard
from lab.scientist import solve


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.request = Request(0, 1, .001)
        self.normal = solve(0, 1, .001)
        self.stale = solve(0, 1, .001, 'stale')

    def test_normal_accepted_stale_rejected(self):
        self.assertTrue(final_guard(self.request, self.normal)['accepted'])
        self.assertFalse(final_guard(self.request, self.stale)['accepted'])

    def test_state_and_self_report_are_not_authority(self):
        stale = copy.deepcopy(self.stale)
        stale['trace'][-1]['state']['target_tolerance'] = 999
        stale['trace'][-1]['bound_plus_heuristic_rounding'] = 0
        stale['claimed_target_met'] = True
        self.assertFalse(final_guard(self.request, stale)['accepted'])
        self.normal['claimed_target_met'] = False
        self.assertTrue(final_guard(self.request, self.normal)['accepted'])

    def test_no_oracle_in_guard(self):
        with patch('lab.scientist.analytic', side_effect=AssertionError('oracle leaked')):
            self.assertTrue(final_guard(self.request, self.normal)['accepted'])

    def test_request_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.request.tolerance = 99

    def test_malformed_records_fail_closed(self):
        records = [{}, None, {'trace': []}]
        for key, value in [('value', [float('nan'), 0]), ('modes_spent', 0), ('calls', 0)]:
            record = copy.deepcopy(self.normal)
            record[key] = value
            records.append(record)
        for modes in (True, 0, -1, 100000, 1.5):
            record = copy.deepcopy(self.normal)
            record['trace'][0]['modes'] = modes
            records.append(record)
        for record in records:
            self.assertFalse(final_guard(self.request, record)['accepted'])

    def test_wrong_request_type_fails_closed(self):
        for request in (None, {}, object()):
            self.assertFalse(final_guard(request, self.normal)['accepted'])

    def test_unsupported_request_fails_closed(self):
        for request in (Request(-1, 1, .001), Request(0, 0, .001), Request(0, 1, float('nan'))):
            self.assertFalse(final_guard(request, self.normal)['accepted'])

    def test_declared_boundary_not_a_value_integrity_verifier(self):
        # Resolution guard intentionally cannot certify a corrupted solver value.
        self.normal['value'] = [0, 0]
        self.assertTrue(final_guard(self.request, self.normal)['accepted'])

    def test_declared_boundary_forged_accounting_is_not_authenticated(self):
        # Guard is not an execution attestation service; trusted journal required.
        self.stale['trace'][0]['modes'] = 608
        self.stale['modes_spent'] = 608
        self.assertTrue(final_guard(self.request, self.stale)['accepted'])
