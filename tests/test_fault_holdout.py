import unittest
from unittest.mock import patch
from lab.allocation import propose_modes
from lab.executor import execute
from lab.guard import Request
from lab.fault_holdout import inject, verify, FAULTS


class FaultHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.request=Request(1j,1,.001)
        self.receipt=execute(self.request,propose_modes(self.request,.5))

    def test_injections_do_not_mutate_receipt(self):
        before=self.receipt.value
        for fault in FAULTS:
            outcome,cost=inject(self.request,self.receipt,fault,.123)
            self.assertEqual(self.receipt.value,before)
            self.assertEqual(outcome['modes_spent'],self.receipt.modes)
            self.assertGreaterEqual(cost,0)
        self.assertEqual(inject(self.request,self.receipt,'stale_value',.123)[0]['value'],[.123,0])

    def test_phase_and_normalization_faults(self):
        conjugate=complex(*inject(self.request,self.receipt,'conjugate',0)[0]['value'])
        self.assertEqual(conjugate,self.receipt.value.conjugate())
        request=Request(0,1,.001)
        receipt=execute(request,100)
        normalized=complex(*inject(request,receipt,'renormalized',0)[0]['value'])
        self.assertEqual(normalized,1)

    def test_verification_does_not_read_oracle(self):
        with patch('lab.fault_holdout.analytic',side_effect=AssertionError('oracle leaked')):
            supplied,_=inject(self.request,self.receipt,'conjugate',0)
            self.assertFalse(verify(self.request,supplied,'D')['accepted'])

    def test_reject_unregistered_operations(self):
        with self.assertRaises(ValueError):inject(self.request,self.receipt,'invented',0)
        with self.assertRaises(ValueError):verify(self.request,{},'invented')
