import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from lab import extrapolation as e
from lab.executor import request_hash
from lab.guard import Request
from lab.surface_order import surface_response


class ExtrapolationTests(unittest.TestCase):
    def test_clean_and_scale_mesh_identity(self):
        for z in (0, 1+2j, 1e4):
            self.assertEqual(e.graded_model(z, 1., 512, 'sphere'), surface_response(z, 1., 512))
            self.assertEqual(e.graded_model(z, 1., 512, 'model_scale'), surface_response(z, 1000., 512))

    def test_slab_against_closed_form_and_refinement(self):
        for z in (.1, 1+2j, 100):
            q = np.sqrt(3*complex(z))
            reference = np.tanh(q)/q
            coarse = abs(e.graded_model(z, 1., 1024, 'slab') - reference)
            fine = abs(e.graded_model(z, 1., 2048, 'slab') - reference)
            self.assertLess(fine, .3*coarse)
            self.assertLess(abs(e.modal_model(z, 1., 4096, 'slab') - reference), 6e-5)

    def test_high_precision_and_reference_fallback(self):
        request = Request(1+2j, 1., .01)
        reference = complex(*map(float, e.high_precision(request)))
        self.assertLess(abs(reference - complex(e.analytic(request.z, 1.))), 1e-14)
        intake = [dict(task_id=0, cohort='A')]
        self.assertFalse(e.reference_audit(intake, [request])['fallback_all_AB'])
        with patch.object(e, 'analytic', return_value=2+0j):
            self.assertTrue(e.reference_audit(intake, [request])['fallback_all_AB'])

    def test_three_way_combination_distinguishes_model_and_resolution(self):
        self.assertEqual(e.combine_verdict('accept', True, {'hash': True}), ('accept', 'all_gates_pass'))
        self.assertEqual(e.combine_verdict('accept', False, {'hash': True}), ('inconclusive', 'insufficient_resolution'))
        self.assertEqual(e.combine_verdict('accept', True, {'hash': False}), ('violation_detected', 'model_or_request_violation'))
        self.assertEqual(e.combine_verdict('violation_detected', True, {'hash': True}), ('violation_detected', 'numerical_violation'))

    def test_probes_are_fresh_and_shared_hooks_apply(self):
        request = Request(1+1j, 1., .01)
        for model in e.MODELS:
            with patch.object(e, 'modal_model', wraps=e.modal_model) as modal, \
                 patch.object(e, 'graded_model', wraps=e.graded_model) as fv:
                result = e.invariant_probes(request, model)
            self.assertEqual((modal.call_count, fv.call_count), (5, 5))
            self.assertEqual(sum(g['modal_terms'] for g in result.values()), 20480)
            self.assertEqual(sum(g['finite_volume_cells'] for g in result.values()), 13312)
            self.assertEqual(all(p['accepted'] for p in result['high_frequency']['probes'].values()), model == 'sphere')

    def test_evaluator_cannot_consult_reference(self):
        request = Request(1+1j, 1., .01)
        receipt = e.execute(request, e.propose_modes(request, .5))
        outcome, _ = e.inject(request, receipt, 'clean', 0)
        with patch.object(e, 'analytic', side_effect=AssertionError('reference in decision')), \
             patch.object(e, 'high_precision', side_effect=AssertionError('reference in decision')):
            row = e.evaluate(request, request_hash(request), outcome, 'sphere')
        self.assertTrue(row['accepted'])
        self.assertGreaterEqual(row['independent']['steps'][0]['last_grid'], 512)
        self.assertNotIn('observed_order', row['independent']['steps'][0])

    def test_audit_files_closed_before_operational_execution(self):
        request = Request(1+1j, 1., .01)
        intake = [dict(task_id=0, cohort='A', request=e.encode(request), request_sha256=request_hash(request))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'results.json'
            def stop_before_execution(*args):
                self.assertTrue(output.with_name('results-intake.json').is_file())
                self.assertTrue(output.with_name('results-reference-audit.json').is_file())
                raise RuntimeError('intentional predecision stop')
            with patch.object(e, 'metadata', return_value={}), \
                 patch.object(e, 'make_intake', return_value=(intake, [request], None)), \
                 patch.object(e, 'execute', side_effect=stop_before_execution):
                with self.assertRaisesRegex(RuntimeError, 'intentional predecision stop'):
                    e.run(output)
