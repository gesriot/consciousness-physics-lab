import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from lab import pipeline_holdout as study
from lab.guard import Request


def canned_record(task_id=0, cohort='W', ratio=(1, 2)):
    return dict(task_id=task_id, cohort=cohort,
                magnitude_u_ratio=list(ratio), log_magnitude_interval=list(study.LOG_RANGES[cohort]))


class HoldoutTests(unittest.TestCase):
    def test_exact_coordinate_selection_boundary(self):
        at = canned_record(49, 'H', (1, 3))
        self.assertEqual(study.exact_log_coordinate(at), 8)
        self.assertFalse(study.audit_selected(at))
        above = canned_record(49, 'H', (2**53 + 1, 3 * 2**53))
        self.assertTrue(study.audit_selected(above))
        self.assertTrue(study.audit_selected(canned_record(50, 'H', (0, 1))))

    def test_fallback_replaces_every_reference(self):
        records = [canned_record(0), canned_record(1)]
        requests = [Request(1, 1, .01), Request(2, 1, .01)]
        with patch.object(study, 'analytic', return_value=99j), \
             patch.object(study, 'high_precision', side_effect=[['.5', '0'], ['.25', '0']]):
            refs, audit = study.reference_audit(records, requests)
        self.assertEqual(refs, [.5+0j, .25+0j])
        self.assertTrue(audit['fallback_all'])
        self.assertEqual(audit['initial_audited_tasks'], 1)
        self.assertEqual(audit['audited_tasks'], 2)
        self.assertEqual(audit['failed_tasks'], 2)

    def test_success_keeps_analytic_values(self):
        with patch.object(study, 'analytic', return_value=.5+0j), \
             patch.object(study, 'high_precision', return_value=['.5', '0']):
            refs, audit = study.reference_audit([canned_record()], [Request(1, 1, .01)])
        self.assertEqual(refs, [.5+0j])
        self.assertFalse(audit['fallback_all'])

    def test_replay_never_calls_rng(self):
        record = canned_record()
        record.update(condition_order=list(reversed(study.CONDITIONS)), request={'budget': 4096})
        with patch.object(study, 'load_replay', return_value=({'intake_channel':[record]}, ['request'], {'saved': True})), \
             patch.object(study.np.random, 'default_rng', side_effect=AssertionError('must not regenerate')):
            records, requests, provenance = study.make_intake('saved.json')
        self.assertEqual(records[0]['condition_order'], list(reversed(study.CONDITIONS)))
        self.assertEqual(requests, ['request'])
        self.assertTrue(provenance['saved'])
        records[0]['condition_order'][0] = 'bad'
        self.assertNotEqual(records, [record])
        with self.assertRaises(ValueError):
            study.validate_intake(records)

    def test_coverage_distinguishes_model_bias(self):
        step = dict(last_grid=512, uncertainty_estimate=.01, verdict='accept',
                    both_differences_within_rounding=True, converging=True, shadow_C2_converging=False)
        decision = dict(independent=dict(grids=[512], values=[[.5, 0]], steps=[step]),
            third=dict(values=[[1,0], [1,0]], uncertainty=.01, converging=True, verdict='accept'))
        coverage = study.coverage_evidence(decision, 1+0j, 'mixture')
        self.assertFalse(coverage['V2'][0]['covered'])
        self.assertEqual(coverage['V2'][0]['interpretation'], 'model_discrepancy_against_original_sphere')
        self.assertTrue(coverage['V2'][0]['C1_C2_disagreement'])
        self.assertTrue(coverage['third']['covered'])

    def test_mixture_counts_component_solves(self):
        decision = dict(independent=dict(cells_solved=960), probes={'old':dict(modal_terms=8192, finite_volume_cells=1024)},
                        third=dict(matrix_solves=2, degrees=[64,128]))
        work = study.component_work(decision, 'mixture')
        self.assertEqual(work['verification_fv_cells'], 1920)
        self.assertEqual(work['probe_modal_terms'], 16384)
        self.assertEqual(work['third_matrix_solves'], 2)

    def test_scientific_comparison_keeps_flags_and_diagnostics(self):
        a = dict(environment={'platform':'a'}, replay={'file':'x'},
                 runs=[dict(seconds=1., flag=True, error=.2, both_differences_within_rounding=False)])
        b = copy.deepcopy(a)
        b['environment'] = {'platform':'b'}
        b['runs'][0]['seconds'] = 9.
        self.assertEqual(study.scientific_view(a), study.scientific_view(b))
        b['runs'][0]['both_differences_within_rounding'] = True
        self.assertNotEqual(study.scientific_view(a), study.scientific_view(b))

    def test_summarize_paired_unique_denominators(self):
        numerical = dict(verdict='accept')
        decision = dict(independent=numerical, third=dict(verdict='inconclusive'),
                        resolution=dict(accepted=True), checks={'hash': True},
                        policies={'strict':dict(verdict='inconclusive'), 'soft':dict(verdict='accept')})
        step = dict(grid=512, decisive=True, covered=True, intrinsic_numerical_covered=True, model='sphere',
                    both_differences_within_rounding=True, C1_C2_disagreement=False)
        row = dict(task_id=0, condition='clean', model='sphere', accurate=True,
                   decision=decision, coverage={'V2':[step], 'third':{'decisive':False}}, work={'cells':960})
        duplicate = copy.deepcopy(row)
        duplicate['condition'] = 'conjugate'
        summary = study.summarize([row, duplicate])
        self.assertEqual(summary['coverage']['V2_decisive_steps'], 2)
        self.assertEqual(summary['coverage']['unique_task_model_grid_decisive_steps'], 1)
        self.assertEqual(summary['strictness_cost'], 1)
        self.assertEqual(summary['policies']['strict']['abstention_reasons']['third'], 2)


if __name__ == '__main__':
    unittest.main()
