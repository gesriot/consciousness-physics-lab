import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from lab import frozen_pipeline as p, extrapolation as e, matched_mixture as m
from lab import radial_collocation
from lab.executor import request_hash
from lab.guard import Request, final_guard


def request_from(row):
    q = row['request']
    return Request(complex(*q['z']),q['mean_time'],q['tolerance'],q['budget'],q['max_calls'])


class FrozenPipelineTests(unittest.TestCase):
    def test_registered_configuration_and_unchanged_models(self):
        self.assertEqual(p.CONFIG.grids,(64,128,256,512,1024,2048,4096))
        self.assertEqual(p.CONFIG.candidate,'C1')
        self.assertEqual(p.CONFIG.third_degrees,(64,128))
        for model in e.MODELS:
            for kind, solver in (('modal',e.modal_model),('fv',e.graded_model)):
                self.assertEqual(p.model_response(kind,1+2j,1.,512,model),solver(1+2j,1.,512,model))
        self.assertEqual(p.model_response('modal',1+2j,1.,512,'mixture'),
                         m.model_response('modal',1+2j,1.,512,'mixture'))

    def test_archive_e011_c1_paths_gates_and_resolution(self):
        data = json.loads(Path('results/E011-extrapolation/results.json').read_text())
        tasks = {r['task_id']: r for r in data['tasks']}
        with gzip.open('results/E014-convergence-candidates/first/cost-replay.json.gz','rt') as handle:
            selected = json.load(handle)
        expected = {(r['task_id'],r['fault']):r['candidates']['C1']
                    for r in selected['rows'] if r['experiment']=='E011'}
        self.assertEqual(len(data['runs']),1536)
        for row in data['runs']:
            task = tasks[row['task_id']]
            request = request_from(task)
            values = dict(zip(p.CONFIG.grids,[complex(*v) for v in task['shadow_values'][row['model']]]))
            actual = p.check_v2(request,complex(*row['outcome']['value']),row['model'],
                                solver=lambda z,t,n:values[n])
            baseline = expected[row['task_id'],row['fault']]
            for key in ('verdict','stop_reason','cells_solved'):
                self.assertEqual(actual[key],baseline[key])
            for new, old in zip(actual['steps'],baseline['steps']):
                for key in ('last_grid','converging','verdict','futile','blocked_rounding'):
                    self.assertEqual(new[key],old[key])
                for key in ('uncertainty_estimate','discrepancy'):
                    self.assertAlmostEqual(new[key],old[key],delta=1e-12+1e-12*abs(old[key]))
            self.assertEqual(final_guard(request,row['outcome'])['accepted'],row['resolution']['accepted'])
            checks = p.checks_from_probes(row['probes'],request_hash(request),row['original_digest'])
            self.assertEqual(checks,row['checks'])
            self.assertEqual(p.plan_modes(request),task['base_modes'])

    def test_archive_e013_third_and_model_check_decisions(self):
        data = json.loads(Path('results/E013-matched-mixture/results.json').read_text())
        tasks = {r['task_id']:r for r in data['intake_channel']}
        self.assertEqual(len(data['runs']),288)
        for row in data['runs']:
            request = request_from(tasks[row['task_id']])
            checks = p.checks_from_probes(row['probes'],request_hash(request),row['original_digest'])
            self.assertEqual(checks,row['checks'])
            # Replaying the stored numerical prefix isolates composition from BLAS drift.
            values = [complex(*v) for v in row['third']['values']]
            with patch.object(radial_collocation,'response',side_effect=values):
                actual = p.third_check(request.z,request.mean_time,complex(*row['outcome']['value']),request.tolerance)
            for key in ('verdict','converging','matrix_solves','degrees'):
                self.assertEqual(actual[key],row['third'][key])
            self.assertEqual(p.plan_modes(request),row['outcome']['trace'][-1]['modes'])

    def test_fresh_archive_probes_and_third(self):
        from scripts.audit_frozen_pipeline import audit
        result = audit()
        self.assertEqual(result['gate_requests'],864)
        self.assertEqual(result['third_rows'],288)
        self.assertEqual(result['mismatches'],[])

    def test_manifest_matches_delegated_probe_behavior(self):
        request = Request(1+2j,1.,.01)
        evidence = p.model_probes(request)
        for name,row in evidence['old']['probes'].items():
            self.assertEqual(row['normalization_accepted'],row['normalization_error']<=p.CONFIG.normalization_allowance)
            self.assertEqual(row['first_moment_accepted'],row['mean_relative_error']<=p.CONFIG.first_moment_allowance)
            self.assertEqual(row['h'],p.CONFIG.old_probe_step)
        for group,allowance in (('second_moment',p.CONFIG.second_moment_allowance),('high_frequency',p.CONFIG.high_frequency_allowance)):
            for row in evidence[group]['probes'].values():
                self.assertEqual(row['accepted'],row['relative_error']<=allowance)
        self.assertEqual(evidence['old']['finite_volume_cells'],2*p.CONFIG.old_probe_cells)
        self.assertEqual(evidence['high_frequency']['modal_terms'],p.CONFIG.probe_modes)
        for n in p.CONFIG.grids[3:]:
            values = [complex(1+4**(-k)) for k in range(4)]
            step = p.v2_step(values,n,1+0j,.01)
            self.assertEqual(step['uncertainty_estimate'],p.CONFIG.uncertainty_difference_factor*step['last_difference']+step['rounding'])
            self.assertEqual(step['converging'],step['last_difference']<=p.CONFIG.contraction*step['previous_difference'])
        third = p.third_check(request.z,request.mean_time,1+0j,request.tolerance)
        self.assertEqual(tuple(third['degrees']),p.CONFIG.third_degrees)
        self.assertEqual(third['converging'],third['difference']<=request.tolerance/p.CONFIG.third_contraction_divisor)

    def test_policy_difference_is_third_abstention_only(self):
        for numerical in ('accept','inconclusive','violation_detected'):
            for resolution in (True,False):
                for valid in (True,False):
                    for third in ('accept','inconclusive','violation_detected'):
                        result = p.combine(numerical,resolution,{'hash':valid},third)
                        self.assertFalse(result['strict']['accepted'] and not result['soft']['accepted'])
                        different = result['strict'] != result['soft']
                        self.assertEqual(different,numerical=='accept' and resolution and valid and third=='inconclusive')
        result = p.combine('inconclusive',False,{'hash':True},'violation_detected')
        self.assertEqual(result['strict']['verdict'],'violation_detected')

    def test_rounding_diagnostics_do_not_change_c1(self):
        step = p.v2_step([1+0j,1+1e-12,1+1.1e-12,1+1.18e-12],512,1+0j,.01)
        self.assertTrue(step['both_differences_within_rounding'])
        self.assertFalse(step['converging'])
        self.assertTrue(step['shadow_C2_converging'])
        self.assertEqual(step['verdict'],'inconclusive')
