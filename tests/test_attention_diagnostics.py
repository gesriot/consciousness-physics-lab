import copy
import tempfile
import unittest
from pathlib import Path
from attention import a001, a002, a003, a004
from attention import compare_a004, compare_a005, compare_a006, compare_a007
from attention.io import write_json
from attention.audit_a006_portability import diagnostic_atol


class DiagnosticComparisonTests(unittest.TestCase):
    def fixture(self):
        fields=('experiment','intake_channel','global_permutations','intake_sha256',
                'self_model_sha256','onset_package','calibration','summary',
                'full_pipeline_parity_failures','request_overlap','sample_library_sha256',
                'sample_library_sources','sample_library_counts','intervention_audit')
        data=dict.fromkeys(fields,[])
        data['packages']=[dict(package_id=8,step=dict(shadow_C2_converging=True,
            converging=True,verdict='accept',terminal=True,cost=1.,budget=2.))]
        return data

    def test_all_stream_comparators_separate_shadow_and_keep_working_fields_exact(self):
        for module in (compare_a004,compare_a005,compare_a006,compare_a007):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                a,b=Path(directory)/'a.json',Path(directory)/'b.json'
                original=self.fixture();changed=copy.deepcopy(original)
                changed['packages'][0]['step']['shadow_C2_converging']=False
                write_json(a,original);write_json(b,changed)
                result=module.compare_paths(a,b)
                self.assertTrue(result['decisions_pass'])
                self.assertEqual(len(result['shadow_diagnostics']),1)
                self.assertEqual(result['shadow_diagnostics'][0]['path'],
                    '/packages/0/step/shadow_C2_converging')
                for key,value in [('converging',False),('verdict','inconclusive'),
                                  ('terminal',False),('cost',1.00000000001),('budget',3.)]:
                    mutation=copy.deepcopy(changed);mutation['packages'][0]['step'][key]=value
                    write_json(b,mutation)
                    self.assertFalse(module.compare_paths(a,b)['decisions_pass'],key)

    def test_early_decision_views(self):
        original=self.fixture();changed=copy.deepcopy(original)
        changed['packages'][0]['step']['shadow_C2_converging']=False
        for module in (a002,a003,a004):
            result=module.compare(original,changed)
            self.assertTrue(result['decisions_pass'])
            self.assertEqual(len(result['shadow_diagnostics']),1)
        self.assertEqual(a001.control_flags(original),a001.control_flags(changed))

    def test_owner_sample82_and_negative_controls(self):
        from attention.io import read_json
        row=read_json('attention/sample_library_a006.json')['samples'][82]
        probe=row['clean_result']['original']['probes']['fv']
        allowance=diagnostic_atol('second_moment','error',probe)
        self.assertLess(abs(.00021622200640969425-probe['error']),allowance)
        self.assertGreater(.001,allowance)
        # Conditioning is family-specific, not one global relaxed tolerance.
        self.assertGreater(allowance,diagnostic_atol('normalization','error',probe))
        self.assertGreater(diagnostic_atol('first_moment','error',probe),allowance)

    def test_allowances_follow_probe_and_grid_conditioning(self):
        import numpy as np
        from attention.audit_a006_portability import endpoint_allowance
        probe=dict(value=[1.,0.])
        endpoint=endpoint_allowance('second_moment',probe)
        self.assertAlmostEqual(diagnostic_atol('second_moment','error',probe),
                               4*endpoint/(.01**2*(20/7)))
        self.assertAlmostEqual(diagnostic_atol('high_frequency','error',probe),
                               100*endpoint/np.sqrt(3/5))
        self.assertEqual(endpoint_allowance('second_moment',dict(value=[10.,0.])),10*endpoint)
        step=dict(last_grid=512,antecedent_ratio=.25,previous_difference=1e-6)
        error=2*endpoint_allowance('v2',step)
        self.assertEqual(diagnostic_atol('v2','antecedent_ratio',step),
                         error*1.25/(4e-6-error))
        self.assertEqual(diagnostic_atol('v2','uncertainty_estimate',step),2*error)
