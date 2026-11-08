import copy
import math
import unittest
from unittest.mock import patch
from attention import active_monitor as active, calibration_samples as samples, agent, s3
from attention.io import read_json
from lab.guard import Request
from lab.executor import request_hash
from lab.scientist import ROOT


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=read_json(ROOT/'attention/sample_library_a006.json')
        cls.weights=s3.load_model()['action_model']['cost_weights']

    def test_all_check_families_and_models(self):
        self.assertEqual(set(self.library['counts']),{'modal','binding','normalization','first_moment','second_moment','high_frequency','collocation','v2'})
        for family in samples.FAMILIES:
            rows=[s for s in self.library['samples'] if s['family']==family]
            self.assertTrue(any(s['expected']=='accept' for s in rows))
            self.assertTrue(any(s['expected']!='accept' for s in rows))
            self.assertTrue({'sphere','slab','mixture','model_scale'}<={s['model'] for s in rows})

    def test_all_frozen_samples_execute_clean(self):
        for row in self.library['samples']:
            with self.subTest(sample=row['sample_id']):
                actual=samples.execute_sample(row,self.weights)
                self.assertEqual(actual['verdict'],row['expected'])
                self.assert_diagnostics_close(actual,row['clean_result'])
                self.assertLessEqual(actual['cost'],row['maximum_cost'])

    def assert_diagnostics_close(self,actual,expected):
        if isinstance(expected,dict):
            self.assertEqual(set(actual),set(expected))
            for key in expected:self.assert_diagnostics_close(actual[key],expected[key])
        elif isinstance(expected,list):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):self.assert_diagnostics_close(a,b)
        elif isinstance(expected,float):
            self.assertTrue(math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-11),(actual,expected))
        else:self.assertEqual(actual,expected)

    def test_selection_has_no_fault_feedback(self):
        import inspect
        from attention import build_a006_library
        source=inspect.getsource(build_a006_library.build)
        self.assertIn('execute_sample(candidate,weights)',source)
        self.assertNotIn('underestimated',source)

    def test_frozen_library_digest(self):
        from lab.replay import canonical_sha256
        self.assertEqual(canonical_sha256(self.library['samples']),self.library['samples_sha256'])

    def test_expected_intervention_volume(self):
        d=active.State(1.,'D',self.library);dp=active.State(1.,'Dprime',self.library);ba=active.State(1.,'BA',self.library)
        self.assertEqual(d.expectations,dp.unused_expectations)
        self.assertEqual(ba.expectations,dp.expectations)
        self.assertEqual(ba.unused_expectations,d.unused_expectations)
        self.assertNotEqual(d.expectations,ba.expectations)

    def test_persistent_alternation(self):
        state=active.State(1.,'BA',self.library)
        for family in samples.FAMILIES:
            for counter in range(12):
                row=state.next_sample(family)
                self.assertEqual(row['expected']=='accept',counter%2==0)
                state.observe_sample(row,dict(verdict=row['expected']),dict(package_id=counter//3))
        self.assertFalse(state.alerts)
        self.assertEqual(state.total_samples,36)

    def test_reject_sample_detects_fail_open(self):
        for family,fault in [('collocation','permissive'),('second_moment','uninformative')]:
            state=active.State(1.,'BA',self.library);state.counts[family]=1
            row=state.next_sample(family);result=samples.execute_sample(row,self.weights,fault)
            state.observe_sample(row,result,{})
            self.assertEqual(state.suspect,{family})

    def test_d_clean_false_alarm_is_preserved(self):
        state=active.State(1.,'D',self.library);state.counts['second_moment']=1
        row=state.next_sample('second_moment');result=samples.execute_sample(row,self.weights)
        state.observe_sample(row,result,{})
        self.assertEqual(state.suspect,{'second_moment'})

    def test_v2_hook_reclassifies_without_changing_truth(self):
        original=dict(verdict='inconclusive',terminal=False,value=[1.,0.],step=dict(last_grid=512,
            uncertainty_estimate=.2,converging=True,discrepancy=.08,verdict='inconclusive',futile=False,
            last_difference=.1,previous_difference=.4,antecedent_ratio=.25))
        saved=copy.deepcopy(original);actual=samples.spoil_v2(original,.1)
        self.assertEqual(actual['step']['uncertainty_estimate'],.002)
        self.assertEqual(actual['verdict'],'accept');self.assertTrue(actual['terminal'])
        self.assertEqual(actual['value'],original['value']);self.assertEqual(original,saved)

    def test_v2_decisive_monotonicity(self):
        for discrepancy,expected in [(0.,'accept'),(2.,'violation_detected')]:
            original=dict(verdict=expected,terminal=True,value=[0.,0.],step=dict(last_grid=512,
                uncertainty_estimate=.01,converging=True,discrepancy=discrepancy,verdict=expected,futile=False,
                last_difference=.1,previous_difference=.4,antecedent_ratio=.25))
            self.assertEqual(samples.spoil_v2(original,.1)['verdict'],expected)

    def test_zero_budget_and_equal_quota(self):
        model=s3.load_model();r=Request(1,1,.01)
        for policy in active.POLICIES:
            state=active.State(1.,policy,self.library)
            result=active.run_batch([agent.Executor(r,request_hash(r),'clean',1+0j)],policy,0.,model,state,0,'none',4,.2)
            self.assertEqual(result['spent'],0.)
            self.assertEqual(result['sample_allocation']['executed'],0)

    def test_sample_expectation_not_used_for_execution(self):
        row=copy.deepcopy(self.library['samples'][0]);expected=samples.execute_sample(row,self.weights)
        row['expected']='not-a-verdict';row['clean_result']={}
        self.assertEqual(samples.execute_sample(row,self.weights),expected)

    def test_runtime_override_flag(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-m','attention.a006','--help'],capture_output=True,text=True,check=True)
        self.assertIn('--runtime-override',result.stdout)
        self.assertIn('--last-bit-noise',result.stdout)

    def test_stream_writer_matches_canonical_gzip(self):
        import tempfile
        from pathlib import Path
        from attention.a006 import write_result
        from attention.io import write_json
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.json.gz',Path(directory)/'b.json.gz'
            value={'example':[1,2.,True,'calibration']}
            write_result(a,value);write_json(b,value)
            self.assertEqual(a.read_bytes(),b.read_bytes())
