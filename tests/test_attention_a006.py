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
                self.assert_diagnostics_close(actual,row['clean_result'],row['family'])
                self.assertLessEqual(actual['cost'],row['maximum_cost'])

    def assert_diagnostics_close(self,actual,expected,family=None,atol=1e-11):
        from attention.audit_a006_portability import diagnostic_atol
        if isinstance(expected,dict):
            self.assertEqual(set(actual),set(expected))
            for key in expected:self.assert_diagnostics_close(actual[key],expected[key],family,diagnostic_atol(family,key,expected))
        elif isinstance(expected,list):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):self.assert_diagnostics_close(a,b,family,atol)
        elif isinstance(expected,float):
            self.assertTrue(math.isclose(actual,expected,rel_tol=1e-9,abs_tol=atol),(actual,expected))
        else:self.assertEqual(actual,expected)

    def test_conditioned_diagnostics_keep_verdicts_exact(self):
        row=copy.deepcopy(self.library['samples'][80])
        changed=copy.deepcopy(row['clean_result'])
        changed['original']['probes']['fv']['error']=0.00022717023440055328
        self.assert_diagnostics_close(changed,row['clean_result'],row['family'])
        changed['verdict']='inconclusive'
        with self.assertRaises(AssertionError):
            self.assert_diagnostics_close(changed,row['clean_result'],row['family'])
        row=copy.deepcopy(self.library['samples'][108])
        changed=copy.deepcopy(row['clean_result'])
        changed['original']['step']['antecedent_ratio']=0.249999450922046
        self.assert_diagnostics_close(changed,row['clean_result'],row['family'])
        changed['original']['step']['antecedent_ratio']=.3
        with self.assertRaises(AssertionError):
            self.assert_diagnostics_close(changed,row['clean_result'],row['family'])

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

    def test_selected_root_fields_match_full_reader(self):
        import tempfile
        from pathlib import Path
        from attention.a006 import read_root_fields
        from attention.io import write_json
        for large_name in ('runs','packages'):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'fixture.json.gz'
                value={'experiment':'A006','intake_channel':[{'a':1}],large_name:[{'nested':{'a':[1,2,3]}}]*100,'tail':{'last':True}}
                write_json(path,value)
                self.assertEqual(read_root_fields(path,('experiment','intake_channel','tail')),
                    {k:value[k] for k in ('experiment','intake_channel','tail')})

    def test_selective_reader_rejects_duplicate_and_truncated_roots(self):
        import tempfile
        from pathlib import Path
        from attention.a006 import read_root_fields
        examples=['{\n  "x": 1,\n  "x": 2\n}\n','{\n  "x": 1,\n  "packages": [\n    {\n      "a": 2\n',
            '{\n  "x": 1,\n}\n','{\n  "x": 1\n}\ntrailing']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for raw in examples:
                path.write_text(raw)
                with self.assertRaises(ValueError):read_root_fields(path,('x',))

    def test_streamed_source_hash_preserves_historical_semantics(self):
        import tempfile,hashlib
        from pathlib import Path
        from attention.a006 import source_sha256
        from attention.io import source_bytes,write_json
        with tempfile.TemporaryDirectory() as directory:
            compressed=Path(directory)/'fixture.json.gz';historical=Path(directory)/'fixture.json'
            write_json(compressed,{'x':list(range(10000))})
            for path in (compressed,historical):
                self.assertEqual(source_sha256(path),hashlib.sha256(source_bytes(path)).hexdigest())
                self.assertEqual(source_sha256(path,False),hashlib.sha256(compressed.read_bytes()).hexdigest())

    def test_selective_replay_matches_full_loader(self):
        import tempfile,copy
        from pathlib import Path
        from attention.a006 import load_a006_replay
        from attention.io import write_json,load_replay
        from lab.executor import request_hash
        from lab.common_mode import encode
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.json.gz';request=Request(1,1,.01)
            records=[dict(task_id=i,request=encode(request),request_sha256=request_hash(request)) for i in range(60)]
            value=dict(experiment='A006',intake_channel=records,global_permutations=[],intake_sha256='fixture',packages=[{'large':[1]*10000}])
            write_json(path,value)
            old,old_requests,old_provenance=load_replay(path,'A006',60)
            new,new_requests,new_provenance=load_a006_replay(path)
            self.assertEqual(new,{k:old[k] for k in new})
            self.assertEqual(new_requests,old_requests);self.assertEqual(new_provenance,old_provenance)
            value['intake_channel'][0]['request_sha256']='wrong';write_json(path,value)
            with self.assertRaises(ValueError):load_a006_replay(path)
