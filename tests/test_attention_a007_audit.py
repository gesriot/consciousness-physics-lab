import copy
import json
import tempfile
import unittest
from pathlib import Path
from attention.audit_a005 import Audit
from attention.audit_a007 import audit_run, verify_hook, verify_guard
from attention.compare_a007 import compare_paths, decision_view
from attention import agent, s3
from attention.channel_monitor import run_batch
from attention.io import read_json
from lab.guard import Request
from lab.executor import request_hash
from lab.scientist import ROOT, analytic


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=s3.load_model();cls.weights=cls.model['action_model']['cost_weights']
        cls.library=read_json(ROOT/'attention/sample_library_a007.json')
        cls.r=Request(1,1,.01)
        cls.spec=dict(kind='none',target_channel='collocation:0',onset_action=20,circle_phase=4,random_seed=12)
        e=agent.Executor(cls.r,request_hash(cls.r),'clean',1+0j)
        e.perform('modal');value=e.outcome['value'];reference=complex(analytic(cls.r.z,cls.r.mean_time))
        error=abs(complex(*value)-reference)
        cls.infos=[dict(task_id=0,outcome=e.outcome,reference=[reference.real,reference.imag],absolute_error=error,accurate=error<=cls.r.tolerance)]
        from lab.common_mode import encode
        cls.records=[dict(task_id=0,request=encode(cls.r))]
        cls.runs={}
        for policy in ('BA','BU','C2A','C2R','C2','D','Dprime'):
            run=run_batch([agent.Executor(cls.r,request_hash(cls.r),'clean',1+0j)],policy,100.,cls.model,cls.library,0,cls.spec,.4)
            for item in run['inputs']:
                accurate=cls.infos[0]['accurate'];item['correct']=(accurate and item['verdict']=='accept') or (not accurate and item['verdict']=='violation_detected')
                item['false_accept']=not accurate and item['verdict']=='accept'
            cls.runs[policy]=run

    def check_run(self,run):
        check=Audit();audit_run(check,run,self.infos,self.records,self.spec,'0.4',self.weights,self.library,'development smoke')
        return check

    def test_all_policy_development_smokes(self):
        for policy,run in self.runs.items():
            with self.subTest(policy=policy):self.assertEqual(self.check_run(run).errors,[])

    def test_accounting_mutation_detected(self):
        run=copy.deepcopy(self.runs['BA']);run['actions'][0]['cost']+=1
        self.assertTrue(self.check_run(run).errors)

    def test_verdict_mutation_detected(self):
        run=copy.deepcopy(self.runs['BA']);item=run['inputs'][0]
        item['verdict']='inconclusive' if item['verdict']=='accept' else 'accept'
        self.assertTrue(self.check_run(run).errors)

    def test_detection_mutation_detected(self):
        run=copy.deepcopy(self.runs['BA']);run['detected']=True
        self.assertTrue(self.check_run(run).errors)

    def test_raw_risk_mutation_detected(self):
        run=copy.deepcopy(self.runs['BA'])
        row=next(a for a in run['actions'] if a.get('score_details'))
        row['score_details']['raw']['binding']+=1.
        self.assertTrue(self.check_run(run).errors)

    def test_sample_timing_mutation_detected(self):
        run=copy.deepcopy(self.runs['BA'])
        next(a for a in run['actions'] if a['sample'])['sample_reason']='exhausted'
        self.assertTrue(self.check_run(run).errors)

    def test_noisy_learned_risk_reconstruction(self):
        run=run_batch([agent.Executor(self.r,request_hash(self.r),'clean',1+0j)],'BA',100.,self.model,self.library,0,self.spec,.4,noise=True)
        for i,original in zip(run['inputs'],self.runs['BA']['inputs']):
            i['correct']=original['correct'];i['false_accept']=original['false_accept']
        check=Audit();audit_run(check,run,self.infos,self.records,self.spec,'0.4',self.weights,self.library,'noisy development',noise=True)
        self.assertEqual(check.errors,[])

    def test_local_hook_rejects_unrelated_change(self):
        check=Audit();verify_hook(check,'binding',{'verdict':'accept'},{'verdict':'violation_detected'},.01,False,'permissive','synthetic')
        self.assertTrue(check.errors)

    def test_comparator_decisions_and_prices_exact(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.json';b=Path(td)/'b.json'
            base=dict(experiment='A007',intake_sha256='a',summary={},packages=[dict(cost=1.,verdict='accept',diagnostic=.5)])
            a.write_text(json.dumps(base,indent=2)+'\n')
            changed=copy.deepcopy(base);changed['packages'][0]['diagnostic']+=1e-12
            b.write_text(json.dumps(changed,indent=2)+'\n');self.assertTrue(compare_paths(a,b)['decisions_pass'])
            changed['packages'][0]['cost']+=1e-12
            b.write_text(json.dumps(changed,indent=2)+'\n');self.assertFalse(compare_paths(a,b)['decisions_pass'])
            changed=copy.deepcopy(base);changed['packages'][0]['verdict']='inconclusive'
            b.write_text(json.dumps(changed,indent=2)+'\n');self.assertFalse(compare_paths(a,b)['decisions_pass'])

    def test_comparator_rejects_root_decision_difference(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[Path(td)/x for x in ('a.json','b.json')]
            for path,n in zip(paths,(0,1)):path.write_text(json.dumps(dict(experiment='A007',summary={'false_accepts':n},packages=[]),indent=2)+'\n')
            self.assertFalse(compare_paths(*paths)['decisions_pass'])
