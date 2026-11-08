"""Independent audit checks on synthetic traces, never held-out outcomes."""
import copy
import unittest
import numpy as np
from attention import factorial, s3
from attention.audit_a004 import (ACTIONS, likelihoods, posterior_checks,
    intervention_checks, nested_close, forecast, summary_checks, SCHEMES,
    POLICIES, FRACTIONS, recompute_calibration)
from lab.guard import Request
from lab import frozen_pipeline as P


def fixture():
    model=s3.load_model()
    rows=[]
    for i in range(9):
        values=[(0 if a=='v2_512' else 2 if a.startswith('v2_') and int(a[3:])<512 else -1 if a.startswith('v2_') else 0) for a in ACTIONS]
        if i%3:values[ACTIONS.index('normalization' if i%3==1 else 'high_frequency')]=1
        rows.append(dict(group_index=0,accurate=i%3==0,full_verdict='accept' if i%3==0 else 'violation_detected',outcomes=values))
    model['groups']=[dict(id='synthetic:0',features=[0.]*6)]
    model['neighbors']=1;model['trajectories']=rows
    request=Request(1,1,.01)
    class Executor:
        def __init__(self,row):self.request=request;self.row=row
        def perform(self,action):return s3.observation(action,self.row['outcomes'][ACTIONS.index(action)])
    infos=[dict(prediction=dict(neighbor_ids=['synthetic:0']),outcome=dict(modes_spent=P.plan_modes(request))) for _ in rows[:3]]
    return model,rows,Executor,infos


class A004AuditTests(unittest.TestCase):
    def test_independent_emission_channel_and_equal_corruption(self):
        model,_,_,_=fixture()
        raw,base=likelihoods(model);_,changed=likelihoods(model,True)
        j=ACTIONS.index('normalization')
        self.assertEqual(base[0,j,:2].tolist(),[.99,.01])
        self.assertTrue(np.all(changed[base>0]>0))
        self.assertTrue(np.all(base[raw<0]==0))
        result=factorial.corruption_audit(model)
        self.assertEqual(intervention_checks(result,model),[])
        result['D_L1']+=1
        self.assertTrue(intervention_checks(result,model))

    def test_all_factorial_variants_pass_independent_evidence_filter(self):
        model,rows,Executor,infos=fixture()
        for policy in SCHEMES:
            run=factorial.run_batch([Executor(r) for r in rows[:3]],policy,12.,model)
            self.assertEqual(posterior_checks(run,infos,model,policy),[],policy)

    def test_forecast_and_fallback_tampering_detected(self):
        model,rows,Executor,infos=fixture()
        run=factorial.run_batch([Executor(r) for r in rows[:3]],'U1O',12.,model)
        wrong=copy.deepcopy(run)
        wrong['predictions'][3]['full_probabilities']['accept']+=.01
        self.assertTrue(posterior_checks(wrong,infos,model,'U1O'))
        wrong=copy.deepcopy(run);wrong['predictions'][3]['prior_reset']=True
        self.assertIn('prior fallback flag',posterior_checks(wrong,infos,model,'U1O'))
        wrong=copy.deepcopy(run);wrong['support_audit']['prior_resets']+=1
        self.assertTrue(posterior_checks(wrong,infos,model,'U1O'))

    def test_hybrid_skip_and_decision_probability_tampering_detected(self):
        model,rows,Executor,infos=fixture()
        run=factorial.run_batch([Executor(r) for r in rows[:3]],'H',12.,model)
        self.assertTrue(run['skipped'])
        wrong=copy.deepcopy(run);wrong['skipped'][0]['decisive_probability']+=.1
        self.assertIn('hybrid skipped audit',posterior_checks(wrong,infos,model,'H'))

    def test_missing_category_is_not_observation(self):
        model,_,_,_=fixture();raw,matrix=likelihoods(model)
        row=forecast(np.ones(9)/9,matrix,'v2_1024')
        self.assertEqual(row['eligible_mass'],0)
        self.assertTrue(row['missing_prediction_extension'])
        self.assertEqual(row['probabilities'],[.25]*4)

    def test_probability_tolerance_does_not_hide_structure(self):
        self.assertTrue(nested_close({'p':[.5]}, {'p':[.5+1e-14]}))
        self.assertFalse(nested_close({'p':[.5]}, {'p':[.5,.5]}))
        self.assertFalse(nested_close({'p':[.5]}, {'q':[.5]}))

    def test_summary_negative_control_and_stop_sensitivity(self):
        from attention.a004 import summarize, stopping, calibration_summary
        packages=[]
        for i in range(2):
            policies={}
            for f in FRACTIONS:
                policies[f]={}
                for p in POLICIES:
                    count=1+int(p=='U1O' and f=='0.2')+int(p=='U1T' and f=='0.4')
                    policies[f][p]=dict(correct=count,false_accepts=0,accepts=count,correct_accepts=count,correct_detections=0,
                        spent=1.,actions=[],predictions=[])
            packages.append(dict(package_id=i,inputs=[],policies=policies,oracle={f:dict(correct=3) for f in FRACTIONS}))
        data=dict(packages=packages,summary=summarize(packages),calibration=calibration_summary(packages))
        data['stopping_rule']=stopping(data['summary'])
        totals={f:{p:{k:v for k,v in row.items() if k!='oracle_fraction'} for p,row in data['summary'][f]['policies'].items()} for f in FRACTIONS}
        ceilings={f:[3,3] for f in FRACTIONS};changed={f:{p:0 for p in ('D','Dprime','HD','HDprime')} for f in FRACTIONS}
        self.assertEqual(summary_checks(data,totals,ceilings,changed),[])
        self.assertTrue(data['stopping_rule']['close_budget_series'])
        self.assertEqual(data['stopping_rule']['variants_beating_C2_combined'],['U1O','U1T'])
        wrong=copy.deepcopy(data);wrong['summary']['0.2']['policies']['H']['correct']+=1
        self.assertTrue(summary_checks(wrong,totals,ceilings,changed))
        wrong=copy.deepcopy(data);wrong['stopping_rule']['close_budget_series']=False
        self.assertTrue(summary_checks(wrong,totals,ceilings,changed))


if __name__=='__main__':unittest.main()
