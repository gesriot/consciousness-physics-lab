"""Independent safety and audit negative controls using constructed data only."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from attention import audit_a005 as audit
from lab.guard import Request
from lab.replay import canonical_sha256


class TinyRuntime:
    def __init__(self,model):
        self.groups=np.array([0]);self.raw=np.zeros((1,len(audit.ORDER)),dtype=int)
    def neighbors(self,request):return [0],[]


def fixture():
    model={'action_model':{'cost_weights':{'fv_cell':1,'modal_term':1,'binding':1,'collocation_degree_cubed':1}}}
    records=[dict(task_id=i,request=dict(z=[1.,0.],mean_time=1.,tolerance=.01,budget=4096,max_calls=3)) for i in range(60)]
    packages=[]
    empty_input=dict(verdict='inconclusive',observed={},replacements={},spent=0.,already_closed=False,correct=False,false_accept=False)
    empty_run=dict(policy='',budget=0.,spent=0.,inputs=[dict(input_index=i,**copy.deepcopy(empty_input)) for i in range(45)],actions=[],alerts=[],suspect=[],monitor_history=[],touched_inputs=0,target_exposures=0,replacement_cost=0.,correct=0,false_accepts=0)
    for k in range(12):
        infos=[dict(task_id=k*5+i//9,full={'actions':[]},outcome={'value':[1.,0.]},reference=[1.,0.],absolute_error=0.,tolerance=.01,accurate=True) for i in range(45)]
        conditions={fault:{f:{p:dict(copy.deepcopy(empty_run),policy=p) for p in audit.POLICIES} for f in audit.FRACTIONS} for fault in audit.FAULTS}
        packages.append(dict(package_id=k,inputs=infos,reference_cost=0.,conditions=conditions))
    summary={}
    for fault in audit.FAULTS:
        summary[fault]={}
        for f in audit.FRACTIONS:
            summary[fault][f]={}
            for p in audit.POLICIES:
                summary[fault][f][p]=dict(correct=0,correct_by_package=[0]*12,false_accepts_before=0,false_accepts_after=0,false_accepts_by_package=[0]*12,alerts=0,packages_with_new_alert=0,preonset_target_alert=False,detected=False if fault!='none' else None,delay=None,censored=True if fault!='none' else None,detected_within_90=False if fault!='none' else None,spent=0.,replacement_cost=0.,followup={'inputs':0,'actions':0,'target_observations':0})
    delta=60*(abs(.999-(.98+.02/3))+abs(.999-(.98+.02/2)))
    data=dict(experiment='A005',intake_channel=records,packages=packages,onset_package=int(np.random.default_rng(20261144).integers(4,9)),global_permutations=[],intake_sha256=canonical_sha256([records,[]]),full_pipeline_parity_failures=[],self_model_sha256=canonical_sha256(model),calibration={'threshold':2.,'maximum':1.},summary=summary,intervention_audit={'D_changed_numbers':120,'Dprime_changed_numbers':120,'D_L1':delta,'Dprime_L1':delta})
    return data,model


class IndependentAuditTests(unittest.TestCase):
    def test_accept_requires_base_and_extra(self):
        self.assertEqual(audit.verdict({}, {'collocation':{'verdict':'accept'}}, {'collocation'}),'inconclusive')
        evidence={a:{'verdict':'accept'} for a in ('modal','binding')+audit.FEATURES+('collocation',)}
        evidence['v2_512']={'verdict':'accept','terminal':True}
        self.assertEqual(audit.verdict(evidence,{},set()),'accept')
        self.assertEqual(audit.verdict(evidence,{}, {'collocation'}),'inconclusive')
        self.assertEqual(audit.verdict(evidence,{'collocation':{'verdict':'violation_detected'}},{'collocation'}),'violation_detected')
    def test_prerequisites(self):
        self.assertNotIn('collocation',audit.available({}))
        self.assertNotIn('v2_128',audit.available({'modal':{'verdict':'accept'}}))
        self.assertIn('v2_128',audit.available({'modal':{'verdict':'accept'},'v2_64':{'verdict':'inconclusive','terminal':False}}))
    def test_fv_charge(self):
        self.assertEqual(audit.price('collocation',Request(1,1,.01),{'fv_cell':2},True),122880)
    def test_negative_control_summary(self):
        data,model=fixture()
        with patch.object(audit.s3,'Runtime',TinyRuntime),patch.object(audit.s3,'load_model',return_value=model):
            good=audit.audit(data)
            self.assertTrue(good['passed'],good['errors'])
            data['summary']['permissive']['0.2']['BM']['correct']=1
            bad=audit.audit(data)
            self.assertFalse(bad['passed'])
            self.assertIn('permissive/0.2/BM: summary correct',bad['errors'])
    def test_negative_control_unsupported_acceptance(self):
        data,model=fixture()
        data['packages'][0]['conditions']['none']['0.2']['B']['inputs'][0]['verdict']='accept'
        with patch.object(audit.s3,'Runtime',TinyRuntime),patch.object(audit.s3,'load_model',return_value=model):
            result=audit.audit(data)
        self.assertFalse(result['passed'])
        self.assertTrue(any('final verdict' in e for e in result['errors']))
