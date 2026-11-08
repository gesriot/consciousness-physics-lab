"""Constructed-data controls for the independent active-calibration audit."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from attention import audit_a006 as audit
from attention.io import read_json
from lab.replay import canonical_sha256
from lab.scientist import ROOT


def fixture():
    model=audit.s3.load_model();library=read_json(ROOT/audit.LIBRARY)
    weights=model['action_model']['cost_weights'];maximum=max(audit.max_price(s,weights) for s in library['samples'] if s['family'] in audit.FAMILIES)
    records=[dict(task_id=i,request=dict(z=[1.,0.],mean_time=1.,tolerance=.01,budget=4096,max_calls=3)) for i in range(60)]
    empty_input=dict(verdict='inconclusive',observed={},replacements={},spent=0.,already_closed=False,correct=False,false_accept=False)
    empty_run=dict(policy='',budget=0.,spent=0.,inputs=[dict(input_index=i,**copy.deepcopy(empty_input)) for i in range(45)],actions=[],alerts=[],suspect=[],monitor_history=[],touched_inputs=0,target_exposures=0,target_samples=0,replacement_cost=0.,sample_cost=0.,ordinary_cost=0.,correct=0,false_accepts=0,fault_exposure=dict(ordinary_target_actions=0,target_samples=0,changed_ordinary_categories=0,changed_sample_categories=0,changed_ordinary_v2_steps=0,changed_sample_v2_steps=0))
    packages=[];summary={}
    for k in range(12):
        infos=[dict(task_id=k*5+i//9,full={'actions':[]},outcome={'value':[1.,0.]},reference=[1.,0.],absolute_error=0.,tolerance=.01,accurate=True) for i in range(45)]
        conditions={}
        for fault in audit.FAULTS:
            conditions[fault]={}
            for f in audit.FRACTIONS:
                conditions[fault][f]={}
                for p in audit.POLICIES:
                    allocation=dict(quota=0,requested={'0.2':3,'0.4':6,'0.6':9,'1.0':15}[f],maximum_price=maximum,initial_reserve=0.,ordinary_budget=0.,executed=0,counts_before=dict.fromkeys(audit.FAMILIES,0),counts_after=dict.fromkeys(audit.FAMILIES,0))
                    conditions[fault][f][p]=dict(copy.deepcopy(empty_run),policy=p,sample_allocation=allocation)
        packages.append(dict(package_id=k,inputs=infos,reference_cost=0.,conditions=conditions))
    for fault in audit.FAULTS:
        summary[fault]={}
        for f in audit.FRACTIONS:
            summary[fault][f]={}
            for p in audit.POLICIES:
                summary[fault][f][p]=dict(correct=0,correct_by_package=[0]*12,false_accepts_before=0,false_accepts_after=0,false_accepts_by_package=[0]*12,alerts=0,alerts_before=0,packages_with_new_alert=0,preonset_target_alert=False,detected=False if fault!='none' else None,delay=None,censored=True if fault!='none' else None,detected_within_90=False if fault!='none' else None,spent=0.,replacement_cost=0.,sample_cost=0.,samples=0,postonset_samples=0,target_samples=0,changed_sample_verdicts=0,changed_v2_steps=0,sample_expectation_mismatches=0,followup={'inputs':0,'actions':0,'target_observations':0,'target_samples':0})
    codes={'accept':0,'violation_detected':1,'inconclusive':2}
    changes=[codes[s['expected']] for s in library['samples'] if s['family'] in audit.FAMILIES[:2] and s['expected']!='accept']
    intervention=dict(D_changed_numbers=len(changes),Dprime_changed_numbers=len(changes),D_L1=sum(changes),Dprime_L1=sum(changes),equal_changes=True)
    data=dict(experiment='A006',intake_channel=records,packages=packages,onset_package=int(np.random.default_rng(20261148).integers(4,9)),global_permutations=[],intake_sha256=canonical_sha256([records,[]]),full_pipeline_parity_failures=[],self_model_sha256=canonical_sha256(model),calibration={'threshold':2.,'maximum':1.},summary=summary,intervention_audit=intervention,sample_library_sha256=canonical_sha256(library),sample_library_sources=library['source_sha256'],sample_library_counts=library['counts'])
    return data,library


class ActiveAuditTests(unittest.TestCase):
    def run_fixture(self,data,library):
        with patch.object(audit,'forecasts',return_value={}):return audit.audit(data,library)

    def test_constructed_empty_archive_passes(self):
        data,library=fixture();result=self.run_fixture(data,library)
        self.assertTrue(result['passed'],result['errors'])

    def test_unsupported_acceptance_fails(self):
        data,library=fixture();data['packages'][0]['conditions']['none']['0.2']['B']['inputs'][0]['verdict']='accept'
        result=self.run_fixture(data,library)
        self.assertFalse(result['passed']);self.assertTrue(any('final verdict' in e for e in result['errors']))

    def test_quota_tampering_fails(self):
        data,library=fixture();data['packages'][0]['conditions']['none']['0.2']['C2A']['sample_allocation']['executed']=1
        result=self.run_fixture(data,library)
        self.assertFalse(result['passed']);self.assertTrue(any('allocation executed' in e for e in result['errors']))

    def test_summary_tampering_fails(self):
        data,library=fixture();data['summary']['permissive']['0.4']['BA']['sample_cost']=1
        result=self.run_fixture(data,library)
        self.assertFalse(result['passed']);self.assertTrue(any('summary sample_cost' in e for e in result['errors']))

    def test_library_label_tampering_fails(self):
        data,library=fixture();library['samples'][0]['expected']='invented'
        result=self.run_fixture(data,library)
        self.assertFalse(result['passed']);self.assertIn('sample recipes digest',result['errors'])

    def test_v2_algebra_detects_modified_uncertainty(self):
        library=read_json(ROOT/audit.LIBRARY);model=audit.s3.load_model();weights=model['action_model']['cost_weights']
        sample=next(s for s in library['samples'] if s['family']=='v2');trace=copy.deepcopy(sample['clean_result']['trace'])
        good=audit.Audit();audit.verify_trace(good,trace,complex(*sample['candidate']),sample['request']['tolerance'],weights,False,'synthetic audit')
        self.assertFalse(good.errors,good.errors)
        trace[-1]['observed']['step']['uncertainty_estimate']+=1
        bad=audit.Audit();audit.verify_trace(bad,trace,complex(*sample['candidate']),sample['request']['tolerance'],weights,False,'synthetic audit')
        self.assertTrue(any('observed uncertainty_estimate' in e for e in bad.errors))

    def test_uncertainty_reduction_preserves_decisive_labels(self):
        for uncertainty in (.001,.01,.1):
            for discrepancy in (0.,.05,.2):
                label=audit.classify(True,discrepancy,uncertainty,.1)
                if label!='inconclusive':self.assertEqual(label,audit.classify(True,discrepancy,uncertainty/100,.1))

    def test_family_and_base_requirements(self):
        self.assertEqual(audit.family('v2_512'),'v2')
        self.assertEqual(audit.verdict({}, {'v2':{'verdict':'accept'}}, {'v2'}),'inconclusive')
        self.assertNotIn('collocation',audit.available({}))
        self.assertNotIn('v2_128',audit.available({'modal':{'verdict':'accept'}}))
