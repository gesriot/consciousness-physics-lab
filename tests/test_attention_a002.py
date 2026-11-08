"""Archived-shape and calibration tests without held-out generation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from attention import a002, s2
from lab.guard import Request
from lab.replay import canonical_sha256

class A002Tests(unittest.TestCase):
    def test_brier_definition(self):
        self.assertEqual(a002.score(dict(accept=1.,violation_detected=0.,inconclusive=0.),'accept'),0)
        self.assertEqual(a002.score(dict(accept=1.,violation_detected=0.,inconclusive=0.),'inconclusive'),2)

    def test_no_wrong_replay(self):
        with self.assertRaises(ValueError):a002.validate([],[])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(dict(experiment='A001',intake_channel=[])))
            with self.assertRaises(ValueError):a002.intake(p)

    def test_comparison_float_information_and_exact_costs(self):
        d=dict(intake_channel=[dict(request=1.0)],global_permutations=[],intake_sha256='x',self_model_sha256='s',
            packages=[dict(spent=1.,flag=True,observation=dict(value=1.,verdict='accept'))])
        other=copy.deepcopy(d);other['packages'][0]['observation']['value']+=.01
        self.assertTrue(a002.compare(d,other)['decisions_pass'])
        for key,value in [('spent',1.1),('flag',False),('extra',2)]:
            other=copy.deepcopy(d);other['packages'][0][key]=value
            self.assertFalse(a002.compare(d,other)['decisions_pass'])
        other=copy.deepcopy(d);other['intake_channel'][0]['request']=2.
        self.assertFalse(a002.compare(d,other)['decisions_pass'])

    def test_comparison_preserves_audit_and_summary(self):
        d=dict(intake_channel=[],global_permutations=[],intake_sha256='x',self_model_sha256='s',packages=[],
               summary={'correct':2,'spent_by_package':[1.]},full_pipeline_parity_failures=[],reference_audit={'fallback_all':False})
        for key,value in [('summary',{'correct':3}),('summary',{'correct':2,'spent_by_package':[1.1]}),('full_pipeline_parity_failures',[[1,'clean']]),
                          ('reference_audit',{'fallback_all':True})]:
            other=copy.deepcopy(d);other[key]=value
            self.assertFalse(a002.compare(d,other)['decisions_pass'])

    def test_frozen_s2_probability_and_corruption(self):
        m=s2.load_model(); r=Request(1+1j,1,.01)
        original=s2.predict(r,model=m)
        self.assertAlmostEqual(sum(original['full_probabilities'].values()),1)
        self.assertEqual(original,s2.predict(r,model=s2.corrupt(m,'Dprime')))
        d,dp=s2.corrupt(m,'D'),s2.corrupt(m,'Dprime')
        change=lambda a,b:[abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra['counts'],rb['counts']) if x!=y]
        self.assertEqual(change(m['records'],d['records']),change(m['unused_records'],dp['unused_records']))
        self.assertEqual(len(change(m['records'],d['records'])),540)
        p=s2.predict(Request(1,1,1e-7),model=m)
        self.assertTrue(p['feature_details']['collocation_accept_forbidden'])

if __name__=='__main__':unittest.main()
