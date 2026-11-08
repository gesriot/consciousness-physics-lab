"""A003 archive contracts without sampling held-out inputs."""
import copy
import tempfile
import unittest
from pathlib import Path
from attention import a003
from attention.io import write_json


class A003Tests(unittest.TestCase):
    def test_reject_incorrect_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong.json.gz'
            write_json(p,dict(experiment='A002',intake_channel=[]))
            with self.assertRaises(ValueError):a003.intake(p)

    def test_comparison_exact_decisions_costs_and_protocol(self):
        d=dict(intake_channel=[],global_permutations=[],intake_sha256='i',self_model_sha256='m',
            protocol_sha256='p',static_model_sha256='s',packages=[dict(spent=1.,flag=True,
            expected_cost=2.,planning_seconds=1.,observation=dict(value=1.,verdict='accept'))])
        other=copy.deepcopy(d);other['packages'][0]['planning_seconds']=100.
        other['packages'][0]['observation']['value']+=1e-6
        self.assertTrue(a003.compare(d,other)['decisions_pass'])
        for key,value in [('spent',1.1),('expected_cost',2.1),('flag',False)]:
            other=copy.deepcopy(d);other['packages'][0][key]=value
            self.assertFalse(a003.compare(d,other)['decisions_pass'])
        for key in ('protocol_sha256','static_model_sha256','self_model_sha256'):
            other=copy.deepcopy(d);other[key]='changed'
            self.assertFalse(a003.compare(d,other)['decisions_pass'])

    def test_changed_actions_ignore_model_plan_metadata(self):
        run=dict(correct=1,false_accepts=0,accepts=1,spent=1.,
                 actions=[dict(input_index=0,action='modal',plan=dict(score=1.))])
        other=copy.deepcopy(run);other['actions'][0]['plan']['score']=2.
        package=dict(oracle={str(f):dict(correct=1) for f in a003.FRACTIONS},
                     policies={str(f):{p:copy.deepcopy(run) for p in a003.POLICIES} for f in a003.FRACTIONS})
        package['policies']['0.2']['D']=other
        result=a003.summarize([package])
        self.assertEqual(result['0.2']['changed_action_packages']['D'],0)
        other['actions'][0]['action']='binding'
        self.assertEqual(a003.summarize([package])['0.2']['changed_action_packages']['D'],1)

    def test_oracle_minimum_prerequisite_cost(self):
        def row(action,cost,verdict,terminal=False):
            return dict(action=action,cost=cost,observation=dict(verdict=verdict,terminal=terminal))
        infos=[dict(accurate=False,full=dict(verdict='violation_detected',actions=[
            row('modal',1,'accept'),row('v2_64',2,'inconclusive'),row('v2_128',4,'violation_detected',True),
            row('collocation',10,'violation_detected')]))]
        cert,oracle=a003.oracle_summary(infos,20.)
        self.assertEqual(cert[0]['cost'],7.)
        self.assertEqual(oracle['0.2']['correct'],0)
        self.assertEqual(oracle['0.4']['correct'],1)

    def test_calibration_retains_missing_support(self):
        predictions=[dict(input_index=0,step=0,full_probabilities=dict(accept=1.,violation_detected=0.,inconclusive=0.)),
                     dict(input_index=0,step=1,full_probabilities=None)]
        packages=[dict(package_id=0,inputs=[dict(full=dict(verdict='accept'))],
                       policies={str(f):{p:dict(predictions=predictions) for p in ('B2','D','Dprime')} for f in a003.FRACTIONS})]
        result=a003.calibration_summary(packages)['0.2']['B2']['by_step']
        self.assertEqual(result['0']['brier'],0.)
        self.assertEqual(result['1']['missing'],1)
        self.assertIsNone(result['1']['brier'])


if __name__=='__main__':unittest.main()
