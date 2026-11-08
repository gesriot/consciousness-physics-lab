"""Independent A003 audit primitives use only synthetic evidence."""
import copy
import unittest
from attention.audit_a003 import decision, available, minimum_certificate, scientific, recompute_calibration, REQUIRED


def action(name, price, verdict, terminal=False):
    return dict(action=name,cost=price,observation=dict(verdict=verdict,terminal=terminal))


class A003AuditTests(unittest.TestCase):
    def test_acceptance_requires_every_basis(self):
        obs={a:dict(verdict='accept') for a in REQUIRED}
        self.assertEqual(decision(obs),'inconclusive')
        obs['v2_512']=dict(verdict='accept',terminal=True)
        self.assertEqual(decision(obs),'accept')
        obs['modal']['verdict']='inconclusive'
        self.assertEqual(decision(obs),'inconclusive')
        obs['normalization']['verdict']='violation_detected'
        self.assertEqual(decision(obs),'violation_detected')

    def test_grid_dependencies_and_terminal(self):
        self.assertNotIn('v2_64',available({}))
        obs={'modal':dict(verdict='accept')}
        self.assertIn('v2_64',available(obs))
        self.assertNotIn('v2_128',available(obs))
        obs['v2_64']=dict(verdict='inconclusive',terminal=False)
        self.assertIn('v2_128',available(obs))
        obs['v2_128']=dict(verdict='inconclusive',terminal=True)
        self.assertNotIn('v2_256',available(obs))

    def test_oracle_charges_prerequisite_chain(self):
        info=dict(accurate=False,full=dict(actions=[action('modal',1,'accept'),action('v2_64',2,'inconclusive'),action('v2_128',4,'violation_detected',True),action('collocation',10,'violation_detected')]))
        self.assertEqual(minimum_certificate(info),7.)
        info['full']['actions'].append(action('normalization',3,'violation_detected'))
        self.assertEqual(minimum_certificate(info),3.)
        info['accurate']=True
        self.assertIsNone(minimum_certificate(info))

    def test_no_detector_is_unreachable(self):
        info=dict(accurate=False,full=dict(actions=[action('modal',1,'inconclusive')]))
        self.assertIsNone(minimum_certificate(info))

    def test_only_timing_and_policy_excluded(self):
        a=dict(policy='B2',planning_seconds=1.,nested=dict(spent=3.,flag=True))
        b=copy.deepcopy(a);b['policy']='Dprime';b['planning_seconds']=2.
        self.assertEqual(scientific(a),scientific(b))
        b['nested']['spent']=3.1
        self.assertNotEqual(scientific(a),scientific(b))

    def test_brier_denominators_and_missing(self):
        rows=[dict(input_index=0,step=0,full_probabilities=dict(accept=.5,violation_detected=.5,inconclusive=0)),dict(input_index=0,step=1,full_probabilities=None)]
        packages=[dict(package_id=0,inputs=[dict(full=dict(verdict='accept'))],policies={'0.2':{'B2':dict(predictions=rows)}})]
        answer=recompute_calibration(packages,'0.2','B2')
        self.assertEqual(answer['by_step']['0'],dict(count=1,missing=0,brier=.5))
        self.assertEqual(answer['by_step']['1'],dict(count=0,missing=1,brier=None))
        self.assertEqual(answer['by_package'][0]['missing'],1)


if __name__=='__main__':unittest.main()
