import unittest
from unittest.mock import patch
import numpy as np
from lab import convergence_candidates as m
from lab import coverage_map as base
from lab.three_way import GRIDS,step_evidence


class CandidateTests(unittest.TestCase):
    def test_shift_and_bounds(self):
        keys={k for c in m.base_cells() for k in base.corners(c)}
        self.assertEqual(len(keys),362*38)
        self.assertEqual(m.coordinate((0,0))[:2],(-6,-np.pi/2))
        self.assertEqual(m.coordinate((5760,576))[:2],(12,np.pi/2))
        self.assertAlmostEqual(m.coordinate((8,8))[0],-5.975)
        cells=[(0,0,8,8)]
        for _ in range(3):cells=[d for c in cells for d in base.children(c)]
        self.assertTrue(all(c-a==1 and d-b==1 for a,b,c,d in cells))

    def test_candidate_negative_controls(self):
        # All increments within rounding but no contraction: only C0 and resolved C3.
        values=[0.,1e-12,2e-12,3e-12]
        self.assertEqual(m.flags(values,512,.9),dict(C0=True,C1=False,C2=False,C3=False))
        self.assertTrue(m.flags(values,512,1.)['C3'])
        # Previous pair contracted, current pair did not: C2 rounding can qualify.
        values=[0.,4e-12,5e-12,7e-12]
        self.assertTrue(m.flags(values,512,.1)['C2'])
        self.assertFalse(m.flags(values,512,.1)['C1'])
        # Earlier contraction cannot stand in for immediately previous contraction.
        values=[0.,4e-12,5e-12,7e-12,10e-12]
        self.assertFalse(m.flags(values,1024,.1)['C2'])
        self.assertFalse(m.flags([0.,1.,2.,3.],512,2.)['C3'])

    def test_frozen_exact_and_local_refinement(self):
        row=m.evaluate((2000,288),0,complex(base.analytic(m.coordinate((2000,288))[2],1)))
        values=[complex(*v) for v in row['values']]
        for i,s in enumerate(row['steps'],3):
            expected=step_evidence(values[:i+1],GRIDS[i],0j,1.)
            self.assertEqual(s['candidates']['C0'],expected['converging'])
            self.assertEqual(s['uncertainty'],expected['uncertainty_estimate'])
        cell=(0,0,8,8)
        records={k:dict(steps=[dict(covered=True,candidates=dict.fromkeys(m.CANDIDATES,True))]) for k in base.corners(cell)}
        self.assertFalse(m.needs_refinement(cell,records))
        records[(0,0)]['steps'][0]['candidates']['C2']=False
        self.assertTrue(m.needs_refinement(cell,records))

    def test_reference_threshold_and_fallback(self):
        keys=[(4480,288),(4481,288)]
        _,audit=m.reference_batch(keys,np.random.default_rng(1))
        self.assertEqual([r['coordinate'] for r in audit['records']],[[4481,288]])
        with patch.object(base,'analytic',return_value=2+0j):
            _,audit=m.reference_batch(keys,np.random.default_rng(1))
        self.assertTrue(audit['fallback']);self.assertEqual(audit['audited'],2)

    def test_prefix_and_futility(self):
        layers=dict.fromkeys(GRIDS,2.)
        values=[0.,.1,.12,.125,999.,888.,777.]
        r=m.replay_path(values,.125,1.,layers,'C1')
        self.assertEqual(r['steps'][-1]['last_grid'],512)
        values=[0.,1.,2.,3.,4.,5.,6.]
        r=m.replay_path(values,0j,.001,layers,'C1')
        self.assertEqual(r['stop_reason'],'futility')
        self.assertEqual(r['steps'][-1]['last_grid'],1024)
        values=[i*1e-12 for i in range(7)]
        r=m.replay_path(values,0j,1.,layers,'C1')
        self.assertEqual(r['stop_reason'],'rounding-limit')

    def test_selection(self):
        summary={c:{'512':dict(dangerous_misses=int(c=='C0'))} for c in m.CANDIDATES}
        costs={'combined':{c:dict(lost_accurate_accepts=0) for c in m.CANDIDATES}}
        self.assertEqual(m.select(summary,costs),'C1')
        costs['combined']['C1']['lost_accurate_accepts']=1
        self.assertEqual(m.select(summary,costs),'C3')
