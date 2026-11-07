import unittest
from unittest.mock import patch
from lab.guard import Request
from lab.three_way import classify,check_three_way,full_path,shadow_evidence,step_evidence,GRIDS
from lab.allocation import refine_value


class ThreeWayTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify(True,.75,.25,1),'accept')
        self.assertEqual(classify(True,1.25,.25,1),'inconclusive')
        self.assertEqual(classify(True,1.25001,.25,1),'violation_detected')
        self.assertEqual(classify(False,10,.01,1),'inconclusive')
        self.assertEqual(classify(True,float('nan'),.1,1),'inconclusive')

    def test_decisive_stop(self):
        req=Request(1,1,.01)
        calls=[]
        def constant(z,t,n):
            calls.append(n);return 1+0j
        result=check_three_way(req,1,constant)
        self.assertEqual(result['verdict'],'accept')
        self.assertEqual(calls,[64,128,256,512])
        self.assertEqual(check_three_way(req,2,constant)['verdict'],'violation_detected')

    def test_futility_only_inconclusive(self):
        req=Request(1,1,.001)
        def slow(z,t,n):return 1+1/(n/64)**.1
        result=check_three_way(req,slow(1,1,1024),slow)
        self.assertEqual(result['verdict'],'inconclusive')
        self.assertEqual(result['stop_reason'],'futility')
        self.assertEqual(result['grids'][-1],1024)

    def test_full_shadow_no_stop(self):
        req=Request(1,1,.01)
        calls=[]
        def constant(z,t,n):calls.append(n);return 1+0j
        values=full_path(req,constant)
        self.assertEqual(calls,list(GRIDS))
        evidence=shadow_evidence(req,1,values)
        self.assertTrue(evidence['any_accept'])
        self.assertEqual(evidence['first_decisive_grid'],512)

    def test_u_matches_frozen_D(self):
        req=Request(2+3j,.7,.001)
        frozen=refine_value(req,10)
        values=[complex(*v) for v in frozen['values']]
        for old in frozen['steps']:
            i=frozen['grids'].index(old['last_grid'])
            new=step_evidence(values[:i+1],old['last_grid'],10,req.tolerance)
            for key in ('converging','uncertainty_estimate','discrepancy'):
                self.assertEqual(new[key],old[key])


if __name__=='__main__':unittest.main()
