import inspect
import unittest
from unittest.mock import patch
import mpmath as mp
import numpy as np
from lab import matched_mixture as m
from lab import radial_collocation as r
from lab.executor import request_hash
from lab.geometry_invariants import modal_test_hook, fv_test_hook
from lab.guard import Request
from lab.scientist import analytic


class MatchedMixtureTests(unittest.TestCase):
    def test_two_slab_equations_and_branch(self):
        p = m.derive_parameters()
        with mp.workdps(60):
            w = list(map(mp.mpf,p['weights']))
            x = list(map(mp.mpf,p['times']))
            self.assertTrue(all(v > 0 for v in w+x))
            self.assertLess(abs(sum(w)-1),mp.mpf('1e-55'))
            self.assertLess(abs(sum(a*b for a,b in zip(w,x))-1),mp.mpf('1e-50'))
            self.assertLess(abs(sum(a*b*b for a,b in zip(w,x))-mp.mpf(25)/21),mp.mpf('1e-50'))
            self.assertLess(abs(sum(a/mp.sqrt(b) for a,b in zip(w,x))-3/mp.sqrt(5)),mp.mpf('1e-50'))
        self.assertAlmostEqual(float(w[0]),.18979,places=4)
        self.assertAlmostEqual(float(x[0]),.09826,places=4)
        self.assertAlmostEqual(float(x[1]),1.21123,places=4)

    def test_three_slab_positive_construction(self):
        p = m.derive_three_slabs()
        self.assertTrue(p['positive'],p)
        self.assertLess(max(abs(mp.mpf(v)) for v in p['residuals']),mp.mpf('1e-45'))

    def test_clean_hook_preserves_existing_calls(self):
        for kind, function in (('modal',modal_test_hook),('fv',fv_test_hook)):
            for z in (0,1+2j,1e4):
                self.assertEqual(m.model_response(kind,z,1.,512,'clean'),function(z,1.,512,3))

    def test_common_mixture_hook_uses_two_components(self):
        with patch.object(m,'modal_test_hook',return_value=1+0j) as modal:
            self.assertAlmostEqual(m.model_response('modal',1,1,512,'mixture'),1.)
            self.assertEqual(modal.call_count,2)
        with patch.object(m,'fv_test_hook',return_value=1+0j) as fv:
            self.assertAlmostEqual(m.model_response('fv',1,1,512,'mixture'),1.)
            self.assertEqual(fv.call_count,2)

    def test_third_equation_and_quadrature(self):
        for n in (64,128):
            nodes, d2, w = r.operators(n)
            self.assertLess(abs(sum(w)-1),1e-14)
            self.assertLess(abs(np.dot(w,nodes**2)-1/3),1e-14)
            self.assertLess(max(abs(d2@(nodes**3)-6*nodes)),1e-7)
            for z in (0,1,1+2j,100):
                self.assertLess(abs(r.response(z,1,n)-complex(analytic(z,1))),1e-8)

    def test_third_does_not_import_existing_solvers(self):
        source = inspect.getsource(r)
        self.assertNotIn('from lab.',source)
        self.assertNotIn('import lab',source)
        with patch.object(m,'model_response',side_effect=AssertionError('hook entered')):
            result = r.check(1,1,complex(analytic(1,1)),.001)
        self.assertEqual(result['verdict'],'accept')
        self.assertEqual(result['matrix_solves'],2)

    def test_probe_gates_and_control(self):
        req = Request(1,1,.001)
        mixture = m.evaluate(req,request_hash(req),'mixture')
        slab = m.evaluate(req,request_hash(req),'slab')
        self.assertTrue(all(mixture['checks'].values()),mixture['checks'])
        self.assertFalse(slab['checks']['second_moment'])
        self.assertFalse(slab['checks']['high_frequency'])
        self.assertEqual(mixture['third']['verdict'],'violation_detected')
        self.assertFalse(mixture['accepted']['D_gates_third'])
        self.assertEqual(mixture['work']['probes_modal_terms'],40960)
        self.assertEqual(mixture['work']['probes_fv_cells'],26624)

    def test_model_rejection_and_abstention_are_distinct(self):
        req = Request(1,1,.001)
        with patch.object(m,'third_check',return_value=dict(verdict='inconclusive',matrix_solves=2,seconds=0)):
            result = m.evaluate(req,request_hash(req),'slab')
        self.assertEqual(result['third']['verdict'],'inconclusive')
        self.assertEqual(result['verdicts']['D_gates_third'],'reject')
        self.assertEqual(result['verdicts']['third'],'inconclusive')


if __name__ == '__main__':
    unittest.main()
