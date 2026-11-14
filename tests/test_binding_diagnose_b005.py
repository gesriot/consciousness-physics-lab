"""Synthetic policy and paid-service checks; no registered series are used."""
import copy
import unittest
from unittest.mock import patch
from binding import diagnose_b005 as D
from binding import source_lab_b005 as L


class DiagnosisB005Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={f:L.build_case(L.intake(1),f) for f in L.FAULTS}

    def test_memory_contains_no_evaluator_answers(self):
        c=self.cases['units']
        poison=copy.deepcopy(c)
        poison['truth']={'secret':123}
        poison['culprit']='secret'
        poison['affected']=['secret']
        poison['inaccurate']=['secret']
        poison['catalog']={'secret':456}
        poison['measurements']={'secret':789}
        for p in D.POLICIES:
            ids=D.signal(c,'measurement')
            m=D.Memory(c,p,ids)
            changed=D.Memory(poison,p,ids)
            self.assertEqual(vars(m),vars(changed))
            self.assertFalse(any(k in vars(m) for k in ('truth','culprit','catalog','measurements','case')))

    def test_actions_are_invariant_to_hidden_truth_and_fault_labels(self):
        c=self.cases['units']
        poisoned=copy.deepcopy(c)
        poisoned['truth']={cid:v+1 for cid,v in c['truth'].items()}
        poisoned['culprit']='not-the-culprit'
        poisoned['inaccurate']=[]
        poisoned['affected']=[]
        poisoned['fault']='arbitrary-evaluator-label'
        for p in ('G','L','F','T','N'):
            a=D.simulate(c,p,8,'measurement')
            b=D.simulate(poisoned,p,8,'measurement')
            for key in ('actions','checked','culprit','accusations','repaired','cost','resolved','initial_candidates'):
                self.assertEqual(a['protected'][key],b['protected'][key])
            self.assertEqual(a['diagnostics']['final_values'],b['diagnostics']['final_values'])

    def test_all_costs_are_charged_and_never_overdraw(self):
        for f,c in self.cases.items():
            for p in D.POLICIES:
                for budget in D.BUDGETS:
                    for branch in D.SIGNALS:
                        o=D.simulate(c,p,budget,branch)
                        running=0
                        for action in o['trace']:
                            self.assertGreater(action['cost'],0)
                            running+=action['cost']
                            self.assertLessEqual(running,o['protected']['budget'])
                        self.assertEqual(running,o['metrics']['cost'])
                        self.assertEqual(running+o['metrics']['unused_budget'],o['protected']['budget'])
                        self.assertEqual(o['metrics']['withdrawn_good'],0)

    def test_unaffordable_repair_leaves_accepted_value_unchanged(self):
        c=self.cases['version']
        with patch.object(D.Service,'repair_cost',return_value=10**9):
            out=D.simulate(c,'G',2,'measurement')
        self.assertEqual(out['protected']['culprit'],c['culprit'])
        self.assertEqual(out['protected']['repaired'],[])
        self.assertFalse(any(a['kind']=='repair' for a in out['trace']))
        self.assertEqual(out['diagnostics']['final_values'],{r['id']:r['value'] for r in c['records']})
        self.assertEqual(out['metrics']['remaining_bad'],len(c['inaccurate']))
        self.assertGreater(out['metrics']['suspected'],0)

    def test_local_check_consistency_does_not_validate_ancestors(self):
        c=self.cases['units']
        cid=c['branches'][0]['primary'][1]
        svc=D.Service(c)
        bad,source,cost,value=svc.check_claim(cid)
        self.assertFalse(bad)
        self.assertFalse(L.close(value,c['truth'][cid]))
        self.assertEqual(source['id'],L.record_map(c)[cid]['step_id'])
        self.assertNotEqual(source['id'],c['culprit'])
        self.assertEqual(cost,L.OPERATION_COST['inverse']+1)
        mem=D.Memory(c,'G',[cid])
        candidates=mem.grounds(cid)
        candidates.discard(source['id'])
        self.assertIn(c['culprit'],candidates)

    def test_local_check_detects_only_bad_step(self):
        c=self.cases['intermediate']
        svc=D.Service(c)
        cid=c['branches'][0]['primary'][1]
        bad,source,cost,value=svc.check_claim(cid)
        self.assertTrue(bad)
        self.assertEqual(source['id'],c['culprit'])
        self.assertTrue(L.close(value,c['truth'][cid]))
        self.assertEqual(cost,67)
        # A downstream step consistently consumes a bad parent and is innocent.
        downstream=c['branches'][0]['primary'][2]
        self.assertFalse(svc.check_claim(downstream)[0])

    def test_service_checks_read_only_requested_record(self):
        c=self.cases['version']
        svc=D.Service(c)
        good=next(s for s in svc.sources if s!=c['culprit'])
        bad,fresh,cost=svc.check_source(good)
        self.assertFalse(bad)
        self.assertEqual(fresh['id'],good)
        self.assertNotEqual(fresh['id'],c['culprit'])
        self.assertIn(cost,(30,100))
        self.assertEqual(set(fresh),set(c['catalog'][good]))

    def test_service_actions_do_not_read_hidden_evaluator(self):
        c=copy.deepcopy(self.cases['intermediate'])
        svc=D.Service(c)
        cid=c['branches'][0]['primary'][1]
        expected=svc.check_claim(cid)
        for k in ('truth','culprit','affected','inaccurate','fault'):
            del c[k]
        self.assertEqual(svc.check_claim(cid),expected)

    def test_graph_and_inherited_labels_have_same_information(self):
        for c in self.cases.values():
            ids=D.signal(c,'measurement')
            g,l=D.Memory(c,'G',ids),D.Memory(c,'L',ids)
            self.assertFalse(l.parents)
            self.assertFalse(g.labels)
            for cid in g.rows:
                self.assertEqual(g.grounds(cid),l.grounds(cid))
            self.assertEqual(g.capacity,l.capacity)
            self.assertLessEqual(g.bytes,g.capacity)
            self.assertLessEqual(l.bytes,l.capacity)

    def test_graph_and_labels_choose_identical_actions(self):
        for c in self.cases.values():
            for budget in D.BUDGETS:
                for branch in D.SIGNALS:
                    g=D.simulate(c,'G',budget,branch)
                    l=D.simulate(c,'L',budget,branch)
                    self.assertEqual(g['protected'],l['protected'])
                    self.assertEqual(g['trace'],l['trace'])
                    for metric in ('remaining_bad','withdrawn_good','false_accusations','cost','resolved'):
                        self.assertEqual(g['metrics'][metric],l['metrics'][metric])

    def test_irrelevant_changes_leave_actions_unchanged(self):
        for c in self.cases.values():
            for base,placebo in (('G','Gxprime'),('L','Lxprime')):
                for branch in D.SIGNALS:
                    a=D.simulate(c,base,8,branch)
                    b=D.simulate(c,placebo,8,branch)
                    self.assertEqual(a['protected'],b['protected'])

    def test_relevant_changes_can_change_actions(self):
        c=self.cases['version']
        for base,changed in (('G','Gx'),('L','Lx')):
            a=D.simulate(c,base,8,'measurement')
            b=D.simulate(c,changed,8,'measurement')
            self.assertNotEqual(a['protected']['actions'],b['protected']['actions'])

    def test_no_revision_preserves_all_claims_without_cost(self):
        for c in self.cases.values():
            o=D.simulate(c,'N',32,'measurement')
            self.assertEqual(o['metrics']['cost'],0)
            self.assertEqual(o['metrics']['remaining_bad'],len(c['inaccurate']))
            self.assertEqual(o['metrics']['withdrawn_good'],0)
            self.assertEqual(o['protected']['actions'],[])

    def test_clean_cases_trigger_no_actions_or_accusations(self):
        c=self.cases['clean']
        for p in D.POLICIES:
            for branch in D.SIGNALS:
                o=D.simulate(c,p,32,branch)
                self.assertEqual(o['protected']['signal_ids'],[])
                self.assertEqual(o['protected']['actions'],[])
                self.assertEqual(o['metrics']['false_accusations'],0)

    def test_false_measurement_is_repeated_without_accusing_source(self):
        c=self.cases['false_measurement']
        for p in ('G','L'):
            o=D.simulate(c,p,32,'measurement')
            self.assertEqual(o['metrics']['remaining_bad'],0)
            self.assertEqual(o['metrics']['false_accusations'],0)
            self.assertEqual(o['metrics']['repeated_measurement'],1)
            self.assertTrue(o['protected']['resolved'])
            self.assertIsNone(o['protected']['culprit'])
            self.assertEqual(o['trace'][-1]['kind'],'repeat_measurement')
            self.assertEqual(o['trace'][-1]['cost'],2*c['budget_unit'])

    def test_signal_contains_only_claim_ids(self):
        for c in self.cases.values():
            for branch in D.SIGNALS:
                ids=D.signal(c,branch)
                self.assertTrue(set(ids)<=set(L.record_map(c)))
                self.assertNotIn(c['culprit'],ids)

    def test_ties_are_deterministic_after_rounding(self):
        c=self.cases['units']
        m=D.Memory(c,'G',[])
        tests=[{'kind':'check_source','id':'b','candidate':'b','cost':30},
               {'kind':'check_source','id':'a','candidate':'a','cost':30}]
        a=D.choose(m,copy.deepcopy(tests),{'a','b'})
        b=D.choose(m,list(reversed(copy.deepcopy(tests))),{'a','b'},True)
        self.assertEqual(a['id'],'a')
        self.assertEqual(a,b)

    def test_noise_preserves_protected_decisions_on_synthetic_cases(self):
        for f in L.FAULTS:
            clean=L.build_case(L.intake(2),f)
            noisy=L.build_case(L.intake(2),f,noise=True)
            for p in ('G','L','F','T'):
                a=D.simulate(clean,p,8,'measurement')
                b=D.simulate(noisy,p,8,'measurement',noise=True)
                self.assertEqual(a['protected'],b['protected'])

    def test_wrong_conditions_are_rejected(self):
        c=self.cases['clean']
        for args in (('other',8,'measurement'),('G',7,'measurement'),('G',8,'other')):
            with self.assertRaises(ValueError):
                D.simulate(c,*args)


if __name__=='__main__':
    unittest.main()
