import copy
import unittest
from unittest.mock import patch
from binding import diagnose_b007 as B


class B007PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series=B.lab.make_series(17,'low_moderate',episodes=4)

    def test_four_weak_priors(self):
        self.assertEqual(len(B.P_CONFIGS),4)
        for tp,fp,t in B.P_CONFIGS.values():
            self.assertLessEqual(sum(tp),5)
            self.assertLessEqual(sum(fp),5)

    def test_no_prior_only_clearing(self):
        test=dict(kind='check_source',id='a',candidate='a',cost=1)
        p={'observation':.2,'a':.1,'b':.7}
        self.assertGreater(B.threshold_distance(p,test,set(),.8),0)
        self.assertEqual(B.threshold_distance(p,test,set(),.8,{'a'}),0)

    def test_no_exclusion_only_accusation_potential(self):
        test=dict(kind='check_source',id='a',candidate='a',cost=1)
        p={'observation':.01,'a':.99}
        self.assertGreater(B.threshold_distance(p,test,set(),.8),0)
        self.assertEqual(B.threshold_distance(p,test,{'a'},.8),0)

    def test_exact_model_impossible_branches_omitted(self):
        model={k:dict(tp=1.,fp=0.) for k in B.KINDS}
        test=dict(kind='check_source',id='a',candidate='a',cost=1)
        self.assertEqual(len(list(B.outcomes({'a':1.,'observation':0.},test,model))),1)
        self.assertFalse(B.informative({'a':1.,'observation':0.},test,model))

    def test_information_fallback_never_stops_for_rounding(self):
        model={k:dict(tp=.8,fp=.2) for k in B.KINDS}
        test=dict(kind='check_source',id='a',candidate='a',cost=1)
        with patch.object(B,'action_score',return_value=0.),patch.object(B,'information_score',return_value=0.):
            selected,score=B.choose_probabilistic({'a':.5,'observation':.5},[test],test,set(),.95,model,5)
        self.assertEqual(selected,test)

    def test_no_action_when_unaffordable_or_noninformative(self):
        model={k:dict(tp=.8,fp=.2) for k in B.KINDS}
        test=dict(kind='check_source',id='a',candidate='a',cost=3)
        self.assertEqual(B.choose_probabilistic({'a':.5,'observation':.5},[test],test,set(),.95,model,2),(None,0.))
        self.assertEqual(B.choose_probabilistic({'a':1.,'observation':0.},[test],test,{'a'},.95,model,8),(None,0.))

    def test_variants_budget_direct_evidence_and_learning(self):
        for policy in ('D','R','K2of3','P','P0','Pstar','Px','Pxprime','N'):
            outputs=B.simulate_series(self.series,policy,8)
            for out in outputs:
                self.assertLessEqual(out['protected']['cost'],out['protected']['budget'])
                self.assertEqual(out['metrics']['uncorroborated_accusations'],0)
                if policy in ('P0','Pstar','Px'):
                    self.assertEqual(out['state_before'],out['state_after'])
                if not out['protected']['observable_resolved']:
                    self.assertEqual(out['state_before'],out['state_after'])

    def test_placebo_and_noise(self):
        for config in B.P_CONFIGS:
            a=B.simulate_series(self.series,'P',8,config)
            b=B.simulate_series(self.series,'Pxprime',8,config)
            c=B.simulate_series(self.series,'P',8,config,True)
            for x,y,z in zip(a,b,c):
                self.assertEqual(x['protected'],y['protected'])
                self.assertEqual(x['protected'],z['protected'])
                self.assertEqual(x['metrics'],z['metrics'])

    def test_fixed_conflict_repeats(self):
        episode=next(e for e in self.series['episodes'] if e['signal_branch']=='conflict')
        episode=copy.deepcopy(episode)
        if not episode['signal_ids']:
            episode['signal_ids']=B.lab._false_signal(episode['case'],'conflict',0)
        for policy in ('D','K1of1'):
            with patch.object(B.old,'available_tests',return_value=[]):
                out=B.simulate(episode,policy,8)
            self.assertEqual(out['trace'][0]['kind'],'repeat_observation')
