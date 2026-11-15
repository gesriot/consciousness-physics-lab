import copy
import unittest
from unittest.mock import patch
from binding import diagnose_b006 as D
from binding import source_lab_b006 as L


class DiagnosisB006Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series=L.make_series(13,'low_moderate',episodes=8)
        cls.episode=next(e for e in cls.series['episodes'] if e['signal_ids'])

    def test_probabilities_and_normalization(self):
        state=D.initial_state('beta8_t80');p=D.posterior_start({'a','b'},state)
        self.assertEqual(p,{'observation':.2,'a':.4,'b':.4})
        test=dict(kind='check_source',candidate='a',cost=30)
        for report in (True,False):
            post=D.update_posterior(p,test,report,D.model_view(state,'P')['active'])
            self.assertAlmostEqual(sum(post.values()),1)
            self.assertGreaterEqual(min(post.values()),0)
        self.assertGreater(D.risk_score(p,test,D.model_view(state,'P')['active']),0)

    def test_px_and_placebo_model_views(self):
        state=D.initial_state('beta8_t80')
        p=D.model_view(state,'P');x=D.model_view(state,'Px');z=D.model_view(state,'Pxprime')
        self.assertEqual(p['active'],z['active'])
        self.assertEqual(p['auxiliary'],x['auxiliary'])
        self.assertNotEqual(p['active'],x['active'])
        self.assertNotEqual(p['auxiliary'],z['auxiliary'])

    def test_fractional_learning_uses_only_reports(self):
        state=D.initial_state('beta8_t80');old=copy.deepcopy(state)
        D.learn(state,{'a':.75,'observation':.25},[dict(kind='check_source',candidate='a',bad=True)])
        self.assertEqual(state['prior'],[2.25,8.75])
        self.assertEqual(state['channels']['check_source']['tp'],[8.75,2.])
        self.assertEqual(state['channels']['check_source']['fp'],[1.25,9.])
        self.assertEqual(old['resolved_episodes'],0)
        self.assertEqual(state['resolved_episodes'],1)

    def test_budget_ledger_and_no_unpaid_repair(self):
        for policy in ('D','R','K1of1','K2of3','K3of5','P','Px','Pxprime','N'):
            for budget in D.BUDGETS:
                out=D.simulate(self.episode,policy,budget)
                self.assertEqual(sum(t['cost'] for t in out['trace']),out['metrics']['cost'])
                self.assertLessEqual(out['metrics']['cost'],out['protected']['budget'])
                self.assertEqual(len(out['protected']['repaired']),out['metrics']['repaired'])
                self.assertTrue(all(t['cost']>0 for t in out['trace']))

    def test_no_revision_retains_errors_without_accusation(self):
        out=D.simulate(self.episode,'N',32)
        self.assertEqual(out['metrics']['cost'],0)
        self.assertEqual(out['metrics']['remaining_bad'],out['metrics']['initial_bad'])
        self.assertEqual(out['metrics']['false_accusations'],0)

    def test_placebo_decisions_identical(self):
        for config in D.P_CONFIGS:
            a=D.simulate_series(self.series,'P',8,config)
            b=D.simulate_series(self.series,'Pxprime',8,config)
            self.assertEqual([o['protected'] for o in a],[o['protected'] for o in b])
            self.assertEqual([o['state_after'] for o in a],[o['state_after'] for o in b])

    def test_noise_preserves_decisions_on_fixture(self):
        for policy in ('D','R','K2of3','P','Px'):
            a=D.simulate(self.episode,policy,8,noise=False)
            b=D.simulate(self.episode,policy,8,noise=True)
            self.assertEqual(a['protected'],b['protected'])
            self.assertEqual(a['metrics'],b['metrics'])

    def test_unresolved_episode_never_trains(self):
        out=D.simulate(self.episode,'P',2)
        if not out['protected']['observable_resolved']:
            self.assertEqual(out['state_before'],out['state_after'])
        # A threshold-only prior verdict is not paid observational evidence.
        state=D.initial_state('beta8_t80');state['prior']=[1000.,1.]
        out=D.simulate(self.episode,'P',2,state)
        self.assertEqual(out['protected']['decision'],'observation')
        self.assertFalse(out['protected']['observable_resolved'])
        self.assertEqual(out['state_before'],out['state_after'])

    def test_no_affordable_rebuild_never_trains(self):
        with patch.object(L.NoisyService,'repair_cost',return_value=10**12):
            out=D.simulate(self.episode,'P',32,config='beta99_t80')
        self.assertEqual(out['protected']['repaired'],[])
        if out['protected']['decision'] not in (None,'observation'):
            self.assertFalse(out['protected']['observable_resolved'])
            self.assertEqual(out['state_before'],out['state_after'])

    def test_series_state_is_sequential_and_resets(self):
        out=D.simulate_series(self.series,'P',8)
        for before,after in zip(out,out[1:]):
            self.assertEqual(before['state_after'],after['state_before'])
        self.assertEqual(out,D.simulate_series(self.series,'P',8))

    def test_evaluator_truth_does_not_enter_policy_state(self):
        original=L.NoisyService.evaluate
        def poisoned(service):
            result=original(service)
            result['culprit']='secret';result['kind']='false_objection'
            result['remaining_bad_ids']=[];result['initial_bad_ids']=[]
            return result
        for policy in ('D','P','Px'):
            a=D.simulate(self.episode,policy,8)
            with patch.object(L.NoisyService,'evaluate',poisoned):b=D.simulate(self.episode,policy,8)
            self.assertEqual(a['protected'],b['protected'])
            self.assertEqual(a['state_after'],b['state_after'])

    def test_k_one_is_d(self):
        for episode in self.series['episodes']:
            self.assertEqual(D.simulate(episode,'D',8),D.simulate(episode,'K1of1',8))

    def test_invalid_inputs(self):
        for policy,budget in (('unknown',8),('P',3)):
            with self.assertRaises(ValueError):D.simulate(self.episode,policy,budget)
