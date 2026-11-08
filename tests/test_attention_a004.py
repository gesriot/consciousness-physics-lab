"""Synthetic contracts for the fixed-archive factorial allocator."""
import copy
import inspect
import tempfile
from pathlib import Path
from attention.io import write_json
import unittest
import numpy as np
from attention import agent, s3, factorial, a004
from lab.guard import Request
from test_attention_updated import trace, synthetic, FakeExecutor


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.model=synthetic([trace(),trace(False,{'normalization':1}),trace(False,{'high_frequency':1})])
        self.runtime=factorial.Runtime(self.model)
        self.request=Request(1,1,.01)

    def test_common_fixed_archive_and_positive_channel(self):
        original=copy.deepcopy(self.model)
        for corruption in ('none','D','Dprime'):
            runtime=factorial.Runtime(self.model,corruption)
            np.testing.assert_array_equal(runtime.raw,self.runtime.raw)
            for action,j in s3.INDEX.items():
                k=factorial.alphabet_size(action);present=runtime.raw[:,j]>=0
                self.assertTrue(np.all(runtime.likelihood[present,j,:k]>0))
                np.testing.assert_allclose(runtime.likelihood[present,j].sum(axis=1),1,atol=1e-15)
                self.assertTrue(np.all(runtime.likelihood[~present,j]==0))
        self.assertEqual(self.model,original)
        audit=factorial.corruption_audit(self.model)
        self.assertTrue(audit['equal_changes']);self.assertTrue(audit['Dprime_active_unchanged'])
        self.assertGreater(audit['D_L1'],0)

    def test_update_factor_changes_only_weights(self):
        on=factorial.Posterior(self.request,self.runtime,True)
        off=factorial.Posterior(self.request,self.runtime,False)
        before=off.weights.copy()
        on.update('normalization',{'verdict':'accept'});off.update('normalization',{'verdict':'accept'})
        np.testing.assert_array_equal(off.weights,before)
        self.assertLess(on.weights[1],on.weights[0])
        self.assertEqual(on.snapshot()['support_count'],3)
        np.testing.assert_array_equal(off.conditioned(off.weights,'high_frequency',1)[0],before)
        self.assertEqual(on.evidence,off.evidence)
        with self.assertRaises(ValueError):on.update('normalization',{'verdict':'accept'})

    def test_horizon_has_common_direct_options(self):
        for update in (False,True):
            p=factorial.Posterior(self.request,self.runtime,update)
            prices={a:1. for a in agent.ORDER}
            one=factorial.Plans({},p,prices,False);two=factorial.Plans({},p,prices,True)
            self.assertEqual(one.direct,two.direct)
            self.assertFalse(one.trees);self.assertTrue(two.trees)
            for budget in (.99,1.,2.,100.,1.):
                for plans in (one,two):
                    option=plans.best(budget)
                    if budget<1:self.assertIsNone(option)
                    else:self.assertLessEqual(option['expected_cost'],budget)

    def test_hypothetical_plans_do_not_update_actual_evidence(self):
        p=factorial.Posterior(self.request,self.runtime,True)
        weights=p.weights.copy();before=p.snapshot()
        plans=factorial.Plans({},p,{a:1. for a in agent.ORDER},True)
        np.testing.assert_array_equal(p.weights,weights)
        self.assertEqual(p.evidence,{})
        self.assertEqual(p.snapshot(),before)
        for tree in plans.trees:
            self.assertAlmostEqual(sum(b['probability'] for b in tree['branches']),1.)
            self.assertTrue(all(0<=b['eligible_mass']<=1+1e-12 for b in tree['branches']))
        for budget in (1.,2.,10.):
            chosen=plans.best(budget)
            if chosen and chosen['kind']=='TREE':
                for branch in chosen['branches']:
                    continuation=branch['continuation']
                    if continuation:self.assertLessEqual(1.+continuation['expected_cost'],budget)

    def test_missing_prediction_exclusion_and_reset(self):
        rows=[trace(terminal=512),trace(terminal=1024)]
        p=factorial.Posterior(self.request,factorial.Runtime(synthetic(rows)),True)
        report=p.forecast('v2_1024')
        self.assertAlmostEqual(report['eligible_mass'],.5)
        self.assertFalse(report['missing_prediction_extension'])
        p.weights=np.asarray([1.,0.])
        report=p.forecast('v2_1024')
        self.assertEqual(report['eligible_mass'],0)
        self.assertTrue(report['missing_prediction_extension'])
        self.assertEqual(report['probabilities'],[.25]*4)
        result=p.update('v2_1024',{'verdict':'accept','terminal':True})
        self.assertTrue(result['prior_reset'])
        np.testing.assert_array_equal(p.weights,p.prior)
        self.assertEqual(p.real_extensions,1)
        self.assertEqual(p.runtime.raw[0,s3.INDEX['v2_1024']],-1)

    def test_hybrid_threshold_and_stops_without_fallback(self):
        model=synthetic([trace()])
        scheduler=factorial.Scheduler([self.request],model,'H')
        self.assertIsNone(scheduler.choose(1000.))
        self.assertTrue(scheduler.skipped)
        self.assertTrue(all(x['decisive_probability']<.05 for x in scheduler.skipped))
        model=synthetic([trace(False,{'high_frequency':1})])
        schedule=factorial.Scheduler([self.request],model,'H')
        selected=schedule.choose(1000.)
        self.assertEqual(selected[1],'high_frequency')
        changed=factorial.Scheduler([self.request],model,'HD').choose(1000.)
        self.assertEqual(changed[1],'normalization')

    def test_dprime_exact_and_real_verdicts_only(self):
        for base,control in [('U1T','Dprime'),('H','HDprime')]:
            results=[]
            for policy in (base,control):
                result=factorial.run_batch([FakeExecutor(trace()),FakeExecutor(trace(False,{'normalization':1}))],policy,1000.,self.model)
                result.pop('policy');result.pop('planning_seconds');results.append(result)
            self.assertEqual(*results)
            self.assertEqual(results[0]['spent'],sum(a['cost'] for a in results[0]['actions']))
            for r in results[0]['inputs']:self.assertEqual(r['verdict'],agent.verdict(r['observed']))

    def test_boundary_and_invalid_inputs(self):
        self.assertEqual(list(inspect.signature(factorial.Scheduler).parameters),['requests','model','policy'])
        self.assertEqual(list(inspect.signature(factorial.Scheduler.choose).parameters),['self','remaining'])
        self.assertEqual(list(inspect.signature(factorial.Scheduler.update).parameters),['self','index','action','observed'])
        with self.assertRaises(ValueError):factorial.run_batch([],'U1T',-1,self.model)
        with self.assertRaises(ValueError):factorial.run_batch([],'U1T',1,self.model,[[0,'modal']])
        with self.assertRaises(ValueError):factorial.Scheduler([],self.model,'unknown')

    def test_initial_forecast_and_replay_rejection(self):
        initial=a004.initial(self.request,self.model)
        self.assertEqual(set(initial['action_forecasts']),set(agent.ORDER))
        self.assertEqual(len(initial['neighbor_ids']),1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wrong.json.gz'
            write_json(path,dict(experiment='A003',intake_channel=[]))
            with self.assertRaises(ValueError):a004.intake(path)

    def test_decision_comparison_protects_actions_and_costs(self):
        data=dict(intake_channel=[],global_permutations=[],intake_sha256='i',self_model_sha256='m',
            protocol_sha256='p',packages=[dict(spent=1.,expected_cost=2.,planning_seconds=1.,action='modal',
                score=.5,prior_reset=False,observation=dict(verdict='accept'))])
        other=copy.deepcopy(data);other['packages'][0]['planning_seconds']=99.;other['packages'][0]['score']=.6
        self.assertTrue(a004.compare(data,other)['decisions_pass'])
        for key,value in [('action','binding'),('spent',2.),('expected_cost',3.),('prior_reset',True)]:
            other=copy.deepcopy(data);other['packages'][0][key]=value
            self.assertFalse(a004.compare(data,other)['decisions_pass'])

    def test_stopping_requires_strict_win_at_both(self):
        variants=('U0O','U1O','U0T','U1T','H','C2')
        summary={f:dict(policies={p:dict(correct=10) for p in variants}) for f in ('0.2','0.4')}
        self.assertTrue(a004.stopping(summary)['close_budget_series'])
        summary['0.2']['policies']['U1T']['correct']=11
        self.assertTrue(a004.stopping(summary)['close_budget_series'])
        self.assertEqual(a004.stopping(summary)['variants_beating_C2_combined'],['U1T'])
        summary['0.4']['policies']['U1T']['correct']=11
        self.assertFalse(a004.stopping(summary)['close_budget_series'])


if __name__=='__main__':unittest.main()
