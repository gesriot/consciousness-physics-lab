"""Development and synthetic verification of observation-conditioned scheduling."""
import copy
import inspect
import unittest
from unittest.mock import patch
import numpy as np
from attention import agent, s3, updated_batch
from attention.build_s3 import build
from lab.guard import Request


def trace(accurate=True, changes=None, terminal=512):
    outcomes = []
    for action in agent.ORDER:
        if action.startswith('v2_'):
            n = int(action[3:])
            value = 2 if n < terminal else 0 if n == terminal else -1
        else: value = 0
        outcomes.append(value)
    for a,v in (changes or {}).items(): outcomes[s3.INDEX[a]] = v
    evidence = {a:s3.observation(a,v) for a,v in zip(agent.ORDER,outcomes) if v >= 0}
    return dict(group_index=0, accurate=accurate, full_verdict=agent.verdict(evidence), outcomes=outcomes)


def synthetic(rows, neighbors=16, groups=None):
    base=s3.load_model()
    base['trajectories']=copy.deepcopy(rows)
    base['groups']=groups or [dict(id='synthetic:0',features=[0.]*6)]
    base['neighbors']=neighbors
    return base


class FakeExecutor:
    def __init__(self, row):
        self.request=Request(1,1,.01)
        self.row=row
        self.observed={}
    def perform(self,action):
        if action not in agent.available(self.observed): raise AssertionError('Unavailable execution')
        value=self.row['outcomes'][s3.INDEX[action]]
        result=s3.observation(action,value)
        self.observed[action]=result
        return copy.deepcopy(result)


class UpdatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=s3.load_model()

    def test_model_trace_contract_and_rebuild(self):
        self.assertEqual(len(self.model['groups']),120)
        self.assertEqual(len(self.model['trajectories']),1080)
        runtime=s3.Runtime(self.model)
        for i,row in enumerate(self.model['trajectories']):
            self.assertNotIn('condition',row)
            self.assertEqual(s3.VERDICTS[runtime.full[i]],row['full_verdict'])
            evidence={}
            for action,value in zip(agent.ORDER,row['outcomes']):
                if value < 0:
                    self.assertTrue(action.startswith('v2_'))
                    self.assertTrue(any(v.get('terminal') for a,v in evidence.items() if a.startswith('v2_')))
                    continue
                self.assertIn(action,agent.available(evidence))
                evidence[action]=s3.observation(action,value)
        rebuilt=build()
        # libm can change only the last bits of request-derived coordinates.
        for actual, expected in zip(rebuilt['groups'], self.model['groups']):
            self.assertEqual(len(actual['features']), len(expected['features']))
            for x, y in zip(actual['features'], expected['features']):
                self.assertLessEqual(abs(x-y), 1e-12)
            actual['features'] = expected['features']
        self.assertEqual(rebuilt,self.model)

    def test_hard_bayes_and_relevant_mapping(self):
        rows=[trace(),trace(False,{'normalization':1})]
        model=synthetic(rows)
        r=Request(1,1,.01)
        original=s3.Posterior(r,model)
        corrupted=s3.Posterior(r,s3.corrupt(model,'D'))
        normal=original.update('normalization',{'verdict':'accept'})
        wrong=corrupted.update('normalization',{'verdict':'accept'})
        self.assertEqual(normal['support_count'],1)
        self.assertEqual(normal['raw_feature_violation_mass']['normalization'],0)
        self.assertEqual(wrong['raw_feature_violation_mass']['normalization'],1)
        self.assertAlmostEqual(sum(normal['full_probabilities'].values()),1)
        with self.assertRaises(ValueError): original.update('normalization',{'verdict':'accept'})

    def test_support_expansion_and_package_fallback(self):
        rows=[trace(),trace(False,{'normalization':1})]
        rows[1]['group_index']=1
        model=synthetic(rows,1,[dict(id='near',features=[0.]*6),dict(id='far',features=[10.]*6)])
        r=Request(1,1,.01)
        with patch('attention.s3.s2.features',return_value=([0.]*6,{})):
            schedule=updated_batch.Scheduler([r,r],model)
        first=schedule.update(0,'normalization',{'verdict':'violation_detected'})
        self.assertTrue(first['expanded_now'])
        self.assertFalse(schedule.fallback)
        # This synthetic unsupported result forces all empirical support empty.
        failed=schedule.update(1,'binding',{'verdict':'violation_detected'})
        self.assertTrue(schedule.fallback)
        self.assertIsNone(failed['full_probabilities'])
        self.assertIsNotNone(schedule.fallback_trigger)
        schedule2=updated_batch.Scheduler([r,r],model)
        schedule2.update(0,'modal',{'verdict':'inconclusive'})
        self.assertTrue(schedule2.fallback)
        selected=schedule2.choose(1000.)
        candidates=[(i,a) for i,e in enumerate(schedule2.evidence) for a in agent.available(e)]
        cheap=min(candidates,key=lambda p:(schedule2.prices[p[0]][p[1]],p[0],s3.INDEX[p[1]]))
        self.assertEqual(selected[:2],cheap)
        i,a,_=selected
        later=schedule2.update(i,a,{'verdict':'accept'})
        self.assertIsNone(later['full_probabilities'])
        self.assertTrue(later['fallback_active'])
        self.assertEqual(schedule2.choose(1000.)[2]['kind'],'C2_FALLBACK')

    def test_decisive_v2_requires_terminal(self):
        with self.assertRaises(ValueError):
            s3.token('v2_512', {'verdict':'accept','terminal':False})

    def test_missing_v2_is_excluded_not_imputed(self):
        model=synthetic([trace(terminal=512),trace(terminal=1024)])
        p=s3.Posterior(Request(1,1,.01),model)
        evidence={}
        for action in ['modal','v2_64','v2_128','v2_256','v2_512']:
            obs={'verdict':'accept'} if action=='modal' else {'verdict':'inconclusive','terminal':False}
            evidence[action]=obs;p.update(action,obs)
        self.assertEqual(len(p.indices),1)
        self.assertEqual(p.indices.tolist(),[1])
        branches=updated_batch._branches(evidence,p.indices,'v2_1024',p.runtime)
        self.assertEqual(branches[0][0],0)
        with self.assertRaises(AssertionError):
            updated_batch._branches(evidence,np.asarray([0]),'v2_1024',p.runtime)

    def test_conditional_branches_choose_different_continuations(self):
        rows=[trace(False,{'normalization':1}),trace(False,{'modal':2,'high_frequency':1})]
        model=synthetic(rows)
        posterior=s3.Posterior(Request(1,1,.01),model)
        plans=updated_batch.Plans({},posterior,{a:1. for a in agent.ORDER})
        tree=next(t for t in plans.trees if t['action']=='modal')
        choices=[]
        for branch in tree['branches']:
            best=min([o for o in branch['options'] if o['expected_gain']>0],key=lambda o:(-o['score'],*updated_batch._tie(o)))
            choices.append(best['action'])
        self.assertEqual(choices,['normalization','high_frequency'])
        best=plans.best(2.)
        self.assertLessEqual(best['expected_cost'],2.)

    def test_budget_envelope_rechecks_thresholds(self):
        model=synthetic([trace(False,{'normalization':1}),trace(False,{'modal':2,'high_frequency':1})])
        p=s3.Posterior(Request(1,1,.01),model)
        plan=updated_batch.Plans({},p,{a:1. for a in agent.ORDER})
        for budget in [20.,2.,1.,.99,2.,20.]:
            choice=plan.best(budget)
            if budget<1:self.assertIsNone(choice)
            else:self.assertLessEqual(choice['expected_cost'],budget)

    def test_corruption_equal_and_irrelevant_invariance(self):
        d,dp=s3.corrupt(self.model,'D'),s3.corrupt(self.model,'Dprime')
        audit=s3.corruption_audit(self.model,d,dp)
        self.assertEqual(audit['D_changed_numbers'],4)
        self.assertEqual(audit['D_L1'],4)
        self.assertTrue(audit['equal_changes'])
        self.assertTrue(audit['Dprime_active_unchanged'])
        model=synthetic([trace(),trace(False,{'normalization':1})])
        runs=[]
        for policy,m in [('B2',model),('Dprime',s3.corrupt(model,'Dprime'))]:
            run=updated_batch.run_batch([FakeExecutor(trace()),FakeExecutor(trace(False,{'normalization':1}))],policy,1000.,m)
            run.pop('policy');run.pop('planning_seconds');runs.append(run)
        self.assertEqual(*runs)
        for item in runs[0]['inputs']:
            if item['verdict']=='accept':
                self.assertTrue(all(a in item['observed'] for a in ('modal','binding',*agent.FEATURES,'collocation')))
        self.assertEqual(runs[0]['spent'],sum(a['cost'] for a in runs[0]['actions']))
        with self.assertRaises(ValueError):updated_batch.run_batch([],'B2',-1.,model)

    def test_scheduler_boundary_excludes_hidden_values(self):
        self.assertEqual(list(inspect.signature(updated_batch.Scheduler).parameters),['requests','model'])
        self.assertEqual(list(inspect.signature(updated_batch.Scheduler.choose).parameters),['self','remaining'])
        self.assertEqual(list(inspect.signature(updated_batch.Scheduler.update).parameters),['self','index','action','observed'])


if __name__=='__main__':unittest.main()
