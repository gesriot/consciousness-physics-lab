"""Fixed development checks; never draw A001 held-out tasks in the test suite."""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from attention import agent
from attention.a001 import compare, decision_view, intake, validate
from lab import frozen_pipeline as P
from lab.executor import execute, request_hash
from lab.guard import Request
from lab.replay import canonical_sha256
from lab.scientist import ROOT


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads((ROOT/'attention/self_model.json').read_text())
        cls.request=Request(1+1j,1,.01)
        cls.previous=execute(Request(1,1,.01),P.plan_modes(Request(1,1,.01))).value

    def executor(self,condition='clean'):
        return agent.Executor(self.request,request_hash(self.request),condition,self.previous)

    def test_separate_probes_match_frozen_checks(self):
        for model in ('sphere','model_scale','slab','mixture'):
            checks=P.checks_from_probes(P.model_probes(self.request,model),'x','x')
            for name in agent.FEATURES:
                self.assertEqual(agent.feature(self.request,name,model)['verdict']=='accept',checks[name])

    def test_only_selected_probe_calls(self):
        with patch.object(P,'probe_response',wraps=P.probe_response) as probe:
            agent.feature(self.request,'normalization','sphere')
            self.assertEqual(probe.call_count,2)
            self.assertTrue(all(c.args[1]==0 for c in probe.call_args_list))

    def test_full_path_matches_strict_for_all_conditions(self):
        from lab.pipeline_holdout import CONDITIONS
        for c in CONDITIONS:
            e=self.executor(c)
            full=agent.run_policy(e,'C1',float('inf'),self.model,list(agent.ORDER),True)
            expected=P.evaluate(self.request,request_hash(self.request),e.outcome,e.model)
            self.assertEqual(full['verdict'],expected['policies']['strict']['verdict'])
            for a in full['actions']:
                if a['action'].startswith('v2_') and 'step' in a['observation']:
                    matching=[s for s in expected['independent']['steps'] if s['last_grid']==int(a['action'][3:])]
                    self.assertEqual(a['observation']['step'],matching[0])

    def test_dependencies_and_terminal_v2(self):
        self.assertNotIn('v2_64',agent.available({}))
        self.assertNotIn('collocation',agent.available({}))
        e=self.executor()
        with self.assertRaises(ValueError):e.perform('collocation')
        e.perform('modal')
        with self.assertRaises(ValueError):e.perform('v2_128')
        with self.assertRaises(ValueError):e.perform('modal')
        observed={'modal':{'verdict':'accept'},'v2_64':{'terminal':True,'verdict':'inconclusive'}}
        self.assertFalse(any(a.startswith('v2_') for a in agent.available(observed)))

    def test_budget_and_safe_incompleteness(self):
        for p in agent.POLICIES:
            for budget in (0,.01,.1,1.,3.,10.,20.):
                result=agent.run_policy(self.executor(),p,budget,agent.corrupt(self.model,p),list(agent.ORDER))
                self.assertLessEqual(result['spent'],budget)
                if result['verdict']=='accept':
                    names={a['action'] for a in result['actions']}
                    self.assertTrue(set(('modal','binding','collocation')+agent.FEATURES)<=names)
        self.assertEqual(agent.verdict({}),'inconclusive')
        self.assertEqual(agent.run_policy(self.executor(),'B',0,self.model,list(agent.ORDER))['verdict'],'inconclusive')

    def test_empirical_false_accept_and_subset(self):
        from lab.scientist import analytic
        for c in ('clean','near_tolerance','time_units','mixture'):
            e=self.executor(c)
            full=agent.run_policy(e,'C1',float('inf'),self.model,list(agent.ORDER),True)
            accurate=abs(complex(*e.outcome['value'])-analytic(self.request.z,1))<=self.request.tolerance
            for p in agent.POLICIES:
                for f in agent.FRACTIONS:
                    result=agent.run_policy(self.executor(c),p,f*full['spent'],agent.corrupt(self.model,p),list(agent.ORDER))
                    if result['verdict']=='accept':
                        self.assertEqual(full['verdict'],'accept')
                        self.assertTrue(accurate)

    def test_equal_corruption_size_and_direction(self):
        d,dp=agent.corrupt(self.model,'D'),agent.corrupt(self.model,'Dprime')
        def changes(x,y):
            if isinstance(x,dict):return [v for k in x for v in changes(x[k],y[k])]
            return [abs(x-y)] if isinstance(x,(int,float)) and x!=y else []
        self.assertEqual(changes(self.model,d),changes(self.model,dp))
        self.assertEqual(len(changes(self.model,d)),8)
        self.assertEqual(dp['profiles'],self.model['profiles'])
        for band,original in self.model['profiles'].items():
            self.assertGreaterEqual(d['profiles'][band]['normalization']['detection_probability'],original['normalization']['detection_probability'])
            self.assertLessEqual(d['profiles'][band]['high_frequency']['detection_probability'],original['high_frequency']['detection_probability'])

    def test_irrelevant_corruption_preserves_decisions(self):
        for budget in (1,3,10,20):
            b=agent.run_policy(self.executor('slab'),'B',budget,self.model,list(agent.ORDER))
            dp=agent.run_policy(self.executor('slab'),'Dprime',budget,agent.corrupt(self.model,'Dprime'),list(agent.ORDER))
            self.assertEqual(b,dp)

    def test_scheduler_interface_excludes_hidden_fields(self):
        self.assertEqual(list(inspect.signature(agent.choose).parameters),['policy','request','observed','remaining','model','permutation'])
        with patch.object(P,'model_response',side_effect=AssertionError('Should not evaluate hidden outcomes')):
            self.assertIn(agent.choose('B',self.request,{},10,self.model,list(agent.ORDER)),agent.available({}))

    def test_replay_rejects_wrong_experiment_and_bad_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            path.write_text(json.dumps(dict(experiment='E016',intake_channel=[])))
            with self.assertRaises(ValueError):intake(path)
        with self.assertRaises(ValueError):validate([])

    def test_comparison_detects_decisions_and_numbers(self):
        row=dict(task_id=0,condition='clean',accurate=True,full=dict(verdict='accept',spent=1),
            policies={'1.0':{'B':dict(verdict='accept',spent=1,correct=True,actions=[dict(action='declare',verdict='accept')])}})
        original=dict(runs=[row])
        changed=copy.deepcopy(original)
        self.assertTrue(compare(original,changed)['decisions_pass'])
        changed['runs'][0]['policies']['1.0']['B']['verdict']='inconclusive'
        self.assertFalse(compare(original,changed)['decisions_pass'])
        changed=copy.deepcopy(original)
        changed['runs'][0]['full']['spent']=2
        self.assertFalse(compare(original,changed)['numeric_diagnostics_pass'])

    def test_collocation_below_floor_not_skipped(self):
        r=Request(1,1,1e-7)
        observed={'modal':{'verdict':'inconclusive'}}
        self.assertIn('collocation',agent.available(observed))
        self.assertEqual(agent.verdict(observed),'inconclusive')


if __name__=='__main__':unittest.main()
