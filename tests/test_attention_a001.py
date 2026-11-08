"""Fixed development checks; never draw A001 held-out tasks in the test suite."""
import copy
import inspect
import json
import tempfile
import subprocess
import sys
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
        self.assertFalse(compare(original,changed)['passed'])

    def test_comparison_rejects_flags_and_missing_structure(self):
        observed=dict(verdict='inconclusive',terminal=False,step=dict(converging=False,futile=False))
        result=dict(verdict='inconclusive',spent=1,correct=False,
            actions=[dict(action='v2_512',observation=observed)])
        row=dict(task_id=0,condition='clean',accurate=True,
            full=dict(verdict='inconclusive',spent=1),policies={'1.0':{'B':result}})
        original=dict(runs=[row])
        for alteration in ('flag','missing_step','missing_action','added_key'):
            changed=copy.deepcopy(original)
            action=changed['runs'][0]['policies']['1.0']['B']['actions'][0]
            if alteration=='flag':action['observation']['step']['converging']=True
            elif alteration=='missing_step':del action['observation']['step']
            elif alteration=='missing_action':changed['runs'][0]['policies']['1.0']['B']['actions']=[]
            else:action['observation']['new_diagnostic']=1.
            comparison=compare(original,changed)
            self.assertFalse(comparison['numeric_diagnostics_pass'])
            if alteration!='added_key':self.assertFalse(comparison['decisions_pass'])

    def test_comparison_float_drift_is_informational_but_costs_are_exact(self):
        row=dict(task_id=0,condition='clean',accurate=True,
            full=dict(verdict='accept',spent=1.,actions=[dict(action='modal',cost=1.)]),policies={'1.0':{'B':dict(
                verdict='accept',spent=1.,correct=True,actions=[dict(
                    action='modal',cost=1.,observation=dict(value=2.,zero=0.))])}})
        original=dict(runs=[row])
        changed=copy.deepcopy(original)
        changed['runs'][0]['policies']['1.0']['B']['actions'][0]['observation']['value']=4.
        comparison=compare(original,changed)
        self.assertTrue(comparison['passed'])
        self.assertFalse(comparison['numeric_diagnostics_pass'])
        self.assertTrue(comparison['numeric_diagnostics_informational'])
        metrics=comparison['numeric_diagnostic_summary']['$[].policies.1.0.B.actions[].observation.value']
        self.assertEqual(metrics,dict(count=1,changed=1,max_abs=2.,max_rel=.5,nonfinite=0))
        zero=comparison['numeric_diagnostic_summary']['$[].policies.1.0.B.actions[].observation.zero']
        self.assertEqual(zero['max_rel'],0.)
        for field in ('spent','cost','full_cost','full_action_cost','flag_type','integer','structure','list_length','numeric_type'):
            changed=copy.deepcopy(original)
            policy=changed['runs'][0]['policies']['1.0']['B']
            if field=='spent':policy['spent']+=1e-10
            elif field=='cost':policy['actions'][0]['cost']+=1e-10
            elif field=='full_cost':changed['runs'][0]['full']['spent']+=1e-10
            elif field=='full_action_cost':changed['runs'][0]['full']['actions'][0]['cost']+=1e-10
            elif field=='flag_type':policy['correct']=1
            elif field=='integer':changed['runs'][0]['task_id']=1
            elif field=='list_length':changed['runs'].append(copy.deepcopy(row))
            elif field=='numeric_type':policy['actions'][0]['observation']['value']=2
            else:policy['actions'][0]['observation']['extra']=0.
            self.assertFalse(compare(original,changed)['passed'],field)

    def test_comparison_cli_uses_decisions_and_structure(self):
        row=dict(task_id=0,condition='clean',accurate=True,
            full=dict(verdict='accept',spent=1.,diagnostic=1.),policies={})
        with tempfile.TemporaryDirectory() as tmp:
            first,second,output=[Path(tmp)/name for name in ('first.json','second.json','out.json')]
            first.write_text(json.dumps(dict(runs=[row])))
            row['full']['diagnostic']=2.
            second.write_text(json.dumps(dict(runs=[row])))
            command=[sys.executable,'-m','attention.a001','--compare',str(first),str(second),'--output',str(output)]
            run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertFalse(json.loads(output.read_text())['numeric_diagnostics_pass'])
            output.unlink()
            row['full']['new_flag']=False
            second.write_text(json.dumps(dict(runs=[row])))
            run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,1,run.stderr)
            self.assertFalse(json.loads(output.read_text())['passed'])

    def test_collocation_below_floor_not_skipped(self):
        r=Request(1,1,1e-7)
        observed={'modal':{'verdict':'inconclusive'}}
        self.assertIn('collocation',agent.available(observed))
        self.assertEqual(agent.verdict(observed),'inconclusive')


if __name__=='__main__':unittest.main()
