import copy
import unittest
from unittest.mock import patch
from attention import agent, s3
from attention.io import read_json
from attention.calibration_samples import request
from attention import channel_monitor as C


class MonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=read_json('attention/sample_library_a007.json')
        cls.model=s3.load_model()
        cls.weights=cls.model['action_model']['cost_weights']
        cls.recipe=next(s for s in cls.library['samples'] if s['channel']=='v2:2' and s['standard_kind']=='boundary')

    def spec(self,**kw):
        return dict(kind='none',target_channel='v2:2',onset_action=20,circle_phase=7,random_seed=20261204,**kw)

    def executor(self):
        r=request(self.recipe)
        from lab.executor import request_hash
        return agent.Executor(r,request_hash(r),'clean',0j)

    def test_live_boundary_and_local_hook(self):
        intact=C.sample_execution(self.recipe,self.weights)
        broken=C.sample_execution(self.recipe,self.weights,'underestimated')
        self.assertNotEqual(intact['verdict'],'accept')
        self.assertEqual(broken['verdict'],'accept')
        self.assertLessEqual(broken['cost'],C.maximum_cost(self.recipe,self.weights))

    def test_target_detection_and_timeline(self):
        spec=self.spec();spec.update(kind='underestimated',target_channel='v2:2',onset_action=0,circle_phase=2)
        out=C.run_batch([self.executor()],'C2A',200.,self.model,self.library,0,spec,.2)
        self.assertTrue(out['detection']['detected'])
        self.assertEqual(out['detection']['delay']['inputs'],0)
        self.assertEqual(out['detection']['delay']['samples'],1)
        self.assertTrue(out['alerts'][0]['target_detection'])
        self.assertFalse(out['detection']['preonset_target_alert'])
        self.assertEqual(out['alerts'][0]['channel'],'v2:2')

    def test_unseen_hook_on_synthetic_diagnostics_only(self):
        original=dict(verdict='violation_detected',terminal=True,step=dict(last_grid=512,converging=True,
            discrepancy=1.,uncertainty_estimate=.001,last_difference=.0001,previous_difference=.001,
            antecedent_ratio=.1,verdict='violation_detected',futile=False))
        out=C.hook(original,.1,'compressed')
        self.assertEqual(out['step']['discrepancy'],.01)
        self.assertEqual(out['step']['uncertainty_estimate'],.001)
        self.assertEqual(out['verdict'],'accept')
        self.assertEqual(original['step']['discrepancy'],1.)

    def test_pools_are_in_frozen_order(self):
        pools=C.sample_pools(self.library)
        self.assertEqual(set(pools),set(C.CHANNELS))
        self.assertEqual(pools['v2:2'][0]['standard_kind'],'boundary')
        for pool in pools.values():self.assertEqual(len(pool),len({s['sample_id'] for s in pool}))

    def test_equal_quotas_accounting_and_clean_alarm(self):
        for policy in ('C2A','C2R','BU','C2'):
            with self.subTest(policy=policy):
                out=C.run_batch([self.executor()],policy,200.,self.model,self.library,0,self.spec(),.2)
                self.assertEqual(out['sample_allocation']['executed'],0 if policy=='C2' else 2)
                self.assertEqual(out['spent'],sum(r['cost'] for r in out['actions']))
                self.assertEqual(out['spent'],out['sample_cost']+out['ordinary_cost'])
                self.assertLessEqual(out['spent'],out['budget'])
                self.assertFalse(out['alerts'])
                self.assertEqual(out['detection'],dict(detected=False,delay=None,censored=False,preonset_target_alert=False))
                counter=0
                for row in out['actions']:
                    self.assertEqual(row['counter_before'],counter)
                    counter+=int(not row['sample'] and not row['replacement'])
                    self.assertEqual(row['ordinary_after'],counter)
                self.assertEqual(counter,out['ordinary_actions'])
                self.assertIn('closure_counter',out['inputs'][0])
                self.assertIn('suspect_at_closure',out['inputs'][0])

    def test_circle_starts_at_registered_phase(self):
        out=C.run_batch([self.executor()],'C2A',200.,self.model,self.library,0,self.spec(),.2)
        samples=[r for r in out['actions'] if r['sample']]
        self.assertEqual([r['channel'] for r in samples],[C.CHANNELS[7],C.CHANNELS[8]])

    def test_dprime_and_last_bit_decisions(self):
        runs=[C.run_batch([self.executor()],p,200.,self.model,self.library,0,self.spec(),.2,n)
              for p,n in [('BA',False),('Dprime',False),('BA',True)]]
        def decisions(run):
            return [(r['input_index'],r['action'],r['channel'],r['sample'],r['replacement'],r['observation']['verdict'],r['cost']) for r in run['actions']]
        self.assertEqual(decisions(runs[0]),decisions(runs[1]))
        self.assertEqual(decisions(runs[0]),decisions(runs[2]))
        for row in runs[1]['actions']:
            if row.get('score_details'):
                audit=row['score_details']['intervention']
                self.assertEqual(audit['active']['changed'],0)
                self.assertEqual(audit['counterfactual_active'],audit['counterfactual_unused'])

    def test_guard_rejects_unstable_contraction(self):
        e=self.executor();e.perform('modal')
        step=dict(verdict='accept',discrepancy=.001,uncertainty_estimate=.001,rounding=1e-6,last_difference=1e-10,previous_difference=2.1e-10)
        with patch.object(C.P,'v2_step',return_value=step),patch.object(C,'uniform_response',return_value=.1+0j):
            result=C.guarded_uniform(e)
        self.assertEqual(result['verdict'],'inconclusive')
        self.assertFalse(result['safety']['passed'])

    def test_guard_requires_both_strict_margins(self):
        e=self.executor();e.perform('modal')
        tol=e.request.tolerance
        step=dict(verdict='accept',discrepancy=tol/4,uncertainty_estimate=tol/4,rounding=tol/100,last_difference=tol/10,previous_difference=tol)
        with patch.object(C.P,'v2_step',return_value=step),patch.object(C,'uniform_response',return_value=.1+0j):
            result=C.guarded_uniform(e)
        self.assertEqual(result['verdict'],'accept')
        self.assertTrue(result['safety']['passed'])

    def test_cached_scores_equal_reference_on_evolving_development_trace(self):
        pools=C.sample_pools(self.library)
        for policy in ('BA','BU','D','Dprime'):
            for noise in (False,True):
                with self.subTest(policy=policy,noise=noise):
                    executors=[self.executor(),self.executor()]
                    requests=[e.request for e in executors];evidence=[{},{}]
                    runtime=C.planning.Runtime(self.model)
                    posteriors=[C.planning.Posterior(r,runtime,True) for r in requests]
                    last={c:0 for c in C.CHANNELS};cache={};closed={};suspect=set()
                    for step in range(24):
                        args=(requests,evidence,posteriors,closed,self.weights,pools,last,step,policy,suspect,noise)
                        expected=C.score_channels(*args)
                        actual=C.score_channels(*args,cache=cache)
                        self.assertEqual(actual,expected)
                        # Reuse on the unchanged input table must be bit-identical.
                        self.assertEqual(C.score_channels(*args,cache=cache),expected)
                        if step%5==0:last[actual[0]]=step
                        if step==15:suspect.add(actual[0])
                        i=step%2;options=agent.available(evidence[i])
                        if options:
                            action=min(options,key=agent.ORDER.index)
                            observed=executors[i].perform(action)
                            observed=s3.observation(action,s3.token(action,observed))
                            evidence[i][action]=observed;posteriors[i].update(action,observed)
                        if step==20:closed[0]='inconclusive'
                    self.assertLessEqual(len(cache),len(requests))

    def test_last_bit_noise_preserves_structural_zero_risk(self):
        r=request(self.recipe);pools=C.sample_pools(self.library)
        posterior=C.planning.Posterior(r,C.planning.Runtime(self.model),True)
        args=([r],[{}],[posterior],{},self.weights,pools,{c:0 for c in C.CHANNELS},0,'D',set())
        for full,probability in ((0.,.8),(.7,0.),(0.,0.)):
            with self.subTest(full=full,probability=probability), \
                    patch.object(posterior,'snapshot',return_value={'full_probabilities':{'accept':full}}), \
                    patch.object(posterior,'forecast',return_value={'probabilities':[probability]}):
                clean=C.score_channels(*args)
                noisy=C.score_channels(*args,noise=True)
                self.assertEqual(noisy,clean)
                self.assertTrue(all(v==0. for v in noisy[2]['raw'].values()))
                self.assertEqual(noisy[2]['intervention']['active']['changed'],0)

    def test_cache_avoids_unchanged_forecasts(self):
        r=request(self.recipe);pools=C.sample_pools(self.library)
        posterior=C.planning.Posterior(r,C.planning.Runtime(self.model),True)
        args=([r],[{}],[posterior],{},self.weights,pools,{c:0 for c in C.CHANNELS},0,'BA',set())
        cache={}
        with patch.object(posterior,'forecast',wraps=posterior.forecast) as mocked:
            C.score_channels(*args,cache=cache)
            count=mocked.call_count
            self.assertEqual(count,len(C.FAMILIES))
            C.score_channels(*args,cache=cache)
            self.assertEqual(mocked.call_count,count)

    def test_usage_control_does_not_consult_model(self):
        r=request(self.recipe);pools=C.sample_pools(self.library)
        which,scores,detail=C.score_channels([r],[{}],None,{},self.weights,pools,{c:0 for c in C.CHANNELS},0,'BU',set())
        self.assertIn(which,C.dependencies(r))
        self.assertEqual(set(scores),set(C.CHANNELS))
        self.assertTrue(all(v>=0 for v in scores.values()))


if __name__=='__main__':unittest.main()
