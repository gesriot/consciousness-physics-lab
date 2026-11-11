"""Synthetic archive mutations; these tests generate no B003 experimental data."""
import copy
import unittest
from unittest.mock import patch
from binding import audit_b003 as A
from binding import self_checks_b003 as B
from binding.audit import AuditError, read
from lab.guard import Request
from lab.common_mode import encode
from lab.executor import request_hash


def fixture(policy='S',quota=1):
    request=Request(1,1,.01);fields=encode(request);outcome=dict(value=[1.,0.],calls=1,modes_spent=64,trace=[dict(call=1,modes=64)])
    model=read(A.ROOT/'attention/self_model_s3.json');weights=model['action_model']['cost_weights']
    library={'samples':[dict(sample_id=j,channel=c,family=c.split(':')[0],standard_kind='accept',expected='accept',request=fields,candidate=[1.,0.]) for j,c in enumerate(A.CHANNELS)]}
    intake=dict(rr_phase=6,requests=[dict(task_id=0,package_id=0,request=fields,request_sha256=request_hash(request),condition_order=['sphere']*9)])
    full=[dict(action=a,cost=A.native(a,fields,weights,outcome),observation={'verdict':'accept'}) for a in ('modal','binding')+A.FEATURES+('collocation','v2_64')]
    full.sort(key=lambda a:A.ORDER.index(a['action']))
    for a in full:
        if a['action'].startswith('v2_'):a['observation'].update(terminal=True,value=[1.,0.])
    full.append(dict(action='declare',cost=0.,verdict='accept'))
    infos=[dict(task_id=0,condition='sphere',outcome=copy.deepcopy(outcome),tolerance=.01,full_cost=sum(a['cost'] for a in full),full_actions=copy.deepcopy(full),clean_full_verdict='accept') for _ in range(9)]
    class Executor:
        def __init__(self):self.request=request;self.outcome=copy.deepcopy(outcome);self.observed={};self.values=[]
        def perform(self,a):
            if a.startswith('v2_'):self.values.append(1.+0j)
            result={'verdict':'accept'}
            if a.startswith('v2_'):result.update(terminal=True,value=[1.,0.])
            self.observed[a]=result;return copy.deepcopy(result)
    def sample(s,w,kind):
        obs={'verdict':'accept'};trace=[];cost=A.native(s['family'],fields,w) if s['family']!='v2' else 64*w['fv_cell']
        if s['family']=='v2':
            obs.update(terminal=True,value=[1.,0.]);trace=[dict(action='v2_64',cost=cost,original=obs,observed=obs)]
        return dict(verdict='accept',original=obs,observed=copy.deepcopy(obs),trace=trace,cost=cost)
    with patch.object(B.monitor,'sample_execution',sample),patch.object(B,'v2_observation',lambda *a:dict(verdict='accept',terminal=True,value=[1.,0.])):
        state=B.State(policy,quota,6);result=B.run_package([Executor() for _ in infos],infos,intake['requests'],0,sum(i['full_cost'] for i in infos),model,library,state,())
    archive=dict(package_id=0,policy=policy,quota=0 if policy=='N0' else quota,budget=result['budget'],inputs=infos,result=result)
    return archive,intake,dict(candidates={f'0:{i}':outcome for i in range(9)}),weights,library


class ArchiveAuditTests(unittest.TestCase):
    def audit(self,data):return A.check_package(*data)
    def test_all_policies_and_quotas_reconstruct(self):
        for policy in ('S','Sx','Sxprime','RR','BU','N0'):
            for quota in (1,2,4):
                with self.subTest(policy=policy,quota=quota):self.audit(fixture(policy,quota))

    def test_cost_selection_marker_queue_quota_mutations(self):
        for mutation in ('cost','selection','marker','queue','quota','verdict','graph','request_hash','paired_graph'):
            data=fixture();a=data[0];r=a['result'];sample=next(x for x in r['actions'] if x['kind']=='sample')
            if mutation=='cost':sample['cost']+=1e-6
            elif mutation=='selection':sample['channel']='v2:5'
            elif mutation=='marker':sample['selection_snapshot']['markers']['v2:0']=22
            elif mutation=='queue':r['state']['pending'].append({'record_id':'0:0'})
            elif mutation=='quota':a['quota']=2
            elif mutation=='verdict':r['inputs'][0]['verdict']='violation_detected'
            elif mutation=='graph':r['records'][0]['links'][0]['channel']='modal'
            elif mutation=='request_hash':r['records'][0]['links'][-1]['request_hash']='bad'
            else:sample['selection_snapshot']['paired_graphs']['Sxprime']['scores']['v2:0']=100
            with self.subTest(mutation=mutation),self.assertRaises(AuditError):self.audit(data)

    def test_comparison_ignores_guarded_off_raw_verdict_only(self):
        a={'verdict':'inconclusive','unguarded_verdict':'accept','convergence_pass':False,'decision_pass':True,'value':[1.,0.],'cost':1.,'count':2}
        b=copy.deepcopy(a);b['unguarded_verdict']='violation_detected';b['value'][0]+=1e-13
        self.assertTrue(A.compare_values(a,b)['decisions_pass'])
        for key,value in [('verdict','accept'),('convergence_pass',True),('cost',2.),('count',3)]:
            c=copy.deepcopy(b);c[key]=value;self.assertFalse(A.compare_values(a,c)['decisions_pass'])
        b['value'][0]=float('nan')
        with self.assertRaises(AuditError):A.compare_values(a,b)

    def test_truth_mutation_does_not_change_scheduler_but_fails_evaluator(self):
        data=fixture('N0');state=self.audit(data);archive,intake=data[:2];archive['evaluator']={}
        refs=[1.+0j]
        for r in state['records']:
            archive['evaluator'][r['id']]=dict(accurate=True,absolute_error=0.,reference=[1.,0.],affected_channels=[],package_id=0,input_index=r['input_index'])
        A.evaluator(archive,state,intake,refs,{})
        original=copy.deepcopy(archive['result']['actions']);archive['evaluator']['0:0']['accurate']=False
        with self.assertRaises(AuditError):A.evaluator(archive,state,intake,refs,{})
        self.assertEqual(original,archive['result']['actions'])

    def test_stakes_are_distinct_record_not_steps_and_control_is_null(self):
        data=fixture('N0');state=self.audit(data);rows=state['records'];markers=dict.fromkeys(A.CHANNELS,-1);prices=dict.fromkeys(A.CHANNELS,2.)
        rows[0]['links'].append(copy.deepcopy(rows[0]['links'][0]))
        base=A.score_stakes(rows,markers,prices)
        control,edits=A.projection(rows,markers,'Sxprime');self.assertEqual(base,A.score_stakes(control,markers,prices))
        active,ae=A.projection(rows,markers,'Sx');self.assertEqual([e['record_id'] for e in edits],[e['record_id'] for e in ae])
        for c in A.CHANNELS:
            if c.startswith('collocation:'):self.assertEqual(A.score_stakes(active,markers,prices)[0][c],0)

    def test_finite_measured_float_is_diagnostic_but_prices_are_exact(self):
        a={'probes':{'error':.01},'recomputed_error_estimate':.1,'costs':{'sample':1.},'scores':{'c':.5},'accepted':True}
        b=copy.deepcopy(a);b['probes']['error']+=1e-14;b['recomputed_error_estimate']+=1e-14
        self.assertTrue(A.compare_values(a,b)['decisions_pass'])
        b['costs']['sample']+=1e-14;self.assertFalse(A.compare_values(a,b)['decisions_pass'])
        b=copy.deepcopy(a);b['scores']['c']+=1e-9;self.assertFalse(A.compare_values(a,b)['decisions_pass'])

    def test_aggregate_primary_closure_mutation(self):
        from binding.run_b003 import aggregate
        policies=[('N0',0)]+[(p,q) for q in (1,2,4) for p in ('S','Sx','Sxprime','RR','BU')]
        row={}
        for env in ('whole','clean'):
            for p,q in policies:
                row[f'{env}-d4-{p}-q{q}']=dict(false_remaining=0,accurate_withdrawn=0,actual_cost=1.,initial_acceptances=1,correct_final=1,accurate_ordinary_never_obtained=0,detection_delay=4,post_onset_samples=3,detection_censored=True,corrupted_sample_fraction=0.,opportunity_cost=dict(gross_lost=0,gross_gained=0,net=0),clean_false_alarms=0)
        rows=[copy.deepcopy(row) for _ in range(12)];frozen={'single_channel':{'selected_packages':None}}
        report=aggregate(rows,frozen);A.check_aggregate(report,rows,[['whole',4],['clean',4]])
        for mutation in ('closure','count','price','prediction'):
            changed=copy.deepcopy(report)
            if mutation=='closure':changed['closure']['graph_stakes_unnecessary']=False
            elif mutation=='count':changed['cells']['whole-d4-S-q1']['false_remaining']['total']=1
            elif mutation=='price':changed['dominance']['whole-d4-q1']['S_minus_RR_actual_cost']=1.
            else:changed['predictions']['single_channel_status']='tested'
            with self.subTest(mutation=mutation),self.assertRaises(AuditError):A.check_aggregate(changed,rows,[['whole',4],['clean',4]])

    def test_queue_orders_newest_and_deduplicates(self):
        rows=[dict(id=str(i),standing=True,time=2*i+2,links=[dict(kind='channel',channel='collocation:0',time=2*i+1)]) for i in range(4)]
        self.assertEqual(A.queue_for(rows,'collocation:0',-1,20,{'2'}),['3','1','0'])
        self.assertEqual(A.queue_for(rows,'collocation:0',5,20,set()),['3'])

    def test_request_endpoint_and_ambiguous_root_rejected(self):
        data=fixture('N0');state=self.audit(data);r=copy.deepcopy(state['records'][0]);requests={0:data[1]['requests'][0]}
        r['links'][-1]['request_id']=1
        with self.assertRaises(AuditError):A.request_root(r,requests)
        r=copy.deepcopy(state['records'][0]);r['links'].append(copy.deepcopy(r['links'][-1]))
        with self.assertRaises(AuditError):A.request_root(r,requests)


if __name__=='__main__':unittest.main()
