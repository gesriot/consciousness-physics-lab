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


def fixture(policy='S',quota=1,noise=False):
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
        state=B.State(policy,quota,6);result=B.run_package([Executor() for _ in infos],infos,intake['requests'],0,sum(i['full_cost'] for i in infos),model,library,state,(),noise=noise)
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


class PortabilityVerificationTests(unittest.TestCase):
    def case(self):
        old_text={A.COMPATIBLE_SOURCES[0]:"import math\ndef science(x):\n    return x+1\ndef run():\n    return 'old-path'\n",A.COMPATIBLE_SOURCES[1]:"def check_package(x):\n    return x+1\ndef audit_directory():\n    return 'old-path'\n"}
        new_text={p:v.replace('old-path','new-path') for p,v in old_text.items()}
        old={p:A.byte_sha(v.encode()) for p,v in old_text.items()};new={p:A.byte_sha(v.encode()) for p,v in new_text.items()}
        old['binding/self_checks_b003.py']=new['binding/self_checks_b003.py']='fixed-core'
        record=dict(schema='B003-path-verification-compatibility-v1',old_head=A.EXECUTED_BASELINE,allowed_changes={p:dict(old_sha256=old[p],new_sha256=new[p]) for p in A.COMPATIBLE_SOURCES},unchanged_sources_sha256={'binding/self_checks_b003.py':'fixed-core'},unchanged_scientific_functions=dict(runner=A.unchanged_function_proof(old_text[A.COMPATIBLE_SOURCES[0]],new_text[A.COMPATIBLE_SOURCES[0]],A.RUNNER_METADATA_FUNCTIONS,True),auditor=A.unchanged_function_proof(old_text[A.COMPATIBLE_SOURCES[1]],new_text[A.COMPATIBLE_SOURCES[1]],A.AUDIT_METADATA_FUNCTIONS)))
        return record,old,new,old_text,new_text

    def test_exact_bounded_source_transition(self):
        A.validate_compatibility_record(*self.case())

    def test_unmapped_source_record_tamper_and_scientific_edit_fail(self):
        for kind in ('core','unknown_source','pair','proof','baseline','science','module'):
            record,old,new,ot,nt=self.case()
            if kind=='core':new['binding/self_checks_b003.py']='edited-core'
            elif kind=='unknown_source':new['lab/new.py']='extra'
            elif kind=='pair':record['allowed_changes'][A.COMPATIBLE_SOURCES[0]]['new_sha256']='forged'
            elif kind=='proof':record['unchanged_scientific_functions']['runner']['science']['text_sha256']='forged'
            elif kind=='baseline':record['old_head']='wrong'
            else:
                path=A.COMPATIBLE_SOURCES[0]
                nt[path]=nt[path].replace('x+1','x+2') if kind=='science' else nt[path].replace('import math','import sys')
                new[path]=A.byte_sha(nt[path].encode());record['allowed_changes'][path]['new_sha256']=new[path]
            with self.subTest(kind=kind),self.assertRaises(AuditError):A.validate_compatibility_record(record,old,new,ot,nt)

    def test_record_file_must_match_committed_bytes(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/A.COMPATIBILITY;path.parent.mkdir(parents=True);path.write_text('{"tampered":true}')
            with patch.object(A,'ROOT',root),patch.object(A,'git_bytes',return_value=b'{"original":true}'):
                with self.assertRaises(AuditError):A.committed_compatibility()

    def test_missing_development_overlap_source_fails(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);development=root/'binding/results/B003/development-first';sources={}
            for i in range(12):
                path=development/f'series-{i:02d}/intake.json';path.parent.mkdir(parents=True);path.write_text('{"requests":[]}')
                sources[str(path.relative_to(root))]=A.byte_sha(path.read_bytes())
            manifest=dict(development_directory=str(development.relative_to(root)),overlap={'sources':sources})
            with patch.object(A,'ROOT',root):
                self.assertEqual(len(A.verified_overlap_sources(manifest,development)),12)
                del sources[next(iter(sources))]
                with self.assertRaises(AuditError):A.verified_overlap_sources(manifest,development)

    def test_relocated_relative_and_legacy_absolute_sources(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);development=root/'binding/results/B003/development-first';development.mkdir(parents=True);(development/'freeze.json').write_text('{"freeze":"exact"}')
            archive=development.parent/'confirmation-first';archive.mkdir()
            manifest=dict(development_directory='binding/results/B003/development-first',freeze_sha256=A.byte_sha((development/'freeze.json').read_bytes()))
            with patch.object(A,'ROOT',root):
                self.assertEqual(A.resolve_development(archive,manifest),development.resolve())
                self.assertEqual(A.overlap_path(manifest['development_directory']+'/series-00/intake.json',manifest,development),(development/'series-00/intake.json').resolve())
                manifest['development_directory']='/old-machine/binding/results/B003/development-first'
                self.assertEqual(A.resolve_development(archive,manifest),development.resolve())
                self.assertEqual(A.overlap_path(manifest['development_directory']+'/series-00/intake.json',manifest,development),(development/'series-00/intake.json').resolve())
                with self.assertRaises(AuditError):A.overlap_path('/unrelated/private/intake.json',manifest,development)
                manifest['freeze_sha256']='wrong'
                with self.assertRaises(AuditError):A.resolve_development(archive,manifest)


    def test_repository_under_symlink_uses_one_canonical_path_form(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            container=Path(temporary);physical=container/'real-repository';physical.mkdir()
            linked=container/'linked-repository';linked.symlink_to(physical,target_is_directory=True)
            development=linked/'binding/results/B003/development-first';development.mkdir(parents=True)
            (development/'freeze.json').write_text('{"freeze":"exact"}')
            archive=development.parent/'confirmation-first';archive.mkdir()
            relative='binding/results/B003/development-first'
            for i in range(12):
                source=development/f'series-{i:02d}/intake.json';source.parent.mkdir();source.write_text('{"requests":[]}')
            for old_root in (relative,'/old-machine/'+relative):
                sources={old_root+f'/series-{i:02d}/intake.json':A.byte_sha((development/f'series-{i:02d}/intake.json').read_bytes()) for i in range(12)}
                manifest=dict(development_directory=old_root,freeze_sha256=A.byte_sha((development/'freeze.json').read_bytes()),overlap={'sources':sources})
                with self.subTest(old_root=old_root),patch.object(A,'ROOT',linked):
                    self.assertEqual(A.resolve_development(archive,manifest),development.resolve())
                    self.assertEqual(A.resolve_development(archive,manifest,development),development.resolve())
                    paths=A.verified_overlap_sources(manifest,development)
                    self.assertEqual(set(paths),{(development/f'series-{i:02d}/intake.json').resolve() for i in range(12)})
                    self.assertTrue(all(path.is_relative_to(physical.resolve()) for path in paths))


class NoiseVerifierCorrectionTests(unittest.TestCase):
    def test_registered_half_even_boundary(self):
        self.assertEqual(A.score_round(26.5517578125,False),26.551757812)
        self.assertEqual(A.score_round(26.5517578125,True),26.551757813)
        data=fixture('BU',4,True)
        A.check_package(*data,noise=True)
        sample=next(a for a in data[0]['result']['actions'] if a['kind']=='sample')
        sample['selection_snapshot']['scores']['modal']+=1e-9
        with self.assertRaises(AuditError):A.check_package(*data,noise=True)

    def test_noise_stakes_reconstruct_all_paired_policies(self):
        for policy in ('S','Sx','Sxprime','RR','BU'):
            for quota in (1,2,4):
                A.check_package(*fixture(policy,quota,True),noise=True)

    def test_score_comparison_requires_both_independent_cell_checks(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            roots=[Path(temporary)/name for name in ('first','noise')]
            for index,root in enumerate(roots):
                target=root/'series-00/cell/package-00.json.gz';target.parent.mkdir(parents=True)
                import gzip
                with gzip.open(target,'wt') as stream:json.dump(dict(result={'actions':[dict(kind='sample',scores={'modal':26.551757812+index*1e-9},channel='normalization',cost=1.,count=2)]}),stream)
                (root/'manifest.json').write_text(json.dumps(dict(source_sha256={'same':'source'},intake_sha256='intake')))
            with patch.object(A,'verify_sources',return_value={}),patch.object(A,'verify_score_cell',return_value=1) as check:
                result=A.compare_directories(*roots)
                self.assertTrue(result['decisions_pass']);self.assertEqual(check.call_count,2)
                self.assertEqual(result['independently_verified_score_cells'],2)
            with patch.object(A,'verify_sources',return_value={}),patch.object(A,'verify_score_cell',side_effect=AuditError('tampered score')):
                with self.assertRaises(AuditError):A.compare_directories(*roots)

    def test_score_projection_keeps_channel_shape_and_protected_fields(self):
        base=dict(kind='sample',scores={'a':1.,'b':2.},channel='a',verdict='accept',cost=3.,count=1)
        def projected(action):return A.score_diagnostic_projection({'result':{'actions':[action]}})
        numerical=copy.deepcopy(base);numerical['scores']['a']+=1e-9
        self.assertEqual(projected(base),projected(numerical))
        for key,value in [('channel','b'),('verdict','inconclusive'),('cost',4.),('count',2),('scores',{'a':1.})]:
            altered=copy.deepcopy(base);altered[key]=value
            self.assertNotEqual(projected(base),projected(altered))
        with self.assertRaises(AuditError):A.verify_score_cell('/not-read','summary.json')

    def test_unregistered_nested_score_changes_remain_exact(self):
        known={'result':{'actions':[{'kind':'sample','scores':{'a':1.},'selection_snapshot':{'scores':{'a':1.},'paired_graphs':{'S':{'scores':{'a':1.}},'Sx':{'scores':{'a':1.}},'Sxprime':{'scores':{'a':1.}}}}}]}}
        modified=copy.deepcopy(known)
        action=modified['result']['actions'][0];action['scores']['a']=2.;action['selection_snapshot']['scores']['a']=2.
        for paired in action['selection_snapshot']['paired_graphs'].values():paired['scores']['a']=2.
        self.assertEqual(A.score_diagnostic_projection(known),A.score_diagnostic_projection(modified))
        for location in ('root','action_metadata','unknown_pair','ordinary_action'):
            left=copy.deepcopy(known);right=copy.deepcopy(known)
            if location=='root':left['metadata']={'scores':{'a':1.}};right['metadata']={'scores':{'a':2.}}
            elif location=='action_metadata':
                left['result']['actions'][0]['metadata']={'scores':{'a':1.}};right['result']['actions'][0]['metadata']={'scores':{'a':2.}}
            elif location=='ordinary_action':
                left['result']['actions'][0]['kind']='ordinary';right['result']['actions'][0]['kind']='ordinary';right['result']['actions'][0]['scores']['a']=2.
            else:
                left['result']['actions'][0]['selection_snapshot']['paired_graphs']['unknown']={'scores':{'a':1.}};right['result']['actions'][0]['selection_snapshot']['paired_graphs']['unknown']={'scores':{'a':2.}}
            with self.subTest(location=location):self.assertNotEqual(A.score_diagnostic_projection(A.decision_projection(left)),A.score_diagnostic_projection(A.decision_projection(right)))

    def test_exact_correction_record_rejects_core_and_pair_tampering(self):
        record,old,new,ot,nt=PortabilityVerificationTests().case()
        record['schema']='B003-noise-verifier-correction-v1';record['old_head']=A.CORRECTION_BASELINE
        record['unchanged_scientific_functions']['auditor']=A.unchanged_function_proof(ot[A.COMPATIBLE_SOURCES[1]],nt[A.COMPATIBLE_SOURCES[1]],A.CORRECTED_AUDIT_FUNCTIONS)
        A.validate_verifier_correction(record,old,new,ot,nt)
        damaged=copy.deepcopy(record);damaged['allowed_changes'][A.COMPATIBLE_SOURCES[1]]['new_sha256']='wrong'
        with self.assertRaises(AuditError):A.validate_verifier_correction(damaged,old,new,ot,nt)
        changed=copy.deepcopy(new);changed['binding/self_checks_b003.py']='changed-core'
        with self.assertRaises(AuditError):A.validate_verifier_correction(record,old,changed,ot,nt)


if __name__=='__main__':unittest.main()
