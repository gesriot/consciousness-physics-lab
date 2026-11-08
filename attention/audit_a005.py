"""Independent accounting and evidence audit of saved A005 action streams.

This auditor does not call the experiment scheduler, monitor, verdict, or summary
implementation. Physical values require the separate fresh execution repeat.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from attention import s3
from attention.io import read_json, write_json
from lab.guard import Request
from lab.replay import canonical_sha256

FEATURES=('normalization','first_moment','second_moment','high_frequency')
GRIDS=(64,128,256,512,1024,2048,4096)
ORDER=('modal',)+tuple('v2_'+str(n) for n in GRIDS)+('binding',)+FEATURES+('collocation',)
POLICIES=('C2','B','BM','D','Dprime')
FRACTIONS=('0.2','0.4','0.6','1.0')
FAULTS=('none','permissive','uninformative')
TARGETS=('collocation','second_moment')


def base_verdict(evidence):
    if any(v['verdict']=='violation_detected' for v in evidence.values()):
        return 'violation_detected'
    required=('modal','binding')+FEATURES+('collocation',)
    terminal=[v for a,v in evidence.items() if a.startswith('v2_') and v.get('terminal')]
    if all(evidence.get(a,{}).get('verdict')=='accept' for a in required) and terminal and terminal[-1]['verdict']=='accept':
        return 'accept'
    return 'inconclusive'


def verdict(evidence,extras,suspect):
    base=base_verdict(evidence)
    if base!='accept':return base
    if any(v['verdict']=='violation_detected' for v in extras.values()):return 'violation_detected'
    if all(extras.get(a,{}).get('verdict')=='accept' for a in suspect):return 'accept'
    return 'inconclusive'


def available(evidence):
    out=[a for a in ('modal','binding')+FEATURES if a not in evidence]
    if 'modal' not in evidence:return out
    if 'collocation' not in evidence:out.append('collocation')
    for n in GRIDS:
        a='v2_'+str(n)
        if a in evidence:
            if evidence[a].get('terminal'):break
        else:
            out.append(a);break
    return out


def price(action,request,weights,replacement=False):
    if replacement and action=='collocation':return 61440*weights['fv_cell']
    if action=='modal':
        # Frozen allocation law: the archive carries the selected modal count.
        from lab.frozen_pipeline import plan_modes
        return plan_modes(request)*weights['modal_term']
    if action.startswith('v2_'):return int(action[3:])*weights['fv_cell']
    if action=='binding':return weights['binding']
    if action in ('normalization','first_moment'):return 4096*weights['modal_term']+512*weights['fv_cell']
    if action=='second_moment':return 8192*(weights['modal_term']+weights['fv_cell'])
    if action=='high_frequency':return 4096*(weights['modal_term']+weights['fv_cell'])
    if action=='collocation':return (64**3+128**3)*weights['collocation_degree_cubed']
    raise ValueError(action)


def category(action,original,spoiled):
    result={'verdict':'accept' if spoiled else original['verdict']}
    if action.startswith('v2_'):result['terminal']=bool(original.get('terminal'))
    return result


class Audit:
    def __init__(self):self.checks=0;self.errors=[]
    def check(self,condition,message):
        self.checks+=1
        if not condition and len(self.errors)<100:self.errors.append(message)
    def equal(self,actual,expected,message):self.check(actual==expected,message)
    def near(self,actual,expected,message):
        self.check(isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),message)


def audit(data):
    check=Audit();check.equal(data['experiment'],'A005','experiment')
    records=data['intake_channel'];packages=data['packages'];onset=data['onset_package']
    check.equal(len(records),60,'request count');check.equal(len(packages),12,'package count')
    check.equal(onset,int(np.random.default_rng(20261144).integers(4,9)),'onset')
    check.equal(data['intake_sha256'],canonical_sha256([records,data['global_permutations']]),'intake digest')
    check.equal(data['full_pipeline_parity_failures'],[],'full strict parity')
    model=s3.load_model();check.equal(data['self_model_sha256'],canonical_sha256(model),'frozen model digest')
    runtime=s3.Runtime(model);weights=model['action_model']['cost_weights'];forecasts={};requests={}
    for index,record in enumerate(records):
        fields=dict(record['request']);fields['z']=complex(*fields['z']);request=Request(**fields);requests[index]=request
        neighbors,_=runtime.neighbors(request);ids=np.flatnonzero(np.isin(runtime.groups,neighbors));forecasts[index]={}
        for a in ORDER:
            raw=runtime.raw[ids,s3.INDEX[a]];present=raw[raw>=0];k=4 if a.startswith('v2_') else 3 if a in ('modal','collocation') else 2
            p=[float((.98*np.count_nonzero(present==j)+.02/k*len(present))/len(present)) if len(present) else 1/k for j in range(k)]
            if data.get('last_bit_noise'):p=[float(np.nextafter(v,np.inf if j%2 else -np.inf)) if v>0 else v for j,v in enumerate(p)]
            forecasts[index][a]=p
    threshold=data['calibration']['threshold'];check.near(threshold,data['calibration']['maximum']+1,'threshold')
    streams={};derived=defaultdict(list)
    for package in packages:
        package_id=package['package_id'];infos=package['inputs'];check.equal(len(infos),45,f'{package_id}: input count')
        full_cost=sum(sum(row['cost'] for row in x['full']['actions']) for x in infos)
        check.near(package['reference_cost'],full_cost,f'{package_id}: full cost')
        for x in infos:
            error=abs(complex(*x['outcome']['value'])-complex(*x['reference']))
            check.near(x['absolute_error'],error,'truth error');check.equal(x['accurate'],error<=x['tolerance'],'truth classification')
        for fault in FAULTS:
            for f in FRACTIONS:
                runs=package['conditions'][fault][f]
                for policy in POLICIES:
                    run=runs[policy];tag=f'{package_id}/{fault}/{f}/{policy}'
                    state=streams.setdefault((fault,f,policy),{'sums':{},'suspect':set()})
                    sums=state['sums'];suspect=state['suspect'];active=policy in ('BM','D','Dprime')
                    evidence=[{} for _ in infos];extras=[{} for _ in infos];closed={};spent_by=[0.]*len(infos);spent=0.;extra_cost=0.;touched=set();target_count=0;history=[];alerts=[]
                    target='second_moment' if fault=='uninformative' else 'collocation'
                    def close_ready():
                        for i,e in enumerate(evidence):
                            if i in closed:continue
                            v=base_verdict(e)
                            if v=='violation_detected':closed[i]=v
                            elif v=='accept':
                                r=verdict(e,extras[i],suspect)
                                if r!='inconclusive' or not (suspect-set(extras[i])):closed[i]=r
                    for step,row in enumerate(run['actions']):
                        close_ready();i=row['input_index'];a=row['action'];extra=row['replacement'];request=requests[infos[i]['task_id']]
                        check.check(i not in closed,tag+': action after closure');touched.add(i)
                        charge=price(a,request,weights,extra);check.near(row['cost'],charge,tag+': price')
                        spent+=charge;spent_by[i]+=charge;check.check(spent<=run['budget'],tag+': running budget')
                        if extra:
                            check.check(active and a in suspect and a not in extras[i] and base_verdict(evidence[i])=='accept',tag+': replacement eligibility')
                            check.equal(row['forecast'],None,tag+': extra forecast');check.equal(row['monitor'],None,tag+': extra monitor')
                            extras[i][a]=row['observation'];extra_cost+=charge
                        else:
                            check.check(base_verdict(evidence[i])=='inconclusive' and a in available(evidence[i]),tag+': action eligibility')
                            spoiled=package_id>=onset and ((fault=='permissive' and a=='collocation') or (fault=='uninformative' and a=='second_moment'))
                            obs=category(a,row['original_observation'],spoiled)
                            check.equal(row['observation'],obs,tag+': observed channel');evidence[i][a]=obs
                            p=forecasts[infos[i]['task_id']][a]
                            for actual,expected in zip(row['forecast']['probabilities'],p):check.near(actual,expected,tag+': forecast')
                            target_count+=a==target
                            if active:
                                probability=row['forecast']['probabilities'][0];unused=probability
                                if a in TARGETS:
                                    if policy=='D':probability=.999
                                    if policy=='Dprime':unused=.999
                                regime=str(next((j for j,b in enumerate((.1,1e3,1e6)) if abs(request.z*request.mean_time)<=b),3));key=a+':'+regime
                                value=max(0.,sums.get(key,0.)+int(obs['verdict']=='accept')-probability-.02);sums[key]=value
                                m=dict(action=a,regime=regime,probability=probability,unused_probability=unused,statistic=value,package_id=package_id,input_index=i,action_index=step,touched_inputs=len(touched),target_exposures=target_count)
                                check.equal(row['monitor'],m,tag+': monitor arithmetic');history.append(m)
                                if value>threshold and a not in suspect:suspect.add(a);alerts.append(m)
                            else:check.equal(row['monitor'],None,tag+': inactive monitor')
                    close_ready()
                    check.equal(run['monitor_history'],history,tag+': history');check.equal(run['alerts'],alerts,tag+': alerts')
                    check.equal(run['suspect'],sorted(suspect),tag+': suspects');check.equal(run['touched_inputs'],len(touched),tag+': exposure count');check.equal(run['target_exposures'],target_count,tag+': target count')
                    check.near(run['spent'],spent,tag+': spend');check.near(run['replacement_cost'],extra_cost,tag+': replacement spend');check.near(run['budget'],float(f)*full_cost,tag+': budget')
                    correct=false=0
                    for i,(item,truth) in enumerate(zip(run['inputs'],infos)):
                        v=closed.get(i,verdict(evidence[i],extras[i],suspect));c=(truth['accurate'] and v=='accept') or (not truth['accurate'] and v=='violation_detected');fa=not truth['accurate'] and v=='accept'
                        check.equal(item['verdict'],v,tag+': final verdict');check.equal(item['observed'],evidence[i],tag+': final evidence');check.equal(item['replacements'],extras[i],tag+': final extras');check.near(item['spent'],spent_by[i],tag+': input spend');check.equal(item['correct'],c,tag+': correct label');check.equal(item['false_accept'],fa,tag+': false label');correct+=c;false+=fa
                    check.equal(run['correct'],correct,tag+': correct total');check.equal(run['false_accepts'],false,tag+': false total')
                    if package_id<onset or fault=='none':check.equal(false,0,tag+': clean false acceptance')
                    derived[fault,f,policy].append(dict(correct=correct,false=false,spent=spent,extra=extra_cost,alerts=alerts,touched=len(touched),actions=len(run['actions']),target_count=target_count))
                left,right=runs['BM'],runs['Dprime']
                action_view=lambda r:[(x['input_index'],x['action'],x['replacement'],x['observation'],x['cost']) for x in r['actions']]
                check.equal(action_view(left),action_view(right),f'{package_id}/{fault}/{f}: Dprime actions')
                check.equal(left['inputs'],right['inputs'],f'{package_id}/{fault}/{f}: Dprime inputs')
    for (fault,f,policy),runs in derived.items():
        saved=data['summary'][fault][f][policy];all_alerts=[a for r in runs for a in r['alerts']];target='second_moment' if fault=='uninformative' else 'collocation'
        before=[a for a in all_alerts if a['package_id']<onset and a['action']==target];after=[a for a in all_alerts if a['package_id']>=onset and a['action']==target];delay=None
        if after and not before:
            a=after[0];k=a['package_id'];delay=dict(inputs=sum(runs[j]['touched'] for j in range(onset,k))+a['touched_inputs'],actions=sum(runs[j]['actions'] for j in range(onset,k))+a['action_index']+1,target_observations=sum(runs[j]['target_count'] for j in range(onset,k))+a['target_exposures'])
        expected=dict(correct=sum(r['correct'] for r in runs),correct_by_package=[r['correct'] for r in runs],false_accepts_before=sum(r['false'] for r in runs[:onset]),false_accepts_after=sum(r['false'] for r in runs[onset:]),false_accepts_by_package=[r['false'] for r in runs],alerts=len(all_alerts),packages_with_new_alert=sum(bool(r['alerts']) for r in runs),preonset_target_alert=bool(before),detected=bool(after and not before),delay=delay,censored=not bool(after and not before),detected_within_90=bool(delay and delay['inputs']<=90),spent=sum(r['spent'] for r in runs),replacement_cost=sum(r['extra'] for r in runs))
        if fault=='none':
            expected.update(detected=None,delay=None,censored=None,detected_within_90=None)
        expected['followup']=dict(inputs=sum(r['touched'] for r in runs[onset:]),actions=sum(r['actions'] for r in runs[onset:]),target_observations=sum(r['target_count'] for r in runs[onset:]))
        for key,value in expected.items():check.equal(saved[key],value,f'{fault}/{f}/{policy}: summary {key}')
    delta=[abs(.999-forecasts[i][a][0]) for i in range(len(records)) for a in TARGETS];intervention=data['intervention_audit']
    for prefix in ('D','Dprime'):
        check.equal(intervention[prefix+'_changed_numbers'],sum(x!=0 for x in delta),prefix+': intervention count');check.near(intervention[prefix+'_L1'],sum(delta),prefix+': intervention magnitude')
    return dict(passed=not check.errors,checks=check.checks,errors=check.errors,scope='Independent saved-action accounting, evidence, static forecasts, monitor, interventions and summary audit; no physical re-execution or exhaustive planner optimality proof')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=audit(read_json(args.archive));write_json(args.output,result);print(result)
    if not result['passed']:raise SystemExit(1)
