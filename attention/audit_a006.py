"""Independent reconstruction of A006 accounting and calibration evidence.

No experiment planner, monitor, sample executor or result summarizer is called.
Numerical paths are checked algebraically; fresh physical replay is separate.
"""
import argparse
import copy
import hashlib
import math
from collections import Counter, defaultdict
from itertools import chain
from functools import lru_cache
from pathlib import Path
import numpy as np
from attention import s3
from attention.audit_a005 import Audit, FEATURES, GRIDS, ORDER, available, base_verdict, price, verdict
from attention.io import read_json, source_bytes, write_json
from lab.guard import Request
from lab.replay import canonical_sha256
from lab.scientist import ROOT

POLICIES=('B','BM','BA','C2A','C2','D','Dprime')
ACTIVE=('BA','C2A','D','Dprime')
FAMILIES=('collocation','second_moment','v2')
FAULTS=('none','permissive','uninformative','underestimated')
FRACTIONS=('0.2','0.4','0.6','1.0')
LIBRARY='attention/sample_library_a006.json'


def family(action):
    return 'v2' if action.startswith('v2_') else action


def target(fault):
    return {'permissive':'collocation','uninformative':'second_moment','underestimated':'v2','none':'collocation'}[fault]


def decode(fields):
    fields=dict(fields);fields['z']=complex(*fields['z'])
    return Request(**fields)


def ratio(a,b):
    return a/b if b else (math.inf if a else 0.)


def classify(converging, discrepancy, uncertainty, tolerance):
    if not all(math.isfinite(v) for v in (discrepancy,uncertainty,tolerance)):
        return 'inconclusive'
    if converging and discrepancy+uncertainty<=tolerance:return 'accept'
    if converging and discrepancy-uncertainty>tolerance:return 'violation_detected'
    return 'inconclusive'


def categorical(action, observation):
    out={'verdict':observation['verdict']}
    if action.startswith('v2_'):out['terminal']=observation['terminal']
    return out


def max_price(sample,weights):
    if sample['family']=='v2':return sum(GRIDS)*weights['fv_cell']
    return price(sample['family'],decode(sample['request']),weights)


def replacement_max(action,request,weights):
    return sum(GRIDS)*weights['fv_cell'] if action=='v2' else price(action,request,weights,True)


def verify_v2(check,original,observed,values,grid,candidate,tolerance,spoiled,tag):
    """Derive clean and corrupted uncertainty, verdict and stopping flags."""
    check.equal(original['value'],[values[-1].real,values[-1].imag],tag+': value')
    check.equal(observed['value'],original['value'],tag+': unchanged grid value')
    if grid<512:
        check.equal(original,dict(verdict='inconclusive',terminal=False,value=original['value']),tag+': preparatory grid')
        check.equal(observed,original,tag+': preparatory unchanged')
        return
    ds=[abs(b-a) for a,b in zip(values,values[1:])]
    rounding=32*np.finfo(float).eps*grid**2*max(1,abs(values[-1]))
    uncertainty=2*ds[-1]+rounding
    discrepancy=abs(candidate-values[-1]);converging=ds[-1]<=.5*ds[-2]
    for label,obs,scale in [('original',original,1.),('observed',observed,100. if spoiled else 1.)]:
        step=obs['step'];u=uncertainty/scale;v=classify(converging,discrepancy,u,tolerance)
        futile=bool(grid in (1024,2048) and v=='inconclusive' and u>=4*tolerance and ratio(ds[-1],ds[-2])>=.5 and ratio(ds[-2],ds[-3])>=.5)
        for key,value in dict(last_grid=grid,converging=converging,verdict=v,futile=futile,blocked_rounding=bool(ds[-1]<=rounding and not converging)).items():
            check.equal(step[key],value,tag+': '+label+' '+key)
        for key,value in dict(uncertainty_estimate=u,discrepancy=discrepancy,rounding=rounding,last_difference=ds[-1],previous_difference=ds[-2]).items():
            check.near(step[key],value,tag+': '+label+' '+key)
        if 'antecedent_ratio' in step:
            actual=step['antecedent_ratio'];expected=ratio(ds[-2],ds[-3])
            check.check(actual==expected or math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),tag+': '+label+' antecedent ratio')
        check.equal(obs['verdict'],v,tag+': '+label+' verdict')
        check.equal(obs['terminal'],bool(v!='inconclusive' or futile or grid==GRIDS[-1]),tag+': '+label+' terminal')
    unchanged=lambda d:{k:v for k,v in d.items() if k not in ('verdict','terminal','step')}
    check.equal(unchanged(original),unchanged(observed),tag+': unchanged numerical input')


def verify_trace(check,trace,candidate,tolerance,weights,spoiled,tag):
    values=[];total=0.
    check.check(bool(trace),tag+': nonempty trace')
    for i,row in enumerate(trace):
        grid=GRIDS[i];check.equal(row['action'],'v2_'+str(grid),tag+': trace order')
        check.equal(row['cost'],grid*weights['fv_cell'],tag+': trace cost');total+=grid*weights['fv_cell']
        values.append(complex(*row['original']['value']))
        verify_v2(check,row['original'],row['observed'],values,grid,candidate,tolerance,spoiled,tag+': step '+str(grid))
        check.equal(row['observed']['terminal'],i==len(trace)-1,tag+': trace termination')
    return total


def forecasts(model,records,noise):
    runtime=s3.Runtime(model);result={}
    for record in records:
        request=decode(record['request']);neighbors,_=runtime.neighbors(request)
        ids=np.flatnonzero(np.isin(runtime.groups,neighbors));prior={}
        for a in ORDER:
            raw=runtime.raw[ids,s3.INDEX[a]];present=raw[raw>=0]
            k=4 if a.startswith('v2_') else 3 if a in ('modal','collocation') else 2
            probabilities=[float((.98*np.count_nonzero(present==j)+.02/k*len(present))/len(present)) if len(present) else 1/k for j in range(k)]
            if noise:probabilities=[float(np.nextafter(v,np.inf if j%2 else -np.inf)) if v>0 else v for j,v in enumerate(probabilities)]
            prior[a]=probabilities
        result[record['task_id']]=dict(actions=prior,full_accept=float(np.mean(runtime.full[ids]==0)),accuracy=float(np.mean(runtime.accurate[ids])),full_correct=float(np.mean(runtime.correct[ids])))
    return result


def score_choice(evidence,closed,suspect,state,infos,requests,predictions,samples,weights):
    result={}
    for name in FAMILIES:
        loss=0.
        for i,e in enumerate(evidence):
            if i in closed or base_verdict(e)=='violation_detected':continue
            action=name
            if name=='v2':
                pending=[a for a in available(e) if a.startswith('v2_')]
                done=[a for a in e if a.startswith('v2_')]
                action=pending[0] if pending else done[-1] if done else 'v2_512'
            record=predictions[infos[i]['task_id']]
            loss+=record['actions'][action][0]*record['full_accept']*replacement_max(name,requests[infos[i]['task_id']],weights)*(1+len(e)/len(ORDER))
        maximum=max(max_price(s,weights) for s in samples if s['family']==name)
        result[name]=round(loss*(1+state['ordinary']-state['last'][name])/maximum,9)
    if any(a not in suspect for a in FAMILIES):result={a:0. if a in suspect else v for a,v in result.items()}
    return min(FAMILIES,key=lambda a:(-result[a],FAMILIES.index(a))),result


@lru_cache(maxsize=2)
def development_inputs(source,digest):
    from attention.compare_a004 import Archive
    archive=Archive(ROOT/source);result={}
    for package in archive.packages():
        records=archive.metadata['intake_channel']
        for row in package['inputs']:
            result[package['package_id'],row['task_id'],row['condition']]=dict(
                request=records[row['task_id']]['request'],digest=records[row['task_id']]['request_sha256'],
                outcome=row['outcome'],reference=row['reference'],
                actions={a['action']:a['observation'] for a in row['full']['actions'] if a['action']!='declare'})
    return result


def compare_diagnostics(check,actual,expected,tag,key=None):
    """Discrete fields and logical prices are exact; finite diagnostics may drift."""
    exact={'cost','spent','budget','reference_cost','replacement_cost','sample_cost','ordinary_cost','maximum_cost','maximum_price','initial_reserve','ordinary_budget'}
    if isinstance(expected,dict):
        check.check(isinstance(actual,dict),tag+': mapping type')
        if not isinstance(actual,dict):return
        check.equal(set(actual),set(expected),tag+': keys')
        for name in expected.keys()&actual.keys():compare_diagnostics(check,actual[name],expected[name],tag+'/'+name,name)
    elif isinstance(expected,list):
        check.check(isinstance(actual,list),tag+': list type')
        if not isinstance(actual,list):return
        check.equal(len(actual),len(expected),tag+': length')
        for i,(a,b) in enumerate(zip(actual,expected)):compare_diagnostics(check,a,b,tag+'/'+str(i),key)
    elif isinstance(expected,float) and key not in exact:
        check.near(actual,expected,tag)
    else:check.equal(actual,expected,tag)


class TimingPlanner:
    """Independent U0T envelope for the unrecorded pre-sample action choice.

    U0T never updates the prior, so branch rewards are fixed accuracy means.
    This implementation uses only categorical evidence, frozen probabilities,
    action prices and remaining budget, not the production planning classes.
    """
    def __init__(self,predictions,requests,weights):
        self.predictions=predictions
        self.prices={i:{a:price(a,r,weights) for a in ORDER} for i,r in requests.items()}
        self.envelope=lru_cache(maxsize=4096)(self._envelope)

    @staticmethod
    def option(kind,action,gain,cost):
        return dict(kind=kind,action=action,gain=gain,cost=cost,score=round(gain/cost,9) if cost else 0.)

    @staticmethod
    def tie(option):
        return (-option['score'],ORDER.index(option['action']),{'ATOMIC':0,'TREE':1,'FULL':2}[option['kind']])

    def _envelope(self,task,key):
        evidence={a:dict(verdict=v,**({'terminal':t} if t is not None else {})) for a,v,t in key}
        prior=self.predictions[task];prices=self.prices[task]
        def reward(v):
            return prior['accuracy'] if v=='accept' else 1-prior['accuracy'] if v=='violation_detected' else 0.
        def branches(e,a):
            for category,p in enumerate(prior['actions'][a]):
                if p<=0:continue
                out=dict(verdict=('accept' if category==0 else 'violation_detected' if category==1 else 'inconclusive'))
                if a.startswith('v2_'):out['terminal']=category!=2
                yield p,dict(e,**{a:out})
        def direct(e):
            if base_verdict(e)!='inconclusive':return []
            choices=available(e);out=[]
            for a in choices:
                gain=sum(p*reward(base_verdict(b)) for p,b in branches(e,a))
                out.append(self.option('ATOMIC',a,gain,prices[a]))
            if choices:
                first=min(choices,key=lambda a:(prices[a],ORDER.index(a)))
                terminal=any(a.startswith('v2_') and v.get('terminal',False) for a,v in e.items())
                remaining=[a for a in ORDER if a not in e and not (terminal and a.startswith('v2_'))]
                out.append(self.option('FULL',first,prior['full_correct'],sum(prices[a] for a in remaining)))
            return out
        trees=[]
        for a in available(evidence):
            trees.append((a,[(p,reward(base_verdict(b)),direct(b)) for p,b in branches(evidence,a)]))
        return direct(evidence),trees

    def choose(self,evidence,infos,remaining):
        pairs=[(self.prices[infos[i]['task_id']][a],i,ORDER.index(a),a) for i,e in enumerate(evidence) if base_verdict(e)=='inconclusive' for a in available(e) if self.prices[infos[i]['task_id']][a]<=remaining]
        if not pairs:return None
        candidates=[]
        for i in sorted({row[1] for row in pairs}):
            task=infos[i]['task_id'];key=tuple((a,v['verdict'],v.get('terminal')) for a,v in evidence[i].items())
            direct,trees=self.envelope(task,key);choices=[o for o in direct if o['cost']<=remaining]
            for a,branches in trees:
                charge=self.prices[task][a]
                if charge>remaining:continue
                gain=0.;cost=charge
                for probability,reward,continuations in branches:
                    options=[o for o in continuations if charge+o['cost']<=remaining and o['gain']>0]
                    continuation=min(options,key=self.tie) if options else None
                    gain+=probability*reward
                    if continuation:
                        gain+=probability*continuation['gain'];cost+=probability*continuation['cost']
                choices.append(self.option('TREE',a,gain,cost))
            if choices:candidates.append((i,min(choices,key=self.tie)))
        if not candidates or max(o['score'] for _,o in candidates)<=0:
            cheap=min(pairs);return cheap[1],cheap[3]
        i,o=min(candidates,key=lambda pair:(-pair[1]['score'],pair[0],*self.tie(pair[1])[1:]))
        return i,o['action']


def expected_sample_reason(policy,ordinary_cost,deadline,ordinary_since,last_sample_ordinary,chosen_family,insurance,affordable,pending):
    if ordinary_cost>=deadline:return 'deadline'
    if not affordable and pending is None:return 'exhausted'
    choice=affordable[0][3] if affordable else pending[1]
    if policy!='C2A' and ordinary_since>last_sample_ordinary and (family(choice)==chosen_family or bool(insurance)):
        return 'anticipated_dependency'
    return None


def audit_library(check,library,weights):
    samples=library['samples'];check.equal(library['samples_sha256'],canonical_sha256(samples),'sample recipes digest')
    check.equal(len({s['sample_id'] for s in samples}),len(samples),'unique sample ids')
    counts={f:dict(Counter(s['expected'] for s in samples if s['family']==f)) for f in sorted({s['family'] for s in samples})}
    check.equal(library['counts'],counts,'library counts')
    for path,digest in library['source_sha256'].items():
        check.equal(hashlib.sha256(source_bytes(ROOT/path)).hexdigest(),digest,'development source '+path)
    source='attention/results/A005/results.json.gz'
    origins=development_inputs(source,library['source_sha256'][source])
    for sample in samples:
        origin=origins.get((sample['source_package_id'],sample['source_task_id'],sample['condition']))
        check.check(origin is not None,'sample development origin exists')
        if origin is not None:
            for key in ('request','digest','outcome'):check.equal(sample[key],origin[key],'sample source '+key)
            check.equal(sample['candidate'],origin['reference'] if sample['condition']=='clean' else origin['outcome']['value'],'sample source candidate')
            check.equal(sample['model'],sample['condition'] if sample['condition'] in ('slab','mixture','model_scale') else 'sphere','sample source model')
            if sample['family'] not in ('collocation','v2'):
                expected=origin['actions'][sample['family']]
                if sample['family']=='modal':expected={k:v for k,v in expected.items() if k!='outcome'}
                compare_diagnostics(check,sample['clean_result']['original'],expected,'sample original from development')
        tag='library '+str(sample['sample_id']);clean=sample['clean_result']
        check.equal(sample['expected'],clean['verdict'],tag+': label')
        check.equal(clean['verdict'],clean['observed']['verdict'],tag+': clean verdict')
        check.equal(sample['maximum_cost'],max_price(sample,weights),tag+': maximum cost')
        if sample['family']=='v2':
            cost=verify_trace(check,clean['trace'],complex(*sample['candidate']),sample['request']['tolerance'],weights,False,tag)
            check.equal(clean['original'],clean['trace'][-1]['original'],tag+': last original')
            check.equal(clean['observed'],clean['trace'][-1]['observed'],tag+': last observed')
        else:
            cost=max_price(sample,weights);check.equal(clean['original'],clean['observed'],tag+': intact channel')
        check.equal(clean['cost'],cost,tag+': cost')
    return samples


def audit(data,library=None):
    check=Audit();check.equal(data['experiment'],'A006','experiment')
    library=read_json(ROOT/LIBRARY) if library is None else library
    model=s3.load_model();weights=model['action_model']['cost_weights']
    samples=audit_library(check,library,weights);by_id={s['sample_id']:s for s in samples}
    active_samples=[s for s in samples if s['family'] in FAMILIES]
    maximum=max(max_price(s,weights) for s in active_samples)
    check.equal(data['sample_library_sha256'],canonical_sha256(library),'frozen library digest')
    check.equal(data['sample_library_sources'],library['source_sha256'],'library provenance')
    check.equal(data['sample_library_counts'],library['counts'],'library coverage')
    records=data['intake_channel'];requests={r['task_id']:decode(r['request']) for r in records};onset=data['onset_package']
    check.equal(len(records),60,'request count');check.equal(onset,int(np.random.default_rng(20261148).integers(4,9)),'onset')
    check.equal(data['intake_sha256'],canonical_sha256([records,data['global_permutations']]),'intake digest')
    check.equal(data['self_model_sha256'],canonical_sha256(model),'frozen model digest')
    check.equal(data['full_pipeline_parity_failures'],[],'full pipeline parity')
    predictions=forecasts(model,records,data.get('last_bit_noise',False))
    timing=TimingPlanner(predictions,requests,weights)
    threshold=data['calibration']['threshold'];check.near(threshold,data['calibration']['maximum']+1,'passive threshold')
    intervention=data['intervention_audit'];codes={'accept':0,'violation_detected':1,'inconclusive':2}
    changes=[codes[s['expected']] for s in samples if s['family'] in FAMILIES[:2] and s['expected']!='accept']
    for prefix in ('D','Dprime'):
        check.equal(intervention[prefix+'_changed_numbers'],len(changes),prefix+': intervention count')
        check.equal(intervention[prefix+'_L1'],sum(changes),prefix+': intervention magnitude')
    check.equal(intervention['equal_changes'],True,'matched intervention')
    states={};derived=defaultdict(list);package_count=0
    for package in data['packages']:
        package_id=package['package_id'];check.equal(package_id,package_count,'package ordering');package_count+=1
        infos=package['inputs'];check.equal(len(infos),45,'package input count')
        clean_paths=[{r['action']:r['observation'] for r in x['full']['actions'] if r['action']!='declare'} for x in infos]
        full_cost=sum(sum(row['cost'] for row in x['full']['actions']) for x in infos)
        check.equal(package['reference_cost'],full_cost,'full reference cost')
        for info in infos:
            request=requests[info['task_id']];full_spent=0.
            for row in info['full']['actions']:
                charge=0. if row['action']=='declare' else price(row['action'],request,weights)
                check.equal(row['cost'],charge,'full reference action price');full_spent+=charge
            if 'spent' in info['full']:check.equal(info['full']['spent'],full_spent,'full path sum')
            error=abs(complex(*info['outcome']['value'])-complex(*info['reference']))
            check.near(info['absolute_error'],error,'reference error');check.equal(info['accurate'],error<=info['tolerance'],'reference accuracy')
        for fault in FAULTS:
            for fraction in FRACTIONS:
                runs=package['conditions'][fault][fraction]
                for policy in POLICIES:
                    tag=f'{package_id}/{fault}/{fraction}/{policy}';run=runs[policy]
                    check.equal(len(run['inputs']),len(infos),tag+': final input count')
                    state=states.setdefault((fault,fraction,policy),dict(sums={},suspect=set(),counts=dict.fromkeys(FAMILIES,0),last=dict.fromkeys(FAMILIES,0),ordinary=0,total_samples=0))
                    suspect=state['suspect'];active=policy in ACTIVE;passive=policy=='BM'
                    required=suspect if active or passive else set()
                    budget=float(fraction)*full_cost;requested={'0.2':3,'0.4':6,'0.6':9,'1.0':15}[fraction]
                    quota=min(requested,int(budget//maximum)) if active else 0;ordinary_budget=budget-quota*maximum
                    allocation=run['sample_allocation']
                    for key,value in dict(quota=quota,requested=requested,maximum_price=maximum,initial_reserve=quota*maximum,ordinary_budget=ordinary_budget,counts_before=state['counts']).items():
                        check.equal(allocation[key],value,tag+': allocation '+key)
                    evidence=[{} for _ in infos];extras=[{} for _ in infos];closed={};spent_by=[0.]*len(infos);values=[[] for _ in infos]
                    spent=0.;sample_cost=0.;ordinary_cost=0.;extra_cost=0.;count=0;target_samples=0;target_observations=0;ordinary_actions=0
                    touched=set();alerts=[];history=[];changed_samples=0;changed_steps=0;changed_ordinary=0;changed_ordinary_steps=0;mismatches=0;last_sample_ordinary=-1
                    def close_ready():
                        for i,e in enumerate(evidence):
                            if i in closed:continue
                            base=base_verdict(e)
                            if base=='violation_detected':closed[i]=base
                            elif base=='accept':
                                result=verdict(e,extras[i],required)
                                if result!='inconclusive' or not (required-set(extras[i])):closed[i]=result
                    for index,row in enumerate(run['actions']):
                        close_ready();a=row['action'];reserve=(quota-count)*maximum
                        check.equal(row['reserve_before'],reserve,tag+': reserve')
                        corrupted=package_id>=onset
                        if row['sample']:
                            check.check(active and count<quota,tag+': sample eligibility');check.equal(row['input_index'],None,tag+': sample input boundary')
                            sample=by_id[row['sample_id']];check.equal(sample['family'],a,tag+': sample family')
                            choices=[s for s in samples if s['family']==a];positive=[s for s in choices if s['expected']=='accept'];negative=[s for s in choices if s['expected']!='accept']
                            n=state['counts'][a];pool=(positive if n%2==0 else negative) or choices
                            check.equal(row['sample_id'],pool[(n//2)%len(pool)]['sample_id'],tag+': deterministic sample cycle')
                            if policy=='C2A':
                                check.equal(a,FAMILIES[state['total_samples']%3],tag+': fixed family cycle');check.equal(row['sample_scores'],None,tag+': fixed schedule no scores')
                            else:
                                selected,scores=score_choice(evidence,closed,suspect,state,infos,requests,predictions,samples,weights)
                                check.equal(a,selected,tag+': selected calibration family')
                                for name,value in scores.items():check.near(row['sample_scores'][name],value,tag+': calibration score '+name)
                                for value in row['sample_scores'].values():check.equal(value,round(value,9),tag+': quantized calibration score')
                            check.equal(row['sample_deadline'],count/quota*ordinary_budget,tag+': fixed deadline')
                            check.equal(row['ordinary_cost_before'],ordinary_cost,tag+': ordinary cost before sample')
                            deadline=count/quota*ordinary_budget;remaining=max(0.,budget-spent-reserve)
                            insurance=[]
                            for j,e in enumerate(evidence):
                                if j not in closed and base_verdict(e)=='accept':
                                    for name in required-set(extras[j]):
                                        insurance.append((replacement_max(name,requests[infos[j]['task_id']],weights),j,ORDER.index(name) if name in ORDER else len(ORDER),name))
                            affordable=sorted(x for x in insurance if x[0]<=remaining)
                            pending=None
                            if ordinary_cost<deadline and not affordable:
                                pending=timing.choose(evidence,infos,remaining)
                            reason=expected_sample_reason(policy,ordinary_cost,deadline,state['ordinary'],last_sample_ordinary,a,insurance,affordable,pending)
                            check.equal(row['sample_reason'],reason,tag+': reconstructed sample timing')
                            last_sample_ordinary=state['ordinary']
                            expected='accept' if policy=='D' and a in FAMILIES[:2] else sample['expected']
                            check.equal(row['expected'],expected,tag+': expected verdict table')
                            original=row['original_observation'];observed=row['observation']
                            if a=='v2':
                                charge=verify_trace(check,row['sample_trace'],complex(*sample['candidate']),sample['request']['tolerance'],weights,corrupted and fault=='underestimated',tag+': sample')
                                check.equal(original,row['sample_trace'][-1]['original'],tag+': sample final original');check.equal(observed,row['sample_trace'][-1]['observed'],tag+': sample final observed')
                                for j,t in enumerate(row['sample_trace']):
                                    if j<len(sample['clean_result']['trace']):compare_diagnostics(check,t['original']['value'],sample['clean_result']['trace'][j]['original']['value'],tag+': frozen sample numerical value')
                                changed_steps+=sum(t['original'].get('step',{}).get('uncertainty_estimate')!=t['observed'].get('step',{}).get('uncertainty_estimate') for t in row['sample_trace'])
                            else:
                                charge=max_price(sample,weights);compare_diagnostics(check,original,sample['clean_result']['original'],tag+': sample clean original')
                                spoilt=corrupted and ((fault=='permissive' and a=='collocation') or (fault=='uninformative' and a=='second_moment'))
                                check.equal(observed,{'verdict':'accept'} if spoilt else original,tag+': sample fault channel')
                            count+=1;target_samples+=int(a==target(fault));changed_samples+=int(original['verdict']!=observed['verdict'])
                            mismatch=expected!=observed['verdict'];mismatches+=int(mismatch)
                            event=dict(action=a,sample_id=row['sample_id'],expected=expected,observed=observed['verdict'],mismatch=mismatch,package_id=package_id,input_index=None,action_index=index,touched_inputs=len(touched),target_exposures=target_observations,samples=count,target_samples=target_samples,ordinary_actions=ordinary_actions)
                            check.equal(row['monitor'],event,tag+': sample monitor');history.append(event)
                            state['counts'][a]+=1;state['total_samples']+=1;state['last'][a]=state['ordinary']
                            if mismatch and a not in suspect:suspect.add(a);alerts.append(event)
                            sample_cost+=charge
                        else:
                            if active and count<quota:check.check(ordinary_cost<count/quota*ordinary_budget,tag+': no missed sample deadline')
                            i=row['input_index'];request=requests[infos[i]['task_id']]
                            check.check(i not in closed,tag+': action after closure');touched.add(i)
                            insurance=[];remaining=max(0.,budget-spent-reserve)
                            for j,e in enumerate(evidence):
                                if j not in closed and base_verdict(e)=='accept':
                                    for name in required-set(extras[j]):
                                        cost=replacement_max(name,requests[infos[j]['task_id']],weights)
                                        if cost<=remaining:insurance.append((cost,j,ORDER.index(name) if name in ORDER else len(ORDER),name))
                            if insurance:
                                selected=min(insurance);check.equal((row['replacement'],i,a),(True,selected[1],selected[3]),tag+': insurance priority')
                            elif policy in ('C2','C2A'):
                                candidates=[]
                                for j,e in enumerate(evidence):
                                    if j not in closed and base_verdict(e)=='inconclusive':
                                        for name in available(e):
                                            cost=price(name,requests[infos[j]['task_id']],weights)
                                            if cost<=remaining:candidates.append((cost,j,ORDER.index(name),name))
                                check.check(bool(candidates),tag+': cheap candidate exists')
                                if candidates:
                                    selected=min(candidates);check.equal((row['replacement'],i,a),(False,selected[1],selected[3]),tag+': cheap greedy choice')
                            if row['replacement']:
                                check.check((active or passive) and a in required and a not in extras[i] and base_verdict(evidence[i])=='accept',tag+': replacement eligibility')
                                check.equal(row['forecast'],None,tag+': replacement forecast');check.equal(row['monitor'],None,tag+': replacement monitor')
                                check.equal(row['observation'],row['original_observation'],tag+': replacement intact');check.equal(row['hooked_observation'],row['observation'],tag+': replacement unhooked')
                                if a=='v2':
                                    charge=verify_trace(check,row['observation']['trace'],complex(*infos[i]['outcome']['value']),request.tolerance,weights,False,tag+': replacement')
                                    check.equal(row['observation']['verdict'],row['observation']['trace'][-1]['observed']['verdict'],tag+': replacement final')
                                else:
                                    charge=price(a,request,weights,True)
                                    if a=='collocation':
                                        obs=row['observation'];check.equal(obs['grids'],[4096,8192,16384,32768],tag+': replacement grids')
                                        vv=[complex(*v) for v in obs['values']];ds=[abs(b-a) for a,b in zip(vv,vv[1:])]
                                        u=2*ds[-1]+32*np.finfo(float).eps*32768**2*max(1,abs(vv[-1]));d=abs(complex(*infos[i]['outcome']['value'])-vv[-1])
                                        check.equal(obs['verdict'],classify(ds[-1]<=.5*ds[-2],d,u,request.tolerance),tag+': replacement classification')
                                        check.near(obs['step']['uncertainty_estimate'],u,tag+': replacement uncertainty');check.near(obs['step']['discrepancy'],d,tag+': replacement discrepancy')
                                    elif a in clean_paths[i]:
                                        expected=clean_paths[i][a];actual=row['observation']
                                        check.equal(actual,expected,tag+': clean replacement observation')
                                check.check(replacement_max(a,request,weights)<=remaining,tag+': reserved replacement affordability')
                                extras[i][a]=row['observation'];extra_cost+=charge
                            else:
                                check.check(base_verdict(evidence[i])=='inconclusive' and a in available(evidence[i]),tag+': ordinary prerequisites')
                                charge=price(a,request,weights);original=row['original_observation'];hooked=row['hooked_observation']
                                if a.startswith('v2_'):
                                    values[i].append(complex(*original['value']))
                                    verify_v2(check,original,hooked,values[i],int(a[3:]),complex(*infos[i]['outcome']['value']),request.tolerance,corrupted and fault=='underestimated',tag+': ordinary '+a)
                                    if a in clean_paths[i]:check.equal(original['value'],clean_paths[i][a]['value'],tag+': clean original value')
                                else:
                                    check.equal(original,clean_paths[i][a],tag+': clean original observation')
                                    spoilt=corrupted and ((fault=='permissive' and a=='collocation') or (fault=='uninformative' and a=='second_moment'))
                                    check.equal(hooked,{'verdict':'accept'} if spoilt else original,tag+': ordinary fault channel')
                                changed_ordinary+=int(original['verdict']!=hooked['verdict'])
                                changed_ordinary_steps+=int(original.get('step',{}).get('uncertainty_estimate')!=hooked.get('step',{}).get('uncertainty_estimate'))
                                observed=categorical(a,hooked);check.equal(row['observation'],observed,tag+': visible observation');evidence[i][a]=observed
                                expected_p=predictions[infos[i]['task_id']]['actions'][a]
                                for actual,expected in zip(row['forecast']['probabilities'],expected_p):check.near(actual,expected,tag+': static forecast')
                                target_observations+=int(family(a)==target(fault))
                                if passive:
                                    probability=row['forecast']['probabilities'][0]
                                    regime=str(next((j for j,b in enumerate((.1,1e3,1e6)) if abs(request.z*request.mean_time)<=b),3));key=a+':'+regime
                                    statistic=max(0.,state['sums'].get(key,0.)+int(observed['verdict']=='accept')-probability-.02);state['sums'][key]=statistic
                                    event=dict(action=a,regime=regime,probability=probability,unused_probability=probability,statistic=statistic,package_id=package_id,input_index=i,action_index=index,touched_inputs=len(touched),target_exposures=target_observations,samples=count,target_samples=target_samples,ordinary_actions=ordinary_actions+1)
                                    check.equal(row['monitor'],event,tag+': passive monitor');history.append(event)
                                    if statistic>threshold and a not in suspect:suspect.add(a);alerts.append(event)
                                else:check.equal(row['monitor'],None,tag+': no passive monitor')
                                check.check(charge<=remaining,tag+': reserved ordinary affordability')
                            ordinary_actions+=1;state['ordinary']+=1;ordinary_cost+=charge;spent_by[i]+=charge
                        check.equal(row['cost'],charge,tag+': action price');spent+=charge
                        check.check(spent<=budget,tag+': running budget')
                    close_ready()
                    for key,value in dict(executed=count,counts_after=state['counts']).items():check.equal(allocation[key],value,tag+': allocation '+key)
                    check.equal(count,quota,tag+': matched quota completed')
                    exposure=dict(ordinary_target_actions=target_observations,target_samples=target_samples,changed_ordinary_categories=changed_ordinary,changed_sample_categories=changed_samples,changed_ordinary_v2_steps=changed_ordinary_steps,changed_sample_v2_steps=changed_steps)
                    check.equal(run['fault_exposure'],exposure,tag+': fault exposure')
                    for key,value in dict(monitor_history=history,alerts=alerts,suspect=sorted(suspect),touched_inputs=len(touched),target_exposures=target_observations,target_samples=target_samples).items():check.equal(run[key],value,tag+': '+key)
                    for key,value in dict(spent=spent,sample_cost=sample_cost,ordinary_cost=ordinary_cost,replacement_cost=extra_cost,budget=budget).items():check.equal(run[key],value,tag+': '+key)
                    correct=false=0
                    for i,(saved,truth) in enumerate(zip(run['inputs'],infos)):
                        v=closed.get(i,verdict(evidence[i],extras[i],required));c=(truth['accurate'] and v=='accept') or (not truth['accurate'] and v=='violation_detected');fa=not truth['accurate'] and v=='accept'
                        for key,value in dict(input_index=i,verdict=v,observed=evidence[i],replacements=extras[i],already_closed=i in closed,correct=c,false_accept=fa).items():check.equal(saved[key],value,tag+': final '+key)
                        check.equal(saved['spent'],spent_by[i],tag+': input price');correct+=c;false+=fa
                    check.equal(run['correct'],correct,tag+': correct total');check.equal(run['false_accepts'],false,tag+': false total')
                    if package_id<onset or fault=='none':check.equal(false,0,tag+': clean false acceptance')
                    derived[fault,fraction,policy].append(dict(correct=correct,false=false,spent=spent,extra=extra_cost,sample_cost=sample_cost,samples=count,alerts=alerts,touched=len(touched),actions=len(run['actions']),target_observations=target_observations,target_samples=target_samples,changed_samples=changed_samples,changed_steps=changed_steps,mismatches=mismatches))
                check.equal(runs['BA']['sample_allocation']['executed'],runs['C2A']['sample_allocation']['executed'],tag+': paired sample counts')
                left={k:v for k,v in runs['BA'].items() if k!='policy'};right={k:v for k,v in runs['Dprime'].items() if k!='policy'}
                check.equal(left,right,tag+': irrelevant intervention unchanged')
    check.equal(package_count,12,'package count')
    for (fault,fraction,policy),runs in derived.items():
        saved=data['summary'][fault][fraction][policy];alerts=[a for r in runs for a in r['alerts']]
        before=[a for a in alerts if a['package_id']<onset and family(a['action'])==target(fault)]
        after=[a for a in alerts if a['package_id']>=onset and family(a['action'])==target(fault)];delay=None
        if after and not before:
            a=after[0];k=a['package_id']
            delay=dict(inputs=sum(r['touched'] for r in runs[onset:k])+a['touched_inputs'],actions=sum(r['actions'] for r in runs[onset:k])+a['action_index']+1,target_observations=sum(r['target_observations'] for r in runs[onset:k])+a['target_exposures'],samples=sum(r['samples'] for r in runs[onset:k])+a['samples'],target_samples=sum(r['target_samples'] for r in runs[onset:k])+a['target_samples'])
        expected=dict(correct=sum(r['correct'] for r in runs),correct_by_package=[r['correct'] for r in runs],false_accepts_before=sum(r['false'] for r in runs[:onset]),false_accepts_after=sum(r['false'] for r in runs[onset:]),false_accepts_by_package=[r['false'] for r in runs],alerts=len(alerts),alerts_before=sum(a['package_id']<onset for a in alerts),packages_with_new_alert=sum(bool(r['alerts']) for r in runs),preonset_target_alert=bool(before),detected=bool(after and not before),delay=delay,censored=not bool(after and not before),detected_within_90=bool(delay and delay['inputs']<=90),spent=sum(r['spent'] for r in runs),replacement_cost=sum(r['extra'] for r in runs),sample_cost=sum(r['sample_cost'] for r in runs),samples=sum(r['samples'] for r in runs),postonset_samples=sum(r['samples'] for r in runs[onset:]),target_samples=sum(r['target_samples'] for r in runs),changed_sample_verdicts=sum(r['changed_samples'] for r in runs[onset:]),changed_v2_steps=sum(r['changed_steps'] for r in runs[onset:]),sample_expectation_mismatches=sum(r['mismatches'] for r in runs[onset:]))
        expected['followup']=dict(inputs=sum(r['touched'] for r in runs[onset:]),actions=sum(r['actions'] for r in runs[onset:]),target_observations=sum(r['target_observations'] for r in runs[onset:]),target_samples=sum(r['target_samples'] for r in runs[onset:]))
        if fault=='none':expected.update(detected=None,delay=None,censored=None,detected_within_90=None)
        for key,value in expected.items():check.equal(saved[key],value,f'{fault}/{fraction}/{policy}: summary {key}')
    audit_digest=hashlib.sha256((ROOT/'attention/audit_a006.py').read_bytes()).hexdigest()
    generation_digest=data.get('source_sha256',{}).get('attention/audit_a006.py')
    return dict(passed=not check.errors,checks=check.checks,errors=check.errors,audit_source_sha256=audit_digest,generation_audit_source_sha256=generation_digest,auditor_matches_generation=audit_digest==generation_digest if generation_digest else None,scope='Independent frozen-sample, numerical-hook, calibration-choice, budget, evidence, monitor, intervention and summary reconstruction; including pre-sample two-stage timing choice; no fresh physical execution or exhaustive ordinary-planner optimality proof')


def audit_path(path,mutate=None):
    from attention.compare_a004 import Archive
    archive=Archive(path);iterator=archive.packages();first=next(iterator)
    def packages():
        for index,package in enumerate(chain((first,),iterator)):
            if mutate:mutate(package,index)
            yield package
    archive.metadata['packages']=packages();result=audit(archive.metadata)
    if not archive.finished:raise ValueError('Archive not completely consumed')
    return result


def negative_control(path):
    """Change parsed copies; never modify the source archive."""
    def mutate(package,index):
        if index:return
        plain=package['conditions']['none']['0.2']['B'];plain['correct']+=1;plain['actions'][0]['cost']+=1
        sample=next(r for r in package['conditions']['none']['0.2']['BA']['actions'] if r['sample'])
        sample['expected']='inconclusive' if sample['expected']=='accept' else 'accept'
        package['conditions']['none']['0.2']['C2A']['sample_allocation']['executed']+=1
    result=audit_path(path,mutate)
    markers=('correct total','action price','expected verdict table','allocation executed')
    found={m:any(m in error for error in result['errors']) for m in markers}
    return dict(passed=not result['passed'] and all(found.values()),archive_unchanged=True,detected=found,audit_result=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--negative-control',action='store_true');parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=(negative_control if args.negative_control else audit_path)(args.archive);write_json(args.output,result);print(result)
    if not result['passed']:raise SystemExit(1)
