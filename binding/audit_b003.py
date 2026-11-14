"""Independent B003 ledger reconstruction; no scheduler or runner imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from attention.build_a007_library import CHANNELS, channel
from binding.audit import AuditError, equal, close, read, digest
from binding.audit_b002_exposure import action_price, native_price, cross_check
from lab import frozen_pipeline as P
from lab.guard import Request

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (64,128,256,512,1024,2048,4096)
FEATURES = ('normalization','first_moment','second_moment','high_frequency')
ORDER = ('modal',)+tuple('v2_'+str(n) for n in GRIDS)+('binding',)+FEATURES+('collocation',)
CATEGORIES = ('sample','ordinary','replacement','recall','future')


def request_digest(fields):
    return hashlib.sha256(json.dumps(fields,sort_keys=True,allow_nan=False).encode()).hexdigest()


def finite_tree(value, path='root'):
    if isinstance(value,float) and not math.isfinite(value): raise AuditError('Nonfinite '+path)
    if isinstance(value,dict):
        for k,v in value.items(): finite_tree(v,path+'.'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value): finite_tree(v,path+'.'+str(i))


def request_root(record, requests):
    roots=[e for e in record['links'] if e['kind']=='request']
    equal(len(roots),1,'one request root')
    root=roots[0]; equal(root['request_id'],record['task_id'],'request endpoint')
    equal(root['request_hash'],requests[root['request_id']]['request_sha256'],'request hash')
    equal(root['time'],record['time'],'request closure timestamp')
    equal(root['channel'] in ('request','independent:request'),True,'request provenance label')
    return requests[root['request_id']]['request']


def projection(records, markers, policy):
    rows=copy.deepcopy([r for r in records if r['standing']]); edits=[]
    if policy not in ('Sx','Sxprime'): return rows,edits
    for row in rows:
        matches=[e for e in row['links'] if e['kind']=='channel' and e['channel'].startswith('collocation:') and e['time']>markers[e['channel']]]
        if not matches: continue
        roots=[e for e in row['links'] if e['kind']=='request']
        equal(len(matches),1,'unique eligible collocation edge'); equal(len(roots),1,'unique paired root')
        edge=(matches if policy=='Sx' else roots)[0]; before=edge['channel']
        edge['channel']=before.removeprefix('independent:') if before.startswith('independent:') else 'independent:'+before
        edits.append(dict(record_id=row['id'],edge_id=edge['edge_id'],before=before,after=edge['channel'],time=edge['time'],binary_distance=1))
    return rows,edits


def score_stakes(records,markers,prices,noise=False):
    counts=dict.fromkeys(CHANNELS,0)
    for row in records:
        if row['standing']:
            for c in {e['channel'] for e in row['links'] if e['kind']=='channel' and e['channel'] in counts and e['time']>markers[e['channel']]}: counts[c]+=1
    return counts,{c:score_round(counts[c]/prices[c],noise) for c in CHANNELS}


def queue_for(records,channel_name,marker,tick,excluded):
    scored=[]
    for row in records:
        times=[e['time'] for e in row['links'] if e['kind']=='channel' and e['channel']==channel_name and e['time']>marker]
        if row['standing'] and row['time']<tick and row['id'] not in excluded and times:
            scored.append((round(max(times),9),row['time'],row['id']))
    return [r[2] for r in sorted(scored,key=lambda r:(-r[0],-r[1],r[2]))]


def available(obs):
    options=[a for a in ('modal','binding')+FEATURES if a not in obs]
    if 'modal' not in obs:return options
    if 'collocation' not in obs:options.append('collocation')
    for j,n in enumerate(GRIDS):
        a='v2_'+str(n)
        if a in obs:
            if obs[a].get('terminal'):break
            continue
        if j==0 or 'v2_'+str(GRIDS[j-1]) in obs:options.append(a)
        break
    return options


def verdict(obs):
    if any(v['verdict']=='violation_detected' for v in obs.values()):return 'violation_detected'
    v2=[v for a,v in obs.items() if a.startswith('v2_') and v.get('terminal')]
    if all(obs.get(a,{}).get('verdict')=='accept' for a in ('modal','binding')+FEATURES+('collocation',)) and v2 and v2[-1]['verdict']=='accept':return 'accept'
    return 'inconclusive'


def native(action,fields,weights,outcome=None):
    if action=='modal':
        r=Request(complex(*fields['z']),fields['mean_time'],fields['tolerance'],fields['budget'],fields['max_calls'])
        return P.plan_modes(r)*weights['modal_term']
    return action_price(action,weights,{'outcome':outcome})


def replacement_price(c,fields,weights):
    family=c.split(':')[0]
    return native_price(family,weights) if family in ('collocation','v2') else native(family,fields,weights)


def pools_and_prices(library,weights):
    pools={}
    for c in CHANNELS:
        pool=[]
        for kind in ('boundary','nonaccept','accept'):
            candidates=[s for s in library['samples'] if s['channel']==c and s['standard_kind']==kind]
            if candidates and candidates[0]['sample_id'] not in [s['sample_id'] for s in pool]:pool.append(candidates[0])
        equal(bool(pool),True,'nonempty sample pool');pools[c]=pool
    def price(s):
        return sum(GRIDS)*weights['fv_cell'] if s['family']=='v2' else native(s['family'],s['request'],weights)
    return pools,price


def native_observation(result, candidate, tolerance, values=None, grid=None):
    """Check native classifications from archived solver outputs, never solve again."""
    if 'step' in result and values is not None:
        step=P.v2_step(values,grid,complex(*candidate),tolerance)
        for k in ('verdict','converging','futile'):equal(result['step'][k],step[k],'native V2 '+k)
        equal(result['verdict'],step['verdict'],'native V2 protected output')
        equal(result['terminal'],step['verdict']!='inconclusive' or step['futile'] or grid==4096,'native V2 terminal')
    elif 'degrees' in result and 'values' in result:
        vals=[complex(*v) for v in result['values']];equal(result['degrees'],[64,128],'native collocation degrees')
        equal(len(vals),2,'native collocation solve count');difference=abs(vals[1]-vals[0]);rounding=32*math.ulp(1.)*128**4*max(1,abs(vals[1]));uncertainty=2*difference+rounding
        converging=all(math.isfinite(v.real) and math.isfinite(v.imag) for v in vals) and difference<=tolerance/8
        discrepancy=abs(complex(*candidate)-vals[1]);v='inconclusive'
        if converging and discrepancy+uncertainty<=tolerance:v='accept'
        elif converging and discrepancy-uncertainty>tolerance:v='violation_detected'
        equal(result['verdict'],v,'native collocation verdict');equal(result['converging'],converging,'native collocation convergence')
        for k,v in [('difference',difference),('rounding',rounding),('uncertainty',uncertainty),('discrepancy',discrepancy)]:close(result[k],v,'native collocation '+k)


def protected_check(result,fields,outcome):
    if result.get('instrument')=='frozen_candidate_guard':
        from lab.guard import final_guard
        r=Request(complex(*fields['z']),fields['mean_time'],fields['tolerance'],fields['budget'],fields['max_calls'])
        equal(result['outcome'],outcome,'immutable modal candidate')
        original=final_guard(r,outcome);fresh=final_guard(r,result['reexecution_outcome'])
        equal(result['resolution'],original,'original modal flags');equal(result['reexecution_resolution'],fresh,'fresh modal flags')
        equal(result['verdict'],'accept' if original['accepted'] and fresh['accepted'] else 'inconclusive','modal protected conjunction')
    if 'steps' in result and 'values' in result:
        graded=cross_check(result,complex(*outcome['value']),fields['tolerance'])
        equal(result['charged_work'],dict(grids=list(GRIDS)) if graded else dict(degrees=[64,128]),'full qualified charge grids')


def check_package(archive,intake,reference,weights,library,prior=None,noise=False):
    """Replay all chronology from actions, not final state or sample snapshots."""
    finite_tree(archive)
    result=archive['result']; infos=archive['inputs']; package=archive['package_id']
    for key in ('package_id','policy','quota'):equal(result[key],archive[key],'package result '+key)
    requests={r['task_id']:r for r in intake['requests']}
    pools,price=pools_and_prices(library,weights);maximum=max(price(s) for p in pools.values() for s in p)
    policy=archive['policy'];quota=0 if policy=='N0' else archive['quota']
    equal(quota in ((0,) if policy=='N0' else (1,2,4)),True,'registered quota')
    initial=dict(policy=policy,quota=quota,rr_phase=intake['rr_phase'],rr_pointer=intake['rr_phase'],last_success=dict.fromkeys(CHANNELS,-1),counts=dict.fromkeys(CHANNELS,0),suspect=[],records=[],pending=[],recalled=[],tick=0)
    state=copy.deepcopy(prior if prior is not None else initial)
    equal(result['state_before'],state,'continuous initial state')
    records=state['records'];markers=state['last_success'];counts=state['counts'];suspect=set(state['suspect']);recalled=set(state['recalled']);pending=state['pending']
    expected_order=[(r['task_id'],c) for r in intake['requests'][package*5:(package+1)*5] for c in r['condition_order']]
    equal([(x['task_id'],x['condition']) for x in infos],expected_order,'canonical candidate order')
    budget=sum(x['full_cost'] for x in infos)
    equal(archive['budget'],budget,'common allowance');equal(result['budget'],budget,'result allowance')
    equal(budget>=quota*maximum,True,'fund full quota')
    for info in infos:
        fields=requests[info['task_id']]['request']
        equal(info['tolerance'],fields['tolerance'],'original tolerance')
        if 'full_actions' not in info:raise AuditError('Missing common-allowance native ledger')
        clean_cost=0.;clean_evidence={};clean_values=[]
        equal(info['full_actions'][-1]['action'],'declare','full native declaration')
        for a in info['full_actions']:
            if a['action']=='declare':
                equal(a['cost'],0.,'declaration price');equal(available(clean_evidence),[],'full native ledger completeness');equal(a['verdict'],verdict(clean_evidence),'full clean verdict');equal(info['clean_full_verdict'],a['verdict'],'shared full verdict');continue
            equal(a['action'],min(available(clean_evidence),key=ORDER.index),'full native C1 work order')
            p=native(a['action'],fields,weights,info['outcome']);equal(a['cost'],p,'common native action price');clean_cost+=p;clean_evidence[a['action']]=a['observation']
            if a['action'].startswith('v2_') and 'value' in a['observation']:clean_values.append(complex(*a['observation']['value']))
            native_observation(a['observation'],info['outcome']['value'],fields['tolerance'],clean_values,int(a['action'][3:]) if a['action'].startswith('v2_') else None)
        close(clean_cost,info['full_cost'],'independent clean allowance')
    evidence=[{} for _ in infos];extra=[{} for _ in infos];support=[{} for _ in infos];native_values=[[] for _ in infos];closed={};alerts=[];costs=dict.fromkeys(CATEGORIES,0.);samples=0;ordinary=0;bu_last=dict.fromkeys(CHANNELS,0)
    def closures():
        for i,obs in enumerate(evidence):
            if i in closed:continue
            v=verdict(obs);required=set(support[i])&suspect
            xv=[extra[i][c]['verdict'] for c in required if c in extra[i]]
            effective='violation_detected' if v=='violation_detected' or 'violation_detected' in xv else 'accept' if v=='accept' and required<=set(extra[i]) and all(x=='accept' for x in xv) else 'inconclusive'
            if effective=='violation_detected':closed[i]=dict(input_index=i,verdict=effective,required_replacements=sorted(required))
            elif effective=='accept':
                rid=f'{package}:{i}';task=infos[i]['task_id']
                links=[dict(e,edge_id=f'{rid}:{j}',kind='channel') for j,e in enumerate(support[i].values())]
                links.append(dict(edge_id=rid+':request',kind='request',channel='request',time=state['tick'],request_id=task,request_hash=requests[task]['request_sha256']))
                record=dict(id=rid,time=state['tick'],package_id=package,input_index=i,task_id=task,standing=True,links=links,original_links=copy.deepcopy(links),history=[])
                request_root(record,requests);records.append(record)
                closed[i]=dict(input_index=i,verdict='accept',record_id=rid,required_replacements=sorted(required))
    for action in result['actions']:
        closures();spent=sum(costs.values());reserve=(quota-samples)*maximum;remaining=max(0.,budget-spent-reserve)
        ordinary_cost=costs['ordinary']+costs['replacement']+costs['future']
        futures=[];options=[]
        for i,obs in enumerate(evidence):
            if i in closed:continue
            fields=requests[infos[i]['task_id']]['request']
            if verdict(obs)=='accept':
                for c in (set(support[i])&suspect)-set(extra[i]):
                    p=replacement_price(c,fields,weights)
                    if p<=remaining:futures.append((score_round(p,noise),i,CHANNELS.index(c),p,c))
            else:
                for a in available(obs):
                    p=native(a,fields,weights,infos[i]['outcome'])
                    if p<=remaining:options.append((score_round(p,noise),i,ORDER.index(a),p,a))
        choices=futures or options
        due=samples<quota and (ordinary_cost>=samples/quota*(budget-quota*maximum) or not choices)
        kind=action['kind'];tick=state['tick']+1;equal(action['tick'],tick,'half-tick action time');state['tick']+=2
        c=action['channel'];equal(kind=='sample',due,'common due opportunity')
        if kind in ('ordinary','sample'):
            observed=copy.deepcopy(action['original_observation'])
            if action['fault_applied']:observed['verdict']='accept'
            if kind=='sample':equal(action['observation'],observed,'sample native protected verdict')
            else:
                equal(action['hooked_observation'],observed,'registered permissive hook')
                token=dict(verdict=observed['verdict'])
                if action['action'].startswith('v2_'):token['terminal']=observed['verdict']!='inconclusive' or observed.get('terminal',False)
                equal(action['observation'],token,'categorical protected observation')
        if kind=='sample':
            prices={ch:price(pools[ch][counts[ch]%len(pools[ch])]) for ch in CHANNELS}
            rows,edits=projection(records,markers,policy);numerators,scores=score_stakes(rows,markers,prices,noise)
            eligible=[ch for ch in CHANNELS if ch not in suspect];equal(bool(eligible),True,'unexhausted channels')
            snap=action['selection_snapshot'];equal(snap['records'],rows,'authentic typed intervention graph');equal(snap['edits'],edits,'binary paired graph edits');equal(snap['markers'],markers,'successful markers');equal(snap['suspect'],sorted(suspect),'visible suspect state')
            for paired in ('S','Sx','Sxprime'):
                g,ed=projection(records,markers,paired);nu,sc=score_stakes(g,markers,prices,noise)
                equal(snap['paired_graphs'][paired],dict(records=g,edits=ed,numerators=nu,scores=sc),'actual paired snapshot '+paired)
            for forbidden in ('target','onset','accurate','affected','fault_channels','single_target'):equal(forbidden in snap,False,'truth isolation')
            if policy=='BU':
                raw=dict.fromkeys(CHANNELS,0.)
                for i,obs in enumerate(evidence):
                    if i in closed:continue
                    fields=requests[infos[i]['task_id']]['request']
                    for family in ('v2','collocation')+FEATURES+('binding','modal'):
                        ch=channel(family,fields);raw[ch]+=replacement_price(ch,fields,weights)*(1+len(obs)/len(ORDER))
                scores={ch:score_round(raw[ch]*(1+ordinary-bu_last[ch])/max(price(s) for s in pools[ch]),noise) for ch in CHANNELS}
                equal(snap['evidence'],evidence,'BU open evidence');equal(snap['closed'],sorted(closed),'BU closures');equal(snap['bu_last'],bu_last,'BU ages');equal(snap['ordinary_count'],ordinary,'BU action clock')
                equal(snap['details']['raw'],raw,'frozen BU exposure');chosen=min(eligible,key=lambda ch:(-scores[ch],CHANNELS.index(ch)))
            else:
                for k,v in [('numerators',numerators),('denominators',prices),('tie_order',list(CHANNELS))]:equal(snap[k],v,'S '+k)
                control,_=projection(records,markers,'Sxprime');equal(score_stakes(control,markers,prices,noise),score_stakes(records,markers,prices,noise),'all-channel irrelevant control null')
                if policy=='RR':
                    index=next((state['rr_pointer']+k)%len(CHANNELS) for k in range(len(CHANNELS)) if CHANNELS[(state['rr_pointer']+k)%len(CHANNELS)] in eligible)
                    chosen=CHANNELS[index];state['rr_pointer']=(index+1)%len(CHANNELS)
                else:chosen=min(eligible,key=lambda ch:(-scores[ch],markers[ch],CHANNELS.index(ch)))
            equal(c,chosen,'rounded sample selection');equal(snap['scores'],scores,'all-channel scores');equal(action['scores'],scores,'action scores')
            sample=pools[c][counts[c]%len(pools[c])];equal(action['sample_id'],sample['sample_id'],'persistent sample cursor');equal(action['expected'],sample['expected'],'frozen standard')
            charge=price(sample)
            if sample['family']=='v2':
                charge=0.;values=[]
                for j,t in enumerate(action['sample_trace']):
                    grid=GRIDS[j];equal(t['action'],'v2_'+str(grid),'sample native grid order');equal(t['cost'],grid*weights['fv_cell'],'sample internal work price');charge+=t['cost'];values.append(complex(*t['original']['value']))
                    native_observation(t['original'],sample['candidate'],sample['request']['tolerance'],values,grid)
                    expected=copy.deepcopy(t['original'])
                    if action['fault_applied']:expected['verdict']='accept'
                    equal(t['observed'],expected,'sample internal protected observation')
                    equal(t['observed']['terminal'],j==len(action['sample_trace'])-1,'sample first terminal stop')
                equal(action['original_observation'],action['sample_trace'][-1]['original'],'sample final original')
            else:native_observation(action['original_observation'],sample.get('candidate',[0.,0.]),sample['request']['tolerance'])
            equal(action['cost'],charge,'executed native sample charge')
            mismatch=action['observation']['verdict']!=sample['expected'];equal(action['mismatch'],mismatch,'sample disagreement');equal(action['last_success_before'],markers[c],'pre-alarm marker');equal(action['reserve_before'],reserve,'sample reservation');equal(action['ordinary_cost_before'],ordinary_cost,'opportunity work');equal(action['sample_index'],samples,'quota index')
            if mismatch:
                equal(c not in suspect,True,'sampled nonsuspect channel');suspect.add(c)
                notice=dict(channel=c,time=tick,last_success=markers[c]);ids=queue_for(records,c,markers[c],tick,recalled|{p['record_id'] for p in pending})
                queued=[dict(record_id=rid,channel=c,notice=copy.deepcopy(notice)) for rid in ids];pending.extend(queued)
                alerts.append(dict(channel=c,tick=tick,package_id=package,last_success=markers[c],queued=copy.deepcopy(queued),fault_applied=action['fault_applied']))
            else:markers[c]=tick
            counts[c]+=1;bu_last[c]=ordinary;samples+=1
        else:
            head=None
            if len(recalled)<32:
                while pending:
                    q=pending[0];r=next(r for r in records if r['id']==q['record_id'])
                    if r['standing'] and r['id'] not in recalled and any(e['kind']=='channel' and e['channel']==q['channel'] and e['time']>q['notice']['last_success'] for e in r['links']):
                        fields=request_root(r,requests);p=replacement_price(q['channel'],fields,weights)
                        if p<=remaining:head=(q,r,p)
                        break
                    pending.pop(0)
            equal(kind=='recall',head is not None,'strict recall priority')
            if kind=='recall':
                q,r,charge=head;equal(action['record_id'],r['id'],'G queue ordering');equal(c,q['channel'],'recall channel');equal(action['notice'],q['notice'],'authentic notice');equal(len(recalled)<32,True,'series attempt cap')
                # Original candidate is retained in historical intake by the outer auditor.
                outcome=reference['candidates'][r['id']];fields=request_root(r,requests);protected_check(action['observation'],fields,outcome)
                pending.pop(0);recalled.add(r['id']);r['history'].append(dict(tick=tick,channel=c,verdict=action['observation']['verdict'],cost=charge));r['standing']=action['observation']['verdict']=='accept'
                if r['standing']:
                    for e in r['links']:
                        if e['kind']=='channel' and e['channel']==c:e.update(channel='independent:'+c,time=tick)
            else:
                equal(bool(choices),True,'affordable ordinary action');_,i,_,charge,a=min(choices);equal(action['input_index'],i,'price-ranked input')
                fields=requests[infos[i]['task_id']]['request']
                if futures:
                    equal(kind,'future','future replacement priority');equal(c,a,'future suspect check');protected_check(action['observation'],fields,infos[i]['outcome']);extra[i][c]=action['observation']
                    if action['observation']['verdict']=='accept':support[i][c]=dict(channel='independent:'+c,time=tick)
                else:
                    equal(kind,'ordinary','ordinary kind');equal(action['action'],a,'deterministic ordinary action');equal(c,channel(a,fields),'executable channel');evidence[i][a]=action['observation'];ordinary+=1
                    if a.startswith('v2_'):native_values[i].append(complex(*action['original_observation']['value']))
                    native_observation(action['original_observation'],infos[i]['outcome']['value'],fields['tolerance'],native_values[i],int(a[3:]) if a.startswith('v2_') else None)
                    if action['observation']['verdict']=='accept':support[i][c]=dict(channel=c,time=tick)
            equal(action['cost'],charge,'full native action charge');equal(charge<=remaining,True,'unreserved work allowance')
        costs[kind]+=action['cost'];equal(sum(costs.values())<=budget+1e-8,True,'no overdraft')
    closures();equal(samples,quota,'full quota paid')
    remaining=budget-sum(costs.values())
    for i,obs in enumerate(evidence):
        if i in closed:continue
        fields=requests[infos[i]['task_id']]['request']
        if verdict(obs)=='accept':
            for c in (set(support[i])&suspect)-set(extra[i]):equal(replacement_price(c,fields,weights)>remaining,True,'no omitted affordable future check')
        else:
            for a in available(obs):equal(native(a,fields,weights,infos[i]['outcome'])>remaining,True,'no omitted affordable ordinary work')
    if len(recalled)<32:
        while pending:
            q=pending[0];r=next(r for r in records if r['id']==q['record_id'])
            if r['standing'] and r['id'] not in recalled and any(e['kind']=='channel' and e['channel']==q['channel'] and e['time']>q['notice']['last_success'] for e in r['links']):
                equal(replacement_price(q['channel'],request_root(r,requests),weights)>remaining,True,'no omitted affordable recall');break
            pending.pop(0)
    state['tick']+=2
    expected_inputs=[closed.get(i,dict(input_index=i,verdict='inconclusive',required_replacements=sorted(set(support[i])&suspect))) for i in range(len(infos))]
    state.update(suspect=sorted(suspect),recalled=sorted(recalled))
    equal(result['inputs'],expected_inputs,'protected closures');equal(result['alerts'],alerts,'alarm queues');equal(result['records'],records,'authentic final provenance');equal(result['state'],state,'reconstructed persistent state')
    equal(result['costs'],costs,'exclusive cost categories');equal(result['spent'],sum(costs.values()),'actual cost');equal(result['unused_budget'],budget-sum(costs.values()),'unused common allowance')
    return state


def check_intake(data):
    from lab.pipeline_holdout import COHORTS, CONDITIONS, LOG_RANGES
    seed=data['seed'];n=data['packages'];equal(n in (4,8,16),True,'registered duration')
    children=np.random.SeedSequence(seed).spawn(5);rng,order,sampling=map(np.random.default_rng,children[:3])
    equal(data['streams'],{name:dict(entropy=seed,spawn_key=list(s.spawn_key)) for name,s in zip(('requests','condition_order','sample_execution','single_target','rr_phase'),children)},'five isolated streams')
    equal(data['rr_phase'],int(np.random.default_rng(children[4]).integers(len(CHANNELS))),'independent RR phase draw')
    collocation=[c for c in CHANNELS if c.startswith('collocation:')]
    equal(data['single_target'],collocation[int(np.random.default_rng(children[3]).integers(6))],'isolated evaluator target draw')
    equal(len(data['requests']),5*n,'request count')
    for i,row in enumerate(data['requests']):
        cohort=COHORTS[i%5];lo,hi=LOG_RANGES[cohort]
        lt=float(rng.uniform(-2,2));u,v=float(rng.random()),float(rng.random());tol=float(rng.uniform(-6 if cohort=='S' else -4,-4 if cohort=='S' else -1))
        angle=(-np.pi/2+.1*v if v<.5 else np.pi/2-.1*(1-v)) if cohort=='O' else -np.pi/2+np.pi*v
        time=10**lt;z=10**(lo+(hi-lo)*u)*np.exp(1j*angle)/time;r=Request(z,time,10**tol)
        fields=dict(z=[complex(z).real,complex(z).imag],mean_time=r.mean_time,tolerance=r.tolerance,budget=r.budget,max_calls=r.max_calls)
        # Archive coordinates are replay inputs, not a new libm draw. Only the
        # discrete stream and canonical archived request hash are portable checks.
        for k,v in [('task_id',i),('package_id',i//5),('cohort',cohort),('condition_order',[str(c) for c in order.permutation(CONDITIONS)])]:equal(row[k],v,'discrete intake stream '+k)
        finite_tree(row['request']);equal(row['request_sha256'],request_digest(row['request']),'canonical archived request digest')
    equal(data['sample_seeds'],[int(sampling.integers(0,2**31)) for _ in range(n)],'sample stream')


def reference_values(reference,n):
    equal(len(reference['records']),n,'reference records');equal(reference['all_tasks_audited'],n,'reference audit count')
    passed=[]
    for i,r in enumerate(reference['records']):
        equal(r['task_id'],i,'reference identity');hp=complex(*map(float,r['high_precision']));error=abs(complex(*r['analytic'])-hp)
        close(r['absolute_difference'],error,'reference discrepancy');passed.append(error<=1e-13+1e-12*abs(hp));equal(r['passed'],passed[-1],'reference guard')
    fallback=not all(passed);equal(reference['fallback_all'],fallback,'reference fallback');equal(reference['operational_cost'],0,'reference is evaluator overhead')
    return [complex(*map(float,r['high_precision'])) if fallback else complex(*r['analytic']) for r in reference['records']]


def evaluator(archive,state,intake,refs,previous):
    truth=copy.deepcopy(previous)
    for r in state['records']:
        if r['id'] in truth:continue
        info=archive['inputs'][r['input_index']];ref=refs[r['task_id']];error=abs(complex(*info['outcome']['value'])-ref)
        affected=sorted({a['channel'] for a in archive['result']['actions'] if a['kind']=='ordinary' and a['input_index']==r['input_index'] and a['fault_applied'] and any(e['kind']=='channel' and e['channel']==a['channel'] and e['time']==a['tick'] for e in r['original_links'])})
        truth[r['id']]=dict(accurate=error<=intake['requests'][r['task_id']]['request']['tolerance'],absolute_error=error,reference=[ref.real,ref.imag],affected_channels=affected,package_id=r['package_id'],input_index=r['input_index'])
    equal(archive['evaluator'],truth,'independently recovered truth labels')
    return truth


def summary(state,truth,actions,costs,duration,environment,target):
    accepted={r['id'] for r in state['records']};standing={r['id'] for r in state['records'] if r['standing']};accurate={r for r,t in truth.items() if t['accurate']};affected={r for r,t in truth.items() if t['affected_channels']}
    samples=[a for a in actions if a['kind']=='sample' and a['package_id']>=1];alarms=[a for a in actions if a['kind']=='sample' and a['mismatch']]
    corrupted={c for c in CHANNELS if c.startswith('collocation:')} if environment=='whole' else {target} if environment=='single' else set()
    detection=[a['package_id'] for a in alarms if a['channel'] in corrupted and a['package_id']>=1 and a['fault_applied']]
    return dict(initial_acceptances=len(accepted),initial_false_acceptances=len(accepted-accurate),false_remaining=len(standing-accurate),accurate_withdrawn=len((accepted-standing)&accurate),correct_final=len(standing&accurate),correct_final_ids=sorted(standing&accurate),initially_accurate_ids=sorted(accepted&accurate),standing_ids=sorted(standing),affected=len(affected),affected_false=len(affected-accurate),actual_cost=sum(costs.values()),costs=copy.deepcopy(costs),detection_delay=min(detection)-1 if detection else duration,detection_censored=not bool(detection),post_onset_samples=len(samples),corrupted_sample_fraction=sum(a['channel'] in corrupted for a in samples)/len(samples) if samples else None,clean_false_alarms=len(alarms) if environment=='clean' else None,clean_sample_expenditure=costs.get('sample',0.) if environment=='clean' else None,clean_false_alarm_cost=sum(a['cost'] for a in alarms)+sum(a['cost'] for a in actions if a['kind'] in ('recall','future')) if environment=='clean' else None)


def check_aggregate(saved,rows,scenarios):
    """Recalculate every primary aggregate and nonvacuous closure independently."""
    equal(saved['independent_series'],len(rows),'series unit')
    measures=('false_remaining','accurate_withdrawn','actual_cost','initial_acceptances','correct_final','accurate_ordinary_never_obtained','detection_delay','post_onset_samples')
    def stats(values):
        mean=sum(values)/len(values)
        return dict(per_series=values,total=sum(values),mean=mean,standard_error=math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1)/len(values)) if len(values)>1 else None)
    def stats_equal(actual,expected,label):
        equal(set(actual),set(expected),label+' keys')
        for k,v in expected.items():
            if k in ('mean','standard_error') and v is not None:close(actual[k],v,label+' '+k)
            else:equal(actual[k],v,label+' '+k)
    keys=set(rows[0]);equal(set(saved['cells']),keys,'aggregate cells');closures=[]
    for key in sorted(keys):
        cell=saved['cells'][key]
        for name in measures:stats_equal(cell[name],stats([r[key][name] for r in rows]),'aggregate '+name)
        for name in ('detection_censored','corrupted_sample_fraction'):equal(cell[name],[r[key][name] for r in rows],'aggregate '+name)
        for name in ('gross_lost','gross_gained','net'):stats_equal(cell['opportunity_cost'][name],stats([r[key]['opportunity_cost'][name] for r in rows]),'opportunity '+name)
        env,d,policy,q=key.split('-');baseline='-'.join((env,d,'S',q))
        if policy!='N0':
            for name in measures:stats_equal(saved['paired_schedule_minus_S'][key][name],stats([r[key][name]-r[baseline][name] for r in rows]),'paired '+name)
    for env,duration in scenarios:
        for q in (1,2,4):
            s=saved['cells'][f'{env}-d{duration}-S-q{q}'];r=saved['cells'][f'{env}-d{duration}-RR-q{q}']
            rr=all(r[n]['total']<=s[n]['total'] for n in measures[:2]);ss=all(s[n]['total']<=r[n]['total'] for n in measures[:2]);sc=s['actual_cost']['total'];rc=r['actual_cost']['total']
            expected=dict(S_minus_RR_actual_cost=sc-rc,RR_no_worse_on_outcomes=rr,S_no_worse_on_outcomes=ss,S_outcome_advantage_at_no_higher_cost=ss and not rr and sc<=rc,S_outcome_cost_dominance=ss and sc<=rc and (not rr or sc<rc))
            equal(saved['dominance'][f'{env}-d{duration}-q{q}'],expected,'outcome and actual-cost dominance')
            if env!='clean':closures.append(rr)
    unnecessary=bool(closures) and all(closures)
    equal(saved['closure'],dict(primary_cells=len(closures),graph_stakes_unnecessary=unnecessary,statement='Graph stakes are unnecessary for allocating self-checks in this laboratory.' if unnecessary else 'The registered closure criterion was not met.'),'nonvacuous primary closure')
    predictions=dict(S_whole_first_faulty_majority={str(q):sum(r[f'whole-d4-S-q{q}']['detection_delay']==0 and not r[f'whole-d4-S-q{q}']['detection_censored'] for r in rows)>len(rows)/2 for q in (1,2,4)},BU_whole_undetected_series={str(q):sum(r[f'whole-d4-BU-q{q}']['detection_censored'] for r in rows) for q in (1,2,4)},RR_whole_detected_series={str(q):sum(not r[f'whole-d4-RR-q{q}']['detection_censored'] for r in rows) for q in (1,2,4)},RR_later_than_S_quota1_penalized_delay=sum(r['whole-d4-RR-q1']['detection_delay'] for r in rows)>sum(r['whole-d4-S-q1']['detection_delay'] for r in rows),single_channel_status='tested' if any(e=='single' for e,d in scenarios) else 'untested_not_admitted',no_clean_false_alarms=all(v['clean_false_alarms']==0 for row in rows for key,v in row.items() if key.startswith('clean-')))
    single=next((d for e,d in scenarios if e=='single'),None)
    if single:predictions['single_false_acceptance_totals']={str(q):{p:sum(r[f'single-d{single}-{p}-q{q}']['false_remaining'] for r in rows) for p in ('S','Sx','Sxprime','RR')} for q in (1,2)}
    equal(saved['predictions'],predictions,'registered prediction summaries')



EXECUTED_BASELINE = '8fb29c76565dffb853256a2b1719823cce493689'
COMPATIBILITY = 'binding/results/B003/portability-compatibility.json'
COMPATIBLE_SOURCES = ('binding/run_b003.py', 'binding/audit_b003.py')
RUNNER_METADATA_FUNCTIONS = {'preflight', 'run', 'check_overlap'}
AUDIT_METADATA_FUNCTIONS = {'audit_directory', 'compare_directories'}


def source_inventory(head):
    import subprocess
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',head,'--','binding','attention','lab'],cwd=ROOT,text=True).splitlines()
    return sorted(p for p in paths if '/results/' not in p and p.endswith(('.py','.json')))


def git_bytes(head,path):
    import subprocess
    try:return subprocess.check_output(['git','show',f'{head}:{path}'],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:raise AuditError('Missing committed source '+path) from error


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def unchanged_function_proof(old,new,excluded,module_exact=False):
    """Prove both literal scientific text and AST, independently of record claims."""
    import ast
    before,after=ast.parse(old),ast.parse(new)
    old_nodes={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_nodes={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    proof={}
    for name,node in old_nodes.items():
        if name in excluded:continue
        equal(name in new_nodes,True,'scientific function retained '+name)
        left=ast.get_source_segment(old,node);right=ast.get_source_segment(new,new_nodes[name])
        equal(left,right,'scientific function text '+name)
        tree=ast.dump(node,include_attributes=False);equal(tree,ast.dump(new_nodes[name],include_attributes=False),'scientific function AST '+name)
        proof[name]=dict(text_sha256=byte_sha(left.encode()),ast_sha256=byte_sha(tree.encode()))
    if module_exact:
        def outside(tree,source):
            return [(ast.dump(n,include_attributes=False),ast.get_source_segment(source,n)) for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in excluded]
        equal(outside(before,old),outside(after,new),'runner entire module outside metadata functions')
    return proof


def validate_compatibility_record(record,old_sources,current_sources,old_text,new_text):
    """Pure validator used by negative tests and committed verification."""
    equal(record['old_head'],EXECUTED_BASELINE,'exact executed compatibility baseline')
    equal(set(old_sources),set(current_sources),'unchanged complete source inventory')
    equal(set(record['allowed_changes']),set(COMPATIBLE_SOURCES),'bounded two-source compatibility scope')
    changed={p for p in old_sources if old_sources[p]!=current_sources[p]}
    equal(changed,set(COMPATIBLE_SOURCES),'only declared metadata sources changed')
    for path,pair in record['allowed_changes'].items():
        equal(pair,dict(old_sha256=old_sources[path],new_sha256=current_sources[path]),'exact compatibility pair '+path)
        equal(byte_sha(old_text[path].encode()),pair['old_sha256'],'old compatibility source bytes')
        equal(byte_sha(new_text[path].encode()),pair['new_sha256'],'new compatibility source bytes')
    proof=dict(runner=unchanged_function_proof(old_text[COMPATIBLE_SOURCES[0]],new_text[COMPATIBLE_SOURCES[0]],RUNNER_METADATA_FUNCTIONS,True),auditor=unchanged_function_proof(old_text[COMPATIBLE_SOURCES[1]],new_text[COMPATIBLE_SOURCES[1]],AUDIT_METADATA_FUNCTIONS))
    equal(record['unchanged_scientific_functions'],proof,'independent scientific text and AST proof')
    equal(record['unchanged_sources_sha256'],{p:h for p,h in old_sources.items() if p not in COMPATIBLE_SOURCES},'complete unchanged core and instrument hashes')
    equal(record['schema'],'B003-path-verification-compatibility-v1','compatibility schema')
    return proof


def committed_compatibility():
    import subprocess
    path=ROOT/COMPATIBILITY;raw=path.read_bytes()
    equal(raw,git_bytes('HEAD',COMPATIBILITY),'compatibility record committed bytes')
    record=json.loads(raw)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',COMPATIBILITY],cwd=ROOT,text=True).strip()
    equal(subprocess.run(['git','merge-base','--is-ancestor',record['old_head'],commit],cwd=ROOT,capture_output=True).returncode,0,'compatibility chronology')
    return record,dict(source_compatibility_sha256=byte_sha(raw),source_compatibility_commit=commit)


CORRECTION_BASELINE = '72c445a35f2e2c6a10e1075b4b961c8ca00e07f6'
VERIFIER_CORRECTION = 'binding/results/B003/noise-verifier-correction.json'
CORRECTED_RUNNER_FUNCTIONS = {'run'}
CORRECTED_AUDIT_FUNCTIONS = {'score_stakes','check_package','verify_sources','audit_directory','compare_directories'}


def score_round(value,noise=False):
    """Apply the preregistered perturbation before, never after, rounding."""
    return round(float(np.nextafter(float(value),np.inf)) if noise else float(value),9)


def validate_verifier_correction(record,old_sources,current_sources,old_text,new_text):
    equal(record['schema'],'B003-noise-verifier-correction-v1','verifier correction schema')
    equal(record['old_head'],CORRECTION_BASELINE,'exact pre-correction execution source')
    equal(set(old_sources),set(current_sources),'unchanged correction source inventory')
    equal(set(record['allowed_changes']),set(COMPATIBLE_SOURCES),'bounded verifier and replay metadata scope')
    equal({p for p in old_sources if old_sources[p]!=current_sources[p]},set(COMPATIBLE_SOURCES),'only declared correction sources changed')
    for p,pair in record['allowed_changes'].items():
        equal(pair,dict(old_sha256=old_sources[p],new_sha256=current_sources[p]),'exact verifier correction pair '+p)
        equal(byte_sha(old_text[p].encode()),pair['old_sha256'],'historical correction source bytes')
        equal(byte_sha(new_text[p].encode()),pair['new_sha256'],'current correction source bytes')
    proof=dict(runner=unchanged_function_proof(old_text[COMPATIBLE_SOURCES[0]],new_text[COMPATIBLE_SOURCES[0]],CORRECTED_RUNNER_FUNCTIONS,True),auditor=unchanged_function_proof(old_text[COMPATIBLE_SOURCES[1]],new_text[COMPATIBLE_SOURCES[1]],CORRECTED_AUDIT_FUNCTIONS))
    equal(record['unchanged_scientific_functions'],proof,'independent correction scientific-function proof')
    equal(record['unchanged_sources_sha256'],{p:h for p,h in old_sources.items() if p not in COMPATIBLE_SOURCES},'unchanged producer instruments and libraries')
    return proof


def committed_verifier_correction():
    import subprocess
    raw=(ROOT/VERIFIER_CORRECTION).read_bytes()
    equal(raw,git_bytes('HEAD',VERIFIER_CORRECTION),'verifier correction record committed bytes')
    record=json.loads(raw)
    equal(record['original_portability_record_sha256'],byte_sha((ROOT/COMPATIBILITY).read_bytes()),'original portability record remains byte-identical')
    for path,h in record['preserved_original_failure_sha256'].items():equal(byte_sha((ROOT/path).read_bytes()),h,'preserved original failed verification artifact')
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',VERIFIER_CORRECTION],cwd=ROOT,text=True).strip()
    equal(subprocess.run(['git','merge-base','--is-ancestor',CORRECTION_BASELINE,commit],cwd=ROOT,capture_output=True).returncode,0,'post-execution verifier correction chronology')
    return record,dict(verifier_correction_sha256=byte_sha(raw),verifier_correction_commit=commit)


def score_diagnostic_projection(value,path=(),sample_action=False):
    """Normalize only the score locations independently checked by check_package."""
    if isinstance(value,list):return [score_diagnostic_projection(v,path+(str(i),),sample_action) for i,v in enumerate(value)]
    if isinstance(value,dict):
        action=len(path)>=3 and path[:2]==('result','actions') and path[2].isdigit()
        if action and len(path)==3:sample_action=value.get('kind')=='sample'
        registered=action and sample_action and (len(path)==3 or path[3:]==('selection_snapshot',) or (len(path)==6 and path[3:5]==('selection_snapshot','paired_graphs') and path[5] in ('S','Sx','Sxprime')))
        return {k:({c:'verified numerical score' for c in v} if k=='scores' and registered else score_diagnostic_projection(v,path+(k,),sample_action)) for k,v in value.items()}
    return value


def verify_score_cell(directory,relative_path):
    """Reconstruct the complete paired cell before accepting score-only drift."""
    root=Path(directory);relative=Path(relative_path)
    equal(len(relative.parts)==3 and relative.parts[0].startswith('series-') and relative.parts[-1].startswith('package-'),True,'scores belong to a raw package archive')
    manifest=read(root/'manifest.json');data=read(root/relative.parts[0]/'intake.json');check_intake(data)
    weights=read(ROOT/'attention/self_model_s3.json')['action_model']['cost_weights'];library=read(ROOT/'attention/sample_library_a007.json')
    prior=None;candidates={};count=0
    paths=sorted((root/relative.parent).glob('package-*.json.gz'))
    for path in paths:
        archive=read(path);p=archive['package_id'];equal(p,count,'complete score-cell chronology')
        candidates.update({f'{p}:{i}':info['outcome'] for i,info in enumerate(archive['inputs'])})
        prior=check_package(archive,data,dict(candidates=candidates),weights,library,prior,noise=manifest['last_bit_noise']);count+=1
    equal(count,archive['duration'],'complete independently verified score cell')
    return count


def validate_path_normalization(old_sources,current_sources,old_text,new_text):
    """Bound the path fix to its two returns and source-authentication bridge."""
    import ast
    equal(set(old_sources),set(current_sources),'complete protected B003 source inventory')
    auditor='binding/audit_b003.py'
    equal({p for p in old_sources if old_sources[p]!=current_sources[p]},{auditor},'only B003 path verifier changed')
    equal(byte_sha(old_text.encode()),old_sources[auditor],'path baseline source bytes')
    equal(byte_sha(new_text.encode()),current_sources[auditor],'path current source bytes')
    before,after=ast.parse(old_text),ast.parse(new_text)
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    expected=ast.get_source_segment(old_text,function(before,'overlap_path')).replace('return development/saved.relative_to(old_root)','return (development/saved.relative_to(old_root)).resolve()')
    equal(ast.dump(ast.parse(expected),include_attributes=False),ast.dump(ast.Module(body=[function(after,'overlap_path')],type_ignores=[]),include_attributes=False),'exact two path normalization changes')
    def protected(tree,source):
        exempt={'verify_sources','validate_path_normalization','overlap_path'}
        return [(ast.dump(n,include_attributes=False),ast.get_source_segment(source,n)) for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in exempt]
    equal(protected(before,old_text),protected(after,new_text),'all remaining B003 verifier text and AST unchanged')


def verify_sources(manifest):
    """Authenticate B003 history and the bounded path fix, ignoring isolated B004 additions."""
    import subprocess
    path_baseline='a76bde20a630042d14df04b5f67393f886ccaa25'
    head=manifest['git_head'];saved=manifest['source_sha256']
    equal(set(saved),set(source_inventory(head)),'complete executed source inventory')
    protected=set(source_inventory(path_baseline));current_inventory=set(source_inventory('HEAD'))
    equal(protected.issubset(saved) and protected.issubset(current_inventory),True,'all protected B003 sources retained')
    for path in (set(saved)|current_inventory)-protected:
        equal(path.startswith('binding/') and 'b004' in Path(path).stem.lower(),True,'only isolated B004 source additions')
    for path,h in saved.items():equal(byte_sha(git_bytes(head,path)),h,'executed committed source '+path)
    saved={path:saved[path] for path in protected}
    current={path:byte_sha((ROOT/path).read_bytes()) for path in protected}
    for path,h in current.items():equal(h,byte_sha(git_bytes('HEAD',path)),'current committed verifier source '+path)
    portability,port_metadata=committed_compatibility()
    correction,correction_metadata=committed_verifier_correction()
    baseline={p:byte_sha(git_bytes(EXECUTED_BASELINE,p)) for p in source_inventory(EXECUTED_BASELINE)}
    historical={p:byte_sha(git_bytes(CORRECTION_BASELINE,p)) for p in source_inventory(CORRECTION_BASELINE)}
    prepath={p:byte_sha(git_bytes(path_baseline,p)) for p in protected}
    baseline_text={p:git_bytes(EXECUTED_BASELINE,p).decode() for p in COMPATIBLE_SOURCES}
    historical_text={p:git_bytes(CORRECTION_BASELINE,p).decode() for p in COMPATIBLE_SOURCES}
    prepath_text={p:git_bytes(path_baseline,p).decode() for p in COMPATIBLE_SOURCES}
    current_text={p:(ROOT/p).read_text() for p in COMPATIBLE_SOURCES}
    validate_compatibility_record(portability,baseline,historical,baseline_text,historical_text)
    validate_verifier_correction(correction,historical,prepath,historical_text,prepath_text)
    validate_path_normalization(prepath,current,prepath_text['binding/audit_b003.py'],current_text['binding/audit_b003.py'])
    equal(saved in (baseline,historical,prepath,current),True,'exact authenticated source chain')
    metadata=dict(port_metadata,**correction_metadata)
    for group in (port_metadata,correction_metadata):
        sha_key=next(k for k in group if k.endswith('_sha256'));commit_key=next(k for k in group if k.endswith('_commit'))
        if sha_key in manifest:
            equal({k:manifest[k] for k in group},group,'recorded source verification identity')
            equal(subprocess.run(['git','merge-base','--is-ancestor',group[commit_key],head],cwd=ROOT,capture_output=True).returncode,0,'recorded verification committed before execution')
        elif commit_key in manifest:raise AuditError('Incomplete source verification identity')
    return metadata


def resolve_development(directory,manifest,override=None):
    saved=Path(manifest['development_directory'])
    preferred=Path(override) if override is not None else saved if saved.is_absolute() else ROOT/saved
    candidates=[preferred] if override is not None else [preferred,Path(directory).resolve().parent/'development-first',ROOT/'binding/results/B003/development-first']
    matches={p.resolve() for p in candidates if (p/'freeze.json').is_file() and byte_sha((p/'freeze.json').read_bytes())==manifest['freeze_sha256']}
    equal(len(matches),1,'unique development archive with exact freeze hash')
    return matches.pop()


def overlap_path(name,manifest,development):
    saved=Path(name);old_root=Path(manifest['development_directory'])
    if saved.is_absolute() and old_root.is_absolute() and saved.is_relative_to(old_root):
        return (development/saved.relative_to(old_root)).resolve()
    if not saved.is_absolute() and saved.is_relative_to(old_root):
        return (development/saved.relative_to(old_root)).resolve()
    equal(not saved.is_absolute(),True,'external overlap path is not an authenticated development descendant')
    resolved=(ROOT/saved).resolve();equal(resolved.is_relative_to(ROOT.resolve()),True,'overlap path within repository')
    return resolved


def verified_overlap_sources(manifest,development=None):
    required={p.resolve() for experiment in ('B001','B002') for p in (ROOT/'binding/results'/experiment).rglob('intake.json')}
    if development is not None:required.update((development/f'series-{i:02d}/intake.json').resolve() for i in range(12))
    paths={}
    for name,h in manifest['overlap']['sources'].items():
        if development is not None:path=overlap_path(name,manifest,development)
        else:
            path=(ROOT/name).resolve();equal(path.is_relative_to(ROOT.resolve()),True,'historical overlap source within repository')
        equal(path not in paths,True,'unambiguous overlap source paths');paths[path]=h
    equal(set(paths),required,'complete historical and twelve development overlap sources')
    for path,h in paths.items():equal(byte_sha(path.read_bytes()),h,'overlap source hash')
    return paths


def audit_directory(directory, development=None):
    import subprocess
    directory=Path(directory);manifest=read(directory/'manifest.json');stage=manifest['stage'];equal(stage in ('development','confirmation'),True,'registered stage')
    equal(read(directory/'progress.json'),dict(completed_series=12,status='complete'),'completed archive')
    source_verification=verify_sources(manifest)
    weights=read(ROOT/'attention/self_model_s3.json')['action_model']['cost_weights'];library=read(ROOT/'attention/sample_library_a007.json')
    policies=[['N0',0]] if stage=='development' else [['N0',0]]+[[p,q] for q in (1,2,4) for p in ('S','Sx','Sxprime','RR','BU')]
    equal(manifest['policies'],policies,'all registered policies and quotas')
    if stage=='development':expected_scenarios=[['single',16]]
    else:
        development=resolve_development(directory,manifest,development);frozen=read(development/'freeze.json')
        equal(hashlib.sha256((development/'freeze.json').read_bytes()).hexdigest(),manifest['freeze_sha256'],'exact committed admission freeze')
        audit_directory(development)
        single=frozen['single_channel']['selected_packages'];expected_scenarios=[['whole',4]]+([['single',single]] if single else [])
        expected_scenarios += [['clean',d] for d in sorted({d for _,d in expected_scenarios})]
    equal(manifest['scenarios'],expected_scenarios,'complete admitted scenarios')
    equal(manifest['N0_quota_references'],{str(q):'N0-q0' for q in (1,2,4)},'N0 shared references')

    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for key,path in [('protocol_sha256','binding/protocols/B003-self-checks.md'),('qualification_sha256','binding/results/B002/qualification-freeze.json'),('exposure_sha256','binding/results/B002/exposure-first/freeze.json')]:equal(manifest[key],sha(ROOT/path),'inherited '+key)
    def ancestor(a,b):return subprocess.run(['git','merge-base','--is-ancestor',a,b],cwd=ROOT,capture_output=True).returncode==0
    equal(ancestor(manifest['protocol_commit'],manifest['git_head']),True,'protocol-before-source chronology')
    equal(ancestor(manifest['protocol_latest_commit'],manifest['git_head']),True,'latest protocol-before-code chronology')
    if stage=='confirmation':equal(ancestor(manifest['freeze_commit'],manifest['git_head']),True,'freeze-before-confirmation chronology')
    historical=set()
    for exp in ('B001','B002'):
        for p in (ROOT/'binding/results'/exp).rglob('intake.json'):historical.update(r['request_sha256'] for r in read(p)['requests'])
    equal(bool(historical),True,'historical overlap evidence')
    overlap_sources=verified_overlap_sources(manifest,development if stage=='confirmation' else None)
    if stage=='confirmation':
        for path in overlap_sources:
            historical.update(r['request_sha256'] for r in read(path)['requests'])
    rows=[];datasets=[];seen=set();package_count=0
    dirs=sorted(directory.glob('series-*'));equal(len([p for p in dirs if p.is_dir()]),12,'twelve independent series')
    for index in range(12):
        base=directory/f'series-{index:02d}';data=read(base/'intake.json');check_intake(data);datasets.append(data)
        equal(data['seed'],20270201+index+(12 if stage=='confirmation' else 0),'registered seed')
        for r in data['requests']:
            h=r['request_sha256'];equal(h not in historical|seen,True,'exact request disjointness');seen.add(h)
        refs=reference_values(read(base/'reference.json'),len(data['requests']));summaries={};common={}
        expected={(env,d,p,q) for env,d in manifest['scenarios'] for p,q in manifest['policies']}
        equal({p.name for p in base.iterdir() if p.is_dir()},{f'{env}-d{d}-{p}-q{q}' for env,d,p,q in expected},'complete policy cells')
        for env,duration,policy,quota in sorted(expected):
            key=f'{env}-d{duration}-{policy}-q{quota}';state=None;truth={};actions=[];costs=dict.fromkeys(CATEGORIES,0.);candidates={}
            paths=sorted((base/key).glob('package-*.json.gz'));equal(len(paths),duration,'all packages retained')
            for p,path in enumerate(paths):
                a=read(path);package_count+=1
                for k,v in [('package_id',p),('environment',env),('duration',duration),('policy',policy),('quota',quota),('cell',key)]:equal(a[k],v,'archive '+k)
                equal(a['sample_seed'],data['sample_seeds'][p],'archived sample execution seed');equal(a['result']['sample_seed'],a['sample_seed'],'executed sample seed')
                fault=[] if env=='clean' or p==0 else [c for c in CHANNELS if c.startswith('collocation:')] if env=='whole' else [data['single_target']]
                equal(a['fault_channels'],fault,'registered evaluator fault')
                for action in a['result']['actions']:equal(action['fault_applied'],action['kind'] in ('ordinary','sample') and action['channel'] in fault,'injection isolation')
                common.setdefault(p,a['inputs']);equal(a['inputs'],common[p],'paired common native inputs and allowance')
                candidates.update({f'{p}:{i}':info['outcome'] for i,info in enumerate(a['inputs'])})
                state=check_package(a,data,dict(candidates=candidates),weights,library,state,noise=manifest['last_bit_noise'])
                truth=evaluator(a,state,data,refs,truth);actions.extend(dict(x,package_id=p) for x in a['result']['actions'])
                for c in CATEGORIES:costs[c]+=a['result']['costs'][c]
                if p+1==duration or stage=='development' and p+1 in (4,8,16):
                    name=f'{env}-d{p+1}-{policy}-q{quota}';summaries[name]=summary(state,truth,actions,costs,p+1,env,data['single_target'])
                    equal(read(base/(name+'-summary.json')),summaries[name],'per-duration primary summary')
        if stage=='confirmation':
            for key,s in summaries.items():
                env,d,policy,q=key.split('-');baseline=summaries[f'{env}-{d}-N0-q0'];current=set(s['correct_final_ids']);clean=set(baseline['correct_final_ids'])
                s['opportunity_cost']=dict(gross_lost=len(clean-current),gross_gained=len(current-clean),net=len(clean)-len(current))
                clean_ids=set(summaries[f'clean-{d}-N0-q0']['correct_final_ids'])
                s['contrast_against_clean_N0']=dict(gross_lost=len(clean_ids-current),gross_gained=len(current-clean_ids),net=len(clean_ids)-len(current))
                s['accurate_ordinary_never_obtained']=len(set(baseline['initially_accurate_ids'])-set(s['initially_accurate_ids']))
        equal(read(base/'summary.json'),summaries,'series summary');rows.append(summaries)
    equal(manifest['intake_sha256'],digest(datasets),'intake manifest digest');equal(read(directory/'series-summaries.json'),rows,'all independent series summaries')
    if stage=='development':
        freeze=read(directory/'freeze.json');candidates=[]
        for d in (4,8,16):
            counts={name:sum(row[f'single-d{d}-N0-q0'][name] for row in rows) for name in ('initial_acceptances','affected','affected_false')}
            candidates.append(dict(packages=d,**counts,passed=counts['affected']>=10 and counts['affected_false']>=5))
        equal(freeze['single_channel'],dict(candidates=candidates,selected_packages=next((r['packages'] for r in candidates if r['passed']),None)),'complete development admission gates')
        equal(freeze['status'],'freeze_before_confirmation','completed freeze')
        for name in ('source_sha256','protocol_sha256','qualification_sha256','exposure_sha256','intake_sha256'):equal(freeze[name],manifest[name],'frozen '+name)
        for path,h in freeze['artifact_sha256'].items():
            if path!='progress.json':equal(sha(directory/path),h,'frozen development artifact')
    else:
        report=read(directory/'aggregate.json');check_aggregate(report,rows,manifest['scenarios'])
        equal(report['target_strata'],{c:[i for i,d in enumerate(datasets) if d['single_target']==c] for c in CHANNELS if c.startswith('collocation:')},'target strata')
    return dict(passed=True,independent_series=12,packages=package_count,stage=stage,source_verification=source_verification,limitations=['Archived numerical evidence is checked arithmetically, not independently re-solved.'])


# Numerical values and raw diagnostics may drift without changing protected work.
DIAGNOSTICS={'value','values','reference','analytic','high_precision','absolute_error','absolute_difference','discrepancy','uncertainty','uncertainty_estimate','difference','last_difference','previous_difference','antecedent_ratio','rounding','native_rounding','convergence_margin','contraction_margin','decision_margin','threshold','unguarded_verdict','raw_verdict','elapsed_seconds','seconds','kernel_seconds','solver_noise_calls'}


EXACT_NUMERIC={'cost','costs','full_cost','spent','budget','unused_budget','ordinary_cost_before','reserve_before','denominators','scores','numerators','actual_cost','sample_cost','ordinary_cost','replacement_cost','recall_cost','future_cost','clean_sample_expenditure','clean_false_alarm_cost','S_minus_RR_actual_cost','mean','standard_error','total','per_series','corrupted_sample_fraction','maximum_cost'}


def decision_projection(value, path=()):
    if isinstance(value,list):return [decision_projection(v,path) for v in value]
    if isinstance(value,dict):return {k:decision_projection(v,path+(k,)) for k,v in value.items() if k not in DIAGNOSTICS}
    if isinstance(value,float) and not any(k in EXACT_NUMERIC for k in path):return '<finite numerical diagnostic>'
    return value


def compare_values(left,right):
    finite_tree(left);finite_tree(right)
    return dict(decisions_pass=decision_projection(left)==decision_projection(right),diagnostics_equal=left==right)


def compare_directories(left,right):
    left,right=Path(left),Path(right);lm,rm=read(left/'manifest.json'),read(right/'manifest.json')
    source_verification={}
    if lm['source_sha256']!=rm['source_sha256']:
        left_verification,right_verification=verify_sources(lm),verify_sources(rm)
        equal(bool(left_verification),True,'authorized source comparison transition')
        equal(left_verification,right_verification,'same exact compatibility proof for comparison')
        source_verification=left_verification
    else:
        for k in ('source_compatibility_sha256','source_compatibility_commit','verifier_correction_sha256','verifier_correction_commit'):equal(lm.get(k),rm.get(k),'comparison identity '+k)
    for k in ('intake_sha256','protocol_sha256','qualification_sha256','exposure_sha256','freeze_sha256'):equal(lm.get(k),rm.get(k),'comparison identity '+k)
    def files(root):
        return {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz')) and ('series-' in str(p.relative_to(root)) or p.name in ('aggregate.json','series-summaries.json'))}
    lf,rf=files(left),files(right);equal(lf,rf,'comparison complete archive file set');decisions=[];diagnostics=[];score_diagnostics=[];verified_cells=set();verified_source_roots=set()
    for name in sorted(lf):
        a,b=read(left/name),read(right/name);result=compare_values(a,b)
        if not result['decisions_pass']:
            same_operational=score_diagnostic_projection(decision_projection(a))==score_diagnostic_projection(decision_projection(b))
            if same_operational:
                for root in (left,right):
                    cell=(str(root.resolve()),str(Path(name).parent))
                    if cell not in verified_cells:
                        if cell[0] not in verified_source_roots:
                            verify_sources(read(root/'manifest.json'));verified_source_roots.add(cell[0])
                        verify_score_cell(root,name);verified_cells.add(cell)
                score_diagnostics.append(name)
            else:decisions.append(name)
        if not result['diagnostics_equal']:diagnostics.append(name)
    return dict(decisions_pass=not decisions,differing_decisions=decisions,diagnostics_pass=not diagnostics,differing_diagnostics=diagnostics,files_compared=len(lf),source_verification=source_verification,differing_verified_score_diagnostics=score_diagnostics,independently_verified_score_cells=len(verified_cells))


audit=audit_directory
compare=compare_directories


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--compare');parser.add_argument('--development');args=parser.parse_args()
    print(json.dumps(compare(args.directory,args.compare) if args.compare else audit(args.directory,args.development),indent=2))
