"""Independent A004 arithmetic, evidence and finite-likelihood audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from attention.io import read_json, write_json, source_bytes, source_path
from attention.audit_a003 import (decision, available, minimum_certificate, expected_price,
    scientific, close, recompute_calibration, ACTIONS, VERDICTS, CONDITIONS, FRACTIONS)

ROOT=Path(__file__).resolve().parents[1]
SCHEMES=('U0O','U1O','U0T','U1T','H','D','Dprime','HD','HDprime')
POLICIES=SCHEMES+('C2',)


def alphabet(action):
    return 4 if action.startswith('v2_') else 3 if action in ('modal','collocation') else 2


def category(action, observed):
    if observed['verdict']=='accept':return 0
    if observed['verdict']=='violation_detected':return 1
    return 3 if action.startswith('v2_') and observed.get('terminal') else 2


def observation(action, value):
    row=dict(verdict=VERDICTS[value] if value<2 else 'inconclusive')
    if action.startswith('v2_'):row['terminal']=value!=2
    return row


def likelihoods(model, relevant=False):
    """Construct registered categorical channel, independently of runtime code."""
    raw=np.asarray([row['outcomes'] for row in model['trajectories']],dtype=int)
    matrix=np.zeros((len(raw),len(ACTIONS),4))
    for i,values in enumerate(raw):
        for j,(action,value) in enumerate(zip(ACTIONS,values)):
            if value<0:continue
            size=alphabet(action)
            for outcome in range(size):matrix[i,j,outcome]=.02/size+(.98 if outcome==value else 0.)
    if relevant:
        a,b=(ACTIONS.index(x) for x in ('normalization','high_frequency'))
        left,right=matrix[:,a].copy(),matrix[:,b].copy()
        matrix[:,a]=.75*left+.25*right
        matrix[:,b]=.75*right+.25*left
    return raw,matrix


def forecast(weights,matrix,action):
    probabilities=weights@matrix[:,ACTIONS.index(action)]
    mass=float(sum(probabilities));size=alphabet(action)
    extension=mass<=0 or not math.isfinite(mass)
    values=[1/size]*size if extension else (probabilities[:size]/mass).tolist()
    return dict(eligible_mass=mass,missing_prediction_extension=extension,probabilities=values)


def nested_close(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(nested_close(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(nested_close(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):return close(a,b)
    return a==b


def posterior_checks(run,infos,model,policy):
    errors=[]
    def test(ok,label):
        if not ok:errors.append(label)
    raw,matrix=likelihoods(model,policy in ('D','HD'))
    groups=np.asarray([r['group_index'] for r in model['trajectories']])
    full=np.asarray([VERDICTS.index(decision({a:observation(a,int(v)) for a,v in zip(ACTIONS,row) if v>=0})) for row in raw])
    group_ids={r['id']:i for i,r in enumerate(model['groups'])}
    initial=run['predictions'][:len(infos)]
    test(len(initial)==len(infos),'initial prediction count')
    ids=[];weights=[];resets=[0]*len(infos);steps=[0]*len(infos);observed=[{} for _ in infos]
    def snapshot(index,row):
        w=weights[index];ix=ids[index]
        expected=dict(step=steps[index],full_probabilities={v:float(w@(full[ix]==n)) for n,v in enumerate(VERDICTS)},
            support_count=int(np.count_nonzero(w)),effective_support=float(1/(w@w)),fallback_count=resets[index],
            raw_feature_violation_mass={a:float(w@(raw[ix,ACTIONS.index(a)]==1)) for a in ('normalization','high_frequency')})
        test(row['input_index']==index,'forecast input index')
        for key,value in expected.items():test(nested_close(row.get(key),value),f'posterior {index}/{steps[index]}/{key}')
    for i,info in enumerate(infos):
        names=info['prediction']['neighbor_ids']
        test(len(names)==model['neighbors'] and len(set(names))==len(names),'neighbor count')
        near={group_ids[n] for n in names}
        ix=np.flatnonzero(np.isin(groups,list(near)));ids.append(ix)
        weights.append(np.ones(len(ix))/len(ix))
        test(len(ix)==model['neighbors']*9,'initial trace count')
        snapshot(i,initial[i])
    updates=run['predictions'][len(infos):]
    test(len(updates)==len(run['actions']),'one forecast per action')
    hybrid=policy in ('H','HD','HDprime');spent=0.;skipped=[];real_extensions=0
    def hybrid_choice(after_actions):
        candidates=sorted((expected_price(a,infos[i]),i,ACTIONS.index(a),a)
            for i,e in enumerate(observed) if decision(e)=='inconclusive'
            for a in available(e) if expected_price(a,infos[i])<=run['budget']-spent)
        for _,i,_,a in candidates:
            probabilities=forecast(weights[i],matrix[ids[i]],a)['probabilities']
            chance=sum(p for c,p in enumerate(probabilities) if decision(dict(observed[i],**{a:observation(a,c)}))!='inconclusive')
            if chance<.05:
                skipped.append(dict(after_actions=after_actions,input_index=i,action=a,decisive_probability=chance))
            else:return i,a,chance
        return None
    for n,(action,row) in enumerate(zip(run['actions'],updates)):
        i=action['input_index'];a=action['action'];c=category(a,action['observation'])
        if hybrid:
            choice=hybrid_choice(n)
            test(choice is not None and choice[:2]==(i,a),f'hybrid C2 threshold choice {n}')
            if choice:test(close(action['plan']['decisive_probability'],choice[2]),f'hybrid chance {n}')
        before=forecast(weights[i],matrix[ids[i]],a)
        test(nested_close(row['action_forecast_before'],before),'pre-observation forecast')
        real_extensions+=before['missing_prediction_extension']
        steps[i]+=1;reset=False
        if not policy.startswith('U0'):
            values=weights[i]*matrix[ids[i],ACTIONS.index(a),c]
            total=float(sum(values))
            if total<=0 or not math.isfinite(total):
                weights[i]=np.ones(len(ids[i]))/len(ids[i]);reset=True;resets[i]+=1
            else:weights[i]=values/total
        test(row.get('action')==a and row.get('category')==c,'update category')
        test(row.get('prior_reset')==reset,'prior fallback flag')
        test(row.get('reset_reason')==('zero_or_nonfinite_likelihood_mass' if reset else None),'prior fallback reason')
        observed[i][a]=action['observation'];spent+=action['cost'];snapshot(i,row)
        plan=action['plan']
        test(plan['action']==a,'plan action')
        test(0<=plan['expected_cost']<=run['budget']-(spent-action['cost'])+1e-12,'plan affordability')
        if policy in ('U0O','U1O'):test(plan['kind'] in ('ATOMIC','FULL'),'one-stage option set')
        if not hybrid:
            test(close(plan['score'],plan['expected_gain']/plan['expected_cost']),'plan gain per cost')
            test(0<=plan['expected_gain']<=1+1e-12,'plan gain probability')
        if plan['kind']=='TREE':
            parts=plan['branches'];test(close(sum(b['probability'] for b in parts),1),'tree branch probabilities')
            gain=sum(b['probability']*(b['immediate_reward']+(b['continuation']['expected_gain'] if b['continuation'] else 0)) for b in parts)
            cost=action['cost']+sum(b['probability']*(b['continuation']['expected_cost'] if b['continuation'] else 0) for b in parts)
            test(close(plan['expected_gain'],gain) and close(plan['expected_cost'],cost),'tree arithmetic')
    if hybrid:
        test(hybrid_choice(len(run['actions'])) is None,'hybrid final stop')
        test(nested_close(run['skipped'],skipped),'hybrid skipped audit')
    expected=dict(prior_resets=sum(resets),minimum_support=min(p['support_count'] for p in run['predictions']),missing_predictions=0)
    expected['real_prediction_extensions']=real_extensions
    for key,value in expected.items():test(run['support_audit'].get(key)==value,'support aggregate '+key)
    total=run['support_audit']['hypothetical_prediction_queries'];missing=run['support_audit']['hypothetical_prediction_extensions']
    test(isinstance(total,int) and isinstance(missing,int) and 0<=missing<=total,'hypothetical extension count bounds')
    return errors


def intervention_checks(actual,model):
    errors=[]
    raw,base=likelihoods(model);_,relevant=likelihoods(model,True)
    delta=np.abs(relevant-base)
    expected=dict(epsilon=.02,alpha=.25,hybrid_threshold=.05,D_changed_numbers=int(np.count_nonzero(delta)),
        Dprime_changed_numbers=int(np.count_nonzero(delta)),D_L1=float(delta.sum()),Dprime_L1=float(delta.sum()),
        equal_changes=True,Dprime_active_unchanged=True,D_unused_unchanged=True)
    if not nested_close(actual,expected):errors.append('intervention numeric size and inactive invariance')
    if np.any(relevant[base>0]<=0):errors.append('corruption erased originally supported emission')
    return errors


def source_checks(data,check):
    canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    check(data['self_model_sha256']==canonical(read_json(ROOT/'attention/self_model_s3.json')),'fixed S3 identity')
    check(data['static_model_sha256']==canonical(read_json(ROOT/'attention/self_model_s2.json')),'fixed cost-model identity')
    protocol=ROOT/'attention/protocols/A004-factorial-knowledge.md'
    check(data['protocol_sha256']==hashlib.sha256(protocol.read_bytes()).hexdigest(),'protocol digest')
    for path,digest in data['source_sha256'].items():
        check(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,f'source digest {path}')



def summary_checks(data,totals,ceilings,changed):
    errors=[]
    def test(ok,label):
        if not ok:errors.append(label)
    pairs=(('U1O','U0O'),('U1T','U0T'),('H','C2'),('D','Dprime'),('HD','HDprime'))
    for f in FRACTIONS:
        summary=data['summary'][f];oracle=sum(ceilings[f])
        test(summary['oracle_correct']==oracle and summary['oracle_by_package']==ceilings[f],f'oracle summary {f}')
        expected_changed={p+'_versus_'+('H' if p.startswith('H') else 'U1T'):n for p,n in changed[f].items()}
        test(summary['changed_action_packages']==expected_changed,f'changed package summary {f}')
        for p,out in totals[f].items():
            expected=dict(out,oracle_fraction=out['correct']/oracle if oracle else None)
            test(summary['policies'][p]==expected,f'policy summary {f}/{p}')
        expected_pairs={a+'_minus_'+b:[x-y for x,y in zip(totals[f][a]['correct_by_package'],totals[f][b]['correct_by_package'])] for a,b in pairs}
        test(summary['paired']==expected_pairs,f'paired summary {f}')
        for p in SCHEMES:
            test(data['calibration'][f][p]==recompute_calibration(data['packages'],f,p),f'calibration {f}/{p}')
    variants=('U0O','U1O','U0T','U1T','H')
    winners={f:[p for p in variants if totals[f][p]['correct']>totals[f]['C2']['correct']] for f in ('0.2','0.4')}
    both=[p for p in variants if p in winners['0.2'] and p in winners['0.4']]
    combined={p:totals['0.2'][p]['correct']+totals['0.4'][p]['correct'] for p in (*variants,'C2')}
    expected=dict(close_budget_series=not both,variants_beating_C2_at_both=both,variants_beating_C2_by_budget=winners,
        combined_20_40_correct=combined,variants_beating_C2_combined=[p for p in variants if combined[p]>combined['C2']])
    test(data['stopping_rule']==expected,'registered stopping rule and sensitivities')
    return errors


def cheap_order_checks(run,infos):
    observed=[{} for _ in infos];spent=0.;errors=[]
    def choose():
        pairs=[(expected_price(a,infos[i]),i,ACTIONS.index(a),a)
            for i,e in enumerate(observed) if decision(e)=='inconclusive'
            for a in available(e) if expected_price(a,infos[i])<=run['budget']-spent]
        return min(pairs) if pairs else None
    for row in run['actions']:
        chosen=choose()
        if chosen is None or (chosen[1],chosen[3])!=(row['input_index'],row['action']):errors.append('C2 cheapest order')
        observed[row['input_index']][row['action']]=row['observation'];spent+=row['cost']
    if choose() is not None:errors.append('C2 premature termination')
    return errors


def initial_predictions(request,model):
    # The unchanged feature extractor is shared; nearest ordering and forecasts are not.
    from attention import s2
    vector=s2.features(request,s2.load_model())[0]
    nearest=sorted(range(len(model['groups'])),key=lambda i:(sum((x-y)**2 for x,y in zip(vector,model['groups'][i]['features'])),i))[:model['neighbors']]
    ids=np.array([i for i,r in enumerate(model['trajectories']) if r['group_index'] in nearest],dtype=int)
    output=[]
    for relevant in (False,True):
        raw,matrix=likelihoods(model,relevant);weights=np.ones(len(ids))/len(ids)
        full=np.asarray([VERDICTS.index(decision({a:observation(a,int(v)) for a,v in zip(ACTIONS,row) if v>=0})) for row in raw[ids]])
        output.append(dict(step=0,full_probabilities={v:float(weights@(full==i)) for i,v in enumerate(VERDICTS)},
            support_count=len(ids),effective_support=float(1/(weights@weights)),fallback_count=0,
            raw_feature_violation_mass={a:float(weights@(raw[ids,ACTIONS.index(a)]==1)) for a in ('normalization','high_frequency')},
            neighbor_ids=[model['groups'][i]['id'] for i in nearest],feature_vector=vector,
            action_forecasts={a:forecast(weights,matrix[ids],a) for a in ACTIONS}))
    return output


def audit(data, check_sources=True):
    failures=[]; checks=0
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not ok:failures.append(label)
    check(data.get('experiment')=='A004','experiment')
    check([data.get(k) for k in ('task_seed','order_seed','random_policy_seed')]==[20261138,20261139,20261140],'seeds')
    records=data['intake_channel']; packages=data['packages']
    check(len(records)==60 and len(packages)==12,'sample size')
    check(not data['full_pipeline_parity_failures'],'full P parity archive')
    check(data['reference_audit']['all_tasks_audited']==60,'reference count')
    reference_rows=data['reference_audit']['records']
    check(len(reference_rows)==60,'reference record count')
    check(data['reference_audit']['fallback_all']==(not all(r['passed'] for r in reference_rows)),'reference fallback rule')
    for row in reference_rows:
        diff=abs(complex(*row['analytic'])-complex(*map(float,row['high_precision'])))
        check(close(diff,row['absolute_difference']),'reference numerical difference')
        check(row['passed']==(diff<=1e-13+1e-12*abs(complex(*map(float,row['high_precision'])))),'reference tolerance')
    keys=[]
    for i,r in enumerate(records):
        check(r['task_id']==i and r['package_id']==i//5 and r['cohort']=='WHMOS'[i%5],f'intake order {i}')
        check(sorted(r['condition_order'])==sorted(CONDITIONS),f'conditions {i}')
        q=r['request'];check(q['budget']==4096 and q['max_calls']==3,f'limits {i}')
        digest=hashlib.sha256(json.dumps(q,sort_keys=True,allow_nan=False).encode()).hexdigest()
        check(digest==r['request_sha256'],f'request hash {i}')
        keys.append((tuple(q['z']),q['mean_time'],q['tolerance']))
    check(len(set(keys))==len(keys),'unique requests')
    permutations=data['global_permutations']
    expected=sorted((i,a) for i in range(45) for a in ACTIONS)
    check(len(permutations)==12 and all(sorted(tuple(v) for v in p)==expected for p in permutations),'global permutations')
    digest=hashlib.sha256(json.dumps([records,permutations],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    check(digest==data['intake_sha256'],'intake digest')
    totals={f:{p:dict(correct=0,false_accepts=0,accepts=0,correct_accepts=0,correct_detections=0,spent=0.,correct_by_package=[],spent_by_package=[]) for p in POLICIES} for f in FRACTIONS}
    ceilings={f:[] for f in FRACTIONS}; changed={f:{p:0 for p in ('D','Dprime','HD','HDprime')} for f in FRACTIONS}
    model=read_json(ROOT/'attention/self_model_s3.json')
    support_stats={f:{p:dict(prior_resets=0) for p in SCHEMES} for f in FRACTIONS}
    initial_cache={}
    for pi,package in enumerate(packages):
        check(package['package_id']==pi,f'package order {pi}')
        infos=package['inputs'];check(len(infos)==45,f'package length {pi}')
        for j,info in enumerate(infos):
            task=pi*5+j//9
            check(info['task_id']==task and info['condition']==records[task]['condition_order'][j%9],f'input order {pi}/{j}')
            if task not in initial_cache:initial_cache[task]=initial_predictions(records[task]['request'],model)
            for key,pred in zip(('prediction','D_prediction'),initial_cache[task]):
                check(nested_close(info[key],pred),f'initial neighbors and prediction {pi}/{j}/{key}')
            accurate=abs(complex(*info['outcome']['value'])-complex(*info['reference']))<=info['tolerance']
            check(accurate==info['accurate'],f'accuracy {pi}/{j}')
            obs={};full=info['full']
            for row in full['actions']:
                a=row['action']
                if a=='declare':continue
                check(a in available(obs),f'full dependencies {pi}/{j}/{a}')
                check(row['cost']==expected_price(a,info),f'full price {pi}/{j}/{a}')
                obs[a]=row['observation']
            check(decision(obs)==full['verdict'],f'full verdict {pi}/{j}')
            check(sum(r['cost'] for r in full['actions'])==full['spent'],f'full accounting {pi}/{j}')
            if full['verdict']=='accept':check(accurate,f'full false acceptance {pi}/{j}')
        reference_cost=sum(i['full']['spent'] for i in infos)
        check(reference_cost==package['reference_cost'],f'reference cost {pi}')
        prices=[minimum_certificate(i) for i in infos]
        check([r['cost'] for r in package['oracle_certificates']]==prices,f'oracle certificates {pi}')
        for f in FRACTIONS:
            spent=0.;selected=[]
            for price,j in sorted((v,j) for j,v in enumerate(prices) if v is not None):
                if spent+price>float(f)*reference_cost:break
                spent+=price;selected.append(j)
            ceiling=len(selected);ceilings[f].append(ceiling)
            check(package['oracle'][f]==dict(correct=ceiling,spent=spent,selected=selected),f'oracle prefix {pi}/{f}')
            check(set(package['policies'][f])==set(POLICIES),f'policies {pi}/{f}')
            for policy in POLICIES:
                run=package['policies'][f][policy];label=f'{pi}/{f}/{policy}'
                budget=float(f)*reference_cost
                check(run['budget']==budget and 0<=run['spent']<=budget,f'budget {label}')
                check(len(run['inputs'])==45,f'run input length {label}')
                global_rows=[[] for _ in infos];global_spent=0.
                for row in run['actions']:
                    global_rows[row['input_index']].append(row);global_spent+=row['cost']
                check(global_spent==run['spent'],f'global cost {label}')
                count=false=accepts=correct_accepts=correct_detections=0
                for j,(result,info) in enumerate(zip(run['inputs'],infos)):
                    check(result['actions']==global_rows[j],f'global input agreement {label}/{j}')
                    obs={}
                    for row in result['actions']:
                        a=row['action']
                        check(decision(obs)=='inconclusive',f'no action after decision {label}/{j}')
                        check(a in available(obs),f'dependencies {label}/{j}/{a}')
                        check(row['cost']==expected_price(a,info),f'price {label}/{j}/{a}')
                        obs[a]=row['observation']
                        archived=next(r['observation'] for r in info['full']['actions'] if r['action']==a)
                        check(obs[a]==archived,f'fresh observation agreement {label}/{j}/{a}')
                    check(obs==result['observed'],f'observation storage {label}/{j}')
                    verdict=decision(obs)
                    check(verdict==result['verdict'],f'verdict {label}/{j}')
                    correct=(info['accurate'] and verdict=='accept') or (not info['accurate'] and verdict=='violation_detected')
                    bad=(not info['accurate'] and verdict=='accept')
                    check(result['correct']==correct and result['false_accept']==bad,f'correctness {label}/{j}')
                    check(sum(r['cost'] for r in result['actions'])==result['spent'],f'input cost {label}/{j}')
                    count+=correct;false+=bad;accepts+=verdict=='accept'
                    correct_accepts+=bool(info['accurate'] and verdict=='accept')
                    correct_detections+=bool(not info['accurate'] and verdict=='violation_detected')
                    if verdict=='accept':check(info['full']['verdict']=='accept',f'full support {label}/{j}')
                check((count,false,accepts)==(run['correct'],run['false_accepts'],run['accepts']),f'counts {label}')
                check(run['correct_accepts']==correct_accepts and run['correct_detections']==correct_detections,f'secondary counts {label}')
                check(false==0,f'false acceptance {label}')
                check(count<=ceiling,f'ceiling bound {label}')
                check(sum(r['spent'] for r in run['inputs'])==run['spent'],f'summed input cost {label}')
                out=totals[f][policy]
                for k,n in [('correct',count),('false_accepts',false),('accepts',accepts),('correct_accepts',correct_accepts),('correct_detections',correct_detections),('spent',run['spent'])]:out[k]+=n
                out['correct_by_package'].append(count);out['spent_by_package'].append(run['spent'])
                if policy=='C2':
                    check(not cheap_order_checks(run,infos),f'C2 independent ordering {label}')
                if policy in SCHEMES:
                    if model is not None:
                        errors=posterior_checks(run,infos,model,policy)
                        check(not errors,f'Bayesian evidence audit {label}: {errors[:5]}')
                    support_stats[f][policy]['prior_resets'] += run['support_audit']['prior_resets']
            for base,other in [('U1T','Dprime'),('H','HDprime')]:
                check(scientific(package['policies'][f][base])==scientific(package['policies'][f][other]),f'Dprime invariance {pi}/{f}/{base}')
            for p in changed[f]:
                base='H' if p.startswith('H') else 'U1T'
                changed[f][p]+=[(r['input_index'],r['action']) for r in package['policies'][f][base]['actions']]!=[(r['input_index'],r['action']) for r in package['policies'][f][p]['actions']]
    summary_errors=summary_checks(data,totals,ceilings,changed)
    for error in summary_errors:check(False,error)
    check(not summary_errors,'summaries, calibration and stopping rule')
    intervention_errors=intervention_checks(data['intervention_audit'],model)
    for error in intervention_errors:check(False,error)
    check(not intervention_errors,'matched intervention')
    if check_sources:
        source_checks(data,check)
        paths={'A001':'attention/results/A001/results.json','A002':'attention/results/A002/results.json.gz','E016':'results/E016-frozen-pipeline/results.json','A003':'attention/results/A003/results.json.gz'}
        for name,path in paths.items():
            earlier=read_json(ROOT/path); old={(tuple(r['request']['z']),r['request']['mean_time'],r['request']['tolerance']) for r in earlier['intake_channel']}
            matches=sum(k in old for k in keys)
            check(matches==0 and data['request_overlap']['comparisons'][name]['exact_request_matches']==matches,f'overlap {name}')
            check(data['request_overlap']['comparisons'][name]['source_sha256']==hashlib.sha256(source_bytes(ROOT/path)).hexdigest(),f'overlap source {name}')
        check(data['request_overlap']['within_A004_duplicate_requests']==0,'reported internal overlap')
        for name,row in data['request_overlap']['comparisons'].items():
            if row.get('status')!='not_task_intake':check(row['exact_request_matches']==0,f'other overlap status {name}')
    return dict(passed=not failures,checks=checks,failures=failures,totals=totals,oracle_by_package=ceilings,
                changed_action_packages=changed,support_statistics=support_stats,
                scope='Independent archived-evidence arithmetic, support, dependency, oracle and summary audit. Physical solvers and high-precision reference generation are not rerun by this audit.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refuse to overwrite an audit')
    report=audit(read_json(args.archive))
    report['source_sha256']=hashlib.sha256(source_path(args.archive).read_bytes()).hexdigest()
    write_json(args.output,report)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
