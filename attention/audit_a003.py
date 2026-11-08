"""Independent archived-evidence checks for A003; no new solver outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from attention.io import read_json, write_json, source_bytes, source_path

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (64,128,256,512,1024,2048,4096)
FEATURES = ('normalization','first_moment','second_moment','high_frequency')
REQUIRED = ('modal','binding') + FEATURES + ('collocation',)
ACTIONS = ('modal',) + tuple('v2_'+str(n) for n in GRIDS) + ('binding',) + FEATURES + ('collocation',)
VERDICTS = ('accept','violation_detected','inconclusive')
POLICIES = ('B2','B','C2','C4','D','Dprime')
FRACTIONS = ('0.2','0.4','0.6','1.0')
CONDITIONS = ('clean','conjugate','renormalized','time_units','near_tolerance','stale_value','model_scale','slab','mixture')


def decision(observed):
    if any(v['verdict']=='violation_detected' for v in observed.values()):
        return 'violation_detected'
    supported = all(a in observed and observed[a]['verdict']=='accept' for a in REQUIRED)
    terminal = [v for a,v in observed.items() if a.startswith('v2_') and v.get('terminal')]
    return 'accept' if supported and terminal and terminal[-1]['verdict']=='accept' else 'inconclusive'


def available(observed):
    answer = [a for a in ('modal','binding')+FEATURES if a not in observed]
    if 'modal' not in observed:return answer
    if 'collocation' not in observed:answer.append('collocation')
    for grid in GRIDS:
        a='v2_'+str(grid)
        if a in observed:
            if observed[a].get('terminal'):break
        else:
            answer.append(a);break
    return answer


def minimum_certificate(info):
    rows={r['action']:r for r in info['full']['actions'] if r['action']!='declare'}
    if info['accurate']:
        return sum(r['cost'] for r in rows.values()) if decision({a:r['observation'] for a,r in rows.items()})=='accept' else None
    prices=[]
    for a,row in rows.items():
        if row['observation']['verdict']!='violation_detected':continue
        needs={a}
        if a=='collocation':needs.add('modal')
        if a.startswith('v2_'):
            needs.add('modal')
            needs.update('v2_'+str(n) for n in GRIDS if n<=int(a[3:]))
        if needs.issubset(rows):prices.append(sum(rows[n]['cost'] for n in sorted(needs,key=ACTIONS.index)))
    return min(prices) if prices else None


def expected_price(action, info):
    if action=='modal':return info['outcome']['modes_spent']/4096
    if action=='binding':return 1/4096
    if action in ('normalization','first_moment'):return (4096+512)/4096
    if action=='second_moment':return 4.
    if action=='high_frequency':return 2.
    if action=='collocation':return (64**3+128**3)/262144
    return int(action[3:])/4096


def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='policy' and not k.endswith('seconds')}
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


def close(a,b):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12))


def recompute_calibration(packages, fraction, policy):
    steps={}; package_rows=[]
    for package in packages:
        scores=[]; missing=0
        for row in package['policies'][fraction][policy]['predictions']:
            group=steps.setdefault(str(row['step']),dict(count=0,missing=0,total=0.))
            p=row['full_probabilities']
            if p is None:group['missing']+=1;missing+=1;continue
            target=package['inputs'][row['input_index']]['full']['verdict']
            value=sum((p[v]-(v==target))**2 for v in VERDICTS)
            group['count']+=1;group['total']+=value;scores.append(value)
        package_rows.append(dict(package_id=package['package_id'],count=len(scores),missing=missing,brier=sum(scores)/len(scores) if scores else None))
    return dict(by_step={k:dict(count=v['count'],missing=v['missing'],brier=v['total']/v['count'] if v['count'] else None) for k,v in steps.items()},by_package=package_rows)


def posterior_checks(run, infos, model, policy):
    """Rebuild finite Bayes filtering without calling the model or scheduler."""
    errors=[]
    def test(ok,label):
        if not ok:errors.append(label)
    mapped=[]; full=[]; raw=[]; groups=[]; accurate=[]
    for trajectory in model['trajectories']:
        values=list(trajectory['outcomes']);raw.append(values);groups.append(trajectory['group_index'])
        values=[(1-v if policy=='D' and a in ('normalization','high_frequency') and v in (0,1) else v) for a,v in zip(ACTIONS,values)]
        mapped.append(values);accurate.append(trajectory['accurate'])
        evidence={a:dict(verdict=VERDICTS[v] if v<2 else 'inconclusive',**({'terminal':v!=2} if a.startswith('v2_') else {})) for a,v in zip(ACTIONS,values) if v>=0}
        full.append(decision(evidence))
    group_ids={row['id']:i for i,row in enumerate(model['groups'])}
    support=[];evidence=[{} for _ in infos];expanded=[False for _ in infos];steps=[0 for _ in infos]
    initial=run['predictions'][:len(infos)]
    def probabilities(ids):
        return {v:sum(full[i]==v for i in ids)/len(ids) for v in VERDICTS} if ids else None
    def masses(ids):
        return {a:sum(raw[i][ACTIONS.index(a)]==1 for i in ids)/len(ids) for a in ('normalization','high_frequency')} if ids else None
    for j,info in enumerate(infos):
        prediction=info['D_prediction'] if policy=='D' else info['prediction']
        near={group_ids[n] for n in prediction['neighbor_ids']}
        ids=[i for i,g in enumerate(groups) if g in near];support.append(ids)
        row=initial[j]
        test(row['input_index']==j and row['step']==0,'initial forecast index')
        test(row['support_count']==len(ids)==144,'initial support')
        test(row['full_probabilities']==probabilities(ids),'initial probabilities')
        test(row['raw_feature_violation_mass']==masses(ids),'initial feature mass')
    updates=run['predictions'][len(infos):]
    test(len(updates)==len(run['actions']),'one forecast per observed action')
    fallback=False;expansions=[];trigger=None
    for n,(action,row) in enumerate(zip(run['actions'],updates)):
        j=action['input_index'];a=action['action'];obs=action['observation']
        category=(0 if obs['verdict']=='accept' else 1 if obs['verdict']=='violation_detected' else 3 if a.startswith('v2_') and obs.get('terminal') else 2)
        steps[j]+=1
        test(row['input_index']==j and row['step']==steps[j] and row['action']==a and row['category']==category,f'update index {n}')
        if fallback:
            test(row['full_probabilities'] is None and row['support_count'] is None and row['fallback_active'],f'missing after fallback {n}')
            continue
        prior_mass=masses(support[j]);evidence[j][a]=category
        ids=[i for i in support[j] if mapped[i][ACTIONS.index(a)]==category]
        expanded_now=False
        if not ids and not expanded[j]:
            expanded[j]=expanded_now=True
            ids=[i for i in range(len(mapped)) if all(mapped[i][ACTIONS.index(k)]==v for k,v in evidence[j].items())]
            expansions.append(dict(input_index=j,step=steps[j],action=a))
        support[j]=ids
        if not ids:
            fallback=True;trigger=dict(input_index=j,step=steps[j],action=a,category=category,global_action_index=n)
        test(row['support_count']==len(ids) and row['expanded']==expanded[j] and row['expanded_now']==expanded_now and row['failed']==(not ids),f'posterior support {n}')
        test(row['full_probabilities']==probabilities(ids),f'posterior probabilities {n}')
        test(row['raw_feature_violation_mass']==masses(ids) and row['raw_feature_violation_mass_before']==prior_mass,f'feature update {n}')
        test(row['fallback_active']==fallback,f'fallback state {n}')
    expected=dict(expansions=expansions,expanded_inputs=len(expansions),package_fallback=fallback,fallback_trigger=trigger,
                  missing_predictions=sum(p['full_probabilities'] is None for p in run['predictions']))
    test(run['support_audit']==expected,'support audit aggregate')
    return errors


def audit(data, check_sources=True):
    failures=[]; checks=0
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not ok:failures.append(label)
    check(data.get('experiment')=='A003','experiment')
    check([data.get(k) for k in ('task_seed','order_seed','random_policy_seed')]==[20261135,20261136,20261137],'seeds')
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
    totals={f:{p:dict(correct=0,false_accepts=0,accepts=0,spent=0.,correct_by_package=[],spent_by_package=[]) for p in POLICIES} for f in FRACTIONS}
    ceilings={f:[] for f in FRACTIONS}; changed={f:{p:0 for p in ('D','Dprime')} for f in FRACTIONS}
    model=read_json(ROOT/'attention/self_model_s3.json') if check_sources else None
    support_stats={f:{p:dict(missing_predictions=0,expanded_predictions=0) for p in ('B2','D','Dprime')} for f in FRACTIONS}
    for pi,package in enumerate(packages):
        check(package['package_id']==pi,f'package order {pi}')
        infos=package['inputs'];check(len(infos)==45,f'package length {pi}')
        for j,info in enumerate(infos):
            task=pi*5+j//9
            check(info['task_id']==task and info['condition']==records[task]['condition_order'][j%9],f'input order {pi}/{j}')
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
                count=false=accepts=0
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
                    if verdict=='accept':check(info['full']['verdict']=='accept',f'full support {label}/{j}')
                check((count,false,accepts)==(run['correct'],run['false_accepts'],run['accepts']),f'counts {label}')
                check(false==0,f'false acceptance {label}')
                check(count<=ceiling,f'ceiling bound {label}')
                check(sum(r['spent'] for r in run['inputs'])==run['spent'],f'summed input cost {label}')
                out=totals[f][policy]
                for k,n in [('correct',count),('false_accepts',false),('accepts',accepts),('spent',run['spent'])]:out[k]+=n
                out['correct_by_package'].append(count);out['spent_by_package'].append(run['spent'])
                if policy in ('B2','D','Dprime'):
                    if model is not None:
                        errors=posterior_checks(run,infos,model,policy)
                        check(not errors,f'Bayesian evidence audit {label}: {errors[:5]}')
                    seen=set()
                    for row in run['predictions']:
                        key=(row['input_index'],row['step']);check(key not in seen,f'duplicate forecast {label}/{key}');seen.add(key)
                        p=row['full_probabilities']
                        if p is None:support_stats[f][policy]['missing_predictions']+=1
                        else:check(set(p)==set(VERDICTS) and all(0<=v<=1 for v in p.values()) and close(sum(p.values()),1),f'probabilities {label}/{key}')
                        support_stats[f][policy]['expanded_predictions']+=bool(row.get('expanded'))
                    check(all((j,0) in seen for j in range(45)),f'initial predictions {label}')
            b=package['policies'][f]['B2'];dp=package['policies'][f]['Dprime']
            check(scientific(b)==scientific(dp),f'Dprime invariance {pi}/{f}')
            for p in changed[f]:changed[f][p]+=[(r['input_index'],r['action']) for r in b['actions']]!=[(r['input_index'],r['action']) for r in package['policies'][f][p]['actions']]
    for f in FRACTIONS:
        summary=data['summary'][f];oracle=sum(ceilings[f])
        check(summary['oracle_correct']==oracle and summary['oracle_by_package']==ceilings[f],f'oracle summary {f}')
        check(summary['changed_action_packages']==changed[f],f'changed package summary {f}')
        for p,out in totals[f].items():
            out['oracle_fraction']=out['correct']/oracle if oracle else None
            check(summary['policies'][p]==out,f'policy summary {f}/{p}')
            if p!='B2':check(summary['paired_B2_minus'][p]==[a-b for a,b in zip(totals[f]['B2']['correct_by_package'],out['correct_by_package'])],f'paired {f}/{p}')
        check(summary['paired_D_minus_Dprime']==[a-b for a,b in zip(totals[f]['D']['correct_by_package'],totals[f]['Dprime']['correct_by_package'])],f'D pair {f}')
        for p in ('B2','D','Dprime'):
            rebuilt=recompute_calibration(packages,f,p)
            actual=data['calibration'][f][p]
            check(actual==rebuilt,f'calibration {f}/{p}')
    intervention=data['intervention_audit']
    for key in ('D_changed_numbers','Dprime_changed_numbers','D_L1','Dprime_L1'):
        check(intervention.get(key)==4,f'intervention {key}')
    check(intervention.get('equal_changes') is True,'matched corruption')
    if check_sources:
        paths={'A001':'attention/results/A001/results.json','A002':'attention/results/A002/results.json.gz','E016':'results/E016-frozen-pipeline/results.json'}
        for name,path in paths.items():
            earlier=read_json(ROOT/path); old={(tuple(r['request']['z']),r['request']['mean_time'],r['request']['tolerance']) for r in earlier['intake_channel']}
            matches=sum(k in old for k in keys)
            check(matches==0 and data['request_overlap']['comparisons'][name]['exact_request_matches']==matches,f'overlap {name}')
            check(data['request_overlap']['comparisons'][name]['source_sha256']==hashlib.sha256(source_bytes(ROOT/path)).hexdigest(),f'overlap source {name}')
        check(data['request_overlap']['within_A003_duplicate_requests']==0,'reported internal overlap')
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
