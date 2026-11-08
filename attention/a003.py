"""A003 observation-updated allocation with exact archived-input replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from attention import agent, s2, s3
from attention.a001 import reference_audit, control_flags
from attention.io import read_json, write_json as write, load_replay
from attention.batch import run_batch as static_batch
from attention.updated_batch import run_batch as updated_batch
from attention.development_a002 import oracle_input
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES, overlap_audit, request_key
from lab.replay import canonical_sha256, protect_output
from lab.scientist import ROOT
from scripts.compare_experiment_runs import numeric_summary

PROTOCOL = 'attention/protocols/A003-updated-knowledge.md'
MODEL = 'attention/self_model_s3.json'
POLICIES = ('B2','B','C2','C4','D','Dprime')
FRACTIONS = (.2,.4,.6,1.)
from attention.a001 import SOURCES as OLD_SOURCES
SOURCES = sorted(set(OLD_SOURCES + [MODEL,'attention/a002.py','attention/batch.py','attention/s2.py','attention/build_s2.py','attention/io.py','attention/a003.py','attention/s3.py','attention/build_s3.py','attention/updated_batch.py','attention/self_model_s2.json','attention/development_a002.py']))


def validate(records, permutations):
    if len(records)!=60 or len(permutations)!=12:
        raise ValueError('Expected 60 requests in 12 packages')
    expected=sorted((i,a) for i in range(45) for a in agent.ORDER)
    for i,r in enumerate(records):
        if r['task_id']!=i or r['package_id']!=i//5 or r['cohort']!=COHORTS[i%5]:
            raise ValueError('Request/package/cohort order mismatch')
        if sorted(r['condition_order'])!=sorted(CONDITIONS):
            raise ValueError('Condition permutation mismatch')
        if r['request']['budget']!=4096 or r['request']['max_calls']!=3:
            raise ValueError('Request limits mismatch')
    for p in permutations:
        if sorted(tuple(x) for x in p)!=expected:
            raise ValueError('Global action permutation mismatch')


def intake(replay=None):
    if replay is not None:
        saved,requests,provenance=load_replay(replay,'A003',60)
        records=copy.deepcopy(saved['intake_channel'])
        permutations=copy.deepcopy(saved['global_permutations'])
        validate(records,permutations)
        if saved['intake_sha256']!=canonical_sha256([records,permutations]):
            raise ValueError('Intake digest mismatch')
        return records,requests,permutations,provenance
    rng=np.random.default_rng(20261135)
    order=np.random.default_rng(20261136)
    random=np.random.default_rng(20261137)
    records,requests=[],[]
    for i in range(60):
        cohort=COHORTS[i%5]; lo,hi=LOG_RANGES[cohort]
        log_time=float(rng.uniform(-2,2)); u,v=float(rng.random()),float(rng.random())
        log_tol=float(rng.uniform(-6 if cohort=='S' else -4,-4 if cohort=='S' else -1))
        log_mag=lo+(hi-lo)*u
        phase=((-np.pi/2+.1*v if v<.5 else np.pi/2-.1*(1-v)) if cohort=='O' else -np.pi/2+np.pi*v)
        time=10**log_time
        r=Request(10**log_mag*np.exp(1j*phase)/time,time,10**log_tol)
        requests.append(r)
        records.append(dict(task_id=i,package_id=i//5,cohort=cohort,request=encode(r),request_sha256=request_hash(r),
            condition_order=[str(c) for c in order.permutation(CONDITIONS)],sampled_log_time=log_time,
            sampled_log_tolerance=log_tol,sampled_log_magnitude=log_mag,sampled_phase=float(phase)))
    pairs=[(i,a) for i in range(45) for a in agent.ORDER]
    permutations=[[list(pairs[int(i)]) for i in random.permutation(len(pairs))] for _ in range(12)]
    validate(records,permutations)
    return records,requests,permutations,None


def score(p, verdict):
    return sum((p[v]-int(v==verdict))**2 for v in s2.VERDICTS)


def summarize(packages):
    summary={}
    for fraction in FRACTIONS:
        f=str(fraction)
        oracle=sum(b['oracle'][f]['correct'] for b in packages)
        policies={}
        for policy in POLICIES:
            runs=[b['policies'][f][policy] for b in packages]
            count=sum(r['correct'] for r in runs)
            policies[policy]=dict(correct=count,false_accepts=sum(r['false_accepts'] for r in runs),
                accepts=sum(r['accepts'] for r in runs),spent=sum(r['spent'] for r in runs),
                oracle_fraction=count/oracle if oracle else None,
                correct_by_package=[r['correct'] for r in runs],spent_by_package=[r['spent'] for r in runs])
        pairs={p:[b['policies'][f]['B2']['correct']-b['policies'][f][p]['correct'] for b in packages] for p in POLICIES if p!='B2'}
        summary[f]=dict(policies=policies,oracle_correct=oracle,
            oracle_by_package=[b['oracle'][f]['correct'] for b in packages],paired_B2_minus=pairs,
            paired_D_minus_Dprime=[b['policies'][f]['D']['correct']-b['policies'][f]['Dprime']['correct'] for b in packages],
            changed_action_packages={p:sum([(a['input_index'],a['action']) for a in b['policies'][f]['B2']['actions']]!=[(a['input_index'],a['action']) for a in b['policies'][f][p]['actions']] for b in packages) for p in ('D','Dprime')})
    return summary


def oracle_summary(infos,reference_cost):
    certificates=[oracle_input(i) for i in infos]
    ordered=sorted((r['cost'],i) for i,r in enumerate(certificates) if r['cost'] is not None)
    result={}
    for f in FRACTIONS:
        spent=0.; selected=[]
        for price,index in ordered:
            if spent+price>f*reference_cost:break
            spent+=price;selected.append(index)
        result[str(f)]=dict(correct=len(selected),spent=spent,selected=selected)
    return certificates,result


def calibration_summary(packages):
    result={}
    for fraction in FRACTIONS:
        f=str(fraction); result[f]={}
        for policy in ('B2','D','Dprime'):
            steps={}; packages_out=[]
            for b in packages:
                values=[]; missing=0
                for row in b['policies'][f][policy]['predictions']:
                    step=str(row['step']);group=steps.setdefault(step,dict(count=0,missing=0,total=0.))
                    if row['full_probabilities'] is None:
                        group['missing']+=1;missing+=1
                    else:
                        val=score(row['full_probabilities'],b['inputs'][row['input_index']]['full']['verdict'])
                        group['count']+=1;group['total']+=val;values.append(val)
                packages_out.append(dict(package_id=b['package_id'],count=len(values),missing=missing,brier=sum(values)/len(values) if values else None))
            result[f][policy]=dict(by_step={k:dict(count=v['count'],missing=v['missing'],brier=v['total']/v['count'] if v['count'] else None) for k,v in steps.items()},by_package=packages_out)
    return result


def decision_view(data):
    def walk(x):
        if isinstance(x,dict):
            return {k:(v if k in ('spent','cost','budget','full_cost','reference_cost','spent_by_package','allocation_by_initial_prediction','expected_cost','upper_cost') else walk(v)) for k,v in x.items() if not k.endswith('seconds')}
        if isinstance(x,list):return [walk(v) for v in x]
        if isinstance(x,float):return '<numeric diagnostic>'
        return x
    result = dict(intake=data['intake_channel'],permutations=data['global_permutations'],packages=walk(data['packages']),
                intake_sha256=data['intake_sha256'],self_model_sha256=data['self_model_sha256'])
    for key in ('protocol_sha256','static_model_sha256','experiment','task_seed','order_seed','random_policy_seed','summary','calibration',
                'full_pipeline_parity_failures','reference_audit','request_overlap','sampling_unit','intervention_audit'):
        if key in data:
            result[key] = walk(data[key])
    return result


def compare(first,second):
    a,b=decision_view(first),decision_view(second)
    mismatches=[i for i,(x,y) in enumerate(zip(a['packages'],b['packages'])) if x!=y]
    return dict(decisions_pass=a==b,differing_packages=mismatches,numeric_diagnostics=numeric_summary(first['packages'],second['packages']),
                numeric_diagnostics_informational=True)


def run(output,replay=None,runtime_override=None):
    output=Path(output)
    stem=output.name.removesuffix('.gz').removesuffix('.json')
    paths={name:output.with_name(stem+'-'+name+'.json') for name in ('intake','predictions','reference-audit')}
    for path in [output,*paths.values()]:protect_output(path,replay)
    audit=metadata(PROTOCOL,SOURCES,runtime_override)
    records,requests,permutations,provenance=intake(replay)
    if provenance:audit['replay']=provenance
    model=s3.load_model(); static_model=s2.load_model()
    models={p:s3.corrupt(model,p) for p in ('B2','D','Dprime')}
    predictions=[s3.predict(r,model=model) for r in requests]
    static_predictions=[s2.predict(r,model=static_model) for r in requests]
    corrupt_predictions=[s3.predict(r,model=models['D']) for r in requests]
    intervention=s3.corruption_audit(model,models['D'],models['Dprime'])
    output.parent.mkdir(parents=True,exist_ok=True)
    digest=canonical_sha256([records,permutations])
    write(paths['intake'],dict(experiment='A003',**audit,intake_channel=records,global_permutations=permutations,intake_sha256=digest))
    # Persist predictions before any full pipeline or budgeted numerical outcome.
    write(paths['predictions'],dict(intake_sha256=digest,self_model_sha256=canonical_sha256(model),predictions=predictions,static_predictions=static_predictions,D_predictions=corrupt_predictions,intervention_audit=intervention))
    references,reference=reference_audit(requests)
    write(paths['reference-audit'],reference)
    previous_request=Request(1,1,.01)
    previous=execute(previous_request,P.plan_modes(previous_request)).value
    packages=[]; parity=[]
    for package_id in range(12):
        infos=[]; specs=[]
        for task_id in range(package_id*5,(package_id+1)*5):
            record,request,truth=records[task_id],requests[task_id],references[task_id]
            for condition in record['condition_order']:
                e=agent.Executor(request,record['request_sha256'],condition,previous)
                full=agent.run_policy(e,'C1',float('inf'),static_model['action_model'],list(agent.ORDER),full=True)
                original=P.evaluate(request,record['request_sha256'],e.outcome,e.model)
                if full['verdict']!=original['policies']['strict']['verdict']:parity.append([task_id,condition])
                error=abs(complex(*e.outcome['value'])-truth)
                infos.append(dict(task_id=task_id,cohort=record['cohort'],condition=condition,full=full,
                    accurate=bool(error<=request.tolerance),absolute_error=error,tolerance=request.tolerance,
                    outcome=e.outcome,reference=[truth.real,truth.imag],prediction=predictions[task_id],static_prediction=static_predictions[task_id],D_prediction=corrupt_predictions[task_id]))
                specs.append((request,record['request_sha256'],condition,previous))
            previous=execute(request,P.plan_modes(request)).value
        reference_cost=sum(x['full']['spent'] for x in infos)
        policies={}
        for fraction in FRACTIONS:
            policies[str(fraction)]={}
            for policy in POLICIES:
                runner=updated_batch if policy in models else static_batch
                result=runner([agent.Executor(*spec) for spec in specs],policy,fraction*reference_cost,models.get(policy,static_model),permutations[package_id])
                for x,info in zip(result['inputs'],infos):
                    x['correct']=bool((info['accurate'] and x['verdict']=='accept') or (not info['accurate'] and x['verdict']=='violation_detected'))
                    x['false_accept']=bool(not info['accurate'] and x['verdict']=='accept')
                    if x['verdict']=='accept' and info['full']['verdict']!='accept':raise AssertionError('Acceptance lacks full P support')
                if result['spent']>fraction*reference_cost:raise AssertionError('Shared budget exceeded')
                result.update(correct=sum(x['correct'] for x in result['inputs']),false_accepts=sum(x['false_accept'] for x in result['inputs']),accepts=sum(x['verdict']=='accept' for x in result['inputs']))
                policies[str(fraction)][policy]=result
        certificates,oracle=oracle_summary(infos,reference_cost)
        packages.append(dict(package_id=package_id,reference_cost=reference_cost,inputs=infos,policies=policies,oracle_certificates=certificates,oracle=oracle))
        print('Completed package',package_id+1,'of 12',flush=True)
    overlap=overlap_audit(records)
    overlap['within_A003_duplicate_requests']=overlap.pop('within_E016_duplicate_requests')
    for experiment,path in [('E016','results/E016-frozen-pipeline/results.json'),('A001','attention/results/A001/results.json'),('A002','attention/results/A002/results.json.gz')]:
        earlier=read_json(ROOT/path); keys={request_key(r) for r in earlier['intake_channel']}
        overlap['comparisons'][experiment]=dict(exact_request_matches=sum(request_key(r) in keys for r in records),source_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    result=dict(experiment='A003',**audit,task_seed=20261135,order_seed=20261136,random_policy_seed=20261137,
        intake_channel=records,global_permutations=permutations,intake_sha256=digest,self_model_sha256=canonical_sha256(model),static_model_sha256=canonical_sha256(static_model),
        reference_audit=reference,request_overlap=overlap,full_pipeline_parity_failures=parity,intervention_audit=intervention,
        sampling_unit='12 paired packages, 60 requests, 540 dependent inputs; adjacent stale values connect packages',packages=packages,summary=summarize(packages),
        calibration=calibration_summary(packages))
    write(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--runtime-override')
    parser.add_argument('--compare',nargs=2,type=Path)
    args=parser.parse_args()
    if args.compare:
        protect_output(args.output,None)
        result=compare(*(read_json(p) for p in args.compare)); write(args.output,result)
        print(json.dumps(result,indent=2))
        if not result['decisions_pass']:raise SystemExit(1)
    else:
        result=run(args.output,args.replay,args.runtime_override)
        print(json.dumps(dict(summary=result['summary'],calibration=result['calibration']),indent=2))
