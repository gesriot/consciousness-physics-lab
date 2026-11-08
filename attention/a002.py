"""A002 shared-budget allocation with exact archived-input replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from attention import agent, s2
from attention.a001 import reference_audit, write, control_flags
from attention.batch import run_batch
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES, overlap_audit, request_key
from lab.replay import canonical_sha256, load_replay, protect_output
from lab.scientist import ROOT
from scripts.compare_experiment_runs import numeric_summary

PROTOCOL = 'attention/protocols/A002-shared-budget.md'
MODEL = 'attention/self_model_s2.json'
POLICIES = ('B','C1','C2','C3','C4','D','Dprime')
FRACTIONS = (.2,.4,.6,1.)
from attention.a001 import SOURCES as OLD_SOURCES
SOURCES = sorted(set(OLD_SOURCES + [MODEL,'attention/a002.py','attention/batch.py','attention/s2.py','attention/build_s2.py']))


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
        saved,requests,provenance=load_replay(replay,'A002',60)
        records=copy.deepcopy(saved['intake_channel'])
        permutations=copy.deepcopy(saved['global_permutations'])
        validate(records,permutations)
        if saved['intake_sha256']!=canonical_sha256([records,permutations]):
            raise ValueError('Intake digest mismatch')
        return records,requests,permutations,provenance
    rng=np.random.default_rng(20261132)
    order=np.random.default_rng(20261133)
    random=np.random.default_rng(20261134)
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
        policies={}
        for policy in POLICIES:
            runs=[b['policies'][f][policy] for b in packages]
            policies[policy]=dict(correct=sum(r['correct'] for r in runs),false_accepts=sum(r['false_accepts'] for r in runs),
                accepts=sum(r['accepts'] for r in runs),spent=sum(r['spent'] for r in runs),
                correct_by_package=[r['correct'] for r in runs],spent_by_package=[r['spent'] for r in runs])
        pairs={p:[b['policies'][f]['B']['correct']-b['policies'][f][p]['correct'] for b in packages] for p in POLICIES if p!='B'}
        interventions={p:dict(different_action_packages=sum(b['policies'][f]['B']['actions']!=b['policies'][f][p]['actions'] for b in packages),
            correct_difference=0 if p=='B' else sum(pairs[p]),
            allocation_by_initial_prediction={group:sum(x['spent'] for b in packages for x,info in zip(b['policies'][f][p]['inputs'],b['inputs'])
                if (info['prediction']['full_probabilities']['accept']<info['prediction']['full_probabilities']['inconclusive'])==(group=='initially_inconclusive')) for group in ('initially_inconclusive','initially_accepting')}) for p in ('B','D','Dprime')}
        summary[f]=dict(policies=policies,paired_B_minus=pairs,interventions=interventions)
    return summary


def decision_view(data):
    def walk(x):
        if isinstance(x,dict):
            return {k:(v if k in ('spent','cost','budget','full_cost','reference_cost') else walk(v)) for k,v in x.items()}
        if isinstance(x,list):return [walk(v) for v in x]
        if isinstance(x,float):return '<numeric diagnostic>'
        return x
    return dict(intake=data['intake_channel'],permutations=data['global_permutations'],packages=walk(data['packages']),
                intake_sha256=data['intake_sha256'],self_model_sha256=data['self_model_sha256'])


def compare(first,second):
    a,b=decision_view(first),decision_view(second)
    mismatches=[i for i,(x,y) in enumerate(zip(a['packages'],b['packages'])) if x!=y]
    return dict(decisions_pass=a==b,differing_packages=mismatches,numeric_diagnostics=numeric_summary(first['packages'],second['packages']),
                numeric_diagnostics_informational=True)


def run(output,replay=None,runtime_override=None):
    output=Path(output)
    paths={name:output.with_name(output.stem+'-'+name+'.json') for name in ('intake','predictions','reference-audit')}
    for path in [output,*paths.values()]:protect_output(path,replay)
    audit=metadata(PROTOCOL,SOURCES,runtime_override)
    records,requests,permutations,provenance=intake(replay)
    if provenance:audit['replay']=provenance
    model=s2.load_model()
    models={p:s2.corrupt(model,p) for p in POLICIES}
    predictions=[s2.predict(r,model=model) for r in requests]
    corrupt_predictions=[s2.predict(r,model=models['D']) for r in requests]
    output.parent.mkdir(parents=True,exist_ok=True)
    digest=canonical_sha256([records,permutations])
    write(paths['intake'],dict(experiment='A002',**audit,intake_channel=records,global_permutations=permutations,intake_sha256=digest))
    # Persist predictions before any full pipeline or budgeted numerical outcome.
    write(paths['predictions'],dict(intake_sha256=digest,self_model_sha256=canonical_sha256(model),predictions=predictions,D_predictions=corrupt_predictions))
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
                full=agent.run_policy(e,'C1',float('inf'),model['action_model'],list(agent.ORDER),full=True)
                original=P.evaluate(request,record['request_sha256'],e.outcome,e.model)
                if full['verdict']!=original['policies']['strict']['verdict']:parity.append([task_id,condition])
                error=abs(complex(*e.outcome['value'])-truth)
                infos.append(dict(task_id=task_id,cohort=record['cohort'],condition=condition,full=full,
                    accurate=bool(error<=request.tolerance),absolute_error=error,tolerance=request.tolerance,
                    outcome=e.outcome,reference=[truth.real,truth.imag],prediction=predictions[task_id],D_prediction=corrupt_predictions[task_id]))
                specs.append((request,record['request_sha256'],condition,previous))
            previous=execute(request,P.plan_modes(request)).value
        reference_cost=sum(x['full']['spent'] for x in infos)
        policies={}
        for fraction in FRACTIONS:
            policies[str(fraction)]={}
            for policy in POLICIES:
                result=run_batch([agent.Executor(*spec) for spec in specs],policy,fraction*reference_cost,models[policy],permutations[package_id])
                for x,info in zip(result['inputs'],infos):
                    x['correct']=bool((info['accurate'] and x['verdict']=='accept') or (not info['accurate'] and x['verdict']=='violation_detected'))
                    x['false_accept']=bool(not info['accurate'] and x['verdict']=='accept')
                    if x['verdict']=='accept' and info['full']['verdict']!='accept':raise AssertionError('Acceptance lacks full P support')
                if result['spent']>fraction*reference_cost:raise AssertionError('Shared budget exceeded')
                result.update(correct=sum(x['correct'] for x in result['inputs']),false_accepts=sum(x['false_accept'] for x in result['inputs']),accepts=sum(x['verdict']=='accept' for x in result['inputs']))
                policies[str(fraction)][policy]=result
        calibration={name:sum(score(info['prediction'][key],info['full']['verdict']) for info in infos)/45 for name,key in [('S2','full_probabilities'),('base','base_probabilities')]}
        packages.append(dict(package_id=package_id,reference_cost=reference_cost,inputs=infos,policies=policies,calibration=calibration))
        print('Completed package',package_id+1,'of 12',flush=True)
    overlap=overlap_audit(records)
    overlap['within_A002_duplicate_requests']=overlap.pop('within_E016_duplicate_requests')
    for experiment,path in [('E016','results/E016-frozen-pipeline/results.json'),('A001','attention/results/A001/results.json')]:
        earlier=json.loads((ROOT/path).read_text()); keys={request_key(r) for r in earlier['intake_channel']}
        overlap['comparisons'][experiment]=dict(exact_request_matches=sum(request_key(r) in keys for r in records),source_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    result=dict(experiment='A002',**audit,task_seed=20261132,order_seed=20261133,random_policy_seed=20261134,
        intake_channel=records,global_permutations=permutations,intake_sha256=digest,self_model_sha256=canonical_sha256(model),
        reference_audit=reference,request_overlap=overlap,full_pipeline_parity_failures=parity,
        sampling_unit='12 paired packages, 60 requests, 540 dependent inputs; adjacent stale values connect packages',packages=packages,summary=summarize(packages),
        calibration={p:sum(b['calibration'][p] for b in packages)/12 for p in ('S2','base')})
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
        result=compare(*(json.loads(p.read_text()) for p in args.compare)); write(args.output,result)
        print(json.dumps(result,indent=2))
        if not result['decisions_pass']:raise SystemExit(1)
    else:
        result=run(args.output,args.replay,args.runtime_override)
        print(json.dumps(dict(summary=result['summary'],calibration=result['calibration']),indent=2))
