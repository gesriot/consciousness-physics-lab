"""A001 held-out budget allocation and archived-input replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from scripts.compare_experiment_runs import differences, numeric_summary
import numpy as np
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.extrapolation import high_precision
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES, overlap_audit, scientific_view
from lab.replay import canonical_sha256, protect_output
from attention.io import read_json, write_json, load_replay
from lab.scientist import ROOT, analytic
from attention.agent import Executor, ORDER, POLICIES, FRACTIONS, corrupt, run_policy

PROTOCOL='attention/protocols/A001-budgeted-self-model.md'
MODEL='attention/self_model.json'
SOURCES=['attention/agent.py','attention/a001.py','attention/io.py',MODEL]+['lab/'+p for p in (
    'frozen_pipeline.py','pipeline_holdout.py','replay.py','experiment_audit.py',
    'geometry_invariants.py','convergence_candidates.py','radial_collocation.py',
    'matched_mixture.py','extrapolation.py','three_way.py','surface_order.py',
    'fault_holdout.py','common_mode.py','allocation.py','executor.py','guard.py',
    'finite_volume.py','scientist.py')]+['vendor/sphere-thermal-cli/sphere_thermal/response.py']


def write(path,value):
    write_json(path,value)


def validate(records):
    if len(records)!=60:
        raise ValueError('Expected 60 requests')
    for i,r in enumerate(records):
        if r['task_id']!=i or r['cohort']!=COHORTS[i//12]:
            raise ValueError('Task order or cohort mismatch')
        if sorted(r['condition_order'])!=sorted(CONDITIONS):
            raise ValueError('Condition order not a registered permutation')
        if set(r['policy_order'])!=set(CONDITIONS) or any(sorted(v)!=sorted(ORDER) for v in r['policy_order'].values()):
            raise ValueError('Random policy action order mismatch')
        if r['request']['budget']!=4096 or r['request']['max_calls']!=3:
            raise ValueError('Request limits differ from registration')


def intake(replay=None):
    if replay is not None:
        saved,requests,provenance=load_replay(replay,'A001',60)
        records=copy.deepcopy(saved['intake_channel'])
        validate(records)
        if saved['intake_sha256']!=canonical_sha256(records):
            raise ValueError('Intake archive digest mismatch')
        return records,requests,provenance
    rng=np.random.default_rng(20261129)
    order=np.random.default_rng(20261130)
    random=np.random.default_rng(20261131)
    records,requests=[],[]
    for task_id in range(60):
        cohort=COHORTS[task_id//12]
        lo,hi=LOG_RANGES[cohort]
        log_time=float(rng.uniform(-2,2))
        u,v=float(rng.random()),float(rng.random())
        log_tolerance=float(rng.uniform(-6 if cohort=='S' else -4,-4 if cohort=='S' else -1))
        log_magnitude=lo+(hi-lo)*u
        magnitude=10**log_magnitude
        phase=((-np.pi/2+.1*v if v<.5 else np.pi/2-.1*(1-v)) if cohort=='O' else -np.pi/2+np.pi*v)
        mean_time=10**log_time
        request=Request(magnitude*np.exp(1j*phase)/mean_time,mean_time,10**log_tolerance)
        requests.append(request)
        records.append(dict(task_id=task_id,cohort=cohort,request=encode(request),
            request_sha256=request_hash(request),condition_order=[str(c) for c in order.permutation(CONDITIONS)],
            policy_order={c:[str(a) for a in random.permutation(ORDER)] for c in CONDITIONS},
            sampled_log_time=log_time,sampled_log_tolerance=log_tolerance,
            sampled_log_magnitude=log_magnitude,sampled_phase=float(phase)))
    validate(records)
    return records,requests,None


def reference_audit(requests):
    rows=[]
    for i,r in enumerate(requests):
        value=complex(analytic(r.z,r.mean_time))
        exact=high_precision(r)
        precise=complex(*map(float,exact))
        error=abs(value-precise)
        rows.append(dict(task_id=i,analytic=[value.real,value.imag],high_precision=exact,
                         absolute_difference=error,passed=bool(error<=1e-13+1e-12*abs(precise))))
    fallback=not all(r['passed'] for r in rows)
    references=[complex(*map(float,r['high_precision'])) if fallback else complex(*r['analytic']) for r in rows]
    return references,dict(precision_decimal_digits=60,all_tasks_audited=60,fallback_all=fallback,records=rows)


def summary(rows):
    result={}
    for f in FRACTIONS:
        by_policy={}
        for p in POLICIES:
            runs=[r['policies'][str(f)][p] for r in rows]
            by_policy[p]=dict(n=len(rows),correct=sum(x['correct'] for x in runs),
                false_accepts=sum(x['false_accept'] for x in runs),
                accepts=sum(x['verdict']=='accept' for x in runs),
                violations=sum(x['verdict']=='violation_detected' for x in runs),
                inconclusive=sum(x['verdict']=='inconclusive' for x in runs),
                total_cost=sum(x['spent'] for x in runs),
                early_violations=sum(x['early_violation'] for x in runs),
                futile_v2=sum(x['futile_v2'] for x in runs))
        comparisons={}
        for p in ('C1','C2','C3','D','Dprime'):
            pairs=[(r['policies'][str(f)]['B'],r['policies'][str(f)][p]) for r in rows]
            comparisons[p]=dict(different_actions=sum([a['action'] for a in b['actions']]!=[a['action'] for a in d['actions']] for b,d in pairs),
                worse_than_B=sum(b['correct'] and not d['correct'] for b,d in pairs),
                better_than_B=sum(d['correct'] and not b['correct'] for b,d in pairs),
                correct_difference_vs_B=by_policy[p]['correct']-by_policy['B']['correct'])
        result[str(f)]=dict(policies=by_policy,comparisons_to_B=comparisons)
    return result


def control_flags(value, path=''):
    """All discrete boolean decisions, including nested gate and grid flags."""
    if isinstance(value,dict):
        return {p:v for k,item in value.items() for p,v in control_flags(item,path+'/'+k).items()}
    if isinstance(value,list):
        return {p:v for i,item in enumerate(value) for p,v in control_flags(item,path+'/'+str(i)).items()}
    return {path:value} if isinstance(value,bool) else {}


def decision_view(result):
    return [dict(task_id=r['task_id'],condition=r['condition'],accurate=r['accurate'],
       reference_verdict=r['full']['verdict'],full_cost=r['full']['spent'],full_flags=control_flags(r['full']),
       full_actions=[(a['action'],a.get('cost')) for a in r['full'].get('actions',[])],
       policies={f:{p:dict(verdict=x['verdict'],spent=x['spent'],correct=x['correct'],
          actions=[a['action'] for a in x['actions']],
          action_costs=[a.get('cost') for a in x['actions']],
          component_verdicts=[a.get('observation',{}).get('verdict',a.get('verdict')) for a in x['actions']],
          control_flags=control_flags(x))
          for p,x in ps.items()} for f,ps in r['policies'].items()}) for r in result['runs']]


def compare(first,second):
    """Require exact decisions and structure; report measured float drift only."""
    a,b=decision_view(first),decision_view(second)
    mismatches=[]
    for i,(x,y) in enumerate(zip(a,b)):
        if differences(x,y): mismatches.append(i)
    numeric=[]
    structure=[]
    def walk(x,y,path=''):
        if isinstance(x,dict) and isinstance(y,dict):
            if x.keys()!=y.keys():structure.append(path)
            for k in x.keys()&y.keys():
                walk(x[k],y[k],path+'/'+k)
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):structure.append(path)
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif type(x) is float and type(y) is float:
            if not np.isclose(x,y,atol=1e-12,rtol=1e-12):numeric.append(path)
        elif type(x)!=type(y) or x!=y:structure.append(path)
    walk(first['runs'],second['runs'],'runs')
    decisions_pass=bool(len(a)==len(b) and not mismatches)
    structure_pass=not structure
    return dict(passed=decisions_pass and structure_pass,decisions_pass=decisions_pass,
        diagnostic_structure_pass=structure_pass,
        numeric_diagnostic_summary=numeric_summary(first['runs'],second['runs']),
        numeric_diagnostics_informational=True,
        differing_decision_rows=mismatches,numeric_diagnostic_differences=numeric,
        numeric_diagnostics_pass=not numeric and not structure,diagnostic_structure_differences=structure,atol=1e-12,rtol=1e-12,
        scientific_exact=scientific_view(first)==scientific_view(second))


def run(output,replay=None,runtime_override=None):
    output=Path(output)
    stem=output.name.removesuffix('.gz').removesuffix('.json')
    intake_path=output.with_name(stem+'-intake.json')
    audit_path=output.with_name(stem+'-reference-audit.json')
    for path in (output,intake_path,audit_path):protect_output(path,replay)
    audit=metadata(PROTOCOL,SOURCES,runtime_override)
    model=json.loads((ROOT/MODEL).read_text())
    records,requests,provenance=intake(replay)
    if provenance:audit['replay']=provenance
    output.parent.mkdir(parents=True,exist_ok=True)
    write(intake_path,dict(experiment='A001',**audit,intake_channel=records,intake_sha256=canonical_sha256(records)))
    references,ref_audit=reference_audit(requests)
    write(audit_path,ref_audit)
    previous_request=Request(1,1,.01)
    previous=execute(previous_request,P.plan_modes(previous_request)).value
    rows=[]
    models={p:corrupt(model,p) for p in POLICIES}
    parity_failures=[]
    for record,request,truth in zip(records,requests,references):
        for condition in record['condition_order']:
            executor=Executor(request,record['request_sha256'],condition,previous)
            full=run_policy(executor,'C1',float('inf'),model,record['policy_order'][condition],full=True)
            # Independent integration check; its truth never reaches the scheduler.
            original=P.evaluate(request,record['request_sha256'],executor.outcome,executor.model)
            if full['verdict']!=original['policies']['strict']['verdict']:
                parity_failures.append([record['task_id'],condition])
            outcome=executor.outcome
            error=abs(complex(*outcome['value'])-truth)
            accurate=bool(error<=request.tolerance)
            policies={}
            for fraction in FRACTIONS:
                policies[str(fraction)]={}
                for p in POLICIES:
                    e=Executor(request,record['request_sha256'],condition,previous)
                    result=run_policy(e,p,fraction*full['spent'],models[p],record['policy_order'][condition])
                    result['budget']=fraction*full['spent']
                    result['correct']=bool((accurate and result['verdict']=='accept') or (not accurate and result['verdict']=='violation_detected'))
                    result['false_accept']=bool(not accurate and result['verdict']=='accept')
                    if result['verdict']=='accept' and full['verdict']!='accept':
                        raise AssertionError('Budgeted acceptance not supported by full P')
                    if result['spent']>result['budget']:
                        raise AssertionError('Budget exceeded')
                    policies[str(fraction)][p]=result
            rows.append(dict(task_id=record['task_id'],cohort=record['cohort'],condition=condition,
                model=executor.model,outcome=outcome,reference=[truth.real,truth.imag],
                absolute_error=error,tolerance=request.tolerance,accurate=accurate,full=full,policies=policies))
        previous=execute(request,P.plan_modes(request)).value
        print('Completed request',record['task_id']+1,'of',len(records),flush=True)
    overlap=overlap_audit(records)
    earlier=json.loads((ROOT/'results/E016-frozen-pipeline/results.json').read_text())
    from lab.pipeline_holdout import request_key
    keys={request_key(r) for r in earlier['intake_channel']}
    overlap['comparisons']['E016']=dict(exact_request_matches=sum(request_key(r) in keys for r in records),
        source_sha256=hashlib.sha256((ROOT/'results/E016-frozen-pipeline/results.json').read_bytes()).hexdigest())
    overlap['within_A001_duplicate_requests']=overlap.pop('within_E016_duplicate_requests')
    result=dict(experiment='A001',**audit,task_seed=20261129,order_seed=20261130,random_policy_seed=20261131,
        intake_channel=records,intake_sha256=canonical_sha256(records),reference_audit=ref_audit,
        request_overlap=overlap,full_pipeline_parity_failures=parity_failures,
        sampling_unit='60 requests with adjacent stale-value dependence; 540 paired inputs, 18 policy-budget outcomes each',
        self_model_sha256=canonical_sha256(model),runs=rows,summary=summary(rows),
        cohort_summary={c:summary([r for r in rows if r['cohort']==c]) for c in COHORTS},
        condition_summary={c:summary([r for r in rows if r['condition']==c]) for c in CONDITIONS},
        full_reference=dict(total_cost=sum(r['full']['spent'] for r in rows),
            accepts=sum(r['full']['verdict']=='accept' for r in rows),
            false_accepts=sum(r['full']['verdict']=='accept' and not r['accurate'] for r in rows),
            correct=sum((r['accurate'] and r['full']['verdict']=='accept') or (not r['accurate'] and r['full']['verdict']=='violation_detected') for r in rows)))
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
        result=compare(*(read_json(p) for p in args.compare))
        write(args.output,result)
        print(json.dumps(result,indent=2))
        if not result['passed']:raise SystemExit(1)
    else:
        result=run(args.output,args.replay,args.runtime_override)
        print(json.dumps(result['summary'],indent=2))
