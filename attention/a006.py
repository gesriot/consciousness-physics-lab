"""Registered monitoring experiment with exact-input replay."""
import argparse
import copy
import hashlib
import gzip
import json
from pathlib import Path
import numpy as np
from attention import agent, s2, s3, monitor_planning
from attention.active_monitor import State, POLICIES, run_batch, family, target
from attention.calibration_samples import FAMILIES
from attention.a001 import reference_audit
from attention.a005 import SOURCES as OLD_SOURCES
from attention.io import read_json, write_json as write, load_replay, source_bytes
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES, request_key
from lab.replay import canonical_sha256, protect_output
from lab.scientist import ROOT
PROTOCOL='attention/protocols/A006-active-calibration.md'
LIBRARY='attention/sample_library_a006.json'
CALIBRATION='attention/results/A005/calibration.json'
SOURCES=sorted(set(OLD_SOURCES+['attention/a006.py','attention/active_monitor.py','attention/calibration_samples.py','attention/build_a006_library.py','attention/compare_a006.py','attention/audit_a006.py',LIBRARY]))
FRACTIONS=(.2,.4,.6,1.)
FAULTS=('none','permissive','uninformative','underestimated')

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
        saved,requests,provenance=load_replay(replay,'A006',60)
        records=copy.deepcopy(saved['intake_channel'])
        permutations=copy.deepcopy(saved['global_permutations'])
        validate(records,permutations)
        if saved['intake_sha256']!=canonical_sha256([records,permutations]):
            raise ValueError('Intake digest mismatch')
        return records,requests,permutations,provenance
    rng=np.random.default_rng(20261145)
    order=np.random.default_rng(20261146)
    random=np.random.default_rng(20261147)
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


def summarize(packages,onset):
    out={}
    for fault in FAULTS:
        out[fault]={}
        for fraction in FRACTIONS:
            f=str(fraction);out[fault][f]={}
            for policy in POLICIES:
                runs=[p['conditions'][fault][f][policy] for p in packages]
                alerts=[a for r in runs for a in r['alerts']]
                before=[a for a in alerts if a['package_id']<onset and family(a['action'])==target(fault)]
                after=[a for a in alerts if a['package_id']>=onset and family(a['action'])==target(fault)]
                delay=None
                if after and not before:
                    a=after[0];k=a['package_id']
                    delay=dict(inputs=sum(runs[j]['touched_inputs'] for j in range(onset,k))+a['touched_inputs'],
                        actions=sum(len(runs[j]['actions']) for j in range(onset,k))+a['action_index']+1,
                        target_observations=sum(runs[j]['target_exposures'] for j in range(onset,k))+a['target_exposures'],
                        samples=sum(runs[j]['sample_allocation']['executed'] for j in range(onset,k))+a['samples'],
                        target_samples=sum(runs[j]['target_samples'] for j in range(onset,k))+a['target_samples'])
                out[fault][f][policy]=dict(correct=sum(r['correct'] for r in runs),
                    correct_by_package=[r['correct'] for r in runs],
                    false_accepts_before=sum(r['false_accepts'] for r in runs[:onset]),
                    false_accepts_after=sum(r['false_accepts'] for r in runs[onset:]),
                    false_accepts_by_package=[r['false_accepts'] for r in runs],
                    alerts=len(alerts),alerts_before=sum(a['package_id']<onset for a in alerts),packages_with_new_alert=sum(bool(r['alerts']) for r in runs),
                    preonset_target_alert=bool(before),detected=bool(after and not before) if fault!='none' else None,
                    delay=delay if fault!='none' else None,censored=not bool(after and not before) if fault!='none' else None,
                    followup=dict(inputs=sum(r['touched_inputs'] for r in runs[onset:]),actions=sum(len(r['actions']) for r in runs[onset:]),target_observations=sum(r['target_exposures'] for r in runs[onset:]),target_samples=sum(r['target_samples'] for r in runs[onset:])),
                    detected_within_90=bool(delay and delay['inputs']<=90) if fault!='none' else None,
                    spent=sum(r['spent'] for r in runs),replacement_cost=sum(r['replacement_cost'] for r in runs),
                    sample_cost=sum(r['sample_cost'] for r in runs),samples=sum(r['sample_allocation']['executed'] for r in runs),
                    postonset_samples=sum(r['sample_allocation']['executed'] for r in runs[onset:]),
                    target_samples=sum(r['target_samples'] for r in runs),
                    changed_sample_verdicts=sum(row['sample'] and row['original_observation']['verdict']!=row['observation']['verdict'] for r in runs[onset:] for row in r['actions']),
                    sample_expectation_mismatches=sum(row['sample'] and row['monitor']['mismatch'] for r in runs[onset:] for row in r['actions']),
                    changed_v2_steps=sum(bool(t['original'].get('step',{}).get('uncertainty_estimate')!=t['observed'].get('step',{}).get('uncertainty_estimate')) for r in runs[onset:] for row in r['actions'] if row['sample'] for t in row['sample_trace']))
    return out


def intervention_audit(library):
    codes={'accept':0,'violation_detected':1,'inconclusive':2}
    changed=[codes[s['expected']] for s in library['samples'] if s['family'] in ('collocation','second_moment') and s['expected']!='accept']
    return dict(D_changed_numbers=len(changed),Dprime_changed_numbers=len(changed),D_L1=sum(changed),Dprime_L1=sum(changed),
        equal_changes=True,Dprime_active_unchanged=True,scheduling_expectations_unchanged=True,
        scope='Frozen expected verdict table of all collocation and second-moment samples; unused table receives identical changes in Dprime')


def write_result(path,value):
    """Stream the large archive without duplicating its full JSON in memory."""
    path=Path(path)
    def emit(stream):
        chunks=[];size=0
        for chunk in json.JSONEncoder(indent=2,allow_nan=False).iterencode(value):
            chunks.append(chunk);size+=len(chunk)
            if size>=1024*1024:
                stream.write(''.join(chunks).encode('utf-8'));chunks=[];size=0
        stream.write((''.join(chunks)+'\n').encode('utf-8'))
    with path.open('wb') as raw:
        if path.suffix=='.gz':
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=9,mtime=0) as compressed:emit(compressed)
        else:emit(raw)


def run(output,replay=None,noise=False,runtime_override=None):
    monitor_planning.NOISE=noise
    output=Path(output);protect_output(output,replay)
    audit=metadata(PROTOCOL,SOURCES,runtime_override)
    records,requests,permutations,provenance=intake(replay)
    if provenance:audit['replay']=provenance
    onset=int(np.random.default_rng(20261148).integers(4,9))
    model=s3.load_model();static=s2.load_model();calibration=read_json(ROOT/CALIBRATION);library=read_json(ROOT/LIBRARY)
    monitors={(fault,str(f),p):State(calibration['threshold'],p,library) for fault in FAULTS for f in FRACTIONS for p in POLICIES}
    output.parent.mkdir(parents=True,exist_ok=True)
    stem=output.name.removesuffix('.gz').removesuffix('.json')
    intake_path=output.with_name(stem+'-intake.json');protect_output(intake_path,None)
    write(intake_path,dict(experiment='A006',**audit,intake_channel=records,global_permutations=permutations,
        onset_package=onset,intake_sha256=canonical_sha256([records,permutations])))
    runtime=monitor_planning.Runtime(model)
    predictions=[dict(task_id=i,forecasts={a:monitor_planning.Posterior(r,runtime,False).forecast(a) for a in agent.ORDER}) for i,r in enumerate(requests)]
    prediction_path=output.with_name(stem+'-predictions.json');protect_output(prediction_path,None)
    write(prediction_path,dict(predictions=predictions,self_model_sha256=canonical_sha256(model)))
    references,reference=reference_audit(requests)
    previous_request=Request(1,1,.01);previous=execute(previous_request,P.plan_modes(previous_request)).value
    packages=[];parity=[]
    for package_id in range(12):
        infos=[];specs=[]
        for task_id in range(package_id*5,(package_id+1)*5):
            record,r,truth=records[task_id],requests[task_id],references[task_id]
            for condition in record['condition_order']:
                e=agent.Executor(r,record['request_sha256'],condition,previous)
                full=agent.run_policy(e,'C1',float('inf'),static['action_model'],list(agent.ORDER),full=True)
                original=P.evaluate(r,record['request_sha256'],e.outcome,e.model)
                if full['verdict']!=original['policies']['strict']['verdict']:parity.append([task_id,condition])
                error=abs(complex(*e.outcome['value'])-truth)
                infos.append(dict(task_id=task_id,cohort=record['cohort'],condition=condition,full=full,
                    accurate=bool(error<=r.tolerance),absolute_error=error,tolerance=r.tolerance,
                    outcome=e.outcome,reference=[truth.real,truth.imag]))
                specs.append((r,record['request_sha256'],condition,previous))
            previous=execute(r,P.plan_modes(r)).value
        reference_cost=sum(x['full']['spent'] for x in infos);conditions={}
        for fault in FAULTS:
            conditions[fault]={}
            for f in FRACTIONS:
                conditions[fault][str(f)]={}
                for policy in POLICIES:
                    result=run_batch([agent.Executor(*spec) for spec in specs],policy,f*reference_cost,model,monitors[fault,str(f),policy],package_id,fault,onset,f)
                    for x,info in zip(result['inputs'],infos):
                        x['correct']=bool((info['accurate'] and x['verdict']=='accept') or (not info['accurate'] and x['verdict']=='violation_detected'))
                        x['false_accept']=bool(not info['accurate'] and x['verdict']=='accept')
                    if result['spent']>f*reference_cost:raise AssertionError('Shared budget exceeded')
                    result.update(correct=sum(x['correct'] for x in result['inputs']),false_accepts=sum(x['false_accept'] for x in result['inputs']))
                    conditions[fault][str(f)][policy]=result
        packages.append(dict(package_id=package_id,reference_cost=reference_cost,inputs=infos,conditions=conditions))
        print('Completed package',package_id+1,'of 12',flush=True)
    overlap={}
    for experiment,path in [('A001','attention/results/A001/results.json'),('A002','attention/results/A002/results.json.gz'),('A003','attention/results/A003/results.json.gz'),('A004','attention/results/A004/results.json.gz'),('A005','attention/results/A005/results.json.gz')]:
        earlier=read_json(ROOT/path);keys={request_key(r) for r in earlier['intake_channel']}
        overlap[experiment]=dict(exact_request_matches=sum(request_key(r) in keys for r in records),source_sha256=hashlib.sha256(source_bytes(ROOT/path)).hexdigest())
        del earlier
    out=dict(experiment='A006' ,**audit,task_seed=20261145,order_seed=20261146,random_policy_seed=20261147,onset_seed=20261148,
        onset_package=onset,last_bit_noise=noise,request_overlap=overlap,intake_channel=records,global_permutations=permutations,
        intake_sha256=canonical_sha256([records,permutations]),self_model_sha256=canonical_sha256(model),
        calibration=calibration,sample_library_sha256=canonical_sha256(library),sample_library_sources=library['source_sha256'],sample_library_counts=library['counts'],intervention_audit=intervention_audit(library),reference_audit=reference,
        full_pipeline_parity_failures=parity,sampling_unit='Four budget streams, 12 paired packages, 60 requests, 540 dependent inputs; conditions and policies repeated; monitor and stale connect packages',
        packages=packages,summary=summarize(packages,onset))
    write_result(output,out);return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--replay');parser.add_argument('--last-bit-noise',action='store_true');parser.add_argument('--runtime-override');args=parser.parse_args()
    result=run(args.output,args.replay,args.last_bit_noise,args.runtime_override)
    print(json.dumps(result['summary'],indent=2))
