"""Preregistered local-channel calibration with durable package checkpoints."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from attention import agent, s2, s3
from attention.a001 import reference_audit
from attention.a006 import read_root_fields, source_sha256
from attention.a005 import SOURCES as OLD_SOURCES
from attention.io import read_json, write_json
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.executor import execute, request_hash
from lab.experiment_audit import metadata
from lab.guard import Request
from lab.pipeline_holdout import CONDITIONS, COHORTS, LOG_RANGES, request_key
from lab.replay import canonical_sha256, protect_output
from lab.scientist import ROOT

PROTOCOL='attention/protocols/A007-local-channel-calibration.md'
LIBRARY='attention/sample_library_a007.json'
POLICIES=('BA','C2A','C2R','BU','C2','D','Dprime')
FAULTS=('none','permissive','underestimated','compressed')
FRACTIONS=(.2,.4,.6,1.)
SOURCES=sorted(set(OLD_SOURCES+['attention/a007.py','attention/channel_monitor.py',
    'attention/build_a007_library.py','attention/calibration_samples.py',
    'attention/a006.py','attention/compare_a004.py','attention/compare_a007.py',
    'attention/audit_a007.py','scripts/compare_experiment_runs.py',LIBRARY]))


def decode(fields):
    fields=dict(fields);fields['z']=complex(*fields['z']);return Request(**fields)


def validate(records, specs):
    if len(records)!=60 or len(specs)!=12:raise ValueError('Expected 60 requests and 12 fault packages')
    for i,row in enumerate(records):
        if (row['task_id'],row['package_id'],row['cohort'])!=(i,i//5,COHORTS[i%5]):raise ValueError('Intake order')
        r=decode(row['request'])
        if not all(math.isfinite(x) for x in (r.z.real,r.z.imag,r.mean_time,r.tolerance)):raise ValueError('Nonfinite request')
        if (r.budget,r.max_calls)!=(4096,3):raise ValueError('Request budget')
        if request_hash(r)!=row['request_sha256']:raise ValueError('Request digest')
        if sorted(row['condition_order'])!=sorted(CONDITIONS):raise ValueError('Condition order')
    for i,spec in enumerate(specs):
        if set(spec)!=set(FAULTS):raise ValueError('Fault conditions')
        for kind,v in spec.items():
            if v['kind']!=kind or not 20<=v['onset_action']<=60 or not 0<=v['circle_phase']<18:raise ValueError('Fault descriptor')
            family='collocation' if kind in ('none','permissive') else 'v2'
            if v['target_channel'] not in [family+':'+str(j) for j in range(6)]:raise ValueError('Fault target')


def intake(replay=None):
    if replay:
        saved=read_root_fields(replay,('experiment','intake_channel','fault_specs','intake_sha256'))
        if saved['experiment']!='A007':raise ValueError('Replay experiment')
        records,specs=saved['intake_channel'],saved['fault_specs']
        validate(records,specs)
        if saved['intake_sha256']!=canonical_sha256([records,specs]):raise ValueError('Intake digest')
        return records,[decode(r['request']) for r in records],specs,dict(source_sha256=source_sha256(replay,False))
    rng=np.random.default_rng(20261201);order=np.random.default_rng(20261202)
    faults=np.random.default_rng(20261203);phase=np.random.default_rng(20261205)
    records=[];requests=[]
    for i in range(60):
        cohort=COHORTS[i%5];lo,hi=LOG_RANGES[cohort]
        log_time=float(rng.uniform(-2,2));u,v=float(rng.random()),float(rng.random())
        log_tol=float(rng.uniform(-6 if cohort=='S' else -4,-4 if cohort=='S' else -1))
        log_mag=lo+(hi-lo)*u
        angle=((-np.pi/2+.1*v if v<.5 else np.pi/2-.1*(1-v)) if cohort=='O' else -np.pi/2+np.pi*v)
        time=10**log_time;r=Request(10**log_mag*np.exp(1j*angle)/time,time,10**log_tol)
        requests.append(r);records.append(dict(task_id=i,package_id=i//5,cohort=cohort,request=encode(r),request_sha256=request_hash(r),
            condition_order=[str(c) for c in order.permutation(CONDITIONS)],sampled_log_time=log_time,
            sampled_log_tolerance=log_tol,sampled_log_magnitude=log_mag,sampled_phase=float(angle)))
    specs=[]
    for i in range(12):
        onset=int(faults.integers(20,61));circle=int(phase.integers(0,18));row={}
        for kind in FAULTS:
            family='collocation' if kind in ('none','permissive') else 'v2'
            row[kind]=dict(kind=kind,target_channel=family+':'+str(int(faults.integers(0,6))),onset_action=onset,
                circle_phase=circle,random_seed=20261204+1000*i)
        specs.append(row)
    validate(records,specs)
    return records,requests,specs,None


def empty_summary():
    return {fault:{str(f):{p:dict(correct=0,false_accepts_before=0,false_accepts_after=0,alerts=0,
        false_alerts=0,detected_packages=0,censored_packages=0,delay_inputs_penalized=0,delay_samples_penalized=0,
        observed_delays=[],spent=0.,sample_cost=0.,replacement_cost=0.,ordinary_cost=0.,samples=0,
        correct_by_package=[],false_accepts_by_package=[],detection_by_package=[],
        changed_ordinary_categories=0,changed_sample_categories=0,target_observations=0,target_samples=0,
        guarded_replacements=0) for p in POLICIES} for f in FRACTIONS} for fault in FAULTS}


def update_summary(summary,package):
    for fault in FAULTS:
        for f in map(str,FRACTIONS):
            for policy in POLICIES:
                r=package['conditions'][fault][f][policy];s=summary[fault][f][policy];d=r['detection']
                for k in ('correct','false_accepts_before','false_accepts_after','spent','sample_cost','replacement_cost','ordinary_cost'):
                    s[k]+=r[k]
                s['alerts']+=len(r['alerts'])
                s['false_alerts']+=sum(not a.get('target_detection',False) for a in r['alerts'])
                s['samples']+=r['sample_allocation']['executed']
                s['detected_packages']+=int(d['detected']);s['censored_packages']+=int(d['censored'])
                if fault!='none':
                    s['delay_inputs_penalized']+=d['delay']['inputs'] if d['detected'] else 46
                    s['delay_samples_penalized']+=d['delay']['samples'] if d['detected'] else r['sample_allocation']['quota']+1
                if d['detected']:s['observed_delays'].append(dict(package_id=package['package_id'],**d['delay']))
                s['correct_by_package'].append(r['correct'])
                s['false_accepts_by_package'].append(r['false_accepts_after'])
                s['detection_by_package'].append(copy.deepcopy(d))
                for a in r['actions']:
                    if a.get('replacement'):
                        s['guarded_replacements']+=int(a.get('observation',{}).get('safety',{}).get('passed') is False)
                        continue
                    sample=a.get('sample',False)
                    original=a.get('original_observation',{})
                    observed=a.get('hooked_observation',a.get('observation',{}))
                    s['changed_sample_categories' if sample else 'changed_ordinary_categories']+=int(original.get('verdict')!=observed.get('verdict'))
                    if a.get('fault_active') and a.get('channel')==package['fault_specs'][fault]['target_channel']:
                        s['target_samples' if sample else 'target_observations']+=1


def stopping_rule(summary):
    rows=[]
    for f in ('0.2','0.4','0.6'):
        totals={p:{key:sum(summary[fault][f][p][key] for fault in FAULTS if fault!='none')
                   for key in ('false_accepts_after','delay_inputs_penalized')} for p in ('BA','BU','C2A')}
        wins={key:all(totals['BA'][key]<totals[p][key] for p in ('BU','C2A')) for key in ('false_accepts_after','delay_inputs_penalized')}
        rows.append(dict(fraction=f,totals=totals,wins=wins,passes=any(wins.values())))
    passed=sum(r['passes'] for r in rows)
    return dict(budgets=rows,passing_limited_budgets=passed,series_closed=passed<2,
        criterion='BA strictly beats both BU and C2A on the same endpoint in at least two limited budgets')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    def emit(stream):
        chunks=[];size=0
        for chunk in json.JSONEncoder(indent=2,allow_nan=False).iterencode(value):
            chunks.append(chunk);size+=len(chunk)
            if size>=1048576:
                stream.write(''.join(chunks).encode());chunks=[];size=0
        stream.write((''.join(chunks)+'\n').encode())
    with temp.open('wb') as raw:
        if path.suffix=='.gz':
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=9,mtime=0) as f:emit(f)
        else:emit(raw)
        raw.flush();os.fsync(raw.fileno())
    temp.replace(path)


def assemble(path,headers,package_paths,summary):
    """Canonical root framing, only one completed package resident at a time."""
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=9,mtime=0) as binary:
            def emit(text):binary.write(text.encode())
            emit('{\n')
            for key,value in headers.items():
                rendered=json.dumps(value,indent=2,allow_nan=False).splitlines()
                emit('  '+json.dumps(key)+': '+rendered[0])
                for line in rendered[1:]:emit('\n  '+line)
                emit(',\n')
            emit('  "packages": [\n')
            for i,p in enumerate(package_paths):
                with gzip.open(p,'rt') as stream:
                    lines=iter(stream);previous=next(lines)
                    for line in lines:
                        emit('    '+previous);previous=line
                    emit('    '+previous.rstrip('\n')+(',' if i<len(package_paths)-1 else '')+'\n')
            emit('  ],\n')
            rendered=json.dumps(summary,indent=2,allow_nan=False).splitlines()
            emit('  "summary": '+rendered[0])
            for line in rendered[1:]:emit('\n  '+line)
            emit('\n}\n')
        raw.flush();os.fsync(raw.fileno())
    temporary.replace(path)


def run(output,replay=None,noise=False,runtime_override=None):
    from attention.channel_monitor import run_batch
    output=Path(output)
    if output.suffix!='.gz':raise ValueError('A007 result archives must use .json.gz')
    protect_output(output,replay);audit=metadata(PROTOCOL,SOURCES,runtime_override)
    records,requests,specs,provenance=intake(replay)
    if provenance:audit['replay']=provenance
    model=s3.load_model();static=s2.load_model();library=read_json(ROOT/LIBRARY)
    output.parent.mkdir(parents=True,exist_ok=True)
    stem=output.name.removesuffix('.json.gz');checkpoint=output.parent/(stem+'-packages')
    if checkpoint.exists():raise FileExistsError('Checkpoint directory already exists: '+str(checkpoint))
    checkpoint.mkdir()
    headers=dict(experiment='A007',**audit,task_seed=20261201,order_seed=20261202,fault_seed=20261203,
        random_policy_seed=20261204,circle_seed=20261205,last_bit_noise=noise,
        intake_channel=records,fault_specs=specs,intake_sha256=canonical_sha256([records,specs]),
        self_model_sha256=canonical_sha256(model),sample_library_sha256=canonical_sha256(library),
        sample_library_sources=library['source_sha256'],sample_library_counts=library['counts'],
        sampling_unit='12 paired packages; 60 requests, 540 dependent conditions; repeated policies and budgets; stale links adjacent requests')
    atomic_json(output.with_name(stem+'-intake.json'),headers)
    refs,refaudit=reference_audit(requests);headers['reference_audit']=refaudit
    overlap={}
    for i in range(1,7):
        name='A00'+str(i);path=ROOT/('attention/results/'+name+'/results.json.gz')
        earlier=read_root_fields(path,('intake_channel',))
        keys={request_key(r) for r in earlier['intake_channel']}
        overlap[name]=dict(exact_request_matches=sum(request_key(r) in keys for r in records),source_sha256=source_sha256(path))
        del earlier
    headers['request_overlap']=overlap
    previous_request=Request(1,1,.01);previous=execute(previous_request,P.plan_modes(previous_request)).value
    summary=empty_summary();paths=[];parity=[]
    for package_id in range(12):
        infos=[];execution_specs=[]
        for task_id in range(package_id*5,(package_id+1)*5):
            record,r,truth=records[task_id],requests[task_id],refs[task_id]
            for condition in record['condition_order']:
                e=agent.Executor(r,record['request_sha256'],condition,previous)
                full=agent.run_policy(e,'C1',float('inf'),static['action_model'],list(agent.ORDER),full=True)
                original=P.evaluate(r,record['request_sha256'],e.outcome,e.model)
                if full['verdict']!=original['policies']['strict']['verdict']:parity.append([task_id,condition])
                error=abs(complex(*e.outcome['value'])-truth)
                infos.append(dict(task_id=task_id,cohort=record['cohort'],condition=condition,full=full,
                    accurate=bool(error<=r.tolerance),absolute_error=error,tolerance=r.tolerance,
                    outcome=e.outcome,reference=[truth.real,truth.imag]))
                execution_specs.append((r,record['request_sha256'],condition,previous))
            previous=execute(r,P.plan_modes(r)).value
        reference_cost=sum(x['full']['spent'] for x in infos);conditions={}
        for fault in FAULTS:
            conditions[fault]={}
            for f in FRACTIONS:
                conditions[fault][str(f)]={}
                fs=dict(specs[package_id][fault],random_seed=specs[package_id][fault]['random_seed']+int(f*100))
                for policy in POLICIES:
                    result=run_batch([agent.Executor(*s) for s in execution_specs],policy,f*reference_cost,model,library,package_id,fs,f,noise=noise)
                    for x,info in zip(result['inputs'],infos):
                        x['correct']=bool((info['accurate'] and x['verdict']=='accept') or (not info['accurate'] and x['verdict']=='violation_detected'))
                        x['false_accept']=bool(not info['accurate'] and x['verdict']=='accept')
                        x['postonset']=bool(x['closure_counter']>=fs['onset_action'])
                    result.update(correct=sum(x['correct'] for x in result['inputs']),
                        false_accepts_before=sum(x['false_accept'] and not x['postonset'] for x in result['inputs']),
                        false_accepts_after=sum(x['false_accept'] and x['postonset'] for x in result['inputs']))
                    if result['spent']>f*reference_cost+1e-10:raise AssertionError('Shared budget exceeded')
                    conditions[fault][str(f)][policy]=result
        package=dict(package_id=package_id,reference_cost=reference_cost,inputs=infos,fault_specs=specs[package_id],conditions=conditions)
        path=checkpoint/('package-%02d.json.gz'%package_id);atomic_json(path,package);paths.append(path)
        update_summary(summary,package)
        atomic_json(checkpoint/'progress.json',dict(completed_packages=package_id+1,summary=summary))
        del package,conditions
        print('Completed and saved package',package_id+1,'of 12',flush=True)
    headers['full_pipeline_parity_failures']=parity
    headers['stopping_rule']=stopping_rule(summary)
    assemble(output,headers,paths,summary)
    return dict(summary=summary,stopping_rule=headers['stopping_rule'],output=str(output),sha256=source_sha256(output,False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--replay')
    p.add_argument('--last-bit-noise',action='store_true');p.add_argument('--runtime-override');a=p.parse_args()
    result=run(a.output,a.replay,a.last_bit_noise,a.runtime_override)
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2))
