"""Development-only channel partition and executable boundary standards."""
import argparse
import copy
from collections import Counter
from pathlib import Path
import numpy as np
from attention import agent, s3
from attention.a006 import read_root_fields, source_sha256
from attention.calibration_samples import execute_sample, maximum_cost, request
from attention.io import read_json, write_json
from attention.compare_a004 import Archive
from lab import frozen_pipeline as P
from lab.common_mode import encode
from lab.coverage_map import precise
from lab.executor import request_hash
from lab.replay import canonical_sha256

V2_EDGES=(0.0009344424533473637,0.7622440218285347,9.724297155975574,16360.24821532068,691475293.9820151)
TOL_EDGES=(1.166881434659866e-05,0.0002790990323118471,0.0015705174092262033,0.007414398085483424,0.019005124124090125)
CHANNELS=tuple('v2:'+str(i) for i in range(6))+tuple('collocation:'+str(i) for i in range(6))+agent.FEATURES+('binding','modal')
SOURCE='attention/results/A005/results.json.gz'


def channel(family, fields):
    if not isinstance(fields,dict): fields=encode(fields)
    if family.startswith('v2'):
        value=abs(complex(*fields['z'])*fields['mean_time']);edges=V2_EDGES;family='v2'
    elif family=='collocation':value=fields['tolerance'];edges=TOL_EDGES
    else:return family
    return family+':'+str(next((i for i,x in enumerate(edges) if value<=x),5))


def build():
    records=read_root_fields(SOURCE,('intake_channel',))['intake_channel']
    weights=s3.load_model()['action_model']['cost_weights']
    selected={};boundaries={};attempts=[];counts=Counter();truth_records=[]
    def retain(recipe,result,kind):
        ch=channel(recipe['family'],recipe['request']);key=(ch,kind)
        if key not in selected:
            selected[key]=dict(recipe,channel=ch,expected=result['verdict'],clean_result=result,maximum_cost=maximum_cost(recipe,weights),standard_kind=kind)
    old=read_json('attention/sample_library_a006.json')
    for sample in old['samples']:
        # Retain live executions, never substitute the archived answer.
        result=execute_sample(sample,weights)
        retain({k:v for k,v in sample.items() if k not in ('sample_id','clean_result','expected','maximum_cost')},result,'accept' if result['verdict']=='accept' else 'nonaccept')
    for record in records:
        fields=record['request'];r=request({'request':fields});q=r.z*r.mean_time
        hp=precise(q);truth=complex(*map(float,hp))
        truth_records.append(dict(task_id=record['task_id'],high_precision=hp,reference=[truth.real,truth.imag]))
        base=dict(request=copy.deepcopy(fields),model='sphere',source_task_id=record['task_id'],source_package_id=record['package_id'],source=SOURCE,digest=request_hash(r),exact_reference=True)
        for family in ('v2','collocation'):
            counts[channel(family,fields)]+=1
            for label,value in [('accept',truth),('nonaccept',truth+2*r.tolerance)]:
                recipe=dict(base,family=family,candidate=[value.real,value.imag],exact_reference=label=='accept')
                result=execute_sample(recipe,weights)
                retain(recipe,result,'accept' if result['verdict']=='accept' else 'nonaccept')
        ch=channel('v2',fields)
        if ch in boundaries:continue
        values=[]
        for grid in agent.GRIDS:
            values.append(P.model_response('fv',r.z,r.mean_time,grid,'sphere'))
            if grid<P.CONFIG.first_decisive_grid:continue
            error=abs(values[-1]-truth)
            if error<=1e-14:continue
            # Candidate is 1 percent outside the tolerance and aligned with
            # the discretization error; the intact uncertainty must prevent acceptance.
            tol=25*error;value=truth+(values[-1]-truth)/error*(1.01*tol)
            changed=copy.deepcopy(fields);changed['tolerance']=tol
            recipe=dict(base,request=changed,family='v2',candidate=[value.real,value.imag],boundary=True,
                exact_reference=False,source_grid=grid,reference=[truth.real,truth.imag],reference_decimal=hp,
                actual_error=abs(value-truth),error_ratio=abs(value-truth)/tol)
            result=execute_sample(recipe,weights)
            broken=execute_sample(recipe,weights,'underestimated')
            ok=result['verdict']!='accept' and broken['verdict']=='accept' and abs(value-truth)>tol
            attempts.append(dict(task_id=record['task_id'],channel=ch,grid=grid,tolerance=tol,error_ratio=abs(value-truth)/tol,
                intact=result['verdict'],underestimated=broken['verdict'],selected=ok))
            if ok:
                recipe['digest']=request_hash(request(recipe))
                retain(recipe,result,'boundary')
                boundaries[ch]=dict(task_id=record['task_id'],grid=grid,tolerance=tol,error_ratio=abs(value-truth)/tol,
                    intact=result['verdict'],underestimated=broken['verdict'],broken_result=broken)
                break
    samples=[]
    # Boundary/nonaccept standards first, to avoid knowingly spending the first
    # test on an insensitive always-accept standard.
    kinds={'boundary':0,'nonaccept':1,'accept':2}
    for i,((ch,kind),row) in enumerate(sorted(selected.items(),key=lambda x:(CHANNELS.index(x[0][0]),kinds[x[0][1]]))):
        samples.append(dict(sample_id=i,**row))
    missing=[ch for ch in CHANNELS if not any(s['channel']==ch for s in samples)]
    assert not missing,missing
    maxprice=max(s['maximum_cost'] for s in samples)
    development_packages=[]
    for package in Archive(SOURCE).packages():
        vulnerabilities=[]
        for index,row in enumerate(package['inputs']):
            if row['accurate']:continue
            evidence={a['action']:a['observation'] for a in row['full']['actions'] if a['action']!='declare'}
            if 'collocation' not in evidence:continue
            modified=copy.deepcopy(evidence);modified['collocation']={'verdict':'accept'}
            if agent.verdict(modified)=='accept':
                vulnerabilities.append(dict(input_index=index,task_id=row['task_id'],condition=row['condition'],
                    channel=channel('collocation',records[row['task_id']]['request'])))
        development_packages.append(dict(package_id=package['package_id'],reference_cost=package['reference_cost'],
            budgets={str(f):f*package['reference_cost'] for f in (.2,.4,.6,1.)},
            permissive_collocation_vulnerabilities=vulnerabilities))
    report=dict(source=SOURCE,source_sha256=source_sha256(SOURCE,False),requests=len(records),channel_counts=dict(counts),
        boundaries=boundaries,boundary_attempts=attempts,channels_without_boundary=[ch for ch in CHANNELS[:6] if ch not in boundaries],
        quota_reference=dict(fractions=[.2,.4,.6,1.],quotas=[2,4,6,10],channels=len(CHANNELS),
            maximum_sample_cost=maxprice,reserve_by_quota={str(q):q*maxprice for q in (2,4,6,10)},
            complete_cycle_packages={str(q):int(np.ceil(len(CHANNELS)/q)) for q in (2,4,6,10)}),
        development_packages=development_packages,reference_audit=truth_records,selection='A005 development only; fixed first eligible intact response per channel/category; V2 boundary selected for known U/100 fault; no unseen fault evaluated.')
    library=dict(version=1,channels=list(CHANNELS),v2_edges=list(V2_EDGES),tolerance_edges=list(TOL_EDGES),samples=samples,
        counts={ch:dict(Counter(s['expected'] for s in samples if s['channel']==ch)) for ch in CHANNELS},
        boundary_channels=sorted(boundaries),source_sha256={SOURCE:source_sha256(SOURCE,False),'attention/sample_library_a006.json':source_sha256('attention/sample_library_a006.json',False)},
        samples_sha256=canonical_sha256(samples),selection=report['selection'])
    return library,report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='attention/sample_library_a007.json');p.add_argument('--report',default='attention/results/A007/development.json');args=p.parse_args()
    library,report=build();Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,library);write_json(args.report,report)
    print(library['counts']);print('Boundary channels:',library['boundary_channels']);print(report['quota_reference'])
