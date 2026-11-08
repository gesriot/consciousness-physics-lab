"""Select calibration inputs using only clean development-check responses."""
import argparse
import hashlib
from collections import Counter
from attention import agent, s3
from attention.calibration_samples import execute_sample, maximum_cost
from attention.compare_a004 import Archive
from attention.io import source_bytes, write_json
from lab.replay import canonical_sha256
from lab.scientist import ROOT

SOURCE='attention/results/A005/results.json.gz'
DESTINATION='attention/sample_library_a006.json'


def build():
    archive=Archive(ROOT/SOURCE); selected={};inventory=Counter();weights=s3.load_model()['action_model']['cost_weights']
    for package in archive.packages():
        records=archive.metadata['intake_channel']
        for row in package['inputs']:
            source=records[row['task_id']];condition=row['condition'];model=condition if condition in ('slab','mixture','model_scale') else 'sphere'
            recipe=dict(request=source['request'],candidate=row['reference'] if condition=='clean' else row['outcome']['value'],
                model=model,condition=condition,source_task_id=row['task_id'],source_package_id=package['package_id'],
                digest=source['request_sha256'],outcome=row['outcome'],exact_reference=condition=='clean')
            for action in row['full']['actions']:
                if action['action']=='declare':continue
                inventory[(action['action'],condition,action['observation']['verdict'])]+=1
            for family in ('collocation','second_moment','v2','modal','binding','normalization','first_moment','high_frequency'):
                candidate=dict(recipe,family=family)
                # All selection is on the intact channel. No fault hook is exercised here.
                result=execute_sample(candidate,weights)
                key=(family,condition,result['verdict'])
                if key not in selected:
                    selected[key]=dict(candidate,expected=result['verdict'],clean_result=result,maximum_cost=maximum_cost(candidate,weights))
    samples=[]
    for i,(key,row) in enumerate(sorted(selected.items())):samples.append(dict(sample_id=i,**row))
    counts={f:dict(Counter(r['expected'] for r in samples if r['family']==f)) for f in sorted({r['family'] for r in samples})}
    sources={SOURCE:hashlib.sha256(source_bytes(ROOT/SOURCE)).hexdigest(),
             'attention/self_model_s3.json':hashlib.sha256(source_bytes(ROOT/'attention/self_model_s3.json')).hexdigest()}
    return dict(version=1,samples=samples,counts=counts,source_sha256=sources,
        selection='First clean response per family/condition/verdict in A005 package/input order; clean candidate is archived independent reference; no corrupted V2 selection.',
        inventory=[dict(action=a,condition=c,verdict=v,count=n) for (a,c,v),n in sorted(inventory.items())],
        samples_sha256=canonical_sha256(samples))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=DESTINATION);args=parser.parse_args()
    value=build();write_json(args.output,value);print(value['counts'])
