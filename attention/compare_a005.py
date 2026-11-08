"""Bounded-memory replay comparison, separating decisions from diagnostics."""
import argparse
from itertools import zip_longest
from attention.compare_a004 import Archive
from attention.io import write_json
from scripts.compare_experiment_runs import numeric_summary


def _package_view(value):
    if isinstance(value,dict):
        exact={'cost','spent','budget','reference_cost','replacement_cost'}
        return {k:v if k in exact else _package_view(v) for k,v in value.items() if not k.endswith('seconds')}
    if isinstance(value,list):return [_package_view(v) for v in value]
    if isinstance(value,float):return '<numeric diagnostic>'
    return value


def compare_paths(first,second):
    a,b=Archive(first),Archive(second);missing=object();different=[];diagnostics={}
    for i,(left,right) in enumerate(zip_longest(a.packages(),b.packages(),fillvalue=missing)):
        if left is missing or right is missing:
            different.append(i);continue
        if _package_view(left)!=_package_view(right):different.append(i)
        for key,row in numeric_summary([left],[right]).items():
            entry=diagnostics.setdefault(key,dict(count=0,changed=0,max_abs=0.,max_rel=0.,nonfinite=0))
            for field in ('count','changed','nonfinite'):entry[field]+=row[field]
            for field in ('max_abs','max_rel'):entry[field]=max(entry[field],row[field])
    fields=('experiment','intake_channel','global_permutations','intake_sha256','self_model_sha256','onset_package','calibration','summary','full_pipeline_parity_failures','request_overlap')
    root_mismatches=[k for k in fields if a.metadata[k]!=b.metadata[k]]
    return dict(decisions_pass=not different and not root_mismatches,differing_packages=different,
                root_mismatches=root_mismatches,numeric_diagnostics=diagnostics,
                numeric_diagnostics_informational=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=compare_paths(args.first,args.second);write_json(args.output,result)
    print({k:v for k,v in result.items() if k!='numeric_diagnostics'})
    if not result['decisions_pass']:raise SystemExit(1)
