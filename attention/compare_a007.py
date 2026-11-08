"""Bounded-memory exact decision and accounting comparison for A007."""
import argparse
from itertools import zip_longest
from attention.compare_a004 import Archive
from attention.io import write_json
from scripts.compare_experiment_runs import numeric_summary

EXACT_NUMBERS={'cost','spent','budget','reference_cost','replacement_cost','sample_cost',
    'ordinary_cost','maximum_price','initial_reserve','ordinary_budget','reserve_before',
    'sample_deadline','ordinary_cost_before','maximum_cost','charge','spent_before'}
PROVENANCE={'environment','runtime_override','git_head','source_sha256','replay',
    'last_bit_noise','reference_audit','checkpoint_files','output','output_path'}


def decision_view(value):
    if isinstance(value,dict):
        return {k:(v if k in EXACT_NUMBERS else decision_view(v)) for k,v in value.items()
                if not k.endswith('seconds')}
    if isinstance(value,list):return [decision_view(v) for v in value]
    if isinstance(value,float):return '<numeric diagnostic>'
    return value


def compare_paths(first,second):
    left,right=Archive(first),Archive(second);missing=object()
    different=[];diagnostics={}
    for index,(a,b) in enumerate(zip_longest(left.packages(),right.packages(),fillvalue=missing)):
        if a is missing or b is missing:
            different.append(index);continue
        if decision_view(a)!=decision_view(b):different.append(index)
        for key,row in numeric_summary([a],[b]).items():
            entry=diagnostics.setdefault(key,dict(count=0,changed=0,max_abs=0.,max_rel=0.,nonfinite=0))
            for field in ('count','changed','nonfinite'):entry[field]+=row[field]
            for field in ('max_abs','max_rel'):entry[field]=max(entry[field],row[field])
    keys=(set(left.metadata)|set(right.metadata))-PROVENANCE
    root=[key for key in sorted(keys) if left.metadata.get(key,missing)!=right.metadata.get(key,missing)]
    return dict(decisions_pass=not different and not root,differing_packages=different,
        root_mismatches=root,numeric_diagnostics=diagnostics,numeric_diagnostics_informational=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=compare_paths(args.first,args.second);write_json(args.output,result)
    print({k:v for k,v in result.items() if k!='numeric_diagnostics'})
    if not result['decisions_pass']:raise SystemExit(1)
