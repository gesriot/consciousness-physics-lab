"""Read-only classification of frozen protected-projection differences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
parser=argparse.ArgumentParser();parser.add_argument('repository',type=Path);parser.add_argument('first',type=Path);parser.add_argument('noise',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();sys.path.insert(0,str(args.repository.resolve()))
from binding import audit_b003 as A

def differences(a,b,path=()):
    if type(a)!=type(b):return [(path,a,b)]
    if isinstance(a,dict):
        if set(a)!=set(b):return [(path+('<keys>',),list(a),list(b))]
        return [d for k in a for d in differences(a[k],b[k],path+(k,))]
    if isinstance(a,list):
        if len(a)!=len(b):return [(path+('<length>',),len(a),len(b))]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+(str(i),))]
    return [] if a==b else [(path,a,b)]

counts=Counter();examples={};files=0;events=set();changed_files=set()
for path in sorted(args.noise.rglob('*.json*')):
    name=path.relative_to(args.noise)
    if not (str(name).startswith('series-') or path.name=='aggregate.json'):continue
    a,b=A.read(args.first/name),A.read(path);A.finite_tree(a);A.finite_tree(b);files+=1
    for p,x,y in differences(A.decision_projection(a),A.decision_projection(b)):
        kind='scores' if 'scores' in p else '/'.join(v for v in p if not v.isdigit())
        counts[kind]+=1;examples.setdefault(kind,dict(file=str(name),path='/'.join(p),left=x,right=y));changed_files.add(str(name))
        if kind=='scores' and 'actions' in p:events.add((str(name),p[p.index('actions')+1]))
report=dict(classification='read_only_diagnostic_not_official_comparison',files_compared=files,projected_scalar_differences=dict(counts),different_score_sample_snapshots=len(events),different_files=sorted(changed_files),examples=examples,all_numeric_values_finite=True,first_manifest_sha256=A.byte_sha((args.first/'manifest.json').read_bytes()),noise_manifest_sha256=A.byte_sha((args.noise/'manifest.json').read_bytes()),script_sha256=A.byte_sha(Path(__file__).read_bytes()))
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
