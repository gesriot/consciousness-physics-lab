"""Exact E014 repeat and independent recorded-endpoint audit."""
import argparse
import gzip
import json
from pathlib import Path
from lab import convergence_candidates as m


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def audit(first,second):
    a,b=Path(first),Path(second)
    left,right=load(a/'map.json.gz'),load(b/'map.json.gz')
    comparisons={k:left[k]==right[k] for k in
        ('experiment','protocol_sha256','source_sha256','points','batches','refinements','summary')}
    for name in ('cost-replay.json.gz',*[f'reference-level-{i}.json.gz' for i in range(4)]):
        comparisons[name]=load(a/name)==load(b/name)
    for name in ('dimensionless.json','summary.json'):
        comparisons[name]=json.loads((a/name).read_text())==json.loads((b/name).read_text())
    comparisons['recomputed_summary']=m.summarize(left['points'])==left['summary']
    summary=json.loads((a/'summary.json').read_text())
    comparisons['recomputed_selection']=m.select(left['summary'],summary['cost'])==summary['selected_V2']
    dangerous=[]
    for p in left['points']:
        for s in p['steps']:
            if not s['covered'] and any(s['candidates'].values()):
                dangerous.append(dict(coordinate=p['coordinate'],grid=s['grid'],
                    candidates=s['candidates'],error=s['error'],uncertainty=s['uncertainty']))
    return dict(passed=all(comparisons.values()),exact_fields=comparisons,
        excluded=['elapsed seconds','git head and environment provenance'],
        points=len(left['points']),selected_V2=summary['selected_V2'],dangerous_misses=dangerous)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second');parser.add_argument('output')
    args=parser.parse_args();result=audit(args.first,args.second)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='dangerous_misses'},indent=2))
    if not result['passed']:raise SystemExit(1)
