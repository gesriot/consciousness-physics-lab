"""Post-map construction: demonstrate the frozen numerical gate's failure."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.guard import Request
from lab.surface_order import surface_response
from lab.three_way import GRIDS, check_three_way


def construct(source):
    with gzip.open(source,'rt') as f:
        data=json.load(f)
    candidates=[(s['error']/s['uncertainty'],r,s) for r in data['points']
                for s in r['steps'] if s['converging'] and not s['covered']]
    candidates.sort(key=lambda x:x[0],reverse=True)
    for _,r,s in candidates:
        value=complex(*r['values'][GRIDS.index(s['grid'])])
        tolerance=(s['error']+s['uncertainty'])/2
        result=check_three_way(Request(complex(*r['q']),1,tolerance),value,
                               solver=surface_response)
        if result['accepted']:
            return dict(kind='post-map constructed numerical-gate witness; not an additional sampled trial or full pipeline claim',
                point=r,grid=s['grid'],supplied_value=[value.real,value.imag],
                tolerance=tolerance,true_error=s['error'],operational=result)
    raise RuntimeError('No operational numerical-gate witness found')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('map');p.add_argument('output')
    a=p.parse_args();Path(a.output).write_text(json.dumps(construct(a.map),indent=2)+'\n')
