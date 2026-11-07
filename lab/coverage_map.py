"""Registered E012 adaptive map, with unchanged frozen sphere arithmetic."""
import argparse
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import mpmath as mp
import numpy as np
from lab import surface_order
from lab.experiment_audit import metadata
from lab.scientist import analytic, modal_partial_sums
from lab.three_way import GRIDS, step_evidence

PROTOCOL = 'experiments/E012-coverage-map.md'
SOURCES = ['lab/coverage_map.py', 'lab/surface_order.py', 'lab/three_way.py',
           'lab/finite_volume.py', 'lab/scientist.py', 'lab/experiment_audit.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py']


def pair(v):
    return [v.real, v.imag]


def coordinate(key):
    x, y = key
    logq = -6 + x/160
    phase = -math.pi/2 + y*math.pi/288
    return logq, phase, 10**logq * complex(math.cos(phase), math.sin(phase))


def base_cells():
    return [(x,y,x+8,y+8) for x in range(0,2880,8) for y in range(0,288,8)]


def corners(cell):
    a,b,c,d = cell
    return [(a,b),(a,d),(c,b),(c,d)]


def children(cell):
    a,b,c,d = cell
    e,f = (a+c)//2,(b+d)//2
    return [(a,b,e,f),(e,b,c,f),(a,f,e,d),(e,f,c,d)]


def needs_refinement(cell, records):
    flags = [tuple((s['converging'],s['covered']) for s in records[k]['steps'])
             for k in corners(cell)]
    return any(f != flags[0] for f in flags[1:])


def dimensionless_check():
    records = []
    for mag in (.01,1.,1e4):
        for phase in (0.,.7):
            q = mag*np.exp(1j*phase)
            for n in (64,512,4096):
                for t in (.125,3.,8.):
                    a = surface_order.surface_response(q,1.,n)
                    b = surface_order.surface_response(q/t,t,n)
                    limit = 32*np.finfo(float).eps*n*n*max(1,abs(a))
                    records.append(dict(kind='FV',q=pair(q),grid=n,time=t,
                        error=abs(a-b),allowance=limit,passed=bool(abs(a-b)<=limit)))
            a = modal_partial_sums(q,1.,[4096])[0]
            for t in (.125,3.,8.):
                b = modal_partial_sums(q/t,t,[4096])[0]
                limit = 128*np.finfo(float).eps*max(1,abs(a))
                records.append(dict(kind='modal',q=pair(q),grid=4096,time=t,
                    error=abs(a-b),allowance=limit,passed=bool(abs(a-b)<=limit)))
    return records


def precise(q):
    with mp.workdps(60):
        x = mp.sqrt(15*mp.mpc(q.real,q.imag))
        v = 3*(mp.coth(x)/x-1/x**2)
        return [mp.nstr(v.real,60),mp.nstr(v.imag,60)]


def reference_batch(keys, rng):
    low = [k for k in keys if abs(coordinate(k)[2])<=1e8]
    selected = {low[i] for i in rng.choice(len(low),size=len(low)//10,replace=False)}
    selected.update(k for k in keys if abs(coordinate(k)[2])>1e8)
    refs = {k:complex(analytic(coordinate(k)[2],1.)) for k in keys}
    audits = {}
    def check(k):
        q = coordinate(k)[2]
        hp = precise(q)
        with mp.workdps(60):
            err = float(abs(mp.mpc(refs[k].real,refs[k].imag)-mp.mpc(*hp)))
        limit = 1e-13+1e-12*abs(complex(*map(float,hp)))
        return dict(coordinate=list(k),high_precision=hp,analytic=pair(refs[k]),
                    absolute_error=err,passed=bool(math.isfinite(err) and err<=limit))
    for k in sorted(selected):
        audits[k] = check(k)
    fallback = not all(r['passed'] for r in audits.values())
    if fallback:
        for k in keys:
            if k not in audits:
                audits[k] = check(k)
            refs[k] = complex(*map(float,audits[k]['high_precision']))
    return refs, dict(fallback=fallback,random_subset_size=len(low)//10,
        audited=len(audits),failed=sum(not r['passed'] for r in audits.values()),
        max_error=max((r['absolute_error'] for r in audits.values()),default=0),
        records=list(audits.values()))


def evaluate(key, level, reference):
    logq,phase,q = coordinate(key)
    values = [surface_order.surface_response(q,1.,n) for n in GRIDS]
    steps = []
    for i,n in enumerate(GRIDS):
        if n<512:
            continue
        evidence = step_evidence(values[:i+1],n,reference,1.)
        error = abs(values[i]-reference)
        delta = abs(values[i]-values[i-1])
        previous = abs(values[i-1]-values[i-2])
        ratio = delta/previous if previous else None
        rounding = 32*np.finfo(float).eps*n*n*max(1,abs(values[i]))
        covered = error<=evidence['uncertainty_estimate']
        steps.append(dict(grid=n,converging=evidence['converging'],covered=bool(covered),
            error=error,uncertainty=evidence['uncertainty_estimate'],delta=delta,
            previous_delta=previous,ratio=ratio,ratio_deficit=None if ratio is None else ratio-.5,
            zero_previous_delta=previous==0,rounding_exception=bool(delta>.5*previous and delta<=rounding),
            layer_ratio=math.sqrt(1/(15*abs(q)))/(1-surface_order.faces(n)[-2])))
    return dict(coordinate=list(key),level=level,log10_magnitude=logq,phase=phase,
                q=pair(q),reference=pair(reference),values=[pair(v) for v in values],steps=steps)


def write(path, data):
    path = Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'wt') as f:
            json.dump(data,f,allow_nan=False,separators=(',',':'))
    else:
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def summarize(records):
    result = {}
    for n in (512,1024,2048,4096):
        rows = [s for r in records for s in r['steps'] if s['grid']==n]
        misses = [s for s in rows if not s['covered']]
        result[str(n)] = dict(steps=len(rows),misses=len(misses),
            dangerous_misses=sum(s['converging'] for s in misses),
            covered_not_converging=sum(s['covered'] and not s['converging'] for s in rows),
            min_miss_layer_ratio=min((s['layer_ratio'] for s in misses),default=None),
            max_miss_layer_ratio=max((s['layer_ratio'] for s in misses),default=None),
            min_miss_ratio_deficit=min((s['ratio_deficit'] for s in misses if s['ratio_deficit'] is not None),default=None),
            max_error_over_U=max(s['error']/s['uncertainty'] for s in rows))
    return result


def run(directory):
    output = Path(directory)
    if output.exists():
        raise ValueError('Output directory already exists')
    meta = metadata(PROTOCOL,SOURCES)
    output.mkdir(parents=True)
    dimensions = dimensionless_check()
    write(output/'dimensionless.json',dimensions)
    if not all(r['passed'] for r in dimensions):
        raise RuntimeError('Dimensionless test failed; map not evaluated')
    rng = np.random.default_rng(20261124)
    active = base_cells()
    records,refinements,batches = {},[],[]
    started = time.perf_counter()
    original_faces = surface_order.faces
    with patch.object(surface_order,'faces',lru_cache(maxsize=7)(original_faces)):
        for level in range(4):
            keys = sorted({k for c in active for k in corners(c)}-records.keys())
            refs,audit = reference_batch(keys,rng)
            write(output/f'reference-level-{level}.json.gz',audit)
            for k in keys:
                records[k] = evaluate(k,level,refs[k])
            batches.append(dict(level=level,new_points=len(keys),audit_count=audit['audited'],
                reference_failures=audit['failed'],fallback=audit['fallback'],max_reference_error=audit['max_error']))
            print(json.dumps(batches[-1]),flush=True)
            flagged = [c for c in active if needs_refinement(c,records)] if level<3 else []
            refinements.append(dict(level=level,cells_tested=len(active),refined_cells=flagged))
            active = [child for c in flagged for child in children(c)]
    ordered = [records[k] for k in sorted(records)]
    summary = summarize(ordered)
    write(output/'map.json.gz',dict(experiment='E012',**meta,points=ordered,
        batches=batches,refinements=refinements,summary=summary,seconds=time.perf_counter()-started))
    write(output/'summary.json',dict(points=len(ordered),batches=batches,grids=summary))
    with (output/'misses.csv').open('w') as f:
        import csv
        names = ['x','y','log10_magnitude','phase','grid','converging','error','uncertainty','layer_ratio','ratio','ratio_deficit','rounding_exception']
        writer = csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for r in ordered:
            for s in r['steps']:
                if not s['covered']:
                    writer.writerow(dict(x=r['coordinate'][0],y=r['coordinate'][1],
                        log10_magnitude=r['log10_magnitude'],phase=r['phase'],
                        **{k:s[k] for k in names[4:]}))
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(run(args.directory),indent=2))
