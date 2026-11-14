"""Compact sequential B004 execution, freeze gates and replay comparison."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
from binding import reconcile_b004 as B

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'binding/protocols/B004-source-reconciliation.md'
SOURCES = ('binding/source_lab_b004.py', 'binding/reconcile_b004.py', 'binding/run_b004.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py')
DEVELOPMENT = tuple(range(20270301,20270313))
CONFIRMATION = tuple(range(20270313,20270325))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = B.canonical(value)+b'\n'
    if path.suffix == '.gz':
        data = gzip.compress(data, compresslevel=9, mtime=0)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def read(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.system(), machine=platform.machine())


def source_metadata():
    return dict(protocol_sha256=sha_file(ROOT/PROTOCOL),
                source_sha256={p:sha_file(ROOT/p) for p in SOURCES},
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def verify_runtime_manifest(manifest, override=False):
    current = source_metadata()
    for key in ('protocol_sha256','source_sha256'):
        if manifest[key] != current[key]:
            raise ValueError('Replay source or protocol identity mismatch')
    if manifest['runtime'] != runtime() and not override:
        raise ValueError('Runtime differs; explicit --runtime-override required')


def make_archive(inputs, faults, noise=False):
    cases = {}
    for fault in faults:
        series = B.build_series(inputs, fault, noise)
        service = B.RecheckService(series, noise)
        outputs, conditions = {}, {}
        for p in B.POLICIES:
            for budget in B.BUDGETS:
                for signal in B.SIGNALS:
                    output = B.simulate(series,p,budget,signal,noise,service)
                    key = B.digest(output)
                    outputs.setdefault(key,output)
                    conditions[f'{p}|{budget}|{signal}'] = key
        series = {k:v for k,v in series.items() if k != 'intake'}
        cases[fault] = dict(series=series, outputs=outputs, conditions=conditions,
                            physical_verifier_evaluations=service.physical_evaluations)
    return dict(inputs=inputs,intake_sha256=B.digest(inputs),cases=cases)


def summarize(archives):
    cells = {}
    for a in archives:
        for fault, case in a['cases'].items():
            for name, key in case['conditions'].items():
                cells.setdefault(fault+'|'+name, []).append(case['outputs'][key]['metrics'])
    result = {}
    for name, rows in sorted(cells.items()):
        cell = dict(series=len(rows), total={},mean={},standard_error={},depths={})
        for k in rows[0]:
            if k == 'depths':
                continue
            x = [r[k] for r in rows]
            cell['total'][k] = sum(x)
            mean = math.fsum(x)/len(x)
            cell['mean'][k] = mean
            cell['standard_error'][k] = (math.sqrt(math.fsum((v-mean)**2 for v in x)/(len(x)*(len(x)-1)))
                                         if len(x)>1 else 0.)
        for d in ('1','2','3'):
            cell['depths'][d] = {k:sum(r['depths'][d][k] for r in rows) for k in rows[0]['depths'][d]}
        result[name] = cell
    primary = [f'{fault}|{{p}}|{b}|{s}' for fault in B.FAULTS if fault != 'clean'
               for b in B.BUDGETS for s in B.SIGNALS if f'{fault}|G|{b}|{s}' in result]
    failures = []
    for name in primary:
        g,f = (result[name.format(p=p)]['total'] for p in ('G','F'))
        if f['remaining_bad'] > g['remaining_bad'] or f['unnecessary'] > g['unnecessary']:
            failures.append(name.format(p='F-v-G'))
    return dict(cells=result, independent_series=len(archives),
                closure_F_no_worse_in_every_primary_cell=bool(primary) and not failures,
                closure_failure_cells=failures)


def gate(archives):
    exposure = {f:{str(d):dict(affected=0,inaccurate=0) for d in (1,2,3)} for f in B.FAULTS if f!='clean'}
    contrasts = {k:0 for k in ('G_F_deep','G_G1_depth3','G1_F_depth2','G_Gx','F_T','active_N')}
    placebo_mismatches, conflicts = 0,0
    for a in archives:
        for f,case in a['cases'].items():
            s=case['series']; byid={r['id']:r for r in s['records']}
            if f!='clean':
                for c in s['affected']:
                    row=exposure[f][str(byid[c]['depth'])]; row['affected']+=1
                    row['inaccurate']+=not B.close(byid[c]['value'],s['truth'][c])
    admitted=[f for f,depths in exposure.items() if all(x['affected']>=10 and x['inaccurate']>=5 for x in depths.values())]
    for a in archives:
        for f,case in a['cases'].items():
            if f not in admitted:
                continue
            for b in B.BUDGETS:
                for branch in B.SIGNALS:
                    out={p:case['outputs'][case['conditions'][f'{p}|{b}|{branch}']] for p in B.POLICIES}
                    def selected(p,depths):
                        return [e['id'] for e in out[p]['trace'] if e['kind'] in ('repair','withdraw') and e['depth'] in depths]
                    contrasts['G_F_deep']+=selected('G',(2,3))!=selected('F',(2,3))
                    contrasts['G_G1_depth3']+=selected('G',(3,))!=selected('G1',(3,))
                    contrasts['G1_F_depth2']+=selected('G1',(2,))!=selected('F',(2,))
                    contrasts['G_Gx']+=selected('G',(1,2,3))!=selected('Gx',(1,2,3))
                    contrasts['F_T']+=selected('F',(1,2,3))!=selected('T',(1,2,3))
                    contrasts['active_N']+=bool(selected('G',(1,2,3)))
                    placebo_mismatches+=selected('G',(1,2,3))!=selected('Gxprime',(1,2,3))
                    conflicts+=len(out['G']['protected']['conflicts']) if branch=='conflict' and b==2 else 0
    admitted=[f for f,depths in exposure.items() if all(x['affected']>=10 and x['inaccurate']>=5 for x in depths.values())]
    return dict(exposure=exposure,admitted_faults=admitted,discernibility_counts=contrasts,
                placebo_mismatches=placebo_mismatches,conflicts=conflicts,
                pass_gate=bool(admitted) and all(contrasts.values()) and not placebo_mismatches and conflicts>0)


def run(output,phase,replay=None,noise=False,override=False,freeze=None):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Output already contains a run; do not overwrite')
    faults=list(B.FAULTS)
    freeze_record=None
    if phase=='confirmation':
        if freeze is None:
            raise ValueError('Confirmation requires committed freeze')
        freeze_path=Path(freeze).resolve()
        freeze_record=read(freeze_path)
        if not freeze_record['gate']['pass_gate']:
            raise ValueError('Development gate failed')
        rel=str(freeze_path.relative_to(ROOT))
        subprocess.run(['git','ls-files','--error-unmatch',rel],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--exit-code','HEAD','--',rel],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        verify_runtime_manifest(freeze_record['development_manifest'],override)
        faults=['clean']+freeze_record['gate']['admitted_faults']
    seedlist=DEVELOPMENT if phase=='development' else CONFIRMATION
    manifest=dict(schema=1,phase=phase,noise=bool(noise),runtime=runtime(),runtime_override=bool(override),
                  seeds=list(seedlist),faults=faults,**source_metadata(),files={},intakes={})
    if freeze_record:
        manifest['freeze_sha256']=sha_file(freeze)
    if replay:
        previous=read(Path(replay)/'manifest.json'); verify_runtime_manifest(previous,override)
        if previous['phase']!=phase or previous['faults']!=faults or previous['seeds']!=list(seedlist):
            raise ValueError('Replay cohort mismatch')
        manifest['replay_manifest_sha256']=sha_file(Path(replay)/'manifest.json')
    start=time.monotonic(); archives=[]
    for seed in seedlist:
        name=f'series-{seed}.json.gz'
        if replay:
            path=Path(replay)/name
            if sha_file(path)!=previous['files'][name]:
                raise ValueError('Replay file checksum mismatch')
            inp=read(path)['inputs']
            if B.digest(inp)!=previous['intakes'][str(seed)]:
                raise ValueError('Replay input checksum mismatch')
        else:
            inp=B.intake(seed)
        archive=make_archive(inp,faults,noise)
        write(out/name,archive); manifest['files'][name]=sha_file(out/name)
        manifest['intakes'][str(seed)]=archive['intake_sha256']
        archives.append(archive)
        write(out/'manifest.partial.json',manifest)
        print(json.dumps(dict(seed=seed,seconds=round(time.monotonic()-start,3),bytes=(out/name).stat().st_size)),flush=True)
    summary=summarize(archives); write(out/'summary.json',summary)
    if phase=='development':
        write(out/'gate.json',gate(archives))
    manifest['elapsed_seconds']=time.monotonic()-start
    write(out/'manifest.json',manifest); (out/'manifest.partial.json').unlink()
    return summary


def numeric_differences(a,b,path='',out=None):
    out={} if out is None else out
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for k in a: numeric_differences(a[k],b[k],path+'/'+k,out)
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for x,y in zip(a,b): numeric_differences(x,y,path+'/*',out)
    elif isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)):
        if a!=b:
            row=out.setdefault(path,dict(count=0,max_abs=0.,max_rel=0.)); diff=abs(a-b)
            row['count']+=1;row['max_abs']=max(row['max_abs'],diff)
            row['max_rel']=max(row['max_rel'],diff/max(abs(a),abs(b),1e-300))
    elif a!=b:
        raise ValueError('Non-numeric drift at '+path)
    return out


def compare(first,second):
    first,second=Path(first),Path(second);m1,m2=read(first/'manifest.json'),read(second/'manifest.json')
    protected_fields=('phase','seeds','faults','source_sha256','protocol_sha256','intakes','freeze_sha256')
    differences=[k for k in protected_fields if m1.get(k)!=m2.get(k)]
    diagnostics={};count=0
    for name in m1['files']:
        a,b=read(first/name),read(second/name)
        for base,manifest in ((first,m1),(second,m2)):
            if sha_file(base/name)!=manifest['files'].get(name):
                differences.append(name+':checksum')
        if a['inputs']!=b['inputs']:differences.append(name+':inputs')
        for fault,ca in a['cases'].items():
            cb=b['cases'][fault]
            numeric_differences(ca['series'],cb['series'],fault+'/initial',diagnostics)
            if ca['conditions'].keys()!=cb['conditions'].keys():differences.append(name+':conditions')
            for key,ref in ca['conditions'].items():
                x,y=ca['outputs'][ref],cb['outputs'][cb['conditions'][key]];count+=1
                if x['protected']!=y['protected'] or x['metrics']!=y['metrics']:
                    differences.append(name+':'+fault+':'+key)
                numeric_differences(x['trace'],y['trace'],fault+'/trace',diagnostics)
                numeric_differences(x['final'],y['final'],fault+'/final',diagnostics)
    return dict(decisions_pass=not differences,differing_decisions=differences,
                compared_conditions=count,numerical_diagnostics=diagnostics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=('development','confirmation'),default='development')
    p.add_argument('--replay');p.add_argument('--noise',action='store_true')
    p.add_argument('--runtime-override',action='store_true');p.add_argument('--freeze')
    p.add_argument('--compare',nargs=2)
    a=p.parse_args()
    if a.compare:
        result=compare(*a.compare);write(a.output,result)
        if not result['decisions_pass']:raise SystemExit(3)
    else:run(a.output,a.phase,a.replay,a.noise,a.runtime_override,a.freeze)

if __name__=='__main__':main()
