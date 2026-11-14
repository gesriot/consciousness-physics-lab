"""Compact sequential B006 runs, global development selection and replay."""
import argparse
import copy
import math
from pathlib import Path
import time
from binding import run_b005 as common
from binding import diagnose_b006 as B
from binding import source_lab_b006 as lab

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'binding/protocols/B006-fallible-diagnosis.md'
SOURCES = ('binding/source_lab_b006.py','binding/diagnose_b006.py','binding/run_b006.py',
           'binding/source_lab_b005.py','binding/diagnose_b005.py','binding/source_lab_b004.py',
           'binding/run_b005.py','vendor/sphere-thermal-cli/sphere_thermal/response.py')
DEVELOPMENT = tuple(range(20270501,20270513))
CONFIRMATION = tuple(range(20270601,20270613))
canonical, read, write = common.canonical, common.read, common.write
sha_file, file_hashes, verify_file = common.sha_file, common.file_hashes, common.verify_file
runtime, git, committed_file = common.runtime, common.git, common.committed_file


class ArchiveCollection:
    """Re-readable on-disk series; never retain all trajectories in memory."""
    def __init__(self, directory, names):
        self.directory, self.names = Path(directory), list(names)
    def __len__(self):
        return len(self.names)
    def __iter__(self):
        for name in self.names:
            yield read(self.directory/name)


def variants(selected=None):
    fixed=['D','R',*B.K_OPTIONS,'N'] if selected is None else ['D','R',selected['selected_K'],'N']
    configs=B.P_CONFIGS if selected is None else [selected['selected_P']]
    return fixed+[p+'_'+c for c in configs for p in ('P','Px','Pxprime')]


def split_variant(variant):
    for prefix in ('Pxprime_','Px_','P_'):
        if variant.startswith(prefix):
            return prefix[:-1],variant[len(prefix):]
    return variant,'beta8_t80'


def source_metadata():
    for source in SOURCES:
        committed_file(ROOT/source)
    return dict(protocol_sha256=sha_file(ROOT/PROTOCOL),protocol_commit=committed_file(ROOT/PROTOCOL),
                source_sha256={s:sha_file(ROOT/s) for s in SOURCES},git_head=git('rev-parse','HEAD'))


def verify_runtime_manifest(manifest, override=False):
    current=source_metadata()
    for key in ('protocol_sha256','protocol_commit','source_sha256'):
        if manifest[key]!=current[key]:
            raise ValueError('Replay identity mismatch: '+key)
    if manifest['runtime']!=runtime() and not override:
        raise ValueError('Runtime differs; explicit --runtime-override required')


def make_archive(seed, selected=None, noise=False):
    environments={}
    for environment in lab.ENVIRONMENTS:
        series=lab.make_series(seed,environment,noise=noise)
        outputs={}
        for variant in variants(selected):
            policy,config=split_variant(variant)
            for budget in B.BUDGETS:
                outputs[f'{variant}|{budget}']=B.simulate_series(series,policy,budget,config,noise)
        environments[environment]=dict(series=series,outputs=outputs)
    intake=lab.physics.intake(seed)
    return dict(seed=seed,inputs=intake,intake_sha256=common.B.digest(intake),environments=environments)


def selected_actions(out):
    return [{k:v for k,v in a.items() if k in ('kind','id','candidate','cost','attempt','bad')}
            for a in out['protected']['actions']]


def _totals(rows):
    return {k:sum(r[k] for r in rows) for k in rows[0]} if rows else {}


def summarize(archives, confidences=None):
    cells, detail, calibration={}, {}, {}
    for archive in archives:
        for env,body in archive['environments'].items():
            episodes=body['series']['episodes']
            for condition, outputs in body['outputs'].items():
                variant,budget=condition.split('|')
                for branch in ('measurement','conflict'):
                    rows=[o['metrics'] for e,o in zip(episodes,outputs) if e['signal_branch']==branch]
                    key=f'{env}|{variant}|{budget}|{branch}'
                    cells.setdefault(key,[]).append(_totals(rows))
                groups={}
                for episode,out in zip(episodes,outputs):
                    ev=episode['evaluator']
                    key=f"{env}|{variant}|{budget}|{episode['signal_branch']}|{ev['kind']}|{ev['scope']}|{out['metrics']['signal_depth']}"
                    groups.setdefault(key,[]).append(out['metrics'])
                    c=calibration.setdefault(variant,dict(samples=[],abstentions=0,series=set()))
                    c['series'].add(archive['seed'])
                    c['abstentions']+=out['metrics']['abstained']
                    if out['metrics']['diagnosis_made']:
                        probability=out['terminal']['probability']
                        if probability is None:
                            probability=(confidences or {}).get(variant)
                        if probability is not None:
                            c['samples'].append((probability,out['metrics']['diagnosis_correct']))
                for key,rows in groups.items():
                    detail.setdefault(key,[]).append(_totals(rows))
    def cell(rows):
        return common._cell(rows)
    cal={}
    for name,row in calibration.items():
        samples=row['samples'];bins=[]
        for i in range(10):
            ss=[(p,y) for p,y in samples if min(9,int(p*10))==i]
            bins.append(dict(lower=i/10,upper=(i+1)/10,count=len(ss),
                             confidence=sum(p for p,y in ss)/len(ss) if ss else None,
                             accuracy=sum(y for p,y in ss)/len(ss) if ss else None))
        cal[name]=dict(decisions=len(samples),abstentions=row['abstentions'],independent_series=len(row['series']),
                       brier=math.fsum((p-y)**2 for p,y in samples)/len(samples) if samples else None,bins=bins)
    return dict(independent_series=len(archives),primary_cells={k:cell(v) for k,v in sorted(cells.items())},
                detail_cells={k:cell(v) for k,v in sorted(detail.items())},calibration=cal)


def aggregate_rank(summary, variant):
    return tuple(sum(row['total'][m] for key,row in summary['primary_cells'].items()
                     if key.split('|')[1]==variant) for m in B.PRIMARY)+(variant,)


def fixed_confidences(archives):
    counts={}
    for archive in archives:
        for body in archive['environments'].values():
            for condition,outputs in body['outputs'].items():
                variant=condition.split('|')[0]
                if variant not in ('D','R',*B.K_OPTIONS):
                    continue
                row=counts.setdefault(variant,[0,0])
                row[0]+=sum(o['metrics']['diagnosis_correct'] for o in outputs)
                row[1]+=sum(o['metrics']['diagnosis_made'] for o in outputs)
    return {k:v[0]/v[1] if v[1] else 0. for k,v in counts.items()}


def development_gate(archives):
    confidence=fixed_confidences(archives)
    summary=summarize(archives,confidence)
    selected_k=min(B.K_OPTIONS,key=lambda k:aggregate_rank(summary,k))
    fixed=min(('D','R',selected_k),key=lambda k:aggregate_rank(summary,k))
    exposure={f+'|'+s:dict(affected=0,inaccurate=0) for f in lab.FAULTS for s in ('shared','private')}
    depths,branches=set(),set()
    errors={};false_objections={'low':0,'high':0};false_d=0
    for archive in archives:
        for env,body in archive['environments'].items():
            for e in body['series']['episodes']:
                ev=e['evaluator'];case=e['case']
                if ev['kind']=='false_objection':false_objections[env.split('_')[0]]+=1
                if ev['kind']=='true_fault':
                    row=exposure[ev['fault']+'|'+ev['scope']]
                    row['affected']+=len(case['affected']);row['inaccurate']+=len(case['inaccurate'])
                if e['signal_ids']:
                    branches.add(e['signal_branch'])
                    rm={r['id']:r for r in case['records']}
                    depths.update(rm[c]['depth'] for c in e['signal_ids'])
            # Paired errors are counted once using fixed D trajectories, not as independent replicates.
            for budget in B.BUDGETS:
                for out in body['outputs'][f'D|{budget}']:
                    if env.endswith('moderate'):false_d+=out['metrics']['false_accusations']
                    for event in out['evaluation']['events']:
                        if event['error']:
                            direction='false_good' if event['truth_bad'] else 'false_bad'
                            key=env+'|'+event['kind']+'|'+direction
                            errors[key]=errors.get(key,0)+1
    exposure_ok=all(r['affected']>=20 and r['inaccurate']>=1 for r in exposure.values())
    errors_ok=all(any(v for k,v in errors.items() if 'moderate|'+kind+'|'+direction in k)
                  for kind in ('check_source','check_claim') for direction in ('false_good','false_bad'))
    common_ok=exposure_ok and errors_ok and all(false_objections.values()) and false_d>0 and depths>={1,2,3,4} and len(branches)==2
    configs={}
    for config in B.P_CONFIGS:
        variant='P_'+config
        contrasts={k:0 for k in ('D','R',selected_k,'Px_'+config)}
        repeat_differences={k:0 for k in ('D','R',selected_k)}
        placebo=0
        for archive in archives:
            for body in archive['environments'].values():
                for budget in B.BUDGETS:
                    p=body['outputs'][f'{variant}|{budget}']
                    for other in contrasts:
                        for a,b in zip(p,body['outputs'][f'{other}|{budget}']):
                            contrasts[other]+=selected_actions(a)!=selected_actions(b)
                            if other in repeat_differences:
                                def repetitions(o):
                                    return [x for x in selected_actions(o) if x.get('attempt',0)>0 or x['kind']=='repeat_observation']
                                repeat_differences[other]+=repetitions(a)!=repetitions(b)
                    for a,b in zip(p,body['outputs'][f'Pxprime_{config}|{budget}']):
                        placebo+=a['protected']!=b['protected'] or a['metrics']!=b['metrics']
        violations=[]
        for budget in B.BUDGETS:
            for branch in ('measurement','conflict'):
                prefix=f'low_zero|{{}}|{budget}|{branch}'
                p=summary['primary_cells'][prefix.format(variant)]['total']
                d=summary['primary_cells'][prefix.format('D')]['total']
                for metric in B.PRIMARY:
                    if p[metric]>d[metric]:violations.append(dict(budget=budget,branch=branch,metric=metric,P=p[metric],D=d[metric]))
        configs[config]=dict(action_contrasts=contrasts,repeat_contrasts=repeat_differences,placebo_mismatches=placebo,
                             low_zero_failures=violations,qualifies=bool(common_ok and all(contrasts.values()) and all(repeat_differences.values()) and placebo==0 and not violations))
    qualifying=[c for c,row in configs.items() if row['qualifies']]
    selected_p=min(qualifying,key=lambda c:aggregate_rank(summary,'P_'+c)) if qualifying else None
    return dict(pass_gate=selected_p is not None,selected_K=selected_k,selected_P=selected_p,
                best_constant=fixed,fixed_confidences=confidence,exposure=exposure,error_counts=errors,
                false_objections=false_objections,D_false_accusations_moderate=false_d,
                signal_depths=sorted(depths),signal_branches=sorted(branches),common_gate=common_ok,configurations=configs)


def closure(summary, selection):
    failures=[]
    constant=selection['best_constant'];p='P_'+selection['selected_P']
    for env in lab.ENVIRONMENTS:
        for budget in B.BUDGETS:
            for branch in ('measurement','conflict'):
                a=summary['primary_cells'][f'{env}|{constant}|{budget}|{branch}']['total']
                b=summary['primary_cells'][f'{env}|{p}|{budget}|{branch}']['total']
                for metric in B.PRIMARY:
                    if a[metric]>b[metric]:failures.append(dict(environment=env,budget=budget,branch=branch,metric=metric,constant=a[metric],P=b[metric]))
    return dict(best_constant=constant,constant_no_worse_everywhere=not failures,failures=failures)


def run(output, phase, replay=None, noise=False, override=False, freeze=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output directory must be empty')
    if phase not in ('development','confirmation'):raise ValueError('Unknown phase')
    metadata=source_metadata();selection=None
    if phase=='confirmation':
        if freeze is None:raise ValueError('Committed freeze required')
        freeze_commit=committed_file(Path(freeze).resolve());selection=read(freeze)
        if not selection['gate']['pass_gate']:raise ValueError('Development gate failed')
        verify_runtime_manifest(selection['development_manifest'],override)
        selection=selection['gate']
    seeds=DEVELOPMENT if phase=='development' else CONFIRMATION
    manifest=dict(schema=1,phase=phase,noise=bool(noise),runtime=runtime(),runtime_override=bool(override),
                  seeds=list(seeds),variants=variants(selection),environments=list(lab.ENVIRONMENTS),**metadata,files={},intakes={})
    if selection is not None:manifest.update(freeze_sha256=sha_file(freeze),freeze_commit=freeze_commit)
    previous=read(Path(replay)/'manifest.json') if replay else None
    if previous:
        verify_runtime_manifest(previous,override)
        for key in ('phase','seeds','variants','environments','freeze_sha256'):
            if previous.get(key)!=manifest.get(key):raise ValueError('Replay cohort mismatch '+key)
        manifest['replay_manifest_sha256']=sha_file(Path(replay)/'manifest.json')
    names=[];start=time.monotonic()
    for seed in seeds:
        name=f'series-{seed}.json.gz'
        if previous:
            verify_file(Path(replay)/name,previous['files'][name])
            saved=read(Path(replay)/name)
            if saved['seed']!=seed or saved['inputs']!=lab.physics.intake(seed):raise ValueError('Replay input mismatch')
        archive=make_archive(seed,selection,noise)
        if previous and archive['intake_sha256']!=previous['intakes'][str(seed)]:raise ValueError('Replay intake hash mismatch')
        write(out/name,archive);manifest['files'][name]=file_hashes(out/name)
        manifest['intakes'][str(seed)]=archive['intake_sha256'];names.append(name)
        write(out/'manifest.partial.json',manifest)
        print(common.json.dumps(dict(seed=seed,seconds=round(time.monotonic()-start,3),bytes=(out/name).stat().st_size)),flush=True)
    del archive
    archives=ArchiveCollection(out,names)
    confidence=selection['fixed_confidences'] if selection else fixed_confidences(archives)
    summary=summarize(archives,confidence)
    if phase=='development':write(out/'gate.json',development_gate(archives))
    else:summary['closure']=closure(summary,selection)
    write(out/'summary.json',summary);manifest['elapsed_seconds']=time.monotonic()-start
    write(out/'manifest.json',manifest);(out/'manifest.partial.json').unlink()
    return summary


def compare(first, second):
    first,second=Path(first),Path(second);a,b=read(first/'manifest.json'),read(second/'manifest.json')
    differences=[k for k in ('phase','seeds','variants','environments','source_sha256','protocol_sha256','protocol_commit','intakes','freeze_sha256','freeze_commit') if a.get(k)!=b.get(k)]
    diagnostics={};count=0
    if a['phase']=='development':
        if read(first/'gate.json')!=read(second/'gate.json'):differences.append('development_gate')
    if a['files'].keys()!=b['files'].keys():differences.append('file_set')
    for name in sorted(a['files'].keys() & b['files'].keys()):
        for base,m in ((first,a),(second,b)):verify_file(base/name,m['files'][name])
        x,y=read(first/name),read(second/name)
        if x['inputs']!=y['inputs']:differences.append(name+':inputs')
        for env in lab.ENVIRONMENTS:
            xe,ye=x['environments'][env],y['environments'][env]
            xs,ys=copy.deepcopy(xe['series']),copy.deepcopy(ye['series'])
            for series in (xs,ys):
                for episode in series['episodes']:
                    for row in episode['case']['records']:
                        row.pop('value')
            if xs!=ys:differences.append(name+':'+env+':series_inputs')
            try:common.numeric_differences(xe['series'],ye['series'],env+'/series',diagnostics)
            except ValueError as error:differences.append(name+':'+str(error))
            if xe['outputs'].keys()!=ye['outputs'].keys():differences.append(name+':conditions')
            for key in xe['outputs'].keys() & ye['outputs'].keys():
                if len(xe['outputs'][key])!=len(ye['outputs'][key]):differences.append(name+':'+env+':'+key+':length')
                for i,(p,q) in enumerate(zip(xe['outputs'][key],ye['outputs'][key])):
                    count+=1
                    if p['protected']!=q['protected'] or p['metrics']!=q['metrics'] or p['evaluation']!=q['evaluation']:
                        differences.append(f'{name}:{env}:{key}:{i}')
                    for field in ('trace','terminal','state_before','state_after','diagnostics'):
                        try:common.numeric_differences(p[field],q[field],env+'/'+field,diagnostics)
                        except ValueError as error:differences.append(name+':'+str(error))
    return dict(decisions_pass=not differences,differing_decisions=differences,compared_conditions=count,numerical_diagnostics=diagnostics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('development','confirmation'),default='development')
    parser.add_argument('--replay');parser.add_argument('--noise',action='store_true')
    parser.add_argument('--runtime-override',action='store_true');parser.add_argument('--freeze')
    args=parser.parse_args();run(args.output,args.phase,args.replay,args.noise,args.runtime_override,args.freeze)

if __name__=='__main__':main()
