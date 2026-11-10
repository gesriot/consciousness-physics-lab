"""Independent verification of conditional revision, controls and paired metrics."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from binding.audit import AuditError, read, digest, equal, close, guard_check
from binding.audit_b002_exposure import reconstruct, native_price, cross_check
from attention.build_a007_library import CHANNELS
from lab.pipeline_holdout import COHORTS, CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
MEMORIES = ('G','T','A','N','Gx','Gxprime')
BUDGETS = ('2','8','32')
FAMILIES = {'permissive':'collocation','underestimated':'v2'}
METRICS = ('residual_false_accepts','unnecessary_correct_recalls','recall_cost','attempted','withdrawn','confirmed','affected','affected_false','selected_affected','true_withdrawals','accepted','initial_false')


def projection(records, notice, memory, reviewed):
    rows = copy.deepcopy([r for r in records if r['time'] < notice['time'] and r['id'] not in reviewed])
    edits = []
    if memory in ('Gx','Gxprime'):
        for row in rows:
            matches = [i for i,e in enumerate(row['links']) if e['channel']==notice['channel'] and e['time']>notice['last_success']]
            if not matches: continue
            choices = matches if memory=='Gx' else [i for i,e in enumerate(row['links']) if e['channel'].removeprefix('independent:')=='normalization']
            if memory=='Gxprime': equal(len(choices),1,'matched normalization edge')
            edge = row['links'][choices[-1]]
            old = edge['channel']
            edge['channel'] = old[len('independent:'):] if old.startswith('independent:') else 'independent:'+old
            edits.append(dict(record_id=row['id'],edge_id=edge.get('edge_id'),before=old,after=edge['channel'],binary_distance=1))
    if memory in ('T','A','N'): rows=[dict(id=r['id'],time=r['time']) for r in rows]
    scores=[]
    for row in rows:
        relevant=[e['time'] for e in row.get('links',[]) if e['channel']==notice['channel'] and e['time']>notice['last_success']]
        eligible=(memory=='A' or (memory=='T' and row['time']>notice['last_success']) or (memory in ('G','Gx','Gxprime') and bool(relevant)))
        if eligible: scores.append((max(relevant) if memory in ('G','Gx','Gxprime') else row['time'],row['time'],row['id']))
    selected=[x[2] for x in sorted(scores,key=lambda x:(-round(float(x[0]),9),-x[1],x[2]))]
    return rows,edits,selected


def check_cell(saved, records, truth, notices, memory, units, kind, infos, weights):
    reviewed=set();withdrawn=set();considered=set();price=0.;confirmed=0;cursor=0;delays={}
    equal(len(saved['interventions']),len(notices),'intervention count')
    for notice, intervention in zip(notices,saved['interventions']):
        rows,edits,selected=projection(records,notice,memory,reviewed)
        expected=dict(notice=notice,records=rows,edits=edits,changed_edges=len(edits),binary_distance=len(edits),selected=selected)
        equal(intervention,expected,'visible graph and deterministic selection')
        considered.update(selected)
        for rid in selected[:max(0,units-len(reviewed))]:
            if cursor>=len(saved['attempts']):raise AuditError('missing attempt')
            attempt=saved['attempts'][cursor];cursor+=1
            equal((attempt['record_id'],attempt['channel'],attempt['notice_time']),(rid,notice['channel'],notice['time']),'attempt identity')
            delays[notice['time']]=delays.get(notice['time'],0)+1
            equal(attempt['service_delay'],delays[notice['time']],'service slot')
            info=infos[rid];family=notice['channel'].split(':')[0]
            if kind=='cross':
                graded=cross_check(attempt['result'],complex(*info['outcome']['value']),info['tolerance'])
                equal(graded,family=='collocation','correct independent tool')
                charge=native_price(family,weights)
            else:
                guard_check(attempt['result'],info)
                equal(attempt['result']['instrument'],'B001_uniform','counterfactual instrument')
                charge=(4096+8192+16384+32768)*weights['fv_cell']
            equal(attempt['cost'],charge,'exact per-notice charge');price+=charge
            revoke=attempt['result']['verdict']!='accept'
            equal(attempt['withdrawn'],revoke,'retain protected acceptance')
            reviewed.add(rid)
            if revoke:withdrawn.add(rid)
            else:confirmed+=1
    equal(cursor,len(saved['attempts']),'extra attempts')
    for key,value in [('reviewed',sorted(reviewed)),('withdrawn',sorted(withdrawn)),('considered',sorted(considered)),('budget_units',units),('unused_units',units-len(reviewed)),('cost',price)]:equal(saved[key],value,key)
    accepted={r['id'] for r in records}
    affected={r['id'] for r in records if any(r['time']<n['time'] and n['channel'] in truth[r['id']]['affected_channels'] for n in notices)}
    hit=reviewed&affected
    expected=dict(residual_false_accepts=sum(not truth[r]['accurate'] for r in accepted-withdrawn),unnecessary_correct_recalls=sum(truth[r]['accurate'] for r in withdrawn),recall_cost=price,attempted=len(reviewed),withdrawn=len(withdrawn),confirmed=confirmed,confirmation_fraction=confirmed/len(reviewed) if reviewed else None,affected=len(affected),affected_false=sum(not truth[r]['accurate'] for r in affected),selected_affected=len(hit),selection_precision=len(hit)/len(reviewed) if reviewed else None,selection_recall=len(hit)/len(affected) if affected else None,true_withdrawals=len(affected&withdrawn),accepted=len(accepted),initial_false=sum(not truth[r]['accurate'] for r in accepted))
    equal(saved['metrics'],expected,'independent primary and secondary metrics')
    return expected


def aggregate(rows):
    def stats(values):
        n=len(values);mean=sum(values)/n
        return dict(total=float(sum(values)),mean=float(mean),standard_error=math.sqrt(sum((x-mean)**2 for x in values)/(n-1)/n) if n>1 else None,by_series=values)
    summary={}
    for scenario in rows[0]:
        summary[scenario]={}
        for branch in ('external','BU'):
            summary[scenario][branch]={}
            for budget in BUDGETS:
                cells=[r[scenario][branch][budget] for r in rows];out={}
                for memory in MEMORIES:
                    entry={kind:{key:stats([c[memory][kind][key] for c in cells]) for key in METRICS} for kind in ('cross','uniform')}
                    entry['avoided_correct_losses']=stats([c[memory]['uniform']['unnecessary_correct_recalls']-c[memory]['cross']['unnecessary_correct_recalls'] for c in cells])
                    entry['paired_minus_G']={key:stats([c[memory]['cross'][key]-c['G']['cross'][key] for c in cells]) for key in METRICS[:3]}
                    for kind in ('cross','uniform'):
                        s=entry[kind]
                        for key,num,den in [('confirmation_fraction','confirmed','attempted'),('selection_precision','selected_affected','attempted'),('selection_recall','selected_affected','affected')]:s[key]=s[num]['total']/s[den]['total'] if s[den]['total'] else None
                    out[memory]=entry
                summary[scenario][branch][budget]=out
    return summary


def numeric_tree(actual,expected,path='summary'):
    if isinstance(expected,dict):
        equal(set(actual),set(expected),path+' keys')
        for k in expected:numeric_tree(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        equal(len(actual),len(expected),path+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):numeric_tree(a,b,path+str(i))
    elif isinstance(expected,float):close(actual,expected,path)
    else:equal(actual,expected,path)


def audit(directory,freeze_path):
    directory=Path(directory);freeze_path=Path(freeze_path)
    manifest=read(directory/'manifest.json');frozen=read(freeze_path);selected=frozen['selected']
    equal(bool(selected),True,'nonempty admitted comparison');equal(frozen['confirmation_allowed'],True,'exposure admission')
    equal(manifest['experiment'],'B002-confirmation','experiment');equal(manifest['independent_series'],12,'series count')
    equal(manifest['selected'],selected,'selected durations')
    head=manifest['git_head']
    def blob(path):return subprocess.check_output(['git','show',head+':'+path],cwd=ROOT)
    for path,sha in manifest['source_sha256'].items():equal(hashlib.sha256(blob(path)).hexdigest(),sha,'source '+path)
    protocol='binding/protocols/B002-recall.md'
    equal(hashlib.sha256(blob(protocol)).hexdigest(),manifest['protocol_sha256'],'current protocol at run')
    latest=subprocess.check_output(['git','log','-1','--format=%H',head,'--',protocol],cwd=ROOT,text=True).strip()
    for ancestor in (latest,manifest['protocol_commit']):subprocess.run(['git','merge-base','--is-ancestor',ancestor,head],cwd=ROOT,check=True)
    equal(hashlib.sha256(freeze_path.read_bytes()).hexdigest(),manifest['exposure_freeze_sha256'],'exposure freeze hash')
    equal(blob(str(freeze_path.resolve().relative_to(ROOT))),freeze_path.read_bytes(),'freeze committed before run')
    qpath='binding/results/B002/qualification-freeze.json';qbytes=blob(qpath);qualified=json.loads(qbytes)
    equal(hashlib.sha256(qbytes).hexdigest(),manifest['qualification_sha256'],'qualification hash')
    equal(frozen['qualification_sha256'],manifest['qualification_sha256'],'qualification binding')
    for fault,duration in selected.items():
        equal(duration,frozen['candidates'][fault]['selected_packages'],'selected shortest duration')
        equal(frozen['candidates'][fault]['exposure_gate'],True,'exposure gate')
        equal(frozen['candidates'][fault]['instrument'] in qualified['qualified'],True,'qualified tool')
    weights=json.loads(blob('attention/self_model_s3.json'))['action_model']['cost_weights']
    equal(hashlib.sha256((ROOT/'attention/sample_library_a007.json').read_bytes()).hexdigest(),manifest['source_sha256']['attention/sample_library_a007.json'],'auditor sample dependency')
    datasets=[read(directory/f'series-{i:02d}'/'ordinary'/'intake.json') for i in range(12)]
    development=[read(freeze_path.parent/f'series-{i:02d}'/'intake.json') for i in range(12)]
    qualification=[read(ROOT/'binding/results/B001/development-complete'/f'series-{i:02d}'/'intake.json') for i in range(12)]
    for key,rows in [('intake_sha256',datasets),('development_intake_sha256',development),('qualification_intake_sha256',qualification)]:equal(digest(rows),manifest[key],key)
    old={r['request_sha256'] for d in development+qualification for r in d['requests']};new=[r['request_sha256'] for d in datasets for r in d['requests']]
    equal(len(new),len(set(new)),'unique confirmation requests');equal(bool(old&set(new)),False,'development disjointness')
    equal(manifest['requests'],len(new),'request count');equal(manifest['exact_development_overlap'],0,'zero overlap')
    rows=[];total_cells=0;replacement_count=0
    for index,data in enumerate(datasets):
        n=max(selected.values());base=directory/f'series-{index:02d}'/'ordinary'
        equal(data['seed'],20270113+index,'confirmation seed');equal(data['packages'],n,'duration')
        equal(len(data['requests']),n*5,'request count');equal(len(data['sample_seeds']),n,'sample seeds')
        for j,r in enumerate(data['requests']):
            equal((r['task_id'],r['package_id'],r['cohort']),(j,j//5,COHORTS[j%5]),'request order')
            equal(sorted(r['condition_order']),sorted(CONDITIONS),'candidate order')
            equal(hashlib.sha256(json.dumps(r['request'],sort_keys=True,allow_nan=False).encode()).hexdigest(),r['request_sha256'],'request hash')
        reference=read(base/'reference.json');equal(reference['all_tasks_audited'],5*n,'reference count')
        equal(len(reference['records']),5*n,'reference rows')
        for j,r in enumerate(reference['records']):
            equal(r['task_id'],j,'reference identity');precise=complex(*map(float,r['high_precision']))
            error=abs(complex(*r['analytic'])-precise);close(r['absolute_difference'],error,'reference discrepancy')
            equal(r['passed'],error<=1e-13+1e-12*abs(precise),'reference threshold')
        equal(reference['fallback_all'],not all(r['passed'] for r in reference['records']),'reference fallback')
        revisions=read(base.parent/'revision.json.gz');expected_scenarios={s for f in selected for s in (f,'none:'+f,'false_notice:'+f)}
        equal(set(revisions),expected_scenarios,'scenario membership');row={}
        candidates={f:[read(base/f'{f}-{p}.json.gz') for p in range(n)] for f in ['none']+list(selected)}
        infos={f'{p}:{j}':info for p,archive in enumerate(candidates['none']) for j,info in enumerate(archive['inputs'])}
        for fault,duration in selected.items():
            rebuilt={source:reconstruct(candidates[source][:duration],data,reference,weights,source) for source in (fault,'none')}
            ledger=read(base/f'ledger-{duration}.json.gz')
            for source,v in rebuilt.items():
                for name in ('records','evaluator','BU_notices','last_success','package_hashes'):equal(ledger[name][source],v[name],'raw reconstructed '+name)
                equal(ledger['offsets'][source],v['offset'],'clock')
            for scenario,source in ((fault,fault),('none:'+fault,'none'),('false_notice:'+fault,'none')):
                value=revisions[scenario];state=rebuilt[source];row[scenario]={}
                for key,expected in [('fault',fault),('duration',duration),('source',source)]:equal(value[key],expected,'scenario '+key)
                external=[] if scenario.startswith('none:') else [dict(channel=c,time=state['offset'],last_success=state['last_success'][c],last_good_global=min(state['last_success'].values()),source='external') for c in CHANNELS if c.startswith(FAMILIES[fault]+':')]
                equal(set(value['branches']),{'external','BU'},'branch membership')
                for branch,notices in [('external',external),('BU',state['BU_notices'])]:
                    content=value['branches'][branch];equal(content['notices'],notices,'notice chronology');equal(set(content['cells']),set(BUDGETS),'budget membership');row[scenario][branch]={}
                    for budget in BUDGETS:
                        cells=content['cells'][budget];equal(set(cells),set(MEMORIES),'memory membership');row[scenario][branch][budget]={}
                        for memory,paired in cells.items():
                            results={kind:check_cell(paired[kind],state['records'],state['evaluator'],notices,memory,int(budget),kind,infos,weights) for kind in ('cross','uniform')}
                            equal([(a['record_id'],a['channel']) for a in paired['cross']['attempts']],[(a['record_id'],a['channel']) for a in paired['uniform']['attempts']],'matched counterfactual attempts')
                            equal(paired['avoided_correct_losses'],results['uniform']['unnecessary_correct_recalls']-results['cross']['unnecessary_correct_recalls'],'avoided losses')
                            row[scenario][branch][budget][memory]=results;total_cells+=1;replacement_count+=len(paired['cross']['attempts'])
        rows.append(row)
    summary=aggregate(rows);numeric_tree(read(directory/'summary.json'),summary)
    cells=[dict(fault=f,budget=int(b),T_no_worse=all(summary[f]['external'][b]['T']['cross'][k]['total']<=summary[f]['external'][b]['G']['cross'][k]['total'] for k in METRICS[:2])) for f in selected for b in BUDGETS]
    closure=dict(cells=cells,links_unnecessary_here=bool(cells) and all(c['T_no_worse'] for c in cells),tested=bool(cells))
    equal(read(directory/'closure.json'),closure,'nonvacuous registered closure')
    return dict(passed=True,series=12,requests=len(new),cells=total_cells,cross_attempts=replacement_count,closure=closure,protocol_last_revision=latest,limitations=['Archived solver values are recomputed arithmetically, not freshly solved.','Repeated cells, faults, controls and counterfactuals remain dependent within series.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--freeze',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    try:result=audit(args.directory,args.freeze)
    except (AuditError,KeyError,ValueError,FileNotFoundError) as error:result=dict(passed=False,error=str(error))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
