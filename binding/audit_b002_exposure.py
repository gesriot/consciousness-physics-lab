"""Independent reconstruction of B002 development exposure from archived actions."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from binding.audit import AuditError, read, digest, equal, close
from attention.build_a007_library import CHANNELS, channel
from lab.guard import Request
from lab.pipeline_holdout import COHORTS, CONDITIONS

DURATIONS = (4, 8, 16)
FAMILIES = {'permissive': 'collocation', 'underestimated': 'v2'}
INSTRUMENTS = {'permissive': 'graded_v2', 'underestimated': 'collocation'}
ROOT = Path(__file__).resolve().parents[1]


def native_price(family, weights):
    if family == 'collocation':
        return sum((64,128,256,512,1024,2048,4096))*weights['fv_cell']
    if family == 'v2':
        return (64**3+128**3)*weights['collocation_degree_cubed']
    raise AuditError('Unexpected replacement family')


def action_price(action, weights, observation=None):
    if action.startswith('v2_'): return int(action[3:])*weights['fv_cell']
    if action == 'modal': return observation['outcome']['modes_spent']*weights['modal_term']
    if action == 'binding': return weights['binding']
    if action in ('normalization','first_moment'): return 4096*weights['modal_term']+512*weights['fv_cell']
    if action == 'second_moment': return 8192*weights['modal_term']+8192*weights['fv_cell']
    if action == 'high_frequency': return 4096*weights['modal_term']+4096*weights['fv_cell']
    if action == 'collocation': return (64**3+128**3)*weights['collocation_degree_cubed']
    raise AuditError('Unknown paid action')


def injected(original,tolerance,fault,spoiled):
    """Derive registered observation corruption without using injection code."""
    expected=copy.deepcopy(original)
    if not spoiled:return expected
    if fault=='permissive':expected['verdict']='accept'
    elif fault=='underestimated' and 'step' in expected:
        step=expected['step'];step['uncertainty_estimate']/=100.
        d,u=step['discrepancy'],step['uncertainty_estimate']
        verdict='inconclusive'
        if all(math.isfinite(x) for x in (d,u,tolerance)) and step['converging']:
            if d+u<=tolerance:verdict='accept'
            elif d-u>tolerance:verdict='violation_detected'
        step['verdict']=verdict
        numerator,denominator=step['last_difference'],step['previous_difference']
        ratio=numerator/denominator if denominator else 0. if not numerator else math.inf
        step['futile']=bool(step['last_grid'] in (1024,2048) and verdict=='inconclusive' and u>=4*tolerance and ratio>=.5 and step.get('antecedent_ratio',0.)>=.5)
        expected.update(verdict=verdict,terminal=bool(verdict!='inconclusive' or step['futile'] or step['last_grid']==4096))
    return expected


def cross_check(result, value, tolerance):
    """Derive guarded classifications from archived clean solver values."""
    values = [complex(*v) for v in result['values']]
    steps = result['steps']
    equal(bool(steps), True, 'replacement has native steps')
    graded = steps[0]['grid'] is not None
    expected_count = len(values)-3 if graded else 1
    equal(len(steps), expected_count, 'replacement step count')
    verdicts = []
    for j, step in enumerate(steps):
        vals = values[:j+4] if graded else values
        grid = (64,128,256,512,1024,2048,4096)[j+3] if graded else None
        equal(step['grid'], grid, 'native grid')
        difference = abs(vals[-1]-vals[-2])
        rounding = 32*math.ulp(1.)*(grid**2 if graded else 128**4)*max(1.,abs(vals[-1]))
        uncertainty = 2*difference+rounding
        discrepancy = abs(value-vals[-1])
        convergence = .5*abs(vals[-2]-vals[-3])-difference if graded else tolerance/8-difference
        raw = ('accept' if discrepancy+uncertainty <= tolerance else
               'violation_detected' if discrepancy-uncertainty > tolerance else 'inconclusive')
        margin = tolerance-discrepancy-uncertainty if raw == 'accept' else discrepancy-uncertainty-tolerance if raw == 'violation_detected' else None
        equal(step['finite'],True,'finite replacement step')
        equal(all(math.isfinite(x) for x in (rounding,uncertainty,discrepancy,convergence,difference,tolerance)) and (margin is None or math.isfinite(margin)),True,'finite derived replacement quantities')
        equal(step['unguarded_verdict'],raw,'raw replacement classification')
        if margin is None:equal(step['decision_margin'],None,'abstaining decision margin')
        else:close(step['decision_margin'],margin,'decision margin')
        con_pass = convergence > 4*rounding
        dec_pass = margin is not None and margin > 4*rounding
        equal(step['convergence_pass'], con_pass, 'protected convergence')
        equal(step['decision_pass'], dec_pass, 'protected decision')
        verdict = raw if con_pass and dec_pass else 'inconclusive'
        equal(step['verdict'], verdict, 'protected replacement verdict')
        for key, expected in [('native_rounding',rounding),('uncertainty',uncertainty),('discrepancy',discrepancy),('difference',difference),('convergence_margin',convergence)]:
            close(step[key], expected, 'replacement diagnostic '+key)
        verdicts.append(verdict)
    equal(result['verdict'], verdicts[-1], 'final replacement verdict')
    equal(all(v == 'inconclusive' for v in verdicts[:-1]), True, 'first decisive stop')
    return graded


def intake_check(data):
    equal(len(data['requests']),80,'80 master requests')
    equal(len(data['sample_seeds']),16,'16 master sample seeds')
    for i,row in enumerate(data['requests']):
        equal((row['task_id'],row['package_id'],row['cohort']),(i,i//5,COHORTS[i%5]),'master order')
        equal(sorted(row['condition_order']),sorted(CONDITIONS),'candidate order membership')
        h=hashlib.sha256(json.dumps(row['request'],sort_keys=True,allow_nan=False).encode()).hexdigest()
        equal(row['request_sha256'],h,'request identity')


def reference_check(reference):
    equal(reference['all_tasks_audited'],80,'reference count')
    equal(len(reference['records']),80,'reference rows')
    for i,row in enumerate(reference['records']):
        equal(row['task_id'],i,'reference order')
        precise=complex(*map(float,row['high_precision']))
        error=abs(complex(*row['analytic'])-precise)
        close(row['absolute_difference'],error,'reference difference')
        equal(row['passed'],error<=1e-13+1e-12*abs(precise),'reference threshold')
    equal(reference['fallback_all'],not all(r['passed'] for r in reference['records']),'reference fallback')


def reconstruct(packages,intake,reference,weights,fault):
    """Rebuild accepted identities at closure, never consulting saved ledgers."""
    samples={s['sample_id']:s for s in read(ROOT/'attention/sample_library_a007.json')['samples']}
    records,evaluator,notices=[],{},[]
    last=dict.fromkeys(CHANNELS,-1);offset=0;hashes=[];replacements=0
    for package_id,archive in enumerate(packages):
        equal(archive['package_id'],package_id,'package index')
        hashes.append(digest(archive))
        infos,trace,spec=archive['inputs'],archive['result'],archive['fault_spec']
        equal(len(infos),45,'candidate count');equal(len(trace['inputs']),45,'trace inputs')
        expected=[(row['task_id'],condition) for row in intake['requests'][package_id*5:package_id*5+5] for condition in row['condition_order']]
        equal([(i['task_id'],i['condition']) for i in infos],expected,'candidate identities')
        equal(spec['kind'],fault,'fault kind')
        equal(spec['onset_action'],10**9 if package_id==0 else 0,'registered onset')
        equal(spec['random_seed'],intake['sample_seeds'][package_id],'sample random seed')
        equal(trace['policy'],'BU','ordinary policy')
        equal(trace['sample_allocation']['requested'],10,'requested quota')
        equal(trace['sample_allocation']['executed'],trace['sample_allocation']['quota'],'paid sample quota')
        equal(trace['budget'],sum(i['full_cost'] for i in infos),'clean price budget')
        equal(trace['spent'],sum(a['cost'] for a in trace['actions']),'total charged cost')
        if trace['spent']>trace['budget']+1e-8:raise AuditError('package overspend')
        for key,predicate in [('sample',lambda a:a['sample']),('replacement',lambda a:a['replacement']),('ordinary',lambda a:not a['sample'])]:
            equal(trace[key+'_cost'],sum(a['cost'] for a in trace['actions'] if predicate(a)),key+' charged cost')
        for info in infos:
            r=reference['records'][info['task_id']]
            value=list(map(float,r['high_precision'])) if reference['fallback_all'] else r['analytic']
            equal(info['reference'],value,'reference binding')
            tolerance=intake['requests'][info['task_id']]['request']['tolerance']
            equal(info['tolerance'],tolerance,'input tolerance')
            error=abs(complex(*info['outcome']['value'])-complex(*value))
            close(info['absolute_error'],error,'physical error')
            equal(info['accurate'],error<=tolerance,'physical accuracy')
        closes={};support={i:{} for i in range(45)};suspect=set();ordinary=0
        for i,row in enumerate(trace['inputs']):
            equal(row['input_index'],i,'closure identity')
            if not 0<=row['closure_action_index']<=len(trace['actions']):raise AuditError('closure bounds')
            if row['verdict']=='accept':closes.setdefault(row['closure_action_index'],[]).append(i)
        for pos in range(len(trace['actions'])+1):
            for i in closes.get(pos,[]):
                rid=f'{package_id}:{i}';links=[];affected=[]
                for j,(ch,(when,replaced,spoiled)) in enumerate(support[i].items()):
                    name=('independent:' if replaced else '')+ch
                    links.append(dict(edge_id=f'{rid}:{j}',channel=name,region=ch.split(':')[-1],time=when,last_success=last.get(name,-1)))
                    if spoiled and not replaced:affected.append(ch)
                records.append(dict(id=rid,time=offset+2*pos,links=links,last_good_global=min(last.values()),last_success=last.copy()))
                evaluator[rid]=dict(accurate=infos[i]['accurate'],affected_channels=affected,package_id=package_id,input_index=i)
            if pos==len(trace['actions']):break
            action=trace['actions'][pos];ch=action['channel'];when=offset+2*pos+1
            if ch not in CHANNELS:raise AuditError('actual channel identity')
            active=fault!='none' and ordinary>=spec['onset_action']
            spoiled=bool(active and ch.split(':')[0]==FAMILIES.get(fault) and not action['replacement'])
            equal(action['fault_applied'],spoiled,'whole-family injection')
            equal(action['fault_active'],active,'fault onset state')
            if action['sample']:
                sample=samples[action['sample_id']]
                equal(ch,sample['channel'],'sample channel')
                equal(action['expected'],sample['expected'],'frozen sample verdict')
                equal(action['observation'],injected(action['original_observation'],sample['request']['tolerance'],fault,spoiled),'sample corruption semantics')
                for sample_step in action['sample_trace']:
                    equal(sample_step['observed'],injected(sample_step['original'],sample['request']['tolerance'],fault,spoiled),'sample step corruption semantics')
                equal(action['cost'],sum(a['cost'] for a in action['sample_trace']) if action['sample_trace'] else samples[action['sample_id']]['maximum_cost'],'sample trace price')
                for sample_step in action['sample_trace']:
                    equal(sample_step['cost'],action_price(sample_step['action'],weights,sample_step['original']),'native sample price')
                if action['observation']['verdict']==action['expected']:last[ch]=when
                else:
                    suspect.add(ch)
                    if ch.startswith(('v2:','collocation:')) and not any(n['channel']==ch for n in notices):
                        notices.append(dict(channel=ch,time=when,last_success=last[ch],last_good_global=min(last.values()),source='BU'))
                continue
            index=action['input_index'];info=infos[index]
            req=dict(intake['requests'][info['task_id']]['request']);req['z']=complex(*req['z']);req=Request(**req)
            if action['replacement']:
                equal(ch in suspect,True,'replacement after channel alarm')
                equal(action['observation'],action['original_observation'],'clean replacement original')
                equal(action['observation'],action['hooked_observation'],'clean replacement hook')
                equal(action['cost'],native_price(ch.split(':')[0],weights),'fixed native replacement price')
                graded=cross_check(action['observation'],complex(*info['outcome']['value']),req.tolerance)
                equal(graded,ch.startswith('collocation:'),'cross-instrument identity')
                equal(action['observation']['instrument'],'graded_v2' if graded else 'collocation','instrument label')
                equal(action['observation']['charged_work'],dict(grids=[64,128,256,512,1024,2048,4096]) if graded else dict(degrees=[64,128]),'fixed charged work')
                replacements+=1
            else:
                hooked=injected(action['original_observation'],req.tolerance,fault,spoiled)
                equal(action['hooked_observation'],hooked,'ordinary corruption semantics')
                observed=dict(verdict=hooked['verdict'])
                if action['action'].startswith('v2_'):observed['terminal']=hooked['terminal']
                equal(action['observation'],observed,'categorical observation')
                equal(action['cost'],action_price(action['action'],weights,action['original_observation']),'native ordinary price')
                equal(ch,channel(action['action'],req),'input channel identity')
                ordinary+=1
            if action['observation']['verdict']=='accept':
                deps=support[index]
                if action['replacement'] or not deps.get(ch,(None,False,None))[1]:deps[ch]=(when,action['replacement'],spoiled)
        equal(trace['suspect'],sorted(suspect),'suspect channels')
        offset+=2*(len(trace['actions'])+1)
    return dict(records=records,evaluator=evaluator,BU_notices=notices,last_success=last,offset=offset,package_hashes=hashes,replacements=replacements)


def exposure(evaluator):
    affected=[v for v in evaluator.values() if v['affected_channels']]
    return dict(accepted=len(evaluator),initial_false=sum(not v['accurate'] for v in evaluator.values()),affected=len(affected),affected_false=sum(not v['accurate'] for v in affected))


def source_check(manifest):
    head=manifest['git_head']
    def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT,stderr=subprocess.PIPE)
    for path,expected in manifest['source_sha256'].items():equal(hashlib.sha256(blob(head,path)).hexdigest(),expected,'source '+path)
    path='binding/protocols/B002-qualification.md'
    for commit in (head,manifest['protocol_commit']):equal(hashlib.sha256(blob(commit,path)).hexdigest(),manifest['protocol_sha256'],'protocol hash')
    subprocess.run(['git','merge-base','--is-ancestor',manifest['protocol_commit'],head],cwd=ROOT,check=True)
    sample_path='attention/sample_library_a007.json'
    equal(hashlib.sha256((ROOT/sample_path).read_bytes()).hexdigest(),manifest['source_sha256'][sample_path],'auditor sample dependency')
    return dict(source_files=len(manifest['source_sha256']),protocol_ancestry=True)


def freeze_check(rows,admitted,qualification_hash,saved):
    candidates={}
    for fault in admitted:
        scores=[]
        for duration in DURATIONS:
            counts={key:sum(row[str(duration)][fault][key] for row in rows) for key in ('accepted','initial_false','affected','affected_false')}
            scores.append(dict(packages=duration,**counts,exposure_gate=counts['affected']>=10 and counts['affected_false']>=5))
        passed=[s for s in scores if s['exposure_gate']]
        candidates[fault]=dict(instrument=INSTRUMENTS[fault],candidates=scores,selected_packages=passed[0]['packages'] if passed else None,exposure_gate=bool(passed))
    selected={f:c['selected_packages'] for f,c in candidates.items() if c['exposure_gate']}
    expected=dict(experiment='B002-exposure',development_seeds=list(range(20270101,20270113)),qualification_sha256=qualification_hash,candidates=candidates,selected=selected,confirmation_allowed=bool(selected),status='freeze_before_confirmation' if selected else 'stopped_no_passing_exposure')
    equal(saved,expected,'shortest registered exposure freeze')
    return expected


def audit(directory):
    directory=Path(directory);manifest=read(directory/'manifest.json')
    equal(manifest['experiment'],'B002-exposure','experiment')
    equal(manifest['durations'],list(DURATIONS),'durations')
    equal(manifest['independent_series'],12,'independent series')
    sources=source_check(manifest)
    qpath='binding/results/B002/qualification-freeze.json'
    qbytes=subprocess.check_output(['git','show',manifest['git_head']+':'+qpath],cwd=ROOT)
    equal(hashlib.sha256(qbytes).hexdigest(),manifest['qualification_sha256'],'committed qualification')
    qualification=json.loads(qbytes)
    equal(qualification['exposure_allowed'],True,'qualification admission')
    admitted=[f for f,t in INSTRUMENTS.items() if t in qualification['qualified']]
    equal(manifest['admitted'],admitted,'admitted families')
    faults=['none']+admitted
    weights=json.loads(subprocess.check_output(['git','show',manifest['git_head']+':attention/self_model_s3.json'],cwd=ROOT))['action_model']['cost_weights']
    datasets=[read(directory/f'series-{i:02d}'/'intake.json') for i in range(12)]
    equal(digest(datasets),manifest['intake_sha256'],'master intake hash')
    equal([d['seed'] for d in datasets],list(range(20270101,20270113)),'registered seeds')
    all_hashes=[];rows=[];replacements=0
    for i,data in enumerate(datasets):
        intake_check(data);all_hashes.extend(r['request_sha256'] for r in data['requests'])
        base=directory/f'series-{i:02d}';reference=read(base/'reference.json');reference_check(reference)
        row={str(d):{} for d in DURATIONS}
        for fault in faults:
            packages=[read(base/f'{fault}-{p}.json.gz') for p in range(16)]
            for duration in DURATIONS:
                rebuilt=reconstruct(packages[:duration],data,reference,weights,fault)
                ledger=read(base/f'ledger-{duration}.json.gz')
                for name in ('records','evaluator','BU_notices','last_success','package_hashes'):
                    equal(ledger[name][fault],rebuilt[name],'reconstructed '+name)
                equal(ledger['offsets'][fault],rebuilt['offset'],'global action clock')
                row[str(duration)][fault]=exposure(rebuilt['evaluator'])
                equal(read(base/f'exposure-{duration}.json')[fault],row[str(duration)][fault],'exposure counts')
                if duration==16:replacements+=rebuilt['replacements']
        rows.append(row)
    equal(len(set(all_hashes)),960,'unique master requests')
    frozen=freeze_check(rows,admitted,manifest['qualification_sha256'],read(directory/'freeze.json'))
    return dict(passed=True,series=12,unique_requests=960,packages=12*16*len(faults),snapshots=12*3*len(faults),clean_replacements=replacements,sources=sources,freeze=frozen,limitations=['Archived solver values and analytic/high-precision references are checked for consistency, not freshly solved.','Scheduler optimality is not inferred from trace validation.','Durations and fault families share series and are not independent trials.'])


def compare(left,right):
    """Compare paired operational traces exactly and summarize floating drift."""
    from binding.compare import upstream_decisions
    left,right=Path(left),Path(right)
    lm,rm=read(left/'manifest.json'),read(right/'manifest.json')
    result=dict(decisions_pass=True,intake_pass=True,flags_pass=True,differing_decisions=[],changed_numerics=0,max_absolute_difference=0.,series_compared=12)
    for key in ('protocol_sha256','source_sha256','qualification_sha256','intake_sha256','admitted','durations'):
        if lm[key]!=rm[key]:result['intake_pass']=False
    def drift(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            for key in a.keys()&b.keys():drift(a[key],b[key])
        elif isinstance(a,list) and isinstance(b,list):
            for x,y in zip(a,b):drift(x,y)
        elif isinstance(a,float) and isinstance(b,(int,float)) and not isinstance(b,bool) and a!=b:
            if not math.isfinite(a) or not math.isfinite(b):result['flags_pass']=False
            else:
                result['changed_numerics']+=1
                result['max_absolute_difference']=max(result['max_absolute_difference'],abs(a-b))
    def same(a,b,label):
        if a!=b:result['differing_decisions'].append(label)
    def guard_flags(trace):
        return [(a['input_index'],a['channel'],a['observation']['verdict'],[(s['grid'],s['verdict'],s['convergence_pass'],s['decision_pass'],s['finite']) for s in a['observation']['steps']]) for a in trace['actions'] if a['replacement']]
    for i in range(12):
        lb,rb=left/f'series-{i:02d}',right/f'series-{i:02d}'
        same(read(lb/'intake.json'),read(rb/'intake.json'),[i,'intake'])
        for fault in ['none']+lm['admitted']:
            for package in range(16):
                name=f'{fault}-{package}.json.gz';a,b=read(lb/name),read(rb/name)
                same(upstream_decisions(a['result']),upstream_decisions(b['result']),[i,name,'operational'])
                same(guard_flags(a['result']),guard_flags(b['result']),[i,name,'protected_flags'])
                same([(x['task_id'],x['condition'],x['accurate'],x['full_cost']) for x in a['inputs']],[(x['task_id'],x['condition'],x['accurate'],x['full_cost']) for x in b['inputs']],[i,name,'candidate_identity'])
                drift(a,b)
        for duration in DURATIONS:
            name=f'ledger-{duration}.json.gz';a,b=read(lb/name),read(rb/name)
            a.pop('package_hashes');b.pop('package_hashes')
            same(a,b,[i,name])
            name=f'exposure-{duration}.json';same(read(lb/name),read(rb/name),[i,name])
    same(read(left/'freeze.json'),read(right/'freeze.json'),['freeze'])
    result['decisions_pass']=not result['differing_decisions']
    result['flags_pass']=result['flags_pass'] and result['decisions_pass']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--output',required=True);parser.add_argument('--compare')
    args=parser.parse_args()
    try:result=compare(args.directory,args.compare) if args.compare else audit(args.directory)
    except (AuditError,KeyError,ValueError,FileNotFoundError) as error:result=dict(passed=False,error=str(error))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result.get('passed',all(result.get(k,False) for k in ('decisions_pass','intake_pass','flags_pass'))) else 1)
