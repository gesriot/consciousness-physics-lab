"""Independently recompute A002 archived counts, costs and calibration."""
import argparse
import json, math, sys
from attention.io import read_json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
PATH=args.archive
OUT=args.output
if OUT.exists():raise ValueError('Refuse to overwrite an audit')
if not PATH.exists():
 raise FileNotFoundError(PATH)
d=read_json(PATH); failures=[]; lines=[]
def check(ok, message):
 if not ok: failures.append(message)
def same(a,b): return abs(a-b)<=1e-12
V=['accept','violation_detected','inconclusive']; required=['modal','binding','normalization','first_moment','second_moment','high_frequency','collocation']
policies=['B','C1','C2','C3','C4','D','Dprime']; fracs=['0.2','0.4','0.6','1.0']
check(len(d['packages'])==12,'Package count')
check(len(d['intake_channel'])==60,'Request count')
check(not d['full_pipeline_parity_failures'],'Full pipeline parity failures')
check(d['reference_audit']['all_tasks_audited']==60,'Reference audit incomplete')
cal={'S2':[],'base':[]}; totals={f:{p:0 for p in policies} for f in fracs}; false={f:{p:0 for p in policies} for f in fracs}; paired={f:{p:[] for p in policies if p!='B'} for f in fracs}; allocation={f:{p:{g:0. for g in ['initially_inconclusive','initially_accepting']} for p in ['B','D','Dprime']} for f in fracs}
for pack in d['packages']:
 pid=pack['package_id']; infos=pack['inputs'];check(len(infos)==45,f'Package {pid} size')
 fullcost=sum(i['full']['spent'] for i in infos);check(fullcost==pack['reference_cost'],f'Package {pid} reference cost')
 truth=[]
 for i in infos:
  accurate=abs(complex(*i['outcome']['value'])-complex(*i['reference']))<=i['tolerance'];truth.append(accurate)
  check(accurate==i['accurate'],f'Package {pid} stored accuracy')
 for name,key in [('S2','full_probabilities'),('base','base_probabilities')]:
  val=sum(sum((i['prediction'][key][v]-int(i['full']['verdict']==v))**2 for v in V) for i in infos)/45
  cal[name].append(val);check(same(val,pack['calibration'][name]),f'Package {pid} {name} Brier')
 for f in fracs:
  counts={}
  for p in policies:
   run=pack['policies'][f][p];correct=fa=accepts=0;budget=float(f)*fullcost
   check(run['budget']==budget and run['spent']<=budget,f'Package {pid}/{f}/{p} budget')
   check(sum(a['cost'] for a in run['actions'])==run['spent'],f'Package {pid}/{f}/{p} global accounting')
   check(sum(x['spent'] for x in run['inputs'])==run['spent'],f'Package {pid}/{f}/{p} input accounting')
   for j,(x,info,accurate) in enumerate(zip(run['inputs'],infos,truth)):
    obs=x['observed']
    check({a['action']:a['observation'] for a in x['actions']}==obs,f'Package {pid}/{f}/{p}/{j} action evidence consistency')
    if 'modal' in obs:check(obs['modal']['outcome']['value']==info['outcome']['value'],f'Package {pid}/{f}/{p}/{j} modal value reproducibility')
    viol=any(v['verdict']=='violation_detected' for v in obs.values())
    supported=all(a in obs and obs[a]['verdict']=='accept' for a in required) and any(a.startswith('v2_') and o.get('terminal') and o['verdict']=='accept' for a,o in obs.items())
    verdict='violation_detected' if viol else 'accept' if supported else 'inconclusive'
    check(verdict==x['verdict'],f'Package {pid}/{f}/{p}/{j} verdict support')
    c=(accurate and verdict=='accept') or (not accurate and verdict=='violation_detected');bad=not accurate and verdict=='accept'
    correct+=c;fa+=bad;accepts+=verdict=='accept'
    check(x['correct']==c and x['false_accept']==bad,f'Package {pid}/{f}/{p}/{j} correctness')
    check(sum(a['cost'] for a in x['actions'])==x['spent'],f'Package {pid}/{f}/{p}/{j} cost')
    if verdict=='accept':check(info['full']['verdict']=='accept',f'Package {pid}/{f}/{p}/{j} full support')
    if p in allocation[f]:
     probs=info['prediction']['full_probabilities'];group='initially_inconclusive' if probs['accept']<probs['inconclusive'] else 'initially_accepting';allocation[f][p][group]+=x['spent']
   check((correct,fa,accepts)==(run['correct'],run['false_accepts'],run['accepts']),f'Package {pid}/{f}/{p} aggregate counts')
   counts[p]=correct;totals[f][p]+=correct;false[f][p]+=fa
  check(pack['policies'][f]['B']['actions']==pack['policies'][f]['Dprime']['actions'],f'Package {pid}/{f} Dprime action invariance')
  b=pack['policies'][f]['B'].copy();dp=pack['policies'][f]['Dprime'].copy();b.pop('policy');dp.pop('policy');check(b==dp,f'Package {pid}/{f} Dprime full invariance')
  for p in paired[f]:paired[f][p].append(counts['B']-counts[p])
for f in fracs:
 summary=d['summary'][f]
 for p in policies:
  check(totals[f][p]==summary['policies'][p]['correct'] and false[f][p]==summary['policies'][p]['false_accepts'],f'{f}/{p} totals')
 for p in paired[f]:check(paired[f][p]==summary['paired_B_minus'][p],f'{f}/{p} paired differences')
 for p in allocation[f]:
  for g,n in allocation[f][p].items():check(same(n,summary['interventions'][p]['allocation_by_initial_prediction'][g]),f'{f}/{p}/{g} allocation')
 lines.append(f'Budget {f}: correct counts {totals[f]}; B minus C1-C4 {[sum(paired[f][p]) for p in policies[1:5]]}; B minus D {sum(paired[f]["D"])}; false accepts {sum(false[f].values())}.')
 lines.append(f'Allocation {f}: {allocation[f]}')
for name in cal:check(same(sum(cal[name])/12,d['calibration'][name]),f'{name} overall Brier')
m=json.loads((ROOT/'attention/self_model_s2.json').read_text());changes=[abs(r['counts'][0]-r['counts'][2]) for r in m['records'] if r['counts'][0]!=r['counts'][2]];inter=d['intervention_audit']
check(inter['D_changed_numbers']==inter['Dprime_changed_numbers']==2*len(changes),'Corruption changed count');check(inter['D_L1']==inter['Dprime_L1']==2*sum(changes),'Corruption L1');check(inter['equal_changes'],'Corruption equality')
check(len(inter['initial_priority_checks'])==60 and all(x['direction_pass'] and same(x['expected_score_change'],x['actual_score_change']) for x in inter['initial_priority_checks']),'Initial priority directions')
def key(r):
 q=r['request'];return (tuple(q['z']),q['mean_time'],q['tolerance'])
keys=[key(r) for r in d['intake_channel']];check(len(keys)==len(set(keys)),'Duplicate A002 requests')
for label,path in [('A001','attention/results/A001/results.json'),('E016','results/E016-frozen-pipeline/results.json')]:
 old={key(r) for r in json.loads((ROOT/path).read_text())['intake_channel']};n=sum(k in old for k in keys);check(n==0 and n==d['request_overlap']['comparisons'][label]['exact_request_matches'],f'{label} overlap')
for label,row in d['request_overlap']['comparisons'].items():
 if row.get('status')=='not_task_intake': continue
 check(row['exact_request_matches']==0,f'Archived overlap {label}')
 if 'source' in row:
  old_data=json.loads((ROOT/row['source']).read_text());old_keys=set()
  def collect(obj):
   if isinstance(obj,dict):
    if all(k in obj for k in ['z','mean_time','tolerance']):
     try: old_keys.add((tuple(obj['z']),obj['mean_time'],obj['tolerance']))
     except TypeError: pass
    for value in obj.values():collect(value)
   elif isinstance(obj,list):
    for value in obj:collect(value)
  collect(old_data)
  check(bool(old_keys) and not any(k in old_keys for k in keys),f'Independently read overlap {label}')
lines.insert(0,'A002 independent result review: '+('PASS' if not failures else 'FAIL'))
lines += [f'Brier S2={sum(cal["S2"])/12}; base={sum(cal["base"])/12}.',f'Corruption: {2*len(changes)} changed numbers per arm; L1={2*sum(changes)}; initial direction checks pass.', 'All 12 package pairs independently recomputed from input verdicts. No task regeneration or implementation changes. Full-P parity and high-precision reference status were read from archived audit records, not rerun here.', 'Failures: '+repr(failures)]
OUT.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if failures:sys.exit(1)
