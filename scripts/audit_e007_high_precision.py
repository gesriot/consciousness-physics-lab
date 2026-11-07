import json,math
from pathlib import Path
import mpmath as mp
r=json.loads(Path('results/verifier-stress/results.json').read_text())
mp.mp.dps=60
refs={}
for t in r['tasks']:
 q=t['request'];z=mp.mpc(*q['z']);x=mp.sqrt(15*z*mp.mpf(q['mean_time']))
 refs[t['task_id']]=complex(3*(mp.coth(x)/x-1/x**2))
misses=0;maxratio=0;worst=0
bykey={(s['task_id'],s['grid']):s for s in r['coverage_steps']}
for t in r['tasks']:
 vals=[complex(*v) for v in t['shadow_values']]
 for i,n in enumerate((64,128,256,512,1024,2048,4096)):
  if n<512:continue
  u=2*abs(vals[i]-vals[i-1])+32*2**-52*n*n*max(1,abs(vals[i]))
  e=abs(vals[i]-refs[t['task_id']])
  old=bykey[t['task_id'],n]
  assert math.isclose(u,old['uncertainty'],rel_tol=2e-15)
  assert (e<=u)==old['covered']
  misses+=e>u;maxratio=max(maxratio,e/u)
for row in r['runs']:
 e=abs(complex(*row['outcome']['value'])-refs[row['task_id']])
 tol=r['tasks'][row['task_id']]['request']['tolerance']
 assert (e<=tol)==row['accurate']
 worst=max(worst,abs(e-row['absolute_error']))
 if row['condition']=='D3':
  for s in row['independent']['steps']:
   v=('accept' if s['converging'] and s['discrepancy']+s['uncertainty_estimate']<=tol
      else 'violation_detected' if s['converging'] and s['discrepancy']-s['uncertainty_estimate']>tol
      else 'inconclusive')
   assert v==s['verdict']
print('Post-hoc independent audit, not an extra preregistered endpoint.')
print('mpmath',mp.__version__,'precision',mp.mp.dps,'decimal digits.')
print('Independent closed form 3*(coth(sqrt(15*z*T))/sqrt(15*z*T)-1/(15*z*T)).')
print('All 2304 accuracy labels and D3 step inequalities agree; all 768 coverage labels agree.')
print('Coverage misses:',misses,'max error/U:',maxratio)
print('Maximum absolute change of recorded artifact error:',worst)
print('PASS')
