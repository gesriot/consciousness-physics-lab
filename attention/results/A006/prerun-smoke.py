import copy,json
from attention import agent,s3,monitor,active_monitor
from attention.calibration_samples import request
from attention.io import read_json
from lab.executor import request_hash
library=read_json('attention/sample_library_a006.json');model=s3.load_model()
recipe=next(s for s in library['samples'] if s['family']=='collocation' and s['condition']=='clean' and s['expected']=='accept')
r=request(recipe)
conditions=['clean','slab','mixture','model_scale','near_tolerance']
def executors():return [agent.Executor(r,request_hash(r),c,1+0j) for c in conditions]
results={}
for policy in ('B','C2','BM'):
 old=monitor.run_batch(executors(),policy,100000.,model,monitor.Monitor(29.904027777777802,policy),0,'none',4)
 new=active_monitor.run_batch(executors(),policy,100000.,model,active_monitor.State(29.904027777777802,policy,library),0,'none',4,1.)
 sig=lambda x:[(a['input_index'],a['action'],a['cost'],a['observation']) for a in x['actions']]
 assert sig(old)==sig(new),(policy,'action mismatch')
 assert [x['verdict'] for x in old['inputs']]==[x['verdict'] for x in new['inputs']]
 results[policy]={'baseline_match':True,'actions':len(new['actions'])}
active={}
for policy in ('BA','C2A','Dprime'):
 st=active_monitor.State(29.904027777777802,policy,library)
 out=active_monitor.run_batch(executors(),policy,100000.,model,st,0,'none',4,1.)
 assert out['spent']<=100000
 assert not out['alerts']
 active[policy]=out
 results[policy]={'quota':out['sample_allocation']['quota'],'samples':out['sample_allocation']['executed'],'spent':out['spent'],'alerts':len(out['alerts'])}
assert active['BA']['sample_allocation']['executed']==active['C2A']['sample_allocation']['executed']
a=copy.deepcopy(active['BA']);b=copy.deepcopy(active['Dprime']);a.pop('policy');b.pop('policy');assert a==b
print(json.dumps(results,indent=2))
