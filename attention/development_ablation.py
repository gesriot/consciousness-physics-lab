"""Frozen post-result factorial trace replay using A001-only learning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from attention import agent, s2
from attention.batch import remaining_actions
from attention.development_a002 import read, FRACTIONS


def run(data, training, model):
    train_records = {r['task_id']:r for r in training['intake_channel']}
    vectors = {i:s2.features(r['request'],model)[0] for i,r in train_records.items()}
    traces = {i:[] for i in train_records}
    for row in training['runs']:
        actions = {a['action']:a['observation']['verdict'] for a in row['full']['actions'] if a['action']!='declare'}
        traces[row['task_id']].append((row['accurate'],row['full']['verdict'],actions))
    query = {r['task_id']:r['request'] for r in data['intake_channel']}
    priors = {}
    for task_id,r in query.items():
        vector = s2.features(r,model)[0]
        nearest = sorted(vectors,key=lambda i:(sum((a-b)**2 for a,b in zip(vector,vectors[i])),i))[:16]
        paths = [t for i in nearest for t in traces[i]]
        priors[task_id] = dict(neighbors=nearest,
            solved=np.asarray([float((accurate and final=='accept') or (not accurate and final=='violation_detected')) for accurate,final,_ in paths]),
            codes={a:np.asarray([t.get(a,'missing') for _,_,t in paths]) for a in agent.ORDER},
            utility={a:np.asarray([float(not accurate and t.get(a)=='violation_detected')+.05*float(t.get(a)=='accept') for accurate,_,t in paths]) for a in agent.ORDER})
    output = []
    for batch in data['packages']:
        prices = [{a['action']:a['cost'] for a in i['full']['actions'] if a['action']!='declare'} for i in batch['inputs']]
        hidden = [{a['action']:a['observation'] for a in i['full']['actions'] if a['action']!='declare'} for i in batch['inputs']]
        # Costs of inaccessible later grids are still known from the action model.
        from lab.guard import Request
        all_prices=[]
        for info in batch['inputs']:
            fields=dict(query[info['task_id']]);fields['z']=complex(*fields['z']);r=Request(**fields)
            all_prices.append({a:agent.cost(a,r,model['action_model']) for a in agent.ORDER})
        runs={}
        for f in FRACTIONS:
            runs[f]={}
            for update in (False,True):
                for global_priority in (False,True):
                    name=f"update_{int(update)}_global_{int(global_priority)}"
                    observed=[{} for _ in hidden];spent=0.; sequence=[]
                    weights=[np.ones(len(priors[i['task_id']]['solved']))/len(priors[i['task_id']]['solved']) for i in batch['inputs']]
                    scores=[{} for _ in hidden]
                    def refresh(i):
                        prior=priors[batch['inputs'][i]['task_id']];w=weights[i]
                        scores[i]=dict(solvable=float(w@prior['solved']),actions={a:float(w@u)/all_prices[i][a] for a,u in prior['utility'].items()})
                    for i in range(len(hidden)):refresh(i)
                    budget=float(f)*batch['reference_cost']
                    while True:
                        candidates={i:[a for a in agent.available(obs) if all_prices[i][a]<=budget-spent] for i,obs in enumerate(observed) if agent.verdict(obs)=='inconclusive'}
                        candidates={i:a for i,a in candidates.items() if a}
                        if not candidates:break
                        if global_priority:
                            i,a=min(((i,a) for i,opts in candidates.items() for a in opts),key=lambda x:(-scores[x[0]]['actions'][x[1]],x[0],agent.ORDER.index(x[1])))
                        else:
                            i=min(candidates,key=lambda i:(-scores[i]['solvable']/sum(all_prices[i][a] for a in remaining_actions(observed[i])),i))
                            a=min(candidates[i],key=lambda a:(-scores[i]['actions'][a],agent.ORDER.index(a)))
                        obs=hidden[i][a];observed[i][a]=obs;spent+=all_prices[i][a]
                        sequence.append([i,a])
                        if update:
                            prior=priors[batch['inputs'][i]['task_id']]
                            weights[i]*=np.where(prior['codes'][a]==obs['verdict'],.95,.025)
                            weights[i]/=sum(weights[i]);refresh(i)
                    verdicts=[agent.verdict(obs) for obs in observed]
                    correct=[(info['accurate'] and v=='accept') or (not info['accurate'] and v=='violation_detected') for info,v in zip(batch['inputs'],verdicts)]
                    assert not any(not info['accurate'] and v=='accept' for info,v in zip(batch['inputs'],verdicts))
                    runs[f][name]=dict(correct=sum(correct),accepts=verdicts.count('accept'),spent=spent,verdicts=verdicts,actions=sequence)
        output.append(dict(package_id=batch['package_id'],runs=runs))
        print('Development package',batch['package_id']+1,flush=True)
    summary={}
    for f in FRACTIONS:
        totals={name:sum(b['runs'][f][name]['correct'] for b in output) for name in output[0]['runs'][f]}
        u0,u1=totals['update_1_global_0']-totals['update_0_global_0'],totals['update_1_global_1']-totals['update_0_global_1']
        g0,g1=totals['update_0_global_1']-totals['update_0_global_0'],totals['update_1_global_1']-totals['update_1_global_0']
        summary[f]=dict(correct=totals,update_effect_under_task_priority=u0,update_effect_under_global_priority=u1,global_effect_without_update=g0,global_effect_with_update=g1,interaction=u1-u0,
            correct_by_package={name:[b['runs'][f][name]['correct'] for b in output] for name in totals})
    return dict(status='post-result development ablation; not original B; not independent validation',summary=summary,packages=output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.output.exists():raise ValueError('Output already exists')
    sources=['attention/results/A002/results.json.gz','attention/results/A001/results.json','attention/self_model_s2.json','attention/results/A002/development-ablation-plan.md','attention/development_ablation.py']
    result=run(read(sources[0]),read(sources[1]),read(sources[2]))
    result['source_sha256']={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
