"""Pre-outcome A004 control validation on the first archived A003 package."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from attention import a004, agent, factorial, s3
from attention.io import read_json, write_json
from lab.guard import Request


class ArchivedExecutor:
    def __init__(self, request, full):
        self.request=request
        self.trace={a['action']:copy.deepcopy(a['observation']) for a in full['actions'] if a['action']!='declare'}
        self.observed={}

    def perform(self,action):
        if action not in agent.available(self.observed):raise ValueError('Unavailable archived action')
        if agent.verdict(self.observed)!='inconclusive':raise ValueError('Already resolved')
        result=copy.deepcopy(self.trace[action]);self.observed[action]=result
        return result


def run(path):
    data=read_json(path);model=s3.load_model();package=data['packages'][0]
    requests={}
    for record in data['intake_channel']:
        fields=dict(record['request']);fields['z']=complex(*fields['z']);requests[record['task_id']]=Request(**fields)
    # Validate the exact initial-forecast route without generating new intake.
    initial=[a004.initial(requests[r['task_id']],model) for r in package['inputs']]
    outcomes={};differences={};failures=[]
    policies=('U0O','U1O','U0T','U1T','D','Dprime','H','HD','HDprime')
    for fraction in a004.FRACTIONS:
        f=str(fraction);runs={}
        for policy in policies:
            executors=[ArchivedExecutor(requests[r['task_id']],r['full']) for r in package['inputs']]
            result=factorial.run_batch(executors,policy,fraction*package['reference_cost'],model)
            runs[policy]=result
            print('Development',f,policy,'actions',len(result['actions']),flush=True)
        outcomes[f]={p:dict(action_count=len(r['actions']),spent=r['spent'],support=r['support_audit'],
            correct=sum((i['accurate'] and o['verdict']=='accept') or (not i['accurate'] and o['verdict']=='violation_detected') for i,o in zip(package['inputs'],r['inputs']))) for p,r in runs.items()}
        differences[f]={}
        for left,right in (('U1T','D'),('H','HD'),('U1T','Dprime'),('H','HDprime'),('U0O','U1O'),('U0T','U1T')):
            a=[(r['input_index'],r['action']) for r in runs[left]['actions']];b=[(r['input_index'],r['action']) for r in runs[right]['actions']]
            differing=next((i for i in range(max(len(a),len(b))) if (a[i] if i<len(a) else None)!=(b[i] if i<len(b) else None)),None)
            differences[f][left+'_versus_'+right]=dict(changed=a!=b,first_different_index=differing,
                left_action=a[differing] if differing is not None and differing<len(a) else None,
                right_action=b[differing] if differing is not None and differing<len(b) else None)
            if right.endswith('prime') and a!=b:failures.append(f+' irrelevant control changed actions')
        for p in ('D','HD'):
            if runs[p]['support_audit']['prior_resets'] or runs[p]['support_audit']['minimum_support']==0:
                failures.append(f+' '+p+' empty support or fallback')
    if not any(d['U1T_versus_D']['changed'] for d in differences.values()):failures.append('D never changes tree actions')
    if not any(d['H_versus_HD']['changed'] for d in differences.values()):failures.append('D never changes hybrid actions')
    return dict(passed=not failures,failures=failures,source=str(path),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        package_id=0,preregistered_parameters=dict(epsilon=factorial.EPSILON,alpha=factorial.CORRUPTION,threshold=factorial.HYBRID_THRESHOLD),
        alphabets={a:factorial.alphabet_size(a) for a in agent.ORDER},
        ties='Global score descending, input index, action index, ATOMIC/TREE/FULL; C2/H price, input index, action index.',
        intervention_audit=factorial.corruption_audit(model),initial_prediction_count=len(initial),outcomes=outcomes,action_differences=differences,
        prediction='D should change normalization/high_frequency priorities and their continuations. HD should cross the threshold for normalization; Dprime and HDprime must leave actions identical. No retuning follows these checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=Path('attention/results/A003/results.json.gz'));parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refuse overwrite')
    result=run(args.source);write_json(args.output,result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
