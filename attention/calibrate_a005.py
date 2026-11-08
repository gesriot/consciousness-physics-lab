"""Freeze an empirical clean-stream threshold from A004 only."""
import argparse
import hashlib
from attention.io import read_json, write_json, source_bytes
from attention import agent, s3, monitor_planning as planning
from attention.monitor import Monitor
from lab.guard import Request


def calibrate(path='attention/results/A004/results.json.gz'):
    data=read_json(path);model=s3.load_model();maximum=0.;details={}
    for f in ('0.2','0.4','0.6','1.0'):
        monitor=Monitor(float('inf'),'BM')
        for package in data['packages']:
            posts=[];requests=[]
            for item in package['inputs']:
                record=data['intake_channel'][item['task_id']]
                fields=dict(record['request']);fields['z']=complex(*fields['z'])
                request=Request(**fields);requests.append(request)
                posts.append(planning.Posterior(request,planning.Runtime(model),False))
            for row in package['policies'][f]['U0T']['actions']:
                i,a=row['input_index'],row['action'];p=posts[i].forecast(a)['probabilities'][0]
                result=monitor.observe(a,agent.band(requests[i]),p,row['observation'],{})
                maximum=max(maximum,result['statistic'])
        details[f]=dict(maximum=max((r['statistic'] for r in monitor.history),default=0.),observations=len(monitor.history))
    return dict(source=path,source_sha256=hashlib.sha256(source_bytes(path)).hexdigest(),
        maximum=maximum,threshold=maximum+1.,drift=.02,budgets=details,
        interpretation='Empirical development calibration, not a probabilistic false-alarm guarantee')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    write_json(args.output,calibrate())
