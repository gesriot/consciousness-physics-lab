#!/usr/bin/env python3
"""Fresh component decisions on development archives, without new task sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab import frozen_pipeline as p
from lab.executor import request_hash
from lab.guard import Request


def audit():
    result = dict(gate_requests=0,third_rows=0,mismatches=[],archives={})
    for experiment, path in (
            ('E011','results/E011-extrapolation/results.json'),
            ('E013','results/E013-matched-mixture/results.json')):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        tasks = {r['task_id']:r for r in data['intake_channel']}
        seen = set()
        stats = dict(source_sha256=hashlib.sha256(raw).hexdigest(),gate_requests=0,third_rows=0)
        for row in data['runs']:
            q = tasks[row['task_id']]['request']
            request = Request(complex(*q['z']),q['mean_time'],q['tolerance'],q['budget'],q['max_calls'])
            model = row.get('model',row.get('condition'))
            if model == 'clean': model = 'sphere'
            key = (row['task_id'],model)
            if key not in seen:
                checks = p.checks_from_probes(p.model_probes(request,model),request_hash(request),row['original_digest'])
                if checks != row['checks']:
                    result['mismatches'].append(dict(experiment=experiment,task_id=row['task_id'],model=model,
                        component='gates',old=row['checks'],new=checks))
                seen.add(key)
                stats['gate_requests'] += 1
            if experiment == 'E013':
                third = p.third_check(request.z,request.mean_time,complex(*row['outcome']['value']),request.tolerance)
                for field in ('verdict','converging','degrees','matrix_solves'):
                    if third[field] != row['third'][field]:
                        result['mismatches'].append(dict(experiment=experiment,task_id=row['task_id'],model=model,
                            component='third',field=field,old=row['third'][field],new=third[field]))
                stats['third_rows'] += 1
        result['archives'][experiment] = stats
        result['gate_requests'] += stats['gate_requests']
        result['third_rows'] += stats['third_rows']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')
    if result['mismatches']:
        raise SystemExit(1)
