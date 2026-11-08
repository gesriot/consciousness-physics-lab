"""Post-result A006 conditioning and replacement-margin audit."""
import argparse
import platform
import sys
import numpy as np
import scipy
from attention import calibration_samples as samples, s3, agent
from attention.a006 import source_sha256
from attention.compare_a004 import Archive
from attention.io import read_json, write_json
from lab import frozen_pipeline as P
from lab.finite_volume import response as uniform_response
from lab.guard import Request

# A forward diagnostic allowance, not an uncertainty guarantee for the solver.
# Differences amplify two endpoint errors; the second moment amplifies them by
# 2 / (h**2 * (20/7)). Ratios need their own denominator-dependent allowance.
ENDPOINT_ALLOWANCE = 1e-11


def diagnostic_atol(family, key, container):
    if family == 'second_moment' and key == 'error':
        return 4 * ENDPOINT_ALLOWANCE / (.01**2 * (20/7))
    if family == 'v2' and key == 'antecedent_ratio':
        ratio = container[key]
        if ratio > 0:
            denominator = container['previous_difference'] / ratio
            error = 2 * ENDPOINT_ALLOWANCE
            if denominator > error:
                return error * (1 + abs(ratio)) / (denominator - error)
        return ENDPOINT_ALLOWANCE
    if family == 'v2' and key == 'uncertainty_estimate':
        return 4 * ENDPOINT_ALLOWANCE
    return ENDPOINT_ALLOWANCE


def numerical_margin(observation, tolerance, kind='v2'):
    s = observation.get('step', observation)
    d = s['discrepancy']; u = s.get('uncertainty_estimate', s.get('uncertainty'))
    low, high = d-u, d+u
    out = dict(verdict=observation['verdict'], discrepancy=d, uncertainty=u,
               lower=low, upper=high, tolerance=tolerance,
               accept_margin=tolerance-high, violation_margin=low-tolerance,
               nearest_decision_boundary=min(abs(tolerance-high),abs(low-tolerance)),
               rounding=s['rounding'], converging=s['converging'])
    out['boundary_in_rounding_units']=out['nearest_decision_boundary']/s['rounding']
    if kind == 'v2':
        out['contraction_margin']=.5*s['previous_difference']-s['last_difference']
        out['contraction_in_rounding_units']=out['contraction_margin']/s['rounding']
        out['both_differences_within_rounding']=s['both_differences_within_rounding']
    else:
        out['contraction_margin']=tolerance/8-s['difference']
    return out


def library_audit(library):
    rows=[];weights=s3.load_model()['action_model']['cost_weights']
    for sample in library['samples']:
        actual=samples.execute_sample(sample,weights)
        r=samples.request(sample);family=sample['family'];o=sample['clean_result']['original']
        row=dict(sample_id=sample['sample_id'],family=family,expected=sample['expected'],
                 actual_verdict=actual['verdict'],verdict_match=actual['verdict']==sample['expected'])
        if family in agent.FEATURES:
            allowance=getattr(P.CONFIG,family+'_allowance')
            rowspec=[dict(solver=k,error=v['error'],threshold=allowance,
                          margin=abs(v['error']-allowance),
                          diagnostic_allowance=diagnostic_atol(family,'error',v))
                     for k,v in o['probes'].items()]
            row['probe_margins']=rowspec
            row['diagnostic_drift_may_flip']=any(x['margin']<=x['diagnostic_allowance'] for x in rowspec)
        elif family in ('v2','collocation'):
            row['numerical']=numerical_margin(o,r.tolerance,family)
            # A category is stable if both interval boundaries and convergence
            # have more margin than propagated endpoint perturbations.
            m=row['numerical'];a=ENDPOINT_ALLOWANCE
            row['diagnostic_drift_may_flip']=(m['nearest_decision_boundary']<=5*a or abs(m['contraction_margin'])<=3*a)
        else:
            row['diagnostic_drift_may_flip']=False
            row['scope']='Binding hash / integer modal resource guard; no floating diagnostic decision.'
        rows.append(row)
    return dict(count=len(rows),verdict_matches=sum(x['verdict_match'] for x in rows),
                potentially_sensitive_samples=[x['sample_id'] for x in rows if x['diagnostic_drift_may_flip']],samples=rows)


def archive_audit(path):
    replacements=[]; harm=[]; replay={}; values_cache={}
    archive=Archive(path)
    for package in archive.packages():
        p=package['package_id']
        for f,run in package['conditions']['permissive'].items():
            c2=run['C2'];false=[x['input_index'] for x in c2['inputs'] if x['false_accept']]
            if false:
                touched=set();found=[]
                for index,a in enumerate(c2['actions']):
                    if a['sample']: continue
                    touched.add(a['input_index'])
                    if a['input_index'] in false:found.append(dict(action_index=index,input_index=a['input_index'],touched_inputs=len(touched)))
                # Final action on each false-accepted input is the earliest point
                # all its paid evidence was available (no later action exists).
                last={x['input_index']:x for x in found}
                harm.append(dict(package_id=p,fraction=f,false_inputs=false,last_actions=list(last.values()),
                                 total_actions=len(c2['actions']),touched_inputs=c2['touched_inputs']))
        if p!=8:continue
        for fault,fractions in package['conditions'].items():
            for fraction,runs in fractions.items():
                for policy,run in runs.items():
                    for x in run['inputs']:
                        o=x['replacements'].get('collocation')
                        if o is None:continue
                        i=x['input_index'];base=package['inputs'][i]
                        if i not in replay:
                            task=base['task_id']
                            fields=dict(archive.metadata['intake_channel'][task]['request'])
                            fields['z']=complex(*fields['z']);request=Request(**fields)
                            grids=(4096,8192,16384,32768)
                            if task not in values_cache:
                                values_cache[task]=[uniform_response(request.z,request.mean_time,n) for n in grids]
                            actual=P.v2_step(values_cache[task],grids[-1],complex(*base['outcome']['value']),request.tolerance)
                            replay[i]=dict(input_index=i,task_id=task,margin=numerical_margin(dict(verdict=actual['verdict'],step=actual),request.tolerance),
                                           archived_verdict=o['verdict'],verdict_match=actual['verdict']==o['verdict'])
                        replacements.append(dict(package_id=p,fault=fault,fraction=fraction,policy=policy,
                            input_index=i,input={k:base[k] for k in ('task_id','condition','cohort','tolerance','absolute_error','accurate')},final_verdict=x['verdict'],
                            margin=numerical_margin(o,base['tolerance'])))
    return dict(batch8_replacements=replacements,batch8_reexecution=list(replay.values()),c2_permissive_harm=harm)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    library_path='attention/sample_library_a006.json';archive_path='attention/results/A006/results.json.gz'
    out=dict(kind='A006 post-result portability audit',environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),machine=platform.machine()),
             endpoint_allowance=ENDPOINT_ALLOWANCE,library_sha256=source_sha256(library_path),archive_sha256=source_sha256(archive_path),
             library=library_audit(read_json(library_path)))
    write_json(args.output,out)
    out.update(archive_audit(archive_path));write_json(args.output,out)


if __name__=='__main__':main()
