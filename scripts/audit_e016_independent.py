#!/usr/bin/env python3
"""Independently audit saved E016 raw evidence without rerunning its summaries."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.matched_mixture import components


def scientific(value, top=True):
    if isinstance(value, dict):
        return {k:scientific(v, False) for k,v in value.items()
                if not k.endswith('seconds') and not (top and k in
                ('environment','git_head','source_sha256','runtime_override','replay'))}
    if isinstance(value, list):
        return [scientific(v, False) for v in value]
    return value


def intrinsic(q, model):
    with mp.workdps(80):
        z = mp.mpc(*q['z'])
        def one(t, dimension):
            x = mp.sqrt(dimension * (dimension+2) * z * mp.mpf(t))
            return mp.tanh(x)/x if dimension == 1 else 3*(mp.coth(x)/x-1/x**2)
        t = q['mean_time']
        if model == 'mixture':
            value = sum(mp.mpf(w)*one(t*x,1) for w,x,_ in components('mixture'))
        elif model == 'model_scale':
            # Existing sphere model scales the exact binary64 T mathematically.
            x = mp.sqrt(15*z*mp.mpf(t)*1000)
            value = 3*(mp.coth(x)/x-1/x**2)
        else:
            value = one(t,1 if model == 'slab' else 3)
        return complex(float(value.real),float(value.imag)), value


def audit(data):
    failures = []
    def check(test, label):
        if not test:
            failures.append(label)
    records = {r['task_id']:r for r in data['intake_channel']}
    tasks = {r['task_id']:r for r in data['tasks']}
    check(len(records)==240 and len(data['runs'])==2160, 'sample_size')
    truths, max_hp_discrepancy = {}, 0.
    for task_id, task in tasks.items():
        q = records[task_id]['request']
        for model in ('sphere','model_scale','slab','mixture'):
            rounded, high = intrinsic(q, model)
            stored = task['model_references'][model]
            with mp.workdps(80):
                err = float(abs(high-mp.mpc(*stored['high_precision'])))
            max_hp_discrepancy = max(max_hp_discrepancy,err)
            check(err < 1e-52, f'model_precision:{task_id}:{model}')
            check(rounded == complex(*stored['value']),f'model_reference:{task_id}:{model}')
            truths[task_id,model] = rounded
    audited = set()
    for i,r in records.items():
        a,b = r['log_magnitude_interval']; n,d = r['magnitude_u_ratio']
        if Fraction(a)+(b-a)*Fraction(n,d)>8 or i%10==0:
            audited.add(i)
    check(audited == {r['task_id'] for r in data['reference_audit']['records']
                      if r['selection']!='whole_sample_fallback'}, 'audit_selection')
    decoded = []
    for index,row in enumerate(data['runs']):
        tid = row['task_id']; model = row['model']; decision = row['decision']
        q = records[tid]['request']; tol=q['tolerance']; supplied=complex(*row['outcome']['value'])
        reference=complex(*row['reference']); error=abs(supplied-reference)
        accurate=error<=tol
        check(error == row['absolute_error'] and accurate == row['accurate'],f'artifact_accuracy:{index}')
        check((abs(supplied-truths[tid,'sphere'])<=tol)==accurate,f'80_digit_accuracy:{index}')
        num=decision['independent']; values=[complex(*v) for v in num['values']]
        steps=[]
        for s in num['steps']:
            pos=num['grids'].index(s['last_grid']); prefix=values[:pos+1]
            ds=[abs(b-a) for a,b in zip(prefix,prefix[1:])]; n=s['last_grid']
            rounding=32*np.finfo(float).eps*n*n*max(1,abs(prefix[-1])); u=2*ds[-1]+rounding
            c1=ds[-1]<=.5*ds[-2]; c2=c1 or (ds[-1]<=rounding and ds[-2]<=.5*ds[-3])
            discrepancy=abs(supplied-prefix[-1])
            verdict=('accept' if c1 and discrepancy+u<=tol else
                     'violation_detected' if c1 and discrepancy-u>tol else 'inconclusive')
            check(verdict==s['verdict'] and c1==s['converging'] and c2==s['shadow_C2_converging'],f'V2_step:{index}:{n}')
            check(u==s['uncertainty_estimate'],f'V2_U:{index}:{n}')
            noise=bool(ds[-1]<=rounding and ds[-2]<=rounding)
            check(noise==s['both_differences_within_rounding'],f'noise:{index}:{n}')
            steps.append(dict(task_id=tid,model=model,grid=n,decisive=verdict!='inconclusive',
                sphere_covered=abs(prefix[-1]-reference)<=u,
                intrinsic_covered=abs(prefix[-1]-truths[tid,model])<=u,
                noise=noise,disagreement=bool(c1!=c2)))
        third=decision['third']; v0,v1=map(lambda x:complex(*x),third['values'])
        delta=abs(v1-v0); u=2*delta+32*np.finfo(float).eps*128**4*max(1,abs(v1)); d=abs(supplied-v1)
        conv=delta<=tol/8
        verdict=('accept' if conv and d+u<=tol else 'violation_detected' if conv and d-u>tol else 'inconclusive')
        check(verdict==third['verdict'] and conv==third['converging'] and u==third['uncertainty'],f'third_rule:{index}')
        probes=decision['probes']; checks={}
        checks['hash']=decision['operational_digest']==records[tid]['request_sha256']
        checks['normalization']=all(abs(complex(*p['zero'])-1)<=2e-4 for p in probes['old']['probes'].values())
        checks['first_moment']=all(abs((-complex(*p['shifted']).imag/p['h'])/q['mean_time']-1)<=2e-3 for p in probes['old']['probes'].values())
        checks['second_moment']=all(abs((2*(p['zero'][0]-p['shifted'][0])/(p['h']*q['mean_time'])**2)/(20/7)-1)<=.02 for p in probes['second_moment']['probes'].values())
        checks['high_frequency']=all(abs(100*complex(*p['value'])/np.sqrt(3/5)-1)<=.03 for p in probes['high_frequency']['probes'].values())
        check(checks==decision['checks'],f'model_gates:{index}')
        policies={}
        for policy in ('strict','soft'):
            violation=not all(checks.values()) or num['verdict']=='violation_detected' or verdict=='violation_detected'
            accepted=(not violation and num['verdict']=='accept' and decision['resolution']['accepted'] and
                      (verdict=='accept' if policy=='strict' else verdict!='violation_detected'))
            pv='violation_detected' if violation else 'accept' if accepted else 'inconclusive'
            check(pv==decision['policies'][policy]['verdict'],f'policy:{index}:{policy}')
            policies[policy]=pv
        count=2 if model=='mixture' else 1
        work=dict(verification_fv_cells=sum(num['grids'])*count,probe_modal_terms=20480*count,
                  probe_fv_cells=13312*count,third_matrix_solves=2,third_sum_degrees=192)
        check(work==row['work'],f'work:{index}')
        decoded.append(dict(task_id=tid,cohort=row['cohort'],condition=row['condition'],model=model,
            accurate=accurate,policies=policies,steps=steps,third_decisive=verdict!='inconclusive',
            third_covered=abs(v1-reference)<=u,work=work,numerical_violation=(num['verdict']=='violation_detected' or verdict=='violation_detected'),model_violation=not all(checks.values())))
    outputs={}
    for cohort in ('all','W','H','M','O','S'):
        rows=[r for r in decoded if cohort=='all' or r['cohort']==cohort]
        clean=[r for r in rows if r['condition']=='clean']; steps=[s for r in rows for s in r['steps']]
        decisive=[s for s in steps if s['decisive']]; unique={(s['task_id'],s['model'],s['grid']):s for s in decisive}
        third=[r for r in rows if r['third_decisive']]; unique_third={r['task_id']:r for r in third}
        policies={}
        for p in ('strict','soft'):
            policies[p]=dict(verdicts=dict(Counter(r['policies'][p] for r in rows)),
                false_acceptances=sum(r['policies'][p]=='accept' and not r['accurate'] for r in rows),
                clean_accurate_acceptances=sum(r['policies'][p]=='accept' and r['accurate'] for r in clean),
                accurate_numerical_violations=sum(r['accurate'] and r['numerical_violation'] for r in rows),
                accurate_model_violations=sum(r['accurate'] and r['model_violation'] for r in rows))
        out=dict(tasks=len({r['task_id'] for r in rows}),rows=len(rows),clean_accurate=sum(r['accurate'] for r in clean),policies=policies,
            V2_decisive_rows=len(decisive),V2_unique_decisive=len(unique),
            V2_sphere_misses=sum(s['model']=='sphere' and not s['sphere_covered'] for s in decisive),
            V2_intrinsic_misses=sum(not s['intrinsic_covered'] for s in decisive),
            V2_original_sphere_discrepancies=sum(not s['sphere_covered'] for s in decisive),
            third_decisive_rows=len(third),third_unique_decisive=len(unique_third),third_misses=sum(not r['third_covered'] for r in third),
            operational_steps=len(steps),noise_decisive=sum(s['noise'] for s in decisive),
            C1_C2_disagreements=sum(s['disagreement'] for s in steps),
            component_work={k:sum(r['work'][k] for r in rows) for k in rows[0]['work']})
        out['clean_strictness_cost']=policies['soft']['clean_accurate_acceptances']-policies['strict']['clean_accurate_acceptances']
        saved=data['summary'] if cohort=='all' else data['cohort_summary'][cohort]
        check(out['component_work']==saved['component_work'],f'work_summary:{cohort}')
        check(out['noise_decisive']==saved['noise']['decisive_both_differences_within_rounding'],f'noise_summary:{cohort}')
        check(out['V2_unique_decisive']==saved['coverage']['unique_task_model_grid_decisive_steps'],f'unique_summary:{cohort}')
        for p in policies:
            for key,skey in [('false_acceptances','false_acceptances'),('clean_accurate_acceptances','clean_accurate_accepted'),('accurate_numerical_violations','accurate_numerical_violations'),('accurate_model_violations','accurate_model_or_request_violations')]:
                check(policies[p][key]==saved['policies'][p][skey],f'policy_summary:{cohort}:{p}:{key}')
        outputs[cohort]=out
    base=sum(t['base_modes'] for t in tasks.values()); initial=data['work']['initial_stale_modes']
    expected_total=7*base+initial+outputs['all']['component_work']['probe_modal_terms']
    check(expected_total==data['work']['total_modal_terms'],'total_modal_terms')
    check(sum(outputs['all']['component_work'][k] for k in ('verification_fv_cells','probe_fv_cells'))==data['work']['total_fv_cells'],'total_fv_cells')
    return dict(passed=not failures,failures=failures,max_model_80_vs_60_digit_difference=max_hp_discrepancy,
                audited_request_count=len(audited),endpoints=outputs,total_modal_terms=expected_total,
                method='Independent formulas and raw rows; no production summary function invoked')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path);parser.add_argument('repeat',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    original=json.loads(args.original.read_bytes());repeat=json.loads(args.repeat.read_bytes())
    result=audit(original)
    result['repeat_scientific_exact']=scientific(original)==scientific(repeat)
    result['passed']=result['passed'] and result['repeat_scientific_exact']
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.original,args.repeat)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
