"""Paid local diagnosis with equivalent graph and inherited-label views."""
import copy
import hashlib
import json
import math
import numpy as np
from binding import source_lab_b005 as lab

POLICIES=('G','L','F','T','N','Gx','Gxprime','Lx','Lxprime')
FAULTS=lab.FAULTS
SCOPES=lab.SCOPES
BUDGETS=(2,8,32)
SIGNALS=('measurement','conflict')
SLOT_BYTES=16384
ATOL,RTOL=lab.ATOL,lab.RTOL
intake=lab.intake
build_case=lab.build_case
close=lab.close

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def rounded(value,noise=False):
    return round(float(np.nextafter(value,math.inf)) if noise else float(value),9)

def signal(case,branch):
    rows=lab.record_map(case)
    if branch=='measurement':
        failed=[r['id'] for r in case['records'] if not close(r['value'],case['measurements'][r['id']]['value'])]
        groups=[[c] for c in failed]
    else:
        groups=[p for p in case['pairs'] if not close(rows[p[0]]['value'],rows[p[1]]['value'])]
    if not groups:return []
    depths=sorted({rows[g[0]]['depth'] for g in groups})
    depth=depths[case['seed']%len(depths)]
    return sorted({c for g in groups if rows[g[0]]['depth']==depth for c in g})

class Memory:
    """Only public records and provenance; no evaluator or current catalog."""
    def __init__(self,case,policy,signal_ids):
        self.policy=policy
        self.rows={r['id']:{k:copy.deepcopy(r[k]) for k in ('id','value','unit','depth','timestamp','step_id')} for r in case['records']}
        self.sources=copy.deepcopy(case['sources'])
        self.source_order=list(case['source_order'])
        self.direct={r['id']:set(r['source_ids']) for r in case['records']} if policy.startswith('G') or policy=='F' else {}
        self.parents={r['id']:list(r['parents']) for r in case['records']} if policy.startswith('G') else {}
        self.labels={r['id']:lab.ancestry(r['id'],case) for r in case['records']} if policy.startswith('L') else {}
        self.extra={cid:'request:original' for cid in self.rows}
        if policy in ('Gx','Lx'):
            for cid in signal_ids:
                if policy=='Gx':self.parents[cid]=[];self.direct[cid]={'request:masked'}
                else:self.labels[cid]={'request:masked'}
        if policy in ('Gxprime','Lxprime'):
            for cid in signal_ids:self.extra[cid]='request:placebo'
        records=[]
        for cid,row in self.rows.items():
            item=copy.deepcopy(row)
            if policy.startswith('G'):item.update(parents=self.parents[cid],sources=sorted(self.direct[cid]))
            elif policy.startswith('L'):item['labels']=sorted(self.labels[cid])
            elif policy=='F':item['sources']=sorted(self.direct[cid])
            item['request']=self.extra[cid]
            records.append(item)
        self.record_bytes=[len(canonical({k:v for k,v in r.items() if k!='value'}))+8 for r in records]
        if max(self.record_bytes)>SLOT_BYTES:raise ValueError('Memory capacity exceeded')
        self.bytes=sum(self.record_bytes)+len(canonical(self.sources))+len(canonical(self.source_order))
        self.capacity=len(records)*SLOT_BYTES+len(canonical(self.sources))+len(canonical(self.source_order))
    def grounds(self,cid):
        if self.policy.startswith('L'):return set(self.labels[cid])
        if self.policy.startswith('G'):
            found=set();visited=set()
            def walk(c):
                if c in visited:return
                visited.add(c);found.update(self.direct[c])
                for p in self.parents[c]:walk(p)
            walk(cid);return found
        if self.policy=='F':return set(self.direct[cid])
        return set()

class Service:
    """Private physical executor: paid local actions, no culprit-returning oracle."""
    def __init__(self,case,noise=False):
        self.case=case;self.noise=noise
        self.sources=copy.deepcopy(case['sources'])
        self.values={r['id']:r['value'] for r in case['records']}
        self.rows=lab.record_map(case)
    def source_cost(self,sid):
        return 100 if self.sources[sid]['kind']=='method' else 30
    def check_source(self,sid):
        fresh=copy.deepcopy(self.case['catalog'][sid])
        return self.sources[sid]!=fresh,fresh,self.source_cost(sid)
    def claim_cost(self,cid):return lab.OPERATION_COST[self.rows[cid]['operation']]+1
    def check_claim(self,cid):
        sid=self.rows[cid]['step_id'];sources=dict(self.sources)
        sources[sid]=self.case['catalog'][sid]
        fresh,_=lab.rederive(cid,self.case,self.values,sources,self.noise)
        # Only the local method is tested; a wrong parent can still produce a
        # consistent local result and is never exonerated by this action.
        return not close(fresh,self.values[cid]),copy.deepcopy(sources[sid]),self.claim_cost(cid),fresh
    def repair_cost(self,cid):return lab.full_cost(cid,self.case)
    def repair(self,cid):
        visited=set();values=dict(self.values)
        def walk(c):
            if c in visited:return
            for p in self.rows[c]['parents']:walk(p)
            values[c]=lab.rederive(c,self.case,values,self.sources,self.noise)[0];visited.add(c)
        walk(cid)
        self.values[cid]=values[cid]
        return values[cid]
    def repeat_measurements(self,ids):
        return {cid:self.case['measurements'][cid]['repeat_value'] for cid in ids}

def available_tests(memory,service,candidates,checked):
    tests=[]
    for sid in sorted(candidates & set(memory.sources)):
        if sid not in checked:
            tests.append(dict(kind='check_source',id=sid,candidate=sid,cost=service.source_cost(sid)))
            for cid,row in memory.rows.items():
                if row['step_id']==sid:
                    tests.append(dict(kind='check_claim',id=cid,candidate=sid,cost=service.claim_cost(cid)))
    return tests

def choose(memory,tests,candidates,noise=False):
    n=len(candidates)
    for t in tests:t['score']=rounded((1 if n<=1 else min(1,n-1))/t['cost'],noise)
    if memory.policy=='T':
        def key(t):
            position=(len(memory.sources)+memory.rows[t['id']]['timestamp']
                      if t['kind']=='check_claim' else memory.source_order.index(t['candidate']))
            return (-position,t['id'],t['kind'])
    elif memory.policy=='F':
        key=lambda t:(t['candidate'] not in candidates,t['candidate'],t['cost'],t['id'])
    else:key=lambda t:(-t['score'],t['kind']!='check_claim',t['id'],t['candidate'])
    return min(tests,key=key)

def simulate(case,policy,budget,branch,noise=False):
    if policy not in POLICIES or branch not in SIGNALS or budget not in BUDGETS:raise ValueError('Unknown condition')
    ids=signal(case,branch);mem=Memory(case,policy,ids);svc=Service(case,noise)
    limit=budget*case['budget_unit'];spent=0;trace=[];checked=set();culprit=None;accusations=[]
    initial=[mem.grounds(c) for c in ids]
    if initial and branch=='measurement' and policy.startswith(('G','L')):
        candidates=set.intersection(*initial)
    else:candidates=set.union(*initial) if initial else set()
    if policy=='T':candidates=set(mem.sources)
    if policy=='N':candidates=set()
    initial_candidates=sorted(candidates);fallback=False;resolved=not ids;repeat=False;checks=0;repaired=[]
    while ids and policy!='N' and not resolved:
        tests=available_tests(mem,svc,candidates,checked)
        if not tests:
            if policy=='F' and not fallback:
                candidates=set(mem.sources)-checked;fallback=True;continue
            if branch=='measurement':
                cost=2*case['budget_unit']
                if spent+cost<=limit:
                    fresh=svc.repeat_measurements(ids);spent+=cost;repeat=True
                    resolved=all(close(svc.values[c],v) for c,v in fresh.items())
                    trace.append(dict(kind='repeat_measurement',id='measurement',cost=cost,resolved=resolved))
            break
        affordable=[t for t in tests if spent+t['cost']<=limit]
        if not affordable:break
        selected=choose(mem,affordable,candidates,noise);checks+=1;sid=selected['candidate'];cost=selected['cost']
        if selected['kind']=='check_source':bad,fresh,actual=svc.check_source(sid);value=None
        else:bad,fresh,actual,value=svc.check_claim(selected['id'])
        if actual!=cost:raise AssertionError('Uncharged action')
        spent+=cost;checked.add(sid)
        trace.append(dict(**selected,bad=bad,checked_value=value,candidates=sorted(candidates)))
        if bad:
            culprit=sid;accusations.append(sid);svc.sources[sid]=fresh;break
        candidates.discard(sid)
    if culprit is not None:
        if policy=='T':targets=list(mem.rows)
        else:targets=[cid for cid in mem.rows if culprit in mem.grounds(cid)]
        targets.sort(key=lambda c:(mem.rows[c]['depth'],c))
        for cid in targets:
            cost=svc.repair_cost(cid)
            if spent+cost>limit:continue
            value=svc.repair(cid);spent+=cost;repaired.append(cid)
            trace.append(dict(kind='repair',id=cid,cost=cost,value=value))
        resolved=all(close(svc.values[c],case['measurements'][c]['value']) for c in ids) if branch=='measurement' else all(close(svc.values[a],svc.values[b]) for a,b in case['pairs'] if a in ids or b in ids)
        candidates={culprit} if not resolved else set()
    suspected=(sorted(mem.rows) if policy=='T' and candidates else sorted(cid for cid in mem.rows if candidates & mem.grounds(cid))) if not resolved and policy!='N' else []
    # Evaluation is separate from policy choices and is the only truth access.
    bad_ids=sorted(c for c,v in svc.values.items() if not close(v,case['truth'][c]))
    correct_culprit=int(culprit is not None and culprit==case['culprit'])
    metrics=dict(remaining_bad=len(bad_ids),withdrawn_good=0,false_accusations=sum(s!=case['culprit'] for s in accusations),
                 cost=spent,unused_budget=limit-spent,correctly_localized=correct_culprit,checks_to_localization=checks if culprit else 0,
                 checks=checks,suspected=len(suspected),resolved=int(resolved),repeated_measurement=int(repeat),
                 initial_bad=len(case['inaccurate']),repaired=len(repaired),memory_bytes=mem.bytes,capacity_bytes=mem.capacity,
                 signal_depth=max((mem.rows[c]['depth'] for c in ids),default=0),localization_opportunities=int(case['culprit'] is not None and bool(ids)))
    protected=dict(signal_ids=ids,initial_candidates=initial_candidates,checked=sorted(checked),culprit=culprit,
                   accusations=accusations,suspected=suspected,repaired=repaired,resolved=resolved,
                   budget=limit,cost=spent,remaining_bad_ids=bad_ids,
                   actions=[{k:v for k,v in t.items() if k not in ('score','checked_value','value')} for t in trace])
    return dict(protected=protected,metrics=metrics,trace=trace,
                diagnostics=dict(final_values=svc.values,record_bytes=mem.record_bytes,
                    signal_observations={c:case['measurements'][c]['value'] for c in ids} if branch=='measurement' else {}))
