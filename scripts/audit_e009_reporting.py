#!/usr/bin/env python3
"""Derive decisive strata and deduplicated costs from preserved E009 records."""
import json
from pathlib import Path

p=json.loads(Path('results/surface-order/results.json').read_text())
result={}
for cell in p['summary']['cells']:
    ids={(r['task_id'],s['last_grid']) for r in p['runs'] if r['cell']==cell
         for s in r['independent']['steps'] if s['verdict']!='inconclusive'}
    steps=[s for s in p['coverage_steps'] if s['cell']==cell and (s['task_id'],s['grid']) in ids]
    bins={}
    for cohort in ('broad','high'):
        for bin_name in ('[0,.5)','[.5,1)','[1,2)','[2,4)','[4,infinity)'):
            group=[s for s in steps if s['cohort']==cohort and s['stratum']==bin_name]
            bins[cohort+':'+bin_name]=dict(steps=len(group),misses=sum(not s['covered'] for s in group),
                max_ratio=max((s['ratio'] for s in group),default=None))
    result[cell]=dict(unique_decisive_steps=len(steps),max_ratio=max((s['ratio'] for s in steps),default=None),strata=bins)
injections={(r['task_id'],r['fault']):r for r in p['runs']}
work=dict(shadow_cells_per_mesh={m:192*8128 for m in ('uniform','surface')},
          initial_stale_execution_modal_terms=p['initial_stale_modes'],
          total_execution_modal_terms=p['initial_stale_modes']+p['summary']['work']['execution_modal_terms'],
          additional_injection_modal_terms=sum(r['additional_injection_modes'] for r in injections.values()),
          base_execution_seconds=sum(t['base_execution_seconds'] for t in p['tasks']),
          deduplicated_injection_seconds=sum(r['injection_seconds'] for r in injections.values()),
          timing_limits='Per-mesh shadow times and initial stale execution time were not retained; only pooled shadow and per-task execution times are available.')
print(json.dumps(dict(decisive_coverage=result,work=work),indent=2,allow_nan=False))
