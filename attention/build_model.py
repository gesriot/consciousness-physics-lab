"""Reconstruct the frozen development-only table; never load A001 outcomes."""
import argparse
import json, hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1]; data=json.loads((p/'results/E016-frozen-pipeline/results.json').read_text())
def band(r):
 x=abs(complex(*r['request']['z'])*r['request']['mean_time']); return next((i for i,b in enumerate([.1,1e3,1e6]) if x<=b),3)
intake={r['task_id']:r for r in data['intake_channel']}
A=['binding','normalization','first_moment','second_moment','high_frequency','collocation']+[f'v2_{n}' for n in [64,128,256,512,1024,2048,4096]]
profiles={}
for b in range(4):
 rows=[r for r in data['runs'] if band(intake[r['task_id']])==b]
 d={}
 for a in A:
  if a.startswith('v2_'): vals=[r['decision']['independent']['verdict'] for r in rows]
  elif a=='collocation': vals=[r['decision']['third']['verdict'] for r in rows]
  else: vals=['accept' if r['decision']['checks']['hash' if a=='binding' else a] else 'violation_detected' for r in rows]
  d[a]={'detection_probability':sum(v=='violation_detected' for v in vals)/len(vals),'accept_probability':sum(v=='accept' for v in vals)/len(vals),'development_records':len(vals)}
 profiles[str(b)]=d
sources={}
for path in sorted(p.glob('results/**/report.md'))+sorted(p.glob('results/**/summary.md')):
 if any(s in str(path) for s in ['E012','E014','E015','E016','tolerance-allocation','fault-heldout','verifier-stress','common-mode','surface-order','E010','E011','E013']): sources[str(path.relative_to(p))]=hashlib.sha256(path.read_bytes()).hexdigest()
sources['results/E016-frozen-pipeline/results.json']=hashlib.sha256((p/'results/E016-frozen-pipeline/results.json').read_bytes()).hexdigest()
model={'version':1,'development_sources_sha256':sources,'training_source':'results/E016-frozen-pipeline/results.json','training_rows':len(data['runs']),'bands_abs_zT_upper':[.1,1e3,1e6,None],'profiles':profiles,'unused_profiles':json.loads(json.dumps(profiles)),'cost_weights':{'modal_term':1/4096,'fv_cell':1/4096,'collocation_degree_cubed':8/128**3,'binding':1/4096},'score_accept_weight':.05,'reliability_notes':{'V2':'E012/E014: roundoff exception unsafe; P uses C1 contraction and existing futility. Band frequencies are predictions, not guaranteed reliability.','collocation':'E015: uncertainty floor 1.9073486328125e-6 blocks acceptance below that tolerance, not detection; no frequency cutoff justified.','binding':'E008 request binding detects altered requests; E016 nine conditions do not exercise request swaps.','features':'E010 scale/geometry discrepancies; E013 matched mixture escapes four probes. Detection probabilities are empirical E016 marginals, never task labels.'},'collocation_accept_floor':1.9073486328125e-6,'D':'Swap high_frequency and normalization detection_probability in all profiles.','Dprime':'Identical swaps in unused_profiles only.'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise ValueError('Output already exists')
args.output.write_text(json.dumps(model,indent=2)+'\n')
