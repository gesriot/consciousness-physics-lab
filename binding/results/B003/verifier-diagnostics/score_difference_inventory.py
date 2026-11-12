"""Inventory sample-score differences from a saved strict comparison."""
import argparse,gzip,json,hashlib
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('first',type=Path);parser.add_argument('noise',type=Path);parser.add_argument('--comparison',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
first,noise=args.first,args.noise;comparison=json.loads(args.comparison.read_text());rows=[]
for name in comparison['differing_decisions']:
 with gzip.open(first/name,'rt') as f:a=json.load(f)
 with gzip.open(noise/name,'rt') as f:b=json.load(f)
 for i,(x,y) in enumerate(zip(a['result']['actions'],b['result']['actions'])):
  if x['kind']!='sample':continue
  changed={c:{'first':v,'noise':y['scores'][c]} for c,v in x['scores'].items() if v!=y['scores'][c]}
  if changed:
   assert x['channel']==y['channel']
   assert x['selection_snapshot']['scores']==x['scores'] and y['selection_snapshot']['scores']==y['scores']
   rows.append(dict(file=name,action_index=i,sample_index=x['sample_index'],tick=x['tick'],selected_channel=x['channel'],changed_scores=changed,first_file_sha256=hashlib.sha256((first/name).read_bytes()).hexdigest(),noise_file_sha256=hashlib.sha256((noise/name).read_bytes()).hexdigest()))
out={'status':'complete_score_difference_inventory','snapshots':len(rows),'scalar_fields':2*sum(len(r['changed_scores']) for r in rows),'operational_channel_changes':0,'rows':rows}
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(len(rows))
