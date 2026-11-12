import argparse,collections,gzip,json,pathlib,hashlib
parser=argparse.ArgumentParser(description='Describe archived sample choices without rerunning numerical instruments.')
parser.add_argument('--repository',type=pathlib.Path,default=pathlib.Path.cwd())
parser.add_argument('--archive',type=pathlib.Path,default=pathlib.Path('binding/results/B003/confirmation-first'))
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('binding/results/B003/sample-mechanism.json'))
args=parser.parse_args()
R=args.repository.resolve();root=R/args.archive
source_head=json.loads((root/'manifest.json').read_text())['git_head']
out={'status':'post-hoc descriptive analysis, no new experiment or policy retuning','source_head':source_head,'cells':{},'example':None}
for env in ('whole','single','clean'):
 for quota in (1,2,4):
  c=collections.Counter();channels=collections.Counter();families=collections.Counter();ranks=collections.Counter();changed=collections.Counter();margins=[];source=[]
  for series in sorted(root.glob('series-*')):
   if not series.is_dir(): continue
   for p in sorted((series/f'{env}-d4-S-q{quota}').glob('package-*.json.gz')):
    x=json.load(gzip.open(p,'rt')); a=[a for a in x['result']['actions'] if a['kind']=='sample'];source.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for pol in ('Sx','Sxprime'):
     other_path=p.parent.parent/p.parent.name.replace('-S-',f'-{pol}-')/p.name
     source.append({'path':str(other_path.relative_to(R)),'sha256':hashlib.sha256(other_path.read_bytes()).hexdigest()})
     other=json.load(gzip.open(other_path,'rt'))
     b=[b['channel'] for b in other['result']['actions'] if b['kind']=='sample']
     if b != [aa['channel'] for aa in a]: c[pol+'_different_schedule_packages']+=1
    for a in a:
     c['samples']+=1;channel=a['channel'];channels[channel]+=1;families[channel.split(':')[0]]+=1
     if x['package_id']>=1: c['samples_from_package_1_onward']+=1
     snap=a['selection_snapshot'];pair=snap['paired_graphs'];eligible=[ch for ch in snap['tie_order'] if ch not in snap['suspect']]
     def choice(pol):
      return min(eligible,key=lambda ch:(-pair[pol]['scores'][ch],snap['markers'][ch],snap['tie_order'].index(ch)))
     for pol in ('Sx','Sxprime'):
      if pair[pol]['scores']!=pair['S']['scores']: c[pol+'_changed_score_snapshots']+=1
      if choice(pol)!=choice('S'): c[pol+'_changed_choice_snapshots']+=1
      c[pol+'_edge_edit_occurrences']+=len(pair[pol]['edits'])
     scores=pair['S']['scores'];colloc=[ch for ch in eligible if ch.startswith('collocation:')]
     positive=[ch for ch in colloc if pair['S']['numerators'][ch]>0]
     if positive:
      c['snapshots_positive_collocation_stake']+=1
      best=min(colloc,key=lambda ch:(-scores[ch],snap['markers'][ch],snap['tie_order'].index(ch)))
      order=sorted(eligible,key=lambda ch:(-scores[ch],snap['markers'][ch],snap['tie_order'].index(ch)))
      ranks[order.index(best)+1]+=1;margins.append(scores[channel]-scores[best])
     if pair['Sx']['edits']: c['snapshots_with_matched_edits']+=1
     if out['example'] is None and env=='whole' and quota==4 and x['package_id']==1 and a['sample_index']==0:
      out['example']={'path':str(p.relative_to(R)),'package_id':x['package_id'],'sample_index':a['sample_index'],'chosen':channel,'numerators':snap['numerators'],'prices':snap['denominators'],'scores':scores,'Sx_scores':pair['Sx']['scores'],'Sxprime_scores':pair['Sxprime']['scores'],'edits_per_control':len(pair['Sx']['edits'])}
  for metric in ('Sx_different_schedule_packages','Sxprime_different_schedule_packages','Sx_changed_choice_snapshots','Sxprime_changed_choice_snapshots','Sxprime_changed_score_snapshots'):
   c.setdefault(metric,0)
  out['cells'][f'{env}-q{quota}']={'counts':dict(c),'channels':dict(channels),'families':dict(families),'best_positive_collocation_rank_histogram':dict(ranks),'chosen_minus_best_positive_collocation_score_min':min(margins) if margins else None,'chosen_minus_best_positive_collocation_score_max':max(margins) if margins else None,'source_files':source}
path=R/args.output;path.write_text(json.dumps(out,indent=2)+'\n')
for k,v in out['cells'].items():print(k,json.dumps({a:b for a,b in v.items() if a!='source_files'}))
print('example',json.dumps(out['example']))
