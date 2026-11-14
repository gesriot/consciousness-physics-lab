"""Report paired series effects without treating dependent claims as replicates."""
import argparse
import math
from pathlib import Path
from binding.run_b004 import read, write


def analyze(directory):
    directory=Path(directory); manifest=read(directory/'manifest.json')
    archives=[read(directory/name) for name in sorted(manifest['files'])]
    rows={}
    for fault in manifest['faults']:
        for budget in (2,8,32):
            for branch in ('revision','conflict'):
                for opponent in ('G1','F','T','N','Gx','Gxprime'):
                    deltas={k:[] for k in ('remaining_bad','unnecessary','correct_lost','cost')}
                    for archive in archives:
                        case=archive['cases'][fault]
                        a=case['outputs'][case['conditions'][f'G|{budget}|{branch}']]['metrics']
                        b=case['outputs'][case['conditions'][f'{opponent}|{budget}|{branch}']]['metrics']
                        for key in deltas:deltas[key].append(a[key]-b[key])
                    report={}
                    for key,values in deltas.items():
                        n=len(values);mean=math.fsum(values)/n
                        report[key]=dict(mean=mean,standard_error=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n*(n-1))),
                                         G_lower=sum(v<0 for v in values),equal=sum(v==0 for v in values),G_higher=sum(v>0 for v in values))
                    rows[f'{fault}|{budget}|{branch}|G-{opponent}']=report
    return dict(independent_series=len(archives),difference='G minus comparator',paired=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();write(a.output,analyze(a.input))
