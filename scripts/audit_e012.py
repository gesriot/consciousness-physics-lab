"""Exact E012 repeat audit and four-panel discrete coverage map."""
import argparse
import gzip
import json
from pathlib import Path


def load(path):
    with gzip.open(path,'rt') as f:
        return json.load(f)


def compare(a,b):
    a,b = Path(a),Path(b)
    left,right = load(a/'map.json.gz'),load(b/'map.json.gz')
    fields = ('experiment','protocol_sha256','source_sha256','points','batches','refinements','summary')
    comparison = {k:left[k]==right[k] for k in fields}
    comparison['dimensionless'] = json.loads((a/'dimensionless.json').read_text())==json.loads((b/'dimensionless.json').read_text())
    for i in range(4):
        comparison['reference_level_'+str(i)] = load(a/f'reference-level-{i}.json.gz')==load(b/f'reference-level-{i}.json.gz')
    return dict(passed=all(comparison.values()),exact_fields=comparison,
                excluded=['elapsed time','git head and environment provenance'],
                points=len(left['points']))


def plot(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = load(Path(source)/'map.json.gz')['points']
    fig,axes = plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    labels = ('Covered, converging','Covered, not converging','Miss, not converging','Miss, converging')
    colors = ('#b9d9ea','#e6bc57','#b83b46','#54278f')
    for index,ax in enumerate(axes.flat):
        for category in range(4):
            subset = [r for r in data if (0 if r['steps'][index]['covered'] else 2)+(not r['steps'][index]['converging'] if r['steps'][index]['covered'] else r['steps'][index]['converging'])==category]
            ax.scatter([r['log10_magnitude'] for r in subset],
                [r['phase']*180/3.141592653589793 for r in subset],
                s=2,color=colors[category],label=labels[category],rasterized=True)
        ax.set_title('n = '+str(data[0]['steps'][index]['grid']))
        ax.set_xlim(-6,12);ax.set_ylim(-90,90);ax.set_xlabel('log10 |zT|');ax.set_ylabel('Phase (degrees)')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,markerscale=4)
    fig.suptitle('E012: evaluated nodes only; adaptive spacing, not a continuum certificate')
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(destination,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('output')
    args=p.parse_args();result=compare(args.first,args.second)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    plot(args.first,Path(args.output).with_name('coverage-map.png'))
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)
