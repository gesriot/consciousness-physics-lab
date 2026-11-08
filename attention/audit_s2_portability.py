"""Post-result feature-tolerance and rounded-neighbor audit of frozen S2."""
import argparse
import copy
from attention import s2
from attention.build_s2 import build
from attention.io import read_json, write_json


def audit(archive):
    frozen = s2.load_model()
    rebuilt = build()
    deltas = [abs(x-y) for a,b in zip(frozen['records'],rebuilt['records'])
              for x,y in zip(a['features'],b['features'])]
    stripped = copy.deepcopy(rebuilt)
    for key in ('records','unused_records'):
        for a,b in zip(stripped[key],frozen[key]):
            a['features'] = b['features']
    rounded = copy.deepcopy(rebuilt)
    for key in ('records','unused_records'):
        for row in rounded[key]:
            row['features'] = [round(x,12) for x in row['features']]
    rows = []
    for record in archive['intake_channel']:
        request = record['request']
        old = s2.predict(request,model=frozen)
        new = s2.predict(request,model=rounded)
        # Query coordinates are rounded too; the stored model remains unchanged.
        vector, _ = s2.features(request, rounded)
        vector = [round(x,12) for x in vector]
        nearest = sorted(range(len(rounded['records'])), key=lambda i:(
            sum((a-b)**2 for a,b in zip(vector,rounded['records'][i]['features'])),i))[:16]
        ids = [rounded['records'][i]['id'] for i in nearest]
        rows.append(dict(task_id=record['task_id'],model_only_neighbor_order_equal=old['neighbor_ids']==new['neighbor_ids'],
            rounded_query_neighbor_order_equal=old['neighbor_ids']==ids,
            neighbor_sets_equal=set(old['neighbor_ids'])==set(ids),
            predictions_equal=all(old[k]==new[k] for k in ('full_probabilities','solvable_probability',
                'detection_correct_probability','decisive_probability','actions','base_probabilities'))))
    result = dict(analysis='Post-result A002 portability sensitivity, not prospective evidence',
        tolerance=1e-12,feature_count=len(deltas),changed_features=sum(x!=0 for x in deltas),
        maximum_feature_difference=max(deltas),features_within_tolerance=all(x<=1e-12 for x in deltas),
        nonfeature_content_equal=stripped==frozen,queries=len(rows),neighbors=16,
        rounded_feature_precision=12,checks=rows)
    result['passed'] = result['features_within_tolerance'] and result['nonfeature_content_equal'] and all(
        all(r[k] for k in ('model_only_neighbor_order_equal','rounded_query_neighbor_order_equal','neighbor_sets_equal','predictions_equal')) for r in rows)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from pathlib import Path
    if Path(args.output).exists():raise ValueError('Refuse to overwrite an audit')
    result=audit(read_json(args.archive));write_json(args.output,result)
    print({k:v for k,v in result.items() if k!='checks'})
    if not result['passed']:raise SystemExit(1)
