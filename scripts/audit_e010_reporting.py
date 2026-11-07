#!/usr/bin/env python3
"""Derive unambiguous clean-rejection reporting from archived E010 rows."""
import argparse
import json
from pathlib import Path


def correction(result):
    clean = [row for row in result['runs'] if row['geometry'] == 'clean']
    return dict(note='The archived clean_false_rejected field counts all clean rejections, including inaccurate artifacts. This supplement disambiguates it without changing archived decisions or data.',
        policies={policy: dict(
            clean_rejected=sum(not row['accepted'][policy] for row in clean),
            clean_accurate_false_rejected=sum(not row['accepted'][policy] and row['accurate'] for row in clean),
            clean_inaccurate_correctly_rejected=sum(not row['accepted'][policy] and not row['accurate'] for row in clean))
            for policy in result['summary']['clean']['policies']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print(json.dumps(correction(json.loads(args.results.read_text())), indent=2))
