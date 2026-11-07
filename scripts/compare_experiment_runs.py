#!/usr/bin/env python3
"""Exact comparison excluding recorded timings; optionally allow a later HEAD."""
import argparse
import json
from pathlib import Path

def stable(value, ignore_git_head=False):
    if isinstance(value,dict):
        return {key:stable(item,ignore_git_head) for key,item in value.items()
                if not key.endswith('seconds') and not (ignore_git_head and key=='git_head')}
    if isinstance(value,list):
        return [stable(item,ignore_git_head) for item in value]
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path)
    parser.add_argument('repeat',type=Path)
    parser.add_argument('--ignore-git-head',action='store_true',
                        help='Only for reproduction from a later documentation/results commit')
    args=parser.parse_args()
    original=json.loads(args.original.read_text());repeat=json.loads(args.repeat.read_text())
    assert stable(original,args.ignore_git_head)==stable(repeat,args.ignore_git_head), 'Non-timing results differ'
    print('PASS: exact equality after removing fields ending seconds'+
          (' and git_head (later commit).' if args.ignore_git_head else '.'))
    print('Protocol commit/hash, source hashes, environment, seeds, values, decisions and counts retained.')

if __name__=='__main__':main()
