"""Read-only diagnostic; not a replacement for the frozen official audit."""
import argparse
import builtins
import hashlib
import inspect
import json
import sys
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('repository',type=Path)
parser.add_argument('archive',type=Path)
parser.add_argument('--report',type=Path,default=Path('/tmp/B003-noise-score-diagnostic.json'))
args=parser.parse_args();sys.path.insert(0,str(args.repository.resolve()))
from binding import audit_b003 as A
manifest=A.read(args.archive/'manifest.json')
assert manifest['last_bit_noise'] is True
source_hashes={p:A.byte_sha((A.ROOT/p).read_bytes()) for p in manifest['source_sha256']}

def noise_round(value,digits=9):
    assert digits==9
    return builtins.round(float(np.nextafter(float(value),np.inf)),digits)

original_stakes=A.score_stakes

def noise_stakes(records,markers,prices):
    counts,_=original_stakes(records,markers,prices)
    return counts,{c:noise_round(counts[c]/prices[c],9) for c in A.CHANNELS}

source=inspect.getsource(A.check_package)
assert source.count('scores={ch:round(')==1
assert source.count('round(p,9)')==2
modified=source.replace('scores={ch:round(','scores={ch:noise_round(').replace('round(p,9)','noise_round(p,9)')
namespace=dict(vars(A),noise_round=noise_round,score_stakes=noise_stakes)
exec(compile(modified,'<diagnostic in-memory registered noise reconstruction>','exec'),namespace)
corrected=namespace['check_package'];count=0

def instrumented(*args,**kwargs):
    global count
    result=corrected(*args,**kwargs);count+=1
    if count%192==0:print('diagnostic reconstructed packages',count,flush=True)
    return result

A.check_package=instrumented
report=dict(classification='diagnostic_only_not_official_audit',original_frozen_failure='all-channel scores',archive=str(args.archive.resolve()),manifest_sha256=A.byte_sha((args.archive/'manifest.json').read_bytes()),script_sha256=A.byte_sha(Path(__file__).read_bytes()),correction='In memory only: nextafter toward positive infinity before nine-place score and ordinary price-key rounding; no verdict, cost, count, graph, sample, source or archive changes.')
try:
    report['reconstruction']=A.audit_directory(args.archive)
    report['status']='diagnostic_reconstruction_passed'
except Exception as error:
    report.update(status='diagnostic_reconstruction_failed',error_type=type(error).__name__,error=str(error))
    raise
finally:
    report['reconstructed_packages']=count
    report['source_bytes_unchanged']=source_hashes=={p:A.byte_sha((A.ROOT/p).read_bytes()) for p in source_hashes}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
