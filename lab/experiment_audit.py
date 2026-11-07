"""Metadata and exact-environment gates for registered experiments."""
import hashlib
import platform
import subprocess
import numpy as np
import scipy
from lab.scientist import ROOT


def metadata(protocol, sources):
    environment = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    if tuple(environment.values()) != ('3.12.14','2.3.5','1.17.0'):
        raise RuntimeError('Experiment requires pinned Python/NumPy/SciPy environment')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit tracked implementation changes before generation')
    return dict(environment=environment,git_head=git('rev-parse','HEAD'),
                protocol_commit=git('log','--diff-filter=A','--format=%H','--',protocol).splitlines()[-1],
                protocol_sha256=hashlib.sha256((ROOT/protocol).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
