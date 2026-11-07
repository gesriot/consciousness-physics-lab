"""Metadata and exact-environment gates for registered experiments."""
import contextlib
import io
import hashlib
import platform
import subprocess
import numpy as np
import scipy
from lab.scientist import ROOT


def environment_metadata():
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
        scipy.show_config()
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                os=platform.system(), os_release=platform.release(), architecture=platform.machine(),
                blas_lapack_configuration=config.getvalue())


def metadata(protocol, sources, runtime_override=None):
    environment = environment_metadata()
    if runtime_override is not None and not runtime_override.strip():
        raise ValueError('Runtime override requires a nonempty audit reason')
    if tuple(environment[k] for k in ('python', 'numpy', 'scipy')) != ('3.12.14','2.3.5','1.17.0') and runtime_override is None:
        raise RuntimeError('Experiment requires pinned Python/NumPy/SciPy environment')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit tracked implementation changes before generation')
    return dict(environment=environment,runtime_override=runtime_override,git_head=git('rev-parse','HEAD'),
                protocol_commit=git('log','--diff-filter=A','--format=%H','--',protocol).splitlines()[-1],
                protocol_sha256=hashlib.sha256((ROOT/protocol).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
