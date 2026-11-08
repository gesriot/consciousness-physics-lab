"""Deterministic compressed archives and validated attention replay."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from lab.executor import request_hash
from lab.guard import Request


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if path.suffix == '.gz':
        with path.open('wb') as stream:
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=9, mtime=0) as compressed:
                compressed.write(raw)
    else:
        path.write_bytes(raw)


def load_replay(path, experiment, count):
    path = Path(path)
    data = read_json(path)
    records = data.get('intake_channel', [])
    if data.get('experiment') != experiment or len(records) != count:
        raise ValueError('Replay experiment or task count mismatch')
    requests = []
    for index, record in enumerate(records):
        if record['task_id'] != index:
            raise ValueError('Replay task order mismatch')
        fields = dict(record['request'])
        fields['z'] = complex(*fields['z'])
        request = Request(**fields)
        if not all(math.isfinite(v) for v in (request.z.real, request.z.imag,
                                              request.mean_time, request.tolerance)):
            raise ValueError('Nonfinite replay input')
        if 'request_sha256' in record and request_hash(request) != record['request_sha256']:
            raise ValueError('Replay input digest mismatch')
        requests.append(request)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return data, requests, dict(source_sha256=digest.hexdigest(),
                                experiment=experiment, task_count=count)
