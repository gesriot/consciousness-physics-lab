"""Deterministic compressed archives and validated attention replay."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from lab.executor import request_hash
from lab.guard import Request


def source_path(path):
    """Resolve a historical JSON name to its lossless compressed successor."""
    path = Path(path)
    compressed = path.with_name(path.name+'.gz')
    if not path.exists() and path.suffix == '.json' and compressed.is_file():
        return compressed
    return path


def source_bytes(path):
    """Historical JSON identities retain decompressed-byte provenance hashes.

    An explicitly named .gz source still identifies its actual compressed bytes.
    """
    requested = Path(path)
    actual = source_path(requested)
    raw = actual.read_bytes()
    return gzip.decompress(raw) if actual != requested else raw


def read_json(path):
    path = source_path(path)
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
    with source_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return data, requests, dict(source_sha256=digest.hexdigest(),
                                experiment=experiment, task_count=count)
