"""Load archived requests without regenerating floating-point inputs."""
import hashlib
import json
import math
from pathlib import Path
from lab.guard import Request
from lab.executor import request_hash


def load_replay(path, experiment, count):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    key = 'tasks' if experiment == 'E007' else 'intake_channel'
    if data.get('experiment') != experiment or len(data.get(key, [])) != count:
        raise ValueError('Replay experiment or task count mismatch')
    records = data[key]
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
    return data, requests, dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                                experiment=experiment, task_count=count)


def protect_output(output, replay):
    """Never overwrite the source archive through the replay command."""
    if Path(output).exists():
        raise ValueError('Output already exists; choose a new path to preserve archived results')
    if replay is not None and Path(output).resolve() == Path(replay).resolve():
        raise ValueError('Replay output must differ from source archive')
