"""Compact B005 runs, preregistration gates and cross-platform replay."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import zlib

import numpy as np
import scipy
from binding import diagnose_b005 as B

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'binding/protocols/B005-error-localization.md'
SOURCES = ('binding/source_lab_b005.py', 'binding/diagnose_b005.py',
           'binding/run_b005.py', 'binding/source_lab_b004.py',
           'vendor/sphere-thermal-cli/sphere_thermal/response.py')
DEVELOPMENT = tuple(range(20270401, 20270413))
CONFIRMATION = tuple(range(20270413, 20270425))
PRIMARY = ('remaining_bad', 'withdrawn_good', 'false_accusations', 'cost')
CORRUPTIONS = ('units', 'version', 'regime', 'intermediate')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b'\n'
    if path.suffix == '.gz':
        data = gzip.compress(data, compresslevel=9, mtime=0)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def file_hashes(path):
    return dict(compressed_sha256=sha_file(path),
                canonical_json_sha256=hashlib.sha256(canonical(read(path))).hexdigest())


def verify_file(path, expected):
    if file_hashes(path) != expected:
        raise ValueError('Archive integrity mismatch: ' + str(path))


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.system(), machine=platform.machine(), zlib=zlib.ZLIB_VERSION)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def committed_file(path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    git('ls-files', '--error-unmatch', relative)
    git('diff', '--exit-code', 'HEAD', '--', relative)
    return git('log', '-1', '--format=%H', '--', relative)


def source_metadata():
    for source in SOURCES:
        committed_file(ROOT / source)
    protocol_commit = committed_file(ROOT / PROTOCOL)
    return dict(protocol_sha256=sha_file(ROOT / PROTOCOL), protocol_commit=protocol_commit,
                source_sha256={p: sha_file(ROOT / p) for p in SOURCES},
                git_head=git('rev-parse', 'HEAD'))


def verify_runtime_manifest(manifest, override=False):
    current = source_metadata()
    for key in ('protocol_sha256', 'protocol_commit', 'source_sha256'):
        if manifest[key] != current[key]:
            raise ValueError('Replay source or protocol identity mismatch: ' + key)
    if manifest['runtime'] != runtime() and not override:
        raise ValueError('Runtime differs; explicit --runtime-override required')


def make_archive(inputs, faults, noise=False):
    cases = {}
    for fault in faults:
        for scope in B.SCOPES:
            case = B.build_case(inputs, fault, scope, noise)
            outputs, conditions = {}, {}
            for policy in B.POLICIES:
                for budget in B.BUDGETS:
                    for signal in B.SIGNALS:
                        output = B.simulate(case, policy, budget, signal, noise)
                        key = B.digest(output)
                        outputs.setdefault(key, output)
                        conditions[f'{policy}|{budget}|{signal}'] = key
            cases[f'{fault}|{scope}'] = dict(series=case, outputs=outputs, conditions=conditions)
    return dict(inputs=inputs, intake_sha256=B.digest(inputs), cases=cases)


def actions(output):
    """Only selected operations, not intermediate numerical scores."""
    if 'actions' in output['protected']:
        return output['protected']['actions']
    return [{k: e[k] for k in ('kind', 'id', 'target', 'action') if k in e}
            for e in output['trace']]


def _cell(rows):
    if not rows:
        raise ValueError('Empty summary cell')
    totals, means, errors = {}, {}, {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        if not all(isinstance(v, (int, float)) for v in values):
            raise ValueError('Metrics must be flat numeric values: ' + key)
        totals[key] = sum(values)
        mean = math.fsum(values) / len(values)
        means[key] = mean
        errors[key] = (math.sqrt(math.fsum((x-mean)**2 for x in values) /
                                 (len(values)*(len(values)-1))) if len(values)>1 else 0.)
    return dict(series=len(rows), total=totals, mean=means, standard_error=errors)


def summarize(archives):
    scope_cells, primary_rows = {}, {}
    action_differences, action_conditions = 0, 0
    for archive in archives:
        # Scope conditions share the series input; aggregate before standard errors.
        series_primary = {}
        for case_name, case in archive['cases'].items():
            fault, scope = case_name.split('|')
            for condition, ref in case['conditions'].items():
                output = case['outputs'][ref]
                scope_cells.setdefault(case_name+'|'+condition, []).append(output['metrics'])
                key = fault+'|'+condition
                target = series_primary.setdefault(key, {k: 0 for k in output['metrics']})
                for metric, value in output['metrics'].items():
                    target[metric] += value
            for budget in B.BUDGETS:
                for branch in B.SIGNALS:
                    g = case['outputs'][case['conditions'][f'G|{budget}|{branch}']]
                    l = case['outputs'][case['conditions'][f'L|{budget}|{branch}']]
                    action_conditions += 1
                    action_differences += actions(g) != actions(l)
        for key, row in series_primary.items():
            primary_rows.setdefault(key, []).append(row)
    primary_cells = {key: _cell(rows) for key, rows in sorted(primary_rows.items())}
    failures_l, failures_f = [], []
    for fault in B.FAULTS:
        for budget in B.BUDGETS:
            for signal in B.SIGNALS:
                template = f'{fault}|{{p}}|{budget}|{signal}'
                if template.format(p='G') not in primary_cells:
                    continue
                vals = {p: primary_cells[template.format(p=p)]['total'] for p in ('G', 'L', 'F')}
                for measure in PRIMARY:
                    if vals['L'][measure] > vals['G'][measure]:
                        failures_l.append(dict(cell=template, measure=measure))
                    if any(vals['F'][measure] > vals[p][measure] for p in ('G', 'L')):
                        failures_f.append(dict(cell=template, measure=measure))
    return dict(independent_series=len(archives),
                cells={key: _cell(rows) for key, rows in sorted(scope_cells.items())},
                primary_cells=primary_cells,
                closure_L_no_worse_than_G=bool(primary_cells) and not failures_l,
                closure_F_no_worse_than_G_and_L=bool(primary_cells) and not failures_f,
                closure_L_failure_cells=failures_l, closure_F_failure_cells=failures_f,
                G_L_action_differences=action_differences, G_L_action_conditions=action_conditions)


def gate(archives):
    exposure = {f: {scope: {str(d): dict(affected=0, inaccurate=0)
                    for d in ((2, 3, 4) if f=='intermediate' else (1, 2, 3, 4))}
                    for scope in B.SCOPES} for f in CORRUPTIONS}
    contrasts = {k: 0 for k in ('G_F', 'L_F', 'G_Gx', 'L_Lx')}
    placebo, intermediate_checks, signals = 0, 0, set()
    for archive in archives:
        for name, case in archive['cases'].items():
            fault, scope = name.split('|')
            series = case['series']
            records = {r['id']: r for r in series['records']}
            if fault in exposure:
                for ident in series['affected']:
                    record = records[ident]
                    depth = str(record['depth'])
                    if depth not in exposure[fault][scope]:
                        continue
                    row = exposure[fault][scope][depth]
                    row['affected'] += 1
                    row['inaccurate'] += not B.close(record['value'], series['truth'][ident])
            for budget in B.BUDGETS:
                for branch in B.SIGNALS:
                    outputs = {p: case['outputs'][case['conditions'][f'{p}|{budget}|{branch}']]
                               for p in B.POLICIES}
                    for key in contrasts:
                        p, q = key.split('_')
                        contrasts[key] += actions(outputs[p]) != actions(outputs[q])
                    placebo += actions(outputs['G']) != actions(outputs['Gxprime'])
                    placebo += actions(outputs['L']) != actions(outputs['Lxprime'])
                    for p in ('G', 'L'):
                        out = outputs[p]
                        for ident in out['protected'].get('signal_ids', []):
                            if ident in records:
                                signals.add(records[ident]['depth'])
                        for event in out['trace']:
                            ident = event.get('id', event.get('target'))
                            if event.get('kind') in ('check_claim', 'rederive', 'claim_check') and ident in records:
                                intermediate_checks += records[ident]['depth'] >= 2
    admitted = [f for f, scopes in exposure.items()
                if all(row['affected'] >= 10 and row['inaccurate'] >= 5
                       for depths in scopes.values() for row in depths.values())]
    return dict(exposure=exposure, admitted_faults=admitted,
                discernibility_counts=contrasts, placebo_mismatches=placebo,
                intermediate_checks=intermediate_checks, signal_depths=sorted(signals),
                pass_gate=(len(admitted)==len(CORRUPTIONS) and all(contrasts.values()) and
                           placebo==0 and intermediate_checks>0 and {2, 3, 4} <= signals))


def run(output, phase, replay=None, noise=False, override=False, freeze=None):
    if phase not in ('development', 'confirmation'):
        raise ValueError('Unknown phase')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty; existing results are not overwritten')
    metadata = source_metadata()
    freeze_record = None
    faults = list(B.FAULTS)
    if phase=='confirmation':
        if freeze is None:
            raise ValueError('Confirmation requires committed freeze')
        freeze_path = Path(freeze).resolve()
        freeze_commit = committed_file(freeze_path)
        freeze_record = read(freeze_path)
        if not freeze_record['gate']['pass_gate']:
            raise ValueError('Development gate failed')
        verify_runtime_manifest(freeze_record['development_manifest'], override)
        if set(freeze_record['gate']['admitted_faults']) != set(CORRUPTIONS):
            raise ValueError('All prespecified fault families must be admitted')
    seeds = DEVELOPMENT if phase=='development' else CONFIRMATION
    manifest = dict(schema=1, phase=phase, noise=bool(noise), runtime=runtime(),
                    runtime_override=bool(override), seeds=list(seeds), faults=faults,
                    scopes=list(B.SCOPES), **metadata, files={}, intakes={})
    if freeze_record is not None:
        manifest.update(freeze_sha256=sha_file(freeze), freeze_commit=freeze_commit)
    previous = None
    if replay:
        previous = read(Path(replay)/'manifest.json')
        verify_runtime_manifest(previous, override)
        for key in ('phase', 'faults', 'scopes', 'seeds', 'freeze_sha256'):
            if previous.get(key) != manifest.get(key):
                raise ValueError('Replay cohort mismatch: ' + key)
        manifest['replay_manifest_sha256'] = sha_file(Path(replay)/'manifest.json')
    start = time.monotonic()
    archives = []
    for seed in seeds:
        name = f'series-{seed}.json.gz'
        if previous is not None:
            path = Path(replay)/name
            verify_file(path, previous['files'][name])
            inp = read(path)['inputs']
            if B.digest(inp) != previous['intakes'][str(seed)]:
                raise ValueError('Replay input checksum mismatch')
        else:
            inp = B.intake(seed)
        archive = make_archive(inp, faults, noise)
        write(out/name, archive)
        manifest['files'][name] = file_hashes(out/name)
        manifest['intakes'][str(seed)] = archive['intake_sha256']
        archives.append(archive)
        write(out/'manifest.partial.json', manifest)
        print(json.dumps(dict(seed=seed, seconds=round(time.monotonic()-start, 3),
                              bytes=(out/name).stat().st_size)), flush=True)
    summary = summarize(archives)
    write(out/'summary.json', summary)
    if phase=='development':
        write(out/'gate.json', gate(archives))
    manifest['elapsed_seconds'] = time.monotonic()-start
    write(out/'manifest.json', manifest)
    (out/'manifest.partial.json').unlink()
    return summary


def numeric_differences(a, b, path='', out=None):
    out = {} if out is None else out
    if isinstance(a, dict) and isinstance(b, dict) and a.keys()==b.keys():
        for key in a:
            numeric_differences(a[key], b[key], path+'/'+key, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a)==len(b):
        for x, y in zip(a, b):
            numeric_differences(x, y, path+'/*', out)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            row = out.setdefault(path, dict(count=0, max_abs=0., max_rel=0.))
            difference = abs(a-b)
            row['count'] += 1
            row['max_abs'] = max(row['max_abs'], difference)
            row['max_rel'] = max(row['max_rel'], difference/max(abs(a), abs(b), 1e-300))
    elif a != b:
        raise ValueError('Structural or non-numeric drift at ' + path)
    return out


def compare(first, second):
    first, second = Path(first), Path(second)
    m1, m2 = read(first/'manifest.json'), read(second/'manifest.json')
    fields = ('phase', 'seeds', 'faults', 'scopes', 'source_sha256', 'protocol_sha256',
              'protocol_commit', 'intakes', 'freeze_sha256', 'freeze_commit')
    differences = [key for key in fields if m1.get(key) != m2.get(key)]
    diagnostics, count = {}, 0
    if m1['files'].keys() != m2['files'].keys():
        differences.append('file_set')
    for name in sorted(m1['files'].keys() & m2['files'].keys()):
        for base, manifest in ((first, m1), (second, m2)):
            try:
                verify_file(base/name, manifest['files'][name])
            except ValueError:
                differences.append(name+':checksum')
        a, b = read(first/name), read(second/name)
        if a['inputs'] != b['inputs'] or a['intake_sha256'] != b['intake_sha256']:
            differences.append(name+':inputs')
        if a['cases'].keys() != b['cases'].keys():
            differences.append(name+':case_set')
        for case_name in a['cases'].keys() & b['cases'].keys():
            ca, cb = a['cases'][case_name], b['cases'][case_name]
            try:
                numeric_differences(ca['series'], cb['series'], case_name+'/initial', diagnostics)
            except ValueError as error:
                differences.append(name+':'+str(error))
            if ca['conditions'].keys() != cb['conditions'].keys():
                differences.append(name+':conditions')
            for key in ca['conditions'].keys() & cb['conditions'].keys():
                x = ca['outputs'][ca['conditions'][key]]
                y = cb['outputs'][cb['conditions'][key]]
                count += 1
                if x['protected'] != y['protected'] or x['metrics'] != y['metrics'] or actions(x) != actions(y):
                    differences.append(name+':'+case_name+':'+key)
                for field in ('trace', 'diagnostics'):
                    try:
                        numeric_differences(x[field], y[field], case_name+'/'+field, diagnostics)
                    except ValueError as error:
                        differences.append(name+':'+str(error))
    return dict(decisions_pass=not differences, differing_decisions=differences,
                compared_conditions=count, numerical_diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'confirmation'), default='development')
    parser.add_argument('--replay')
    parser.add_argument('--last-bit-noise', '--noise', dest='noise', action='store_true')
    parser.add_argument('--runtime-override', action='store_true')
    parser.add_argument('--freeze')
    parser.add_argument('--compare', nargs=2)
    args = parser.parse_args()
    if args.compare:
        result = compare(*args.compare)
        write(args.output, result)
        if not result['decisions_pass']:
            raise SystemExit(3)
    else:
        run(args.output, args.phase, args.replay, args.noise, args.runtime_override, args.freeze)


if __name__=='__main__':
    main()
