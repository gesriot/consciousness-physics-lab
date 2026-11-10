"""Prospectively frozen cross-instrument qualification on development archives."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from attention.a007 import atomic_json, decode
from attention.io import read_json
from lab import frozen_pipeline as P, radial_collocation
from lab.experiment_audit import metadata
from lab.scientist import ROOT

PROTOCOL = 'binding/protocols/B002-qualification.md'
DEVELOPMENT = 'binding/results/B001/development-complete'
MAPS = {'graded_v2': 'results/E014-convergence-candidates/first/map.json.gz',
        'collocation': 'results/E015-collocation-map/map.json.gz'}
INSTRUMENTS = tuple(MAPS)
SOURCES = ['binding/qualification.py', 'lab/frozen_pipeline.py',
           'lab/convergence_candidates.py', 'lab/surface_order.py',
           'lab/radial_collocation.py', 'lab/extrapolation.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def perturb(value, noise):
    value = complex(value)
    if not noise:
        return value
    return complex(np.nextafter(value.real, np.inf), np.nextafter(value.imag, -np.inf))


def decision(discrepancy, uncertainty, tolerance):
    if discrepancy + uncertainty <= tolerance:
        return 'accept', tolerance - discrepancy - uncertainty
    if discrepancy - uncertainty > tolerance:
        return 'violation_detected', discrepancy - uncertainty - tolerance
    return 'inconclusive', None


def protected_step(instrument, values, value, tolerance, grid=None):
    if instrument not in INSTRUMENTS:
        raise ValueError('Unknown instrument')
    finite = all(math.isfinite(z.real) and math.isfinite(z.imag) for z in values)
    finite &= all(math.isfinite(x) for x in (value.real, value.imag, tolerance))
    if not finite or tolerance <= 0:
        return dict(verdict='inconclusive', convergence_pass=False, decision_pass=False,
                    finite=False, native_rounding=None, uncertainty=None,
                    convergence_margin=None, decision_margin=None)
    difference = abs(values[-1] - values[-2])
    scale = max(1., abs(values[-1]))
    if instrument == 'graded_v2':
        native = P.v2_step(values, grid, value, tolerance)
        rounding = float(native['rounding'])
        contraction = .5 * abs(values[-2] - values[-3]) - difference
        converging = bool(native['converging'])
        uncertainty = float(native['uncertainty_estimate'])
    else:
        rounding = float(32 * np.finfo(float).eps * 128**4 * scale)
        contraction = tolerance / 8 - difference
        converging = difference <= tolerance / 8
        uncertainty = 2 * difference + rounding
    discrepancy = abs(value - values[-1])
    raw, margin = decision(discrepancy, uncertainty, tolerance)
    convergence_pass = bool(converging and contraction > 4 * rounding)
    decision_pass = bool(margin is not None and margin > 4 * rounding)
    return dict(verdict=raw if convergence_pass and decision_pass else 'inconclusive',
                unguarded_verdict=raw, convergence_pass=convergence_pass,
                decision_pass=decision_pass, finite=True, native_rounding=rounding,
                uncertainty=uncertainty, discrepancy=discrepancy,
                difference=difference, convergence_margin=contraction,
                decision_margin=margin, grid=grid)


def replacement(instrument, request, value, noise=False):
    if (not all(math.isfinite(x) for x in (request.z.real, request.z.imag,
                                         request.mean_time, request.tolerance))
            or request.z.real < 0 or request.mean_time <= 0 or request.tolerance <= 0):
        return dict(verdict='inconclusive', reason='outside_domain', steps=[], values=[])
    values, steps = [], []
    if instrument == 'graded_v2':
        for grid in P.CONFIG.grids:
            values.append(perturb(P.model_response('fv', request.z, request.mean_time, grid, 'sphere'), noise))
            if grid >= P.CONFIG.first_decisive_grid:
                steps.append(protected_step(instrument, values, value, request.tolerance, grid))
                if steps[-1]['verdict'] != 'inconclusive':
                    break
    elif instrument == 'collocation':
        values = [perturb(radial_collocation.response(request.z, request.mean_time, n), noise)
                  for n in (64, 128)]
        steps = [protected_step(instrument, values, value, request.tolerance)]
    else:
        raise ValueError('Unknown instrument')
    return dict(verdict=steps[-1]['verdict'], steps=steps,
                values=[[z.real, z.imag] for z in values])


def map_points(path):
    """Stream the archived root points array, checking gzip integrity to its end."""
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        buffer = ''
        while True:
            block = stream.read(65536)
            if not block:
                raise ValueError('Missing points array')
            buffer += block
            match = re.search(r'"points"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > 16777216:
                raise ValueError('Oversized map header')
        expect_item = True
        any_items = False
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                buffer = stream.read(65536)
                if not buffer:
                    raise ValueError('Truncated points array')
                continue
            if buffer[0] == ']':
                if expect_item and any_items:
                    raise ValueError('Trailing comma in points array')
                # Consume the remaining gzip bytes to verify the checksum.
                for _ in iter(lambda: stream.read(65536), ''):
                    pass
                return
            if not expect_item:
                if buffer[0] != ',':
                    raise ValueError('Missing point separator')
                buffer = buffer[1:]
                expect_item = True
                continue
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(65536)
                if not block:
                    raise ValueError('Truncated point') from None
                buffer += block
                if len(buffer) > 16777216:
                    raise ValueError('Oversized map point')
                continue
            if not isinstance(row, dict):
                raise ValueError('Map point must be an object')
            buffer = buffer[end:]
            expect_item = False
            any_items = True
            yield row


def map_audit(instrument, output, noise=False):
    path = ROOT / MAPS[instrument]
    summary = dict(instrument=instrument, source=MAPS[instrument], source_sha256=sha(path),
                   points=0, steps=0, protected_convergence=0, dangerous_misses=0,
                   max_error_over_uncertainty=0., examples=[])
    decisions_hash = hashlib.sha256()
    for index, row in enumerate(map_points(path)):
        values = [perturb(complex(*z), noise) for z in row['values']]
        reference = complex(*row['reference'])
        if instrument == 'graded_v2':
            checks = [(P.CONFIG.grids[i], values[:i+1], 1.) for i in range(3, len(values))]
        else:
            checks = [(None, values, 10.**k) for k in range(-7, 0)]
        for grid, path_values, tolerance in checks:
            step = protected_step(instrument, path_values, reference, tolerance, grid)
            decisions_hash.update(json.dumps([row['coordinate'], grid, tolerance,
                step['verdict'], step['convergence_pass'], step['decision_pass']],
                separators=(',', ':')).encode())
            covered = abs(path_values[-1] - reference) <= step['uncertainty']
            dangerous = step['convergence_pass'] and not covered
            summary['steps'] += 1
            summary['protected_convergence'] += int(step['convergence_pass'])
            summary['dangerous_misses'] += int(dangerous)
            if step['convergence_pass']:
                summary['max_error_over_uncertainty'] = max(summary['max_error_over_uncertainty'],
                    abs(path_values[-1] - reference) / step['uncertainty'])
            if dangerous and len(summary['examples']) < 20:
                summary['examples'].append(dict(coordinate=row['coordinate'], grid=grid,
                                               tolerance=tolerance, step=step))
        summary['points'] = index + 1
        if (index + 1) % 1000 == 0:
            atomic_json(output / (instrument + '-map-progress.json'), summary)
    if not summary['points'] or not summary['protected_convergence']:
        raise ValueError('Empty qualification coverage challenge')
    summary['operational_sha256'] = decisions_hash.hexdigest()
    atomic_json(output / (instrument + '-map.json'), summary)
    return summary


def development_candidates(series):
    directory = ROOT / DEVELOPMENT / ('series-%02d' % series)
    ledger = read_json(directory / 'ledger.json.gz')
    accepted = {record['id'] for records in ledger['records'].values() for record in records}
    intake = read_json(directory / 'intake.json')
    packages = {p: read_json(directory / ('none-%d.json.gz' % p)) for p in range(4)}
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [directory / 'ledger.json.gz',
              directory / 'intake.json', *(directory / ('none-%d.json.gz' % j) for j in range(4))]}
    for rid in sorted(accepted):
        package, index = map(int, rid.split(':'))
        info = packages[package]['inputs'][index]
        request = decode(intake['requests'][info['task_id']]['request'])
        value = complex(*info['outcome']['value'])
        accurate = abs(value - complex(*info['reference'])) <= request.tolerance
        if accurate != info['accurate']:
            raise AssertionError('Archived physical-accuracy mismatch')
        yield dict(id='%02d:%s' % (series, rid), request=request,
                   value=value, accurate=accurate, source_hashes=hashes)


def run(output, noise=False, override=None, replay=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    audit = metadata(PROTOCOL, SOURCES, override)
    input_paths = [ROOT/p for p in MAPS.values()]
    for series in range(12):
        base = ROOT/DEVELOPMENT/('series-%02d' % series)
        input_paths += [base/'ledger.json.gz', base/'intake.json']
        input_paths += [base/('none-%d.json.gz' % i) for i in range(4)]
    audit['input_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if replay:
        saved = read_json(Path(replay)/'manifest.json')
        for key in ('protocol_sha256', 'source_sha256', 'input_sha256'):
            if saved[key] != audit[key]:
                raise ValueError('Replay differs in ' + key)
    output.mkdir(parents=True)
    atomic_json(output / 'manifest.json', dict(experiment='B002-qualification', **audit,
                last_bit_noise=noise, replay=str(replay) if replay else None))
    summary = {name: dict(accurate=0, inaccurate=0, decisive_accurate=0,
                         accurate_accept=0, false_accept=0, false_violation=0,
                         verdicts={v: 0 for v in ('accept', 'inconclusive', 'violation_detected')})
               for name in INSTRUMENTS}
    for series in range(12):
        rows = []
        for candidate in development_candidates(series):
            results = {}
            for name in INSTRUMENTS:
                result = replacement(name, candidate['request'], candidate['value'], noise)
                results[name] = result
                s = summary[name]
                correct, verdict = candidate['accurate'], result['verdict']
                s['accurate' if correct else 'inaccurate'] += 1
                s['decisive_accurate'] += int(correct and verdict != 'inconclusive')
                s['accurate_accept'] += int(correct and verdict == 'accept')
                s['false_accept'] += int(not correct and verdict == 'accept')
                s['false_violation'] += int(correct and verdict == 'violation_detected')
                s['verdicts'][verdict] += 1
            rows.append(dict(id=candidate['id'], accurate=candidate['accurate'],
                             results=results, source_hashes=candidate['source_hashes']))
        atomic_json(output / ('series-%02d.json.gz' % series), rows)
        atomic_json(output / 'progress.json', dict(completed_series=series+1, instruments=summary))
        print('Saved qualification series', series+1, flush=True)
    for name in INSTRUMENTS:
        coverage = map_audit(name, output, noise)
        s = summary[name]
        s['decisive_fraction'] = s['decisive_accurate'] / s['accurate'] if s['accurate'] else None
        s['coverage'] = coverage
        s['gates'] = dict(decisive_fraction=bool(s['accurate'] and 5*s['decisive_accurate'] >= 4*s['accurate']),
                          has_protected_accept=s['accurate_accept'] > 0,
                          no_wrong_decisions=s['false_accept'] == s['false_violation'] == 0,
                          coverage=coverage['dangerous_misses'] == 0)
        s['provisional_pass'] = all(s['gates'].values())
    result = dict(experiment='B002-qualification', instruments=summary,
                  replay_stability='pending', qualified=[], confirmation_allowed=False)
    atomic_json(output / 'summary.json', result)
    if replay:
        comparison = compare(Path(replay), output)
        atomic_json(output / 'comparison.json', comparison)
    return result


def compare(first, second):
    """Exact operational comparison with descriptive finite numerical drift."""
    stats = dict(decisions_pass=True, flags_pass=True, intake_pass=True,
                 changed_numerics=0, max_absolute_difference=0., fields=0,
                 changed_unguarded_verdicts=0, unguarded_examples=[])
    exact = {'verdict', 'convergence_pass', 'decision_pass', 'finite',
             'grid', 'accurate', 'id', 'source_hashes'}
    def walk(a, b, key=''):
        stats['fields'] += 1
        if key == 'unguarded_verdict':
            if a != b:
                stats['changed_unguarded_verdicts'] += 1
                if len(stats['unguarded_examples']) < 10:
                    stats['unguarded_examples'].append([a, b])
            return
        if key in exact:
            if a != b:
                stats['decisions_pass' if 'verdict' in key else 'flags_pass'] = False
            return
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                stats['intake_pass'] = False
                return
            for k in a:
                walk(a[k], b[k], k)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                stats['flags_pass'] = False
                return
            for x, y in zip(a, b):
                walk(x, y, key)
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                stats['flags_pass'] = False
            elif a != b:
                stats['changed_numerics'] += 1
                stats['max_absolute_difference'] = max(stats['max_absolute_difference'], abs(a-b))
        elif a != b:
            stats['flags_pass'] = False
    for series in range(12):
        filename = 'series-%02d.json.gz' % series
        walk(read_json(first/filename), read_json(second/filename))
    a, b = read_json(first/'summary.json'), read_json(second/'summary.json')
    for name in INSTRUMENTS:
        x, y = a['instruments'][name], b['instruments'][name]
        if x['gates'] != y['gates'] or x['verdicts'] != y['verdicts']:
            stats['flags_pass'] = False
        for key in ('points', 'steps', 'protected_convergence', 'dangerous_misses', 'operational_sha256', 'source_sha256'):
            if x['coverage'][key] != y['coverage'][key]:
                stats['flags_pass'] = False
    return stats


def finalize(first, repeat, noise, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    directories = list(map(Path, (first, repeat, noise)))
    manifests = [read_json(d/'manifest.json') for d in directories]
    if [m['last_bit_noise'] for m in manifests] != [False, False, True]:
        raise ValueError('Expected ordinary, ordinary replay, noisy replay in order')
    for key in ('protocol_sha256', 'source_sha256', 'input_sha256'):
        if any(m[key] != manifests[0][key] for m in manifests[1:]):
            raise ValueError('Qualification source or protocol mismatch')
    comparisons = dict(ordinary=compare(directories[0], directories[1]),
                       last_bit_noise=compare(directories[0], directories[2]))
    stable = all(c[k] for c in comparisons.values()
                 for k in ('decisions_pass', 'flags_pass', 'intake_pass'))
    baseline = read_json(directories[0]/'summary.json')
    instruments = baseline['instruments']
    for name, result in instruments.items():
        result['gates']['replay_stability'] = stable
        result['qualified'] = all(result['gates'].values())
    qualified = [name for name, result in instruments.items() if result['qualified']]
    record = dict(experiment='B002-qualification', instruments=instruments,
                  comparisons=comparisons, qualified=qualified,
                  exposure_allowed=bool(qualified), confirmation_allowed=False,
                  status='exposure_required' if qualified else 'stopped_no_qualified_instrument',
                  source_runs=[dict(path=str(d), manifest_sha256=sha(d/'manifest.json'),
                                    summary_sha256=sha(d/'summary.json')) for d in directories])
    atomic_json(output, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--last-bit-noise', action='store_true')
    parser.add_argument('--runtime-override')
    parser.add_argument('--replay')
    parser.add_argument('--finalize', nargs=3, metavar=('FIRST', 'REPEAT', 'NOISE'))
    args = parser.parse_args()
    if args.finalize:
        finalize(*args.finalize, args.output)
    else:
        run(args.output, args.last_bit_noise, args.runtime_override, args.replay)
