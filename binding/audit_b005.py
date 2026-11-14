"""Independent ledger and policy audit of stored B005 trajectories."""
import argparse
import copy
import math
from pathlib import Path
from binding import run_b005 as R
from binding import source_lab_b005 as L


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit_output(case, output, policy, budget, branch, noise=False):
    """Reconstruct decisions from public evidence, never invoke the policy engine."""
    rows = {r['id']: r for r in case['records']}
    require(len(rows) == len(case['records']), 'duplicate claim identifier')
    sources = copy.deepcopy(case['sources'])
    source_order = case['source_order']
    require(len(source_order) == len(sources) and set(source_order) == set(sources), 'source order inventory')
    values = {c: r['value'] for c, r in rows.items()}
    def grounds(cid, seen=None):
        seen = set() if seen is None else seen
        require(cid not in seen, 'cyclic recipe')
        r = rows[cid]
        result = set(r['source_ids'])
        for parent in r['parents']:
            result |= grounds(parent, seen | {cid})
        return result
    original = {c: grounds(c) for c in rows}
    if branch == 'measurement':
        groups = [[c] for c in rows if not L.close(values[c], case['measurements'][c]['value'])]
    else:
        groups = [p for p in case['pairs'] if not L.close(values[p[0]], values[p[1]])]
    depths = sorted({rows[g[0]]['depth'] for g in groups})
    depth = depths[case['seed'] % len(depths)] if depths else None
    ids = sorted({c for g in groups if rows[g[0]]['depth'] == depth for c in g})
    def view(cid):
        if policy in ('Gx', 'Lx'):
            if policy == 'Lx':
                return {'request:masked'} if cid in ids else original[cid]
            def graph(c):
                if c in ids:
                    return {'request:masked'}
                result = set(rows[c]['source_ids'])
                for p in rows[c]['parents']:
                    result |= graph(p)
                return result
            return graph(cid)
        if policy.startswith(('G', 'L')):
            return original[cid]
        return set(rows[cid]['source_ids']) if policy == 'F' else set()
    initial = [view(c) for c in ids]
    candidates = (set.intersection(*initial) if initial and branch == 'measurement' and
                  policy.startswith(('G', 'L')) else set.union(*initial) if initial else set())
    if policy == 'T':
        candidates = set(sources)
    if policy == 'N':
        candidates = set()
    initial_candidates = sorted(candidates)
    limit = budget * case['budget_unit']
    spent, checked, culprit, repaired, checks = 0, set(), None, [], 0
    fallback, repeat, resolved = False, False, not ids
    expected = []
    trace = iter(output['trace'])
    def event(record, numeric=()):
        actual = next(trace, None)
        require(actual is not None, 'missing action')
        require(actual.keys() == record.keys(), 'action fields')
        for key, value in record.items():
            if key in numeric and value is not None:
                require(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-15), 'action numeric '+key)
            else:
                require(actual[key] == value, 'action '+key)
        expected.append(actual)
    while ids and policy != 'N' and not resolved:
        tests = []
        for sid in sorted(candidates & sources.keys() - checked):
            tests.append(dict(kind='check_source', id=sid, candidate=sid,
                              cost=100 if sources[sid]['kind'] == 'method' else 30))
            for c, r in rows.items():
                if r['step_id'] == sid:
                    tests.append(dict(kind='check_claim', id=c, candidate=sid,
                                      cost=L.OPERATION_COST[r['operation']]+1))
        if not tests:
            if policy == 'F' and not fallback:
                candidates = sources.keys() - checked
                fallback = True
                continue
            if branch == 'measurement' and spent + 2*case['budget_unit'] <= limit:
                cost = 2*case['budget_unit']
                spent += cost
                repeat = True
                resolved = all(L.close(values[c], case['measurements'][c]['repeat_value']) for c in ids)
                event(dict(kind='repeat_measurement', id='measurement', cost=cost, resolved=resolved))
            break
        affordable = [t for t in tests if spent+t['cost'] <= limit]
        if not affordable:
            break
        for t in affordable:
            t['score'] = round(1/t['cost'], 9)
        def rank(t):
            if policy == 'T':
                position = (len(sources)+rows[t['id']]['timestamp'] if t['kind'] == 'check_claim'
                            else source_order.index(t['candidate']))
                return (-position, t['id'], t['kind'])
            if policy == 'F':
                return (t['candidate'] not in candidates, t['candidate'], t['cost'], t['id'])
            return (-t['score'], t['kind'] != 'check_claim', t['id'], t['candidate'])
        selected = min(affordable, key=rank)
        sid = selected['candidate']
        fresh = copy.deepcopy(case['catalog'][sid])
        value = None
        if selected['kind'] == 'check_source':
            bad = sources[sid] != fresh
        else:
            corrected = dict(sources)
            corrected[sid] = fresh
            value = L.rederive(selected['id'], case, values, corrected, noise)[0]
            bad = not L.close(value, values[selected['id']])
        checks += 1
        spent += selected['cost']
        checked.add(sid)
        event(dict(**selected, bad=bad, checked_value=value, candidates=sorted(candidates)), ('checked_value',))
        if bad:
            require(sid == case['culprit'], 'false blame')
            culprit = sid
            sources[sid] = fresh
            break
        candidates.discard(sid)
    if culprit is not None:
        targets = list(rows) if policy == 'T' else [c for c in rows if culprit in view(c)]
        for cid in sorted(targets, key=lambda c: (rows[c]['depth'], c)):
            visited, used = set(), set()
            local = dict(values)
            def rebuild(c):
                if c in visited:
                    return
                for p in rows[c]['parents']:
                    rebuild(p)
                used.update(rows[c]['source_ids'])
                local[c] = L.rederive(c, case, local, sources, noise)[0]
                visited.add(c)
            # Count the recipe before executing or granting any repair.
            def recipe(c):
                if c not in visited:
                    visited.add(c)
                    used.update(rows[c]['source_ids'])
                    for p in rows[c]['parents']:
                        recipe(p)
            recipe(cid)
            cost = len(used)+sum(L.OPERATION_COST[rows[c]['operation']] for c in visited)
            if spent+cost > limit:
                continue
            visited.clear()
            rebuild(cid)
            values[cid] = local[cid]
            spent += cost
            repaired.append(cid)
            event(dict(kind='repair', id=cid, cost=cost, value=values[cid]), ('value',))
        resolved = (all(L.close(values[c], case['measurements'][c]['value']) for c in ids)
                    if branch == 'measurement' else all(L.close(values[a], values[b])
                    for a, b in case['pairs'] if a in ids or b in ids))
        candidates = {culprit} if not resolved else set()
    require(next(trace, None) is None, 'extra action')
    suspected = (sorted(rows) if policy == 'T' and candidates else
                 sorted(c for c in rows if candidates & view(c))) if not resolved and policy != 'N' else []
    bad_ids = sorted(c for c in rows if not L.close(values[c], case['truth'][c]))
    protected = dict(signal_ids=ids, initial_candidates=initial_candidates, checked=sorted(checked),
                     culprit=culprit, accusations=[culprit] if culprit else [], suspected=suspected,
                     repaired=repaired, resolved=resolved, budget=limit, cost=spent,
                     remaining_bad_ids=bad_ids,
                     actions=[{k:v for k,v in t.items() if k not in ('score','checked_value','value')} for t in expected])
    require(output['protected'] == protected, 'protected ledger')
    require(0 <= spent <= limit, 'budget')
    metrics = dict(remaining_bad=len(bad_ids), withdrawn_good=0, false_accusations=0,
                   cost=spent, unused_budget=limit-spent, correctly_localized=int(culprit is not None),
                   checks_to_localization=checks if culprit else 0, checks=checks,
                   suspected=len(suspected), resolved=int(resolved), repeated_measurement=int(repeat),
                   initial_bad=len(case['inaccurate']), repaired=len(repaired),
                   signal_depth=max((rows[c]['depth'] for c in ids), default=0),
                   localization_opportunities=int(case['culprit'] is not None and bool(ids)))
    for k,v in metrics.items():
        require(output['metrics'][k] == v, 'metric '+k)
    require(output['diagnostics']['final_values'] == values, 'final values')
    require(output['diagnostics']['signal_observations'] ==
            ({c:case['measurements'][c]['value'] for c in ids} if branch == 'measurement' else {}), 'signal observations')
    sizes = []
    for c, r in rows.items():
        item = {k:r[k] for k in ('id','unit','depth','timestamp','step_id')}
        if policy.startswith('G'):
            item.update(parents=[] if policy == 'Gx' and c in ids else r['parents'],
                        sources=['request:masked'] if policy == 'Gx' and c in ids else sorted(r['source_ids']))
        elif policy.startswith('L'):
            item['labels'] = sorted(view(c))
        elif policy == 'F':
            item['sources'] = sorted(r['source_ids'])
        item['request'] = 'request:placebo' if policy in ('Gxprime','Lxprime') and c in ids else 'request:original'
        sizes.append(len(R.canonical(item))+8)
    require(output['diagnostics']['record_bytes'] == sizes, 'record bytes')
    source_bytes = len(R.canonical(case['sources']))+len(R.canonical(source_order))
    require(output['metrics']['memory_bytes'] == sum(sizes)+source_bytes, 'memory bytes')
    require(output['metrics']['capacity_bytes'] == len(rows)*16384+source_bytes, 'capacity')
    require(max(sizes) <= 16384, 'memory budget')
    return True


def audit_archive(archive, faults, noise=False):
    require(archive['intake_sha256'] == R.B.digest(archive['inputs']), 'intake identity')
    require(set(archive['cases']) == {f+'|'+s for f in faults for s in L.SCOPES}, 'case set')
    count = 0
    conditions = {f'{p}|{b}|{s}' for p in R.B.POLICIES for b in R.B.BUDGETS for s in R.B.SIGNALS}
    for name, item in archive['cases'].items():
        case = item['series']
        require(name == case['fault']+'|'+case['scope'], 'case identity')
        require(case['seed'] == archive['inputs']['seed'], 'seed identity')
        rebuilt = L.build_case(archive['inputs'], case['fault'], case['scope'], noise)
        require(case == rebuilt, 'physical series reconstruction')
        require(set(item['conditions']) == conditions, 'condition set')
        require(set(item['conditions'].values()) == set(item['outputs']), 'output references')
        for ref, output in item['outputs'].items():
            require(R.B.digest(output) == ref, 'output digest')
        for condition, ref in item['conditions'].items():
            policy, budget, branch = condition.split('|')
            audit_output(case, item['outputs'][ref], policy, int(budget), branch, noise)
            count += 1
        for b in R.B.BUDGETS:
            for s in R.B.SIGNALS:
                g,l = [item['outputs'][item['conditions'][f'{p}|{b}|{s}']] for p in ('G','L')]
                require(g['protected'] == l['protected'], 'G/L decisions')
                for k in g['metrics'].keys()-{'memory_bytes','capacity_bytes'}:
                    require(g['metrics'][k] == l['metrics'][k], 'G/L metric '+k)
    return count


def audit(directory):
    directory = Path(directory)
    manifest = R.read(directory/'manifest.json')
    R.verify_runtime_manifest(manifest, override=True)
    require(manifest['phase'] in ('development', 'confirmation'), 'phase')
    R.git('merge-base', '--is-ancestor', manifest['protocol_commit'], manifest['git_head'])
    for path, digest in manifest['source_sha256'].items():
        require(R.hashlib.sha256(R.subprocess.check_output(['git', 'show', manifest['git_head']+':'+path], cwd=R.ROOT)).hexdigest() == digest,
                'committed source identity '+path)
    seeds = R.DEVELOPMENT if manifest['phase'] == 'development' else R.CONFIRMATION
    require(manifest['seeds'] == list(seeds), 'seed cohort')
    require(manifest['faults'] == list(L.FAULTS), 'fault cohort')
    require(manifest['scopes'] == list(L.SCOPES), 'scope cohort')
    require(set(manifest['files']) == {f'series-{s}.json.gz' for s in seeds}, 'file set')
    count = 0
    archives = []
    for seed in seeds:
        path = directory/f'series-{seed}.json.gz'
        R.verify_file(path, manifest['files'][path.name])
        archive = R.read(path)
        require(archive['inputs']['seed'] == seed, 'archive seed')
        require(manifest['intakes'][str(seed)] == archive['intake_sha256'], 'manifest intake')
        count += audit_archive(archive, manifest['faults'], manifest['noise'])
        archives.append(archive)
    require(R.read(directory/'summary.json') == R.summarize(archives), 'summary aggregation')
    if manifest['phase'] == 'development':
        require(R.read(directory/'gate.json') == R.gate(archives), 'development gate accounting')
    return dict(passed=True, series=len(seeds), conditions=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(R.canonical(audit(args.directory)).decode())


if __name__ == '__main__':
    main()
