"""Small synthetic checks, deliberately outside both experimental cohorts."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from binding import diagnose_b005 as B
from binding import audit_b005 as A
from binding import run_b005 as R


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = B.intake(101)
        cls.case = B.build_case(cls.inputs, 'intermediate', 'shared')
        cls.output = B.simulate(cls.case, 'G', 32, 'measurement')

    def test_ledger_independent_of_simulator(self):
        with patch.object(B, 'simulate', side_effect=AssertionError('must not run')):
            self.assertTrue(A.audit_output(self.case, self.output, 'G', 32, 'measurement'))

    def test_all_policies_branches_small_fixture(self):
        for p in B.POLICIES:
            for branch in B.SIGNALS:
                for budget in B.BUDGETS:
                    with self.subTest(p=p, branch=branch, budget=budget):
                        out = B.simulate(self.case, p, budget, branch)
                        A.audit_output(self.case, out, p, budget, branch)

    def test_clean_and_false_measurement(self):
        for fault in ('clean', 'false_measurement'):
            case = B.build_case(self.inputs, fault)
            for p in B.POLICIES:
                out = B.simulate(case, p, 32, 'measurement')
                A.audit_output(case, out, p, 32, 'measurement')

    def test_noisy_fixture(self):
        case = B.build_case(self.inputs, 'version', noise=True)
        out = B.simulate(case, 'G', 8, 'measurement', noise=True)
        A.audit_output(case, out, 'G', 8, 'measurement', noise=True)

    def test_mutated_accounting_and_decisions(self):
        mutations = [
            lambda o:o['metrics'].__setitem__('cost', -1),
            lambda o:o['metrics'].__setitem__('remaining_bad', 999),
            lambda o:o['protected'].__setitem__('budget', 0),
            lambda o:o['protected'].__setitem__('signal_ids', []),
            lambda o:o['protected'].__setitem__('culprit', 'innocent'),
            lambda o:o['trace'][0].__setitem__('cost', 0),
            lambda o:o['trace'][0].__setitem__('candidates', []),
            lambda o:o['trace'][0].__setitem__('id', 'missing'),
            lambda o:o['diagnostics'].__setitem__('signal_observations', {}),
            lambda o:o['metrics'].__setitem__('memory_bytes', 0),
        ]
        for mutate in mutations:
            out = copy.deepcopy(self.output)
            mutate(out)
            with self.assertRaises(ValueError):
                A.audit_output(self.case, out, 'G', 32, 'measurement')

    def test_archive_references(self):
        archive = R.make_archive(self.inputs, ['clean'])
        self.assertEqual(A.audit_archive(archive, ['clean']), 108)
        broken = copy.deepcopy(archive)
        item = next(iter(broken['cases'].values()))
        item['conditions']['G|2|measurement'] = 'absent'
        with self.assertRaises(ValueError):
            A.audit_archive(broken, ['clean'])

    def test_false_blame_is_measured_not_forbidden(self):
        case = copy.deepcopy(self.case)
        case['culprit'] = 'innocent'
        with self.assertRaisesRegex(ValueError, 'metric false_accusations'):
            A.audit_output(case, self.output, 'G', 32, 'measurement')
        output = copy.deepcopy(self.output)
        output['metrics']['false_accusations'] = 1
        output['metrics']['correctly_localized'] = 0
        self.assertTrue(A.audit_output(case, output, 'G', 32, 'measurement'))

    def test_repair_and_physical_series_tampering(self):
        output = copy.deepcopy(self.output)
        repair = next(e for e in output['trace'] if e['kind'] == 'repair')
        repair['value'] += 1
        with self.assertRaises(ValueError):
            A.audit_output(self.case, output, 'G', 32, 'measurement')
        archive = R.make_archive(self.inputs, ['clean'])
        case = next(iter(archive['cases'].values()))['series']
        key = next(iter(case['truth']))
        case['truth'][key] += 1
        with self.assertRaisesRegex(ValueError, 'physical series'):
            A.audit_archive(archive, ['clean'])

    def test_canonical_roundtrip_preserves_time_order(self):
        archive = R.make_archive(self.inputs, ['false_measurement'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json.gz'
            R.write(path, archive)
            restored = R.read(path)
        self.assertEqual(A.audit_archive(restored, ['false_measurement']), 108)
        case = restored['cases']['false_measurement|shared']['series']
        self.assertNotEqual(case['source_order'], list(case['sources']))

    def test_source_order_mutation(self):
        archive = R.make_archive(self.inputs, ['false_measurement'])
        case = archive['cases']['false_measurement|shared']['series']
        case['source_order'].reverse()
        with self.assertRaisesRegex(ValueError, 'physical series'):
            A.audit_archive(archive, ['false_measurement'])
        case = copy.deepcopy(self.case)
        case['source_order'][0] = case['source_order'][1]
        with self.assertRaisesRegex(ValueError, 'source order inventory'):
            A.audit_output(case, self.output, 'G', 32, 'measurement')

    def test_only_physical_outputs_allow_rounding(self):
        modified = copy.deepcopy(self.case)
        modified['records'][0]['value'] = math.nextafter(modified['records'][0]['value'], math.inf)
        cid = next(iter(modified['truth']))
        modified['truth'][cid] = math.nextafter(modified['truth'][cid], math.inf)
        for field in ('value', 'repeat_value'):
            modified['measurements'][cid][field] = math.nextafter(modified['measurements'][cid][field], math.inf)
        A.verify_series(modified, self.case)
        source = next(iter(self.case['sources']))
        mutations = [
            lambda c:c['sources'][source].__setitem__('version', 3),
            lambda c:c['records'][0].__setitem__('depth', 7),
            lambda c:c.__setitem__('budget_unit', c['budget_unit']+1),
            lambda c:c['records'][0].__setitem__('value', c['records'][0]['value']+1e-6),
            lambda c:c['records'].pop(),
        ]
        for mutate in mutations:
            modified = copy.deepcopy(self.case)
            mutate(modified)
            with self.assertRaises(ValueError):
                A.verify_series(modified, self.case)

    def test_freeze_identity_and_ancestry(self):
        development = dict(phase='development', git_head='development-head',
                           source_sha256={'source': 'hash'}, protocol_sha256='protocol-hash',
                           protocol_commit='protocol-commit')
        freeze = dict(gate=dict(pass_gate=True, admitted_faults=list(R.CORRUPTIONS)),
                      development_manifest=development)
        data = R.canonical(freeze)
        manifest = dict(development, phase='confirmation', git_head='run-head',
                        freeze_commit='freeze-commit', freeze_sha256=R.hashlib.sha256(data).hexdigest())
        def git(*args):
            return 'binding/results/B005/freeze.json' if args[0] == 'ls-tree' else ''
        with patch.object(R, 'git', side_effect=git) as command, patch.object(R.subprocess, 'check_output', return_value=data):
            A.verify_freeze(manifest)
            command.assert_any_call('merge-base', '--is-ancestor', 'freeze-commit', 'run-head')
            changed = dict(manifest, freeze_sha256='wrong')
            with self.assertRaisesRegex(ValueError, 'freeze identity'):
                A.verify_freeze(changed)
            A.verify_freeze(dict(manifest, freeze_commit='run-head'))
            command.assert_any_call('merge-base', '--is-ancestor', 'run-head', 'run-head')

    def test_gzip_and_canonical_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json.gz'
            R.write(path, {'fixture': 1})
            hashes = R.file_hashes(path)
            R.verify_file(path, hashes)
            R.write(path, {'fixture': 2})
            with self.assertRaises(ValueError):
                R.verify_file(path, hashes)


if __name__ == '__main__':
    unittest.main()
