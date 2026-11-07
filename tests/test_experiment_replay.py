import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from lab.experiment_audit import environment_metadata, metadata
from lab.executor import request_hash
from lab.replay import load_replay, protect_output
from scripts.compare_experiment_runs import compare

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(environment={'architecture': 'one'}, git_head='a',
            protocol_commit='p', protocol_sha256='h', source_sha256={'x': 'a'},
            runs=[dict(value=1.0, accepted=True, verdict='accept', cells=512,
                       digest='abc', verification_seconds=1.0)])
        self.repeat = copy.deepcopy(self.original)

    def test_exact_retained_and_timing_excluded(self):
        self.repeat['runs'][0]['verification_seconds'] = 8.0
        self.assertFalse(compare(self.original, self.repeat)[0])
        self.repeat['runs'][0]['value'] += 1e-14
        self.assertTrue(compare(self.original, self.repeat)[0])
        self.assertFalse(compare(self.original, self.repeat, 'decisions')[0])

    def test_provenance_separate_not_silently_equal(self):
        self.repeat['environment']['architecture'] = 'two'
        self.repeat['source_sha256']['x'] = 'b'
        errors, provenance = compare(self.original, self.repeat, 'decisions')
        self.assertFalse(errors)
        self.assertEqual(len(provenance), 2)
        self.assertTrue(compare(self.original, self.repeat)[0])

    def test_decisions_counts_hashes_protocols_never_relaxed(self):
        for key, value in [('accepted', False), ('accepted', 1), ('verdict', 'reject'),
                           ('cells', 513), ('digest', 'abd')]:
            other = copy.deepcopy(self.original)
            other['runs'][0][key] = value
            self.assertTrue(compare(self.original, other, 'decisions', .9)[0])
        self.repeat['protocol_sha256'] = 'different'
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])

    def test_tolerance_and_structure(self):
        for value in (-1, 1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compare(self.original, self.repeat, 'decisions', value)
        self.repeat['runs'][0]['value'] = 1.01
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])
        self.original['runs'][0]['value'] = 0.0
        self.repeat['runs'][0]['value'] = 1e-30
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])
        del self.repeat['runs'][0]['accepted']
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])


class ReplayTests(unittest.TestCase):
    def test_archived_inputs_and_digests_are_bit_identical(self):
        for experiment, folder, count, key in [('E007', 'verifier-stress', 192, 'tasks'),
                                              ('E008', 'common-mode', 96, 'intake_channel')]:
            data, requests, provenance = load_replay(ROOT / 'results' / folder / 'results.json', experiment, count)
            self.assertEqual(provenance['task_count'], count)
            for record, request in zip(data[key], requests):
                for a, b in zip(record['request']['z'], (request.z.real, request.z.imag)):
                    self.assertEqual(a.hex(), b.hex())
                self.assertEqual(record['request']['mean_time'].hex(), request.mean_time.hex())
                self.assertEqual(record['request']['tolerance'].hex(), request.tolerance.hex())
                if experiment == 'E008':
                    self.assertEqual(record['request_sha256'], request_hash(request))

    def test_runners_use_saved_requests_before_evaluation(self):
        from lab import common_mode, verifier_stress
        from types import SimpleNamespace
        class Captured(Exception):
            pass
        for module, folder, experiment, count in [
                (common_mode, 'common-mode', 'E008', 96),
                (verifier_stress, 'verifier-stress', 'E007', 192)]:
            path = ROOT / 'results' / folder / 'results.json'
            _, requests, _ = load_replay(path, experiment, count)
            def capture(request, *args):
                self.assertEqual(request, requests[0])
                raise Captured()
            with patch.object(module, 'metadata', return_value={}):
                if experiment == 'E008':
                    with patch.object(module, 'evaluate', side_effect=capture):
                        with self.assertRaises(Captured):
                            module.run(replay=path)
                else:
                    calls = []
                    def execute(request, *args):
                        calls.append(request)
                        if len(calls) == 1:
                            return SimpleNamespace(value=1.0)
                        return capture(request)
                    with patch.object(module, 'execute', side_effect=execute):
                        with self.assertRaises(Captured):
                            module.run(replay=path)

    def test_invalid_archive_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'archive.json'
            path.write_text(json.dumps({'experiment': 'E008', 'intake_channel': []}))
            with self.assertRaises(ValueError):
                load_replay(path, 'E007', 192)
            with self.assertRaises(ValueError):
                protect_output(path, path)
            with self.assertRaises(ValueError):
                protect_output(path, None)
            protect_output(Path(tmp) / 'fresh.json', path)

    def test_environment_and_explicit_override(self):
        env = environment_metadata()
        self.assertTrue(all(env[k] for k in ('os', 'architecture', 'blas_lapack_configuration')))
        env['python'] = '0.0.0'
        with patch('lab.experiment_audit.environment_metadata', return_value=env):
            with self.assertRaises(RuntimeError):
                metadata('README.md', [])
            with self.assertRaises(ValueError):
                metadata('README.md', [], ' ')
            with patch('lab.experiment_audit.subprocess.check_output', side_effect=['', 'head', 'protocol']):
                result = metadata('README.md', [], 'Portability audit')
                self.assertEqual(result['runtime_override'], 'Portability audit')
                self.assertEqual(result['environment']['python'], '0.0.0')
