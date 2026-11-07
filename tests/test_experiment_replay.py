import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from lab.experiment_audit import environment_metadata, metadata
from lab.executor import request_hash
from lab.replay import canonical_sha256, load_replay, protect_output
from scripts.compare_experiment_runs import compare, compare_details, numeric_summary, input_digest

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
        self.assertFalse(compare(self.original, self.repeat, 'decisions')[0])
        self.original['runs'][0]['value'] = 0.0
        self.repeat['runs'][0]['value'] = 1e-30
        self.assertFalse(compare(self.original, self.repeat, 'decisions')[0])
        self.assertFalse(compare(self.original, self.repeat, 'decisions', absolute_tolerance=0)[0])
        del self.repeat['runs'][0]['accepted']
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])


    def test_diagnostic_changes_do_not_mask_operational_changes(self):
        original = dict(experiment='E009', coverage_steps=[dict(converging=True)],
            runs=[dict(accepted=True, verdict='accept', independent=dict(
                cells_solved=512, steps=[dict(converging=True, verdict='accept')]),
                shadow=dict(any_accept=True, steps=[dict(converging=True)]))],
            summary=dict(cells={'uniform/order': dict(coverage=dict(fallback_steps=240),
                                                      decision_cells=512)}))
        repeat = copy.deepcopy(original)
        repeat['coverage_steps'][0]['converging'] = False
        repeat['runs'][0]['shadow']['steps'][0]['converging'] = False
        repeat['summary']['cells']['uniform/order']['coverage']['fallback_steps'] = 239
        failures, _, diagnostics = compare_details(original, repeat, 'decisions')
        self.assertFalse(failures)
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(compare(original, repeat, 'exact')[0])
        repeat['runs'][0]['independent']['steps'][0]['converging'] = False
        self.assertTrue(compare_details(original, repeat, 'decisions')[0])
        repeat = copy.deepcopy(original)
        repeat['summary']['cells']['uniform/order']['decision_cells'] = 513
        self.assertTrue(compare_details(original, repeat, 'decisions')[0])

    def test_e011_diagnostics_are_separate_and_operational_controls_remain_exact(self):
        original = json.loads((ROOT / 'results/E011-extrapolation/results.json').read_text())
        repeat = copy.deepcopy(original)
        repeat['runs'][0]['shadow']['steps'][0]['verdict'] = 'changed diagnostic verdict'
        repeat['runs'][0]['shadow']['steps'][0]['converging'] = not repeat['runs'][0]['shadow']['steps'][0]['converging']
        repeat['coverage_steps'][0]['covered'] = not repeat['coverage_steps'][0]['covered']
        repeat['summary']['A']['coverage']['misses'] += 1
        repeat['tasks'][0]['shadow_cells'] += 1
        repeat['tasks'][0]['shadow_work']['sphere']['cells'] += 1
        repeat['work']['shadow_cells'] += 1
        repeat['work']['shadow_by_model']['sphere']['cells'] += 1
        failures, _, diagnostics = compare_details(original, repeat, 'decisions')
        self.assertFalse(failures)
        self.assertEqual(len(diagnostics), 8)
        self.assertTrue(compare(original, repeat, 'exact')[0])
        for key in ('converging', 'cells_solved'):
            control = copy.deepcopy(repeat)
            independent = control['runs'][0]['independent']
            if key == 'converging':
                independent['steps'][0][key] = not independent['steps'][0][key]
            else:
                independent[key] += 1
            self.assertTrue(compare(original, control, 'decisions')[0])
        control = copy.deepcopy(repeat)
        control['work']['decision_cells'] += 1
        self.assertTrue(compare(original, control, 'decisions')[0])

    def test_absolute_tolerance_is_additive_finite_and_does_not_relax_counts(self):
        self.original['runs'][0]['value'] = 1.0
        self.repeat['runs'][0]['value'] = 1.15
        self.assertFalse(compare(self.original, self.repeat, 'decisions', .1,
                                 absolute_tolerance=.1)[0])
        for tolerance in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compare(self.original, self.repeat, 'decisions', absolute_tolerance=tolerance)
        self.repeat['runs'][0]['cells'] = 513
        self.assertTrue(compare(self.original, self.repeat, 'decisions', .1,
                                absolute_tolerance=1e6)[0])

    def test_exact_inputs_even_when_numeric_outputs_are_unrestricted(self):
        self.original['intake_channel'] = [{'request': {'z': [1.0, 0.0]}}]
        self.repeat = copy.deepcopy(self.original)
        self.repeat['intake_channel'][0]['request']['z'][0] += 1e-14
        self.assertTrue(compare(self.original, self.repeat, 'decisions')[0])
        self.assertNotEqual(input_digest(self.original), input_digest(self.repeat))

    def test_canonical_hash_ignores_object_order_not_content(self):
        self.assertEqual(canonical_sha256({'a': 1, 'b': 2}),
                         canonical_sha256({'b': 2, 'a': 1}))
        self.assertNotEqual(canonical_sha256({'a': 1}), canonical_sha256({'a': 2}))

    def test_numeric_summary_has_no_hidden_threshold(self):
        left = {'x': [1.0, 0.0, float('inf')]}
        right = {'x': [3.0, 1e-30, float('inf')]}
        row = numeric_summary(left, right)['$.x[]']
        self.assertEqual(row['count'], 3)
        self.assertEqual(row['changed'], 2)
        self.assertEqual(row['max_abs'], 2.0)
        self.assertEqual(row['max_rel'], 1.0)
        self.assertEqual(row['nonfinite'], 1)
        self.assertFalse(compare(left, right, 'decisions')[0])

    def test_archives_and_controlled_lower_bit_changes(self):
        import math
        for original_path, repeat_path in [
            ('surface-order/results.json', 'platform-replay/e009-replay.json'),
            ('E011-extrapolation/results.json', 'E011-extrapolation/repeat.json')]:
            original = json.loads((ROOT / 'results' / original_path).read_text())
            repeat = json.loads((ROOT / 'results' / repeat_path).read_text())
            self.assertFalse(compare(original, repeat, 'decisions')[0])
            def perturb(value):
                if isinstance(value, dict):
                    return {k: (v if k in ('request', 'intake_channel', 'sampled_magnitude',
                                          'sampled_phase') else perturb(v)) for k, v in value.items()}
                if isinstance(value, list):
                    return [perturb(v) for v in value]
                return math.nextafter(value, math.inf) if isinstance(value, float) else value
            changed = perturb(repeat)
            self.assertFalse(compare(original, changed, 'decisions')[0])
            self.assertTrue(compare(original, changed, 'exact')[0])
            self.assertGreater(sum(r['changed'] for r in numeric_summary(repeat, changed).values()), 0)
            changed['runs'][0]['accepted'] = not changed['runs'][0]['accepted']
            self.assertTrue(compare(original, changed, 'decisions')[0])


class ReplayTests(unittest.TestCase):
    def test_archived_inputs_and_digests_are_bit_identical(self):
        for experiment, folder, count, key in [('E007', 'verifier-stress', 192, 'tasks'),
                                              ('E008', 'common-mode', 96, 'intake_channel'),
                                              ('E009', 'surface-order', 192, 'tasks')]:
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
        from lab import common_mode, verifier_stress, surface_experiment
        from types import SimpleNamespace
        class Captured(Exception):
            pass
        for module, folder, experiment, count in [
                (common_mode, 'common-mode', 'E008', 96),
                (verifier_stress, 'verifier-stress', 'E007', 192),
                (surface_experiment, 'surface-order', 'E009', 192)]:
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
