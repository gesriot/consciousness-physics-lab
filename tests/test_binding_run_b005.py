"""Replay identity and aggregation checks without experimental data."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding import run_b005 as R


class ArchiveTests(unittest.TestCase):
    def test_hashes_separate_content_and_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory)/'one.json.gz', Path(directory)/'two.json.gz'
            R.write(one, {'x': [1, 2.5]})
            raw = gzip.decompress(one.read_bytes())
            two.write_bytes(gzip.compress(raw, compresslevel=1, mtime=42))
            a, b = R.file_hashes(one), R.file_hashes(two)
            self.assertNotEqual(a['compressed_sha256'], b['compressed_sha256'])
            self.assertEqual(a['canonical_json_sha256'], b['canonical_json_sha256'])
            self.assertEqual(a['canonical_json_sha256'], hashlib.sha256(R.canonical({'x': [1, 2.5]})).hexdigest())
            R.verify_file(one, a)
            with self.assertRaises(ValueError):
                R.verify_file(one, b)

    def test_numeric_drift_is_reported_but_structure_is_not_numeric(self):
        result = R.numeric_differences({'value': 1.}, {'value': 1.00000001})
        self.assertEqual(result['/value']['count'], 1)
        with self.assertRaises(ValueError):
            R.numeric_differences({'decision': 'a'}, {'decision': 'b'})
        with self.assertRaises(ValueError):
            R.numeric_differences([1], [1, 2])
        with self.assertRaises(ValueError):
            R.numeric_differences(True, 2)
        with self.assertRaises(ValueError):
            R.numeric_differences({'depth': 2}, {'depth': 3})

    def _archive(self, cost_l=2, cost_g=2, cost_f=3):
        case = {'series': {'records': []}, 'conditions': {}, 'outputs': {}}
        for p, cost in [('G', cost_g), ('L', cost_l), ('F', cost_f)]:
            output = {'metrics': {'remaining_bad': 0, 'withdrawn_good': 0,
                                  'false_accusations': 0, 'cost': cost},
                      'protected': {'actions': ['same']}, 'trace': [], 'diagnostics': {}}
            case['conditions'][f'{p}|2|measurement'] = p
            case['outputs'][p] = output
        return {'cases': {'units|shared': copy.deepcopy(case), 'units|private': copy.deepcopy(case)}}

    def test_closure_includes_price_and_aggregates_paired_scopes(self):
        with patch.object(R.B, 'BUDGETS', (2,)), patch.object(R.B, 'SIGNALS', ('measurement',)):
            summary = R.summarize([self._archive(), self._archive()])
            self.assertTrue(summary['closure_L_no_worse_than_G'])
            self.assertFalse(summary['closure_F_no_worse_than_G_and_L'])
            cell = summary['primary_cells']['units|G|2|measurement']
            self.assertEqual(cell['series'], 2)
            self.assertEqual(cell['total']['cost'], 8)
            self.assertEqual(summary['G_L_action_differences'], 0)
            costly = R.summarize([self._archive(cost_l=3)])
            self.assertFalse(costly['closure_L_no_worse_than_G'])

    def _write_run(self, root, output, compression_level=9):
        archive = {'inputs': {'seed': 1}, 'intake_sha256': 'input', 'cases': {
            'units|shared': {'series': {'number': 1.}, 'outputs': {'ref': output},
                             'conditions': {'G|2|measurement': 'ref'}}}}
        path = root/'series-1.json.gz'
        path.write_bytes(gzip.compress(R.canonical(archive), compresslevel=compression_level, mtime=0))
        R.write(root/'manifest.json', {'phase': 'development', 'files': {'series-1.json.gz': R.file_hashes(path)}})

    def test_compare_accepts_compression_and_numeric_but_not_decisions(self):
        output = {'protected': {'actions': ['check:a'], 'accepted': [1]},
                  'metrics': {'cost': 1}, 'trace': [{'kind': 'check', 'id': 'a', 'score': 1.}],
                  'diagnostics': {'value': 1.}}
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'first', Path(directory)/'second'
            first.mkdir(); second.mkdir()
            self._write_run(first, output)
            changed = copy.deepcopy(output)
            changed['diagnostics']['value'] = 1.000000001
            self._write_run(second, changed, 1)
            comparison = R.compare(first, second)
            self.assertTrue(comparison['decisions_pass'])
            self.assertTrue(comparison['numerical_diagnostics'])
            changed['protected']['actions'] = ['check:b']
            self._write_run(second, changed)
            self.assertFalse(R.compare(first, second)['decisions_pass'])

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'partial').write_text('evidence')
            with self.assertRaisesRegex(ValueError, 'empty'):
                R.run(directory, 'development')
            self.assertEqual((Path(directory)/'partial').read_text(), 'evidence')

    def test_source_identity_survives_runtime_override(self):
        metadata = {'protocol_sha256': 'p', 'protocol_commit': 'c', 'source_sha256': {'a': 's'}}
        manifest = dict(metadata, runtime={'old': True})
        with patch.object(R, 'source_metadata', return_value=metadata), patch.object(R, 'runtime', return_value={'new': True}):
            with self.assertRaisesRegex(ValueError, 'Runtime'):
                R.verify_runtime_manifest(manifest)
            R.verify_runtime_manifest(manifest, True)
            changed = copy.deepcopy(manifest)
            changed['source_sha256']['a'] = 'forged'
            with self.assertRaisesRegex(ValueError, 'identity'):
                R.verify_runtime_manifest(changed, True)


if __name__=='__main__':
    unittest.main()
