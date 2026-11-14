"""Synthetic fixtures for the descriptive B005 archive report."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from binding.analyze_b005 import aggregate, analyze, canonical


class AnalysisTests(unittest.TestCase):
    def fixture(self, directory):
        metrics = dict(remaining_bad=0, withdrawn_good=0, false_accusations=0,
            cost=3, unused_budget=7, suspected=0, initial_bad=1, resolved=1,
            localization_opportunities=1, correctly_localized=1, checks=2,
            checks_to_localization=2, repeated_measurement=0, signal_depth=3,
            memory_bytes=100, capacity_bytes=1000)
        output = dict(metrics=metrics, trace=[dict(kind='check_source', id='s1', candidate='s1')],
            protected=dict(actions=[dict(kind='check_source', id='s1', candidate='s1')], signal_ids=['c1']),
            diagnostics=dict(record_bytes=[40, 60]))
        cases = {}
        for scope in ('shared', 'private'):
            cases['units|'+scope] = dict(series=dict(fault='units', scope=scope),
                outputs={p: copy.deepcopy(output) for p in ('G', 'L', 'F')},
                conditions={p+'|8|measurement': p for p in ('G', 'L', 'F')})
        archive = dict(cases=cases)
        payload = canonical(archive)
        name = 'series-fixture.json.gz'
        (directory/name).write_bytes(gzip.compress(payload, mtime=1))
        manifest = dict(phase='fixture', files={name: dict(
            canonical_json_sha256=hashlib.sha256(payload).hexdigest(),
            compressed_sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())})
        (directory/'manifest.json').write_text(json.dumps(manifest))
        return archive, name

    def test_series_aggregation_and_localization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            self.fixture(path)
            report = analyze(path)
        row = report['primary_cells']['units|8|measurement|G']
        self.assertEqual(row['independent_series'], 1)
        self.assertEqual(row['total']['cost'], 6)
        loc = report['localization_by_signal_depth']['units|8|measurement|shared|3|G']
        self.assertEqual(loc['correct_localization_rate'], 1)
        self.assertEqual(loc['mean_checks_for_correct_localization'], 2)
        self.assertEqual(report['G_L_total_selected_action_differences'], 0)
        self.assertTrue(report['closure_L_no_worse_than_G'])
        self.assertTrue(report['closure_F_no_worse_than_G_and_L'])
        self.assertEqual(report['storage_bytes']['G']['record_bytes']['minimum'], 40)

    def test_compressed_bytes_are_not_semantic_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            archive, name = self.fixture(path)
            (path/name).write_bytes(gzip.compress(canonical(archive), mtime=2))
            self.assertEqual(analyze(path)['independent_series'], 1)

    def test_wrong_canonical_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            archive, name = self.fixture(path)
            archive['changed'] = True
            (path/name).write_bytes(gzip.compress(canonical(archive)))
            with self.assertRaisesRegex(ValueError, 'Canonical series hash mismatch'):
                analyze(path)

    def test_empty_run_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'manifest.json').write_text('{"files":{}}')
            with self.assertRaisesRegex(ValueError, 'No complete series'):
                analyze(path)

    def test_standard_error_uses_rows(self):
        row = aggregate([dict(n=1), dict(n=3)])
        self.assertEqual(row['mean']['n'], 2)
        self.assertEqual(row['standard_error']['n'], 1)


if __name__ == '__main__':
    unittest.main()
