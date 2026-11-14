"""Mutation tests for independent source-reconciliation reconstruction."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding import reconcile_b004 as engine
from binding import audit_b004 as audit


def build_case(seed=11, fault='regime', noise=False):
    series = engine.build_series(engine.intake(seed), fault, noise=noise)
    outputs, conditions = {}, {}
    service = engine.RecheckService(series, noise=noise)
    for memory in engine.POLICIES:
        for budget in engine.BUDGETS:
            for branch in engine.SIGNALS:
                result = engine.simulate(series, memory, budget, branch, noise=noise, service=service)
                identifier = engine.digest(result)
                outputs[identifier] = result
                conditions[f'{memory}|{budget}|{branch}'] = identifier
    return series, outputs, conditions


class IndependentReconciliationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = build_case()

    def changed(self):
        return copy.deepcopy(self.original)

    def rekey(self, outputs, conditions, condition, mutate):
        old = conditions[condition]
        value = copy.deepcopy(outputs[old])
        mutate(value)
        new = audit.sha(value)
        outputs[new] = value
        conditions[condition] = new
        if old not in conditions.values():
            del outputs[old]

    def test_independent_from_engine_policy(self):
        with patch.object(engine, 'simulate', side_effect=AssertionError('shared policy')), \
             patch.object(engine, 'evaluate', side_effect=AssertionError('shared metrics')), \
             patch.object(engine, 'candidate_order', side_effect=AssertionError('shared order')), \
             patch.object(engine, 'build_series', side_effect=AssertionError('shared generation')):
            self.assertTrue(audit.audit_case(*self.original)['pass'])

    def test_all_corruptions_and_noise(self):
        for fault in engine.FAULTS:
            with self.subTest(fault=fault):
                self.assertTrue(audit.audit_case(*build_case(19, fault, True))['pass'])

    def test_missing_condition(self):
        series, outputs, conditions = self.changed()
        del conditions['G|2|revision']
        with self.assertRaisesRegex(ValueError, 'coverage'):
            audit.audit_case(series, outputs, conditions)

    def test_missing_content_addressed_output(self):
        series, outputs, conditions = self.changed()
        del outputs[conditions['G|2|revision']]
        with self.assertRaisesRegex(ValueError, 'absent output'):
            audit.audit_case(series, outputs, conditions)

    def test_output_hash(self):
        series, outputs, conditions = self.changed()
        outputs[conditions['G|2|revision']]['metrics']['cost'] += 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            audit.audit_case(series, outputs, conditions)

    def test_metric_mutation_with_updated_hash(self):
        series, outputs, conditions = self.changed()
        self.rekey(outputs, conditions, 'G|2|revision', lambda r:r['metrics'].update(cost=0))
        with self.assertRaisesRegex(ValueError, 'metric mismatch'):
            audit.audit_case(series, outputs, conditions)

    def test_false_final_acceptance(self):
        series, outputs, conditions = self.changed()
        def mutate(result):
            cid = next(iter(result['final']))
            result['final'][cid]['accepted'] = not result['final'][cid]['accepted']
        self.rekey(outputs, conditions, 'G|2|revision', mutate)
        with self.assertRaisesRegex(ValueError, 'final state'):
            audit.audit_case(series, outputs, conditions)

    def test_trace_cost(self):
        series, outputs, conditions = self.changed()
        self.rekey(outputs, conditions, 'G|2|revision', lambda r:r['trace'][0].update(cost=0))
        with self.assertRaisesRegex(ValueError, 'action trace'):
            audit.audit_case(series, outputs, conditions)

    def test_wrong_diagnostic_culprit(self):
        series, outputs, conditions = self.changed()
        def mutate(result):
            result['trace'][0]['implicated'] = result['trace'][0]['pair']
        self.rekey(outputs, conditions, 'G|8|conflict', mutate)
        with self.assertRaisesRegex(ValueError, 'action trace'):
            audit.audit_case(series, outputs, conditions)

    def test_graph_edge(self):
        series, outputs, conditions = self.changed()
        child = next(r for r in series['records'] if r['depth'] == 2)
        child['parents'] = []
        with self.assertRaisesRegex(ValueError, 'record structure'):
            audit.audit_case(series, outputs, conditions)

    def test_hidden_reference(self):
        series, outputs, conditions = self.changed()
        cid = next(iter(series['truth']))
        series['truth'][cid] *= 1.01
        with self.assertRaisesRegex(ValueError, 'hidden reference'):
            audit.audit_case(series, outputs, conditions)

    def test_observed_inference(self):
        series, outputs, conditions = self.changed()
        next(r for r in series['records'] if r['depth'] == 2)['value'] *= 1.001
        with self.assertRaisesRegex(ValueError, 'observed chain'):
            audit.audit_case(series, outputs, conditions)

    def test_wrong_source_catalogue(self):
        series, outputs, conditions = self.changed()
        sid = next(iter(series['sources']))
        series['sources'][sid]['radius']['value'] *= 2
        with self.assertRaisesRegex(ValueError, 'catalogue'):
            audit.audit_case(series, outputs, conditions)

    def test_recipe_lineage(self):
        series, outputs, conditions = self.changed()
        recipe = next(r for r in series['recipes'].values() if r['operation'] == 'inverse')
        recipe['parent'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'recipe'):
            audit.audit_case(series, outputs, conditions)

    def test_memory_capacity(self):
        series, outputs, conditions = self.changed()
        self.rekey(outputs, conditions, 'F|8|conflict', lambda r:r['protected'].update(capacity_bytes=1))
        with self.assertRaisesRegex(ValueError, 'protected'):
            audit.audit_case(series, outputs, conditions)

    def test_irrelevant_control_edit(self):
        series, outputs, conditions = self.changed()
        self.rekey(outputs, conditions, 'Gxprime|8|revision', lambda r:r['protected'].update(edits=[]))
        with self.assertRaisesRegex(ValueError, 'protected'):
            audit.audit_case(series, outputs, conditions)

    def test_archive_gzip_and_shared_input_hash(self):
        inputs = engine.intake(11)
        payload = {'inputs':inputs, 'intake_sha256':audit.sha(inputs), 'cases':{}}
        series, outputs, conditions = build_case(11, 'clean')
        del series['intake']
        payload['cases']['clean'] = {'series':series, 'outputs':outputs, 'conditions':conditions,
                                      'physical_verifier_evaluations':1644}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'series.json.gz'
            with gzip.open(path, 'wt') as stream:
                json.dump(payload, stream)
            self.assertTrue(audit.audit_archive(path, faults=('clean',))['pass'])
            payload['intake_sha256'] = '0'*64
            with gzip.open(path, 'wt') as stream:
                json.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, 'intake digest'):
                audit.audit_archive(path, faults=('clean',))

    def test_independent_summary_and_gate(self):
        from binding import run_b004 as runner
        archive = runner.make_archive(engine.intake(23), engine.FAULTS)
        summaries = audit.independent_summary([archive])
        audit.summary_equal(runner.summarize([archive]), summaries)
        self.assertEqual(audit.independent_gate([archive]), runner.gate([archive]))
        # Five copies exercise the gate counters only, never independent evidence.
        self.assertEqual(audit.independent_gate([archive]*5), runner.gate([archive]*5))
        key = next(iter(summaries['cells']))
        summaries['cells'][key]['total']['cost'] += 1
        with self.assertRaisesRegex(ValueError, 'summary integer'):
            audit.summary_equal(runner.summarize([archive]), summaries)

    def test_physical_cache_count_mutation(self):
        from binding import run_b004 as runner
        archive = runner.make_archive(engine.intake(29), ('clean',))
        archive['cases']['clean']['physical_verifier_evaluations'] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'series.json.gz'
            runner.write(path, archive)
            with self.assertRaisesRegex(ValueError, 'cache accounting'):
                audit.audit_archive(path, faults=('clean',))

    def test_manifest_rejects_source_mutation(self):
        from binding import run_b004 as runner
        manifest = dict(schema=1, phase='development', seeds=list(runner.DEVELOPMENT),
                        faults=list(engine.FAULTS), **runner.source_metadata())
        manifest['source_sha256'][runner.SOURCES[0]] = '0'*64
        with tempfile.TemporaryDirectory() as directory:
            runner.write(Path(directory)/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'current source identity'):
                audit.audit_run(directory)

    def test_revision_version_is_protected(self):
        series, outputs, conditions = self.changed()
        sid = next(iter(series['notice_versions']))
        series['notice_versions'][sid] += 1
        with self.assertRaisesRegex(ValueError, 'notice versions'):
            audit.audit_case(series, outputs, conditions)


if __name__ == '__main__':
    unittest.main()
