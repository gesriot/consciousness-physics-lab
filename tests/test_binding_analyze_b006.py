import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from binding import analyze_b006 as analysis
from binding import run_b006 as saved


def fixture(seed=1):
    environments = {}
    for environment in saved.lab.ENVIRONMENTS:
        episodes = [
            {'signal_branch': 'measurement', 'signal_ids': ['a'],
             'evaluator': {'kind': 'false_objection', 'scope': 'shared'}},
            {'signal_branch': 'conflict', 'signal_ids': ['a', 'b'],
             'evaluator': {'kind': 'true_fault', 'scope': 'private'}},
            {'signal_branch': 'measurement', 'signal_ids': [],
             'evaluator': {'kind': 'clean', 'scope': 'shared'}},
        ]
        outputs = {}
        for variant in ('D', 'R', 'K2of3', 'P_beta8_t80', 'P_beta99_t95'):
            for budget in saved.B.BUDGETS:
                rows = []
                for index, episode in enumerate(episodes):
                    signaled = int(bool(episode['signal_ids']))
                    probabilistic = variant.startswith('P_')
                    metrics = {'false_accusations': 0,
                               'remaining_bad': int(index == 1 and not probabilistic),
                               'lost_good': 0, 'cost': (3 if probabilistic else 2)*signaled,
                               'checks': signaled, 'repeated_checks': 0,
                               'observation_repeats': int(index == 0),
                               'rejected_false': int(index == 0),
                               'corrected_counterexample': int(index == 1 and probabilistic),
                               'attempted_false_accusations': 0, 'suspected': 0,
                               'diagnosis_made': signaled, 'diagnosis_correct': signaled,
                               'abstained': 0, 'signal_depth': 2 if signaled else 0}
                    rows.append({'metrics': metrics,
                                 'terminal': {'probability': .75 if probabilistic and signaled else None}})
                outputs[f'{variant}|{budget}'] = rows
        environments[environment] = {'series': {'episodes': episodes}, 'outputs': outputs}
    return {'seed': seed, 'intake_sha256': 'inputs-'+str(seed), 'environments': environments}


def selection():
    return {'pass_gate': False, 'common_gate': True, 'selected_K': 'K2of3',
            'selected_P': None, 'best_constant': 'R',
            'fixed_confidences': {'D': .5, 'R': .8, 'K2of3': .6},
            'configurations': {'beta8_t80': {'qualifies': False, 'low_zero_failures': ['cost']}},
            'exposure': {}, 'false_objections': {'low': 1, 'high': 2},
            'D_false_accusations_moderate': 1}


class AnalysisTests(unittest.TestCase):
    def test_failed_gate_never_claims_confirmation_closure(self):
        out = analysis.analyze_archives([fixture()], 'development', selection())
        self.assertFalse(out['gate_status']['pass_gate'])
        self.assertFalse(out['closure']['evaluated'])
        self.assertNotIn('constant_no_worse_everywhere', out['closure'])
        with self.assertRaises(ValueError):
            analysis.analyze_archives([fixture()], 'confirmation', selection())

    def test_all_global_configs_and_controls_compared(self):
        out = analysis.analyze_archives([fixture()], 'development', selection())
        self.assertEqual(set(out['P_comparisons']), {'P_beta8_t80', 'P_beta99_t95'})
        for comparisons in out['P_comparisons'].values():
            self.assertEqual(set(comparisons), {'D', 'R', 'K2of3'})
            self.assertEqual(len(comparisons['D']['cells']), 24)

    def test_tradeoff_is_not_dominance(self):
        out = analysis.analyze_archives([fixture()], 'development', selection())
        cell = out['P_comparisons']['P_beta8_t80']['D']['cells']['low_zero|8|conflict']
        self.assertEqual(cell['relation'], 'tradeoff')
        self.assertEqual(cell['P_minus_control']['remaining_bad'], -1)
        self.assertEqual(cell['P_minus_control']['cost'], 1)
        self.assertFalse(cell['P_no_worse'])

    def test_independent_series_not_episode_conditions(self):
        out = analysis.analyze_archives([fixture(1), fixture(2)], 'development', selection())
        self.assertEqual(out['independent_series'], 2)
        self.assertEqual(out['dependent_episode_conditions'], 2*4*5*3*3)
        row = out['secondary_by_signal_kind']['low_zero|D|2|measurement|false_objection']
        self.assertEqual(row['independent_series'], 2)
        self.assertEqual(row['dependent_episode_conditions'], 2)

    def test_brier_and_calibration_retain_abstentions(self):
        first, second = fixture(1), fixture(2)
        abstention = second['environments']['low_zero']['outputs']['P_beta8_t80|2'][0]
        abstention['metrics'].update(abstained=1, diagnosis_made=0, diagnosis_correct=0)
        out = analysis.analyze_archives([first, second], 'development', selection())
        row = out['calibration_by_primary_cell']['low_zero|P_beta8_t80|2|measurement']
        self.assertEqual(row['abstentions'], 1)
        self.assertEqual(row['no_signal'], 2)
        self.assertEqual(row['dependent_decisions'], 1)
        self.assertAlmostEqual(row['brier_pooled_descriptive'], .0625)
        self.assertEqual(row['abstention_mean_reported_probability'], .75)
        self.assertEqual(row['bins'][7]['dependent_decisions'], 1)
        self.assertEqual(row['bins'][7]['independent_series'], 1)
        self.assertIsNone(row['brier_series_standard_error'])

    def test_fixed_confidences_are_supplied_not_refitted(self):
        out = analysis.analyze_archives([fixture()], 'development', selection())
        row = out['calibration_by_primary_cell']['low_zero|D|2|measurement']
        self.assertAlmostEqual(row['brier_pooled_descriptive'], .25)
        self.assertEqual(row['fixed_confidence_origin'], 'development_in_sample')

    def test_depth_scope_and_opportunity_denominators(self):
        out = analysis.analyze_archives([fixture()], 'development', selection())
        row = out['secondary_by_depth']['low_zero|P_beta8_t80|2|conflict|2']
        self.assertEqual(row['real_counterexamples_corrected_fraction'], 1)
        row = out['secondary_by_scope']['low_zero|P_beta8_t80|2|measurement|shared']
        self.assertEqual(row['false_objections_rejected_fraction'], 1)
        self.assertEqual(row['total']['episodes'], 2)
        self.assertEqual(row['total']['false_objections'], 1)

    def test_duplicate_series_and_truncation_rejected(self):
        with self.assertRaises(ValueError):
            analysis.analyze_archives([fixture(), fixture()], 'development', selection())
        archive = fixture()
        archive['environments']['low_zero']['outputs']['D|2'].pop()
        with self.assertRaises(ValueError):
            analysis.analyze_archives([archive], 'development', selection())

    def test_confirmation_uses_frozen_selection(self):
        chosen = selection()
        chosen.update(pass_gate=True, selected_P='beta8_t80')
        out = analysis.analyze_archives([fixture()], 'confirmation', chosen)
        self.assertTrue(out['closure']['evaluated'])
        self.assertEqual(out['closure']['best_constant'], 'R')
        self.assertFalse(out['closure']['constant_no_worse_everywhere'])

    def test_analyzer_never_generates_experiment_or_simulates_policy(self):
        with patch.object(saved, 'make_archive', side_effect=AssertionError('generation forbidden')):
            with patch.object(saved.B, 'simulate', side_effect=AssertionError('simulation forbidden')):
                analysis.analyze_archives([fixture()], 'development', selection())

    def test_saved_file_hashes_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = fixture()
            saved.write(root/'series-1.json.gz', archive)
            hashes = saved.file_hashes(root/'series-1.json.gz')
            manifest = {'phase': 'development', 'seeds': [1], 'files': {'series-1.json.gz': hashes},
                        'intakes': {'1': archive['intake_sha256']}}
            saved.write(root/'manifest.json', manifest)
            saved.write(root/'gate.json', selection())
            out = analysis.analyze(root)
            self.assertEqual(out['consumed_files']['series-1.json.gz'], hashes)
            (root/'series-1.json.gz').write_bytes(b'corruption')
            with self.assertRaises((ValueError, OSError)):
                analysis.analyze(root)

    def test_invalid_probability_rejected(self):
        archive = fixture()
        archive['environments']['low_zero']['outputs']['P_beta8_t80|2'][0]['terminal']['probability'] = 2
        with self.assertRaises(ValueError):
            analysis.analyze_archives([archive], 'development', selection())


if __name__ == '__main__':
    unittest.main()
