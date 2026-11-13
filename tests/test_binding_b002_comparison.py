"""Operational equality must survive only guard-vetoed diagnostic drift."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.revision_b002 import compare


class ConfirmationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(instrument='B001_uniform', verdict='inconclusive',
                           unguarded_verdict='accept', grids=[4096, 8192, 16384, 32768],
                           step=dict(verdict='accept', converging=True, futile=False,
                                     rounding=.00001, last_grid=32768),
                           safety=dict(passed=False, multiplier=4,
                                       decision_margin=.000001, contraction_margin=.000001,
                                       threshold=.00004))
        self.base = dict(attempts=[dict(result=self.result, cost=2., withdrawn=True)],
                         reviewed=['0'], withdrawn=['0'], cost=2.,
                         interventions=[dict(selected=['0'])], metrics=dict(confirmed=0))

    def comparison(self, changed, ordinary=False, summary_changed=False):
        manifest = dict.fromkeys(('intake_sha256', 'selected', 'exposure_freeze_sha256',
                                 'qualification_sha256', 'source_sha256', 'protocol_sha256',
                                 'development_intake_sha256', 'qualification_intake_sha256'), 'same')
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/'left', Path(tmp)/'right'
            if ordinary:
                for i in range(12):
                    directory = left/f'series-{i:02d}'/'ordinary'
                    directory.mkdir(parents=True)
                    (directory/'none-0.json.gz').touch()
            def read(path):
                if path.name == 'manifest.json': return manifest
                if path.name in ('summary.json', 'closure.json'):
                    return {'total': int(summary_changed and right in path.parents)}
                if ordinary and path.name == 'revision.json.gz': return {}
                return changed if right in path.parents else self.base
            with patch('binding.revision_b002.read_json', side_effect=read):
                return compare(left, right)

    def test_guard_vetoed_uniform_raw_flip_is_diagnostic(self):
        changed = copy.deepcopy(self.base)
        result = changed['attempts'][0]['result']
        result['unguarded_verdict'] = result['step']['verdict'] = 'inconclusive'
        result['step']['converging'] = False
        result['safety']['decision_margin'] = None
        for ordinary in (False, True):
            stats = self.comparison(changed, ordinary)
            self.assertTrue(stats['decisions_pass'])
            self.assertEqual(stats['diagnostic_unguarded_changes'], 12)
            self.assertEqual(stats['diagnostic_guarded_raw_changes'], 36)

    def test_protected_verdict_flag_and_configuration_stay_exact(self):
        for key, value in (('verdict', 'accept'), ('grids', [1, 2, 3, 4])):
            changed = copy.deepcopy(self.base)
            changed['attempts'][0]['result'][key] = value
            self.assertFalse(self.comparison(changed)['decisions_pass'])
        for key, value in (('passed', True), ('passed', 0), ('multiplier', 5)):
            changed = copy.deepcopy(self.base)
            changed['attempts'][0]['result']['safety'][key] = value
            self.assertFalse(self.comparison(changed)['decisions_pass'])

    def test_active_raw_flags_stay_exact(self):
        self.result['verdict'] = 'accept'
        self.result['safety']['passed'] = True
        changed = copy.deepcopy(self.base)
        changed['attempts'][0]['result']['step']['converging'] = False
        self.assertFalse(self.comparison(changed)['decisions_pass'])

    def test_cross_protected_flags_stay_exact(self):
        self.base['attempts'][0]['result'] = dict(verdict='inconclusive', steps=[dict(
            verdict='inconclusive', unguarded_verdict='accept',
            convergence_pass=False, decision_pass=False, finite=True)])
        for field in ('verdict', 'convergence_pass', 'decision_pass', 'finite'):
            changed = copy.deepcopy(self.base)
            step = changed['attempts'][0]['result']['steps'][0]
            step[field] = 'accept' if field == 'verdict' else not step[field]
            self.assertFalse(self.comparison(changed)['decisions_pass'])

    def test_prices_selection_withdrawal_and_totals_stay_exact(self):
        for field, value in (('cost', 2.000000001), ('withdrawn', False)):
            changed = copy.deepcopy(self.base)
            changed['attempts'][0][field] = value
            self.assertFalse(self.comparison(changed)['decisions_pass'])
        for field, value in (('reviewed', []), ('withdrawn', []), ('cost', 3.),
                             ('interventions', [dict(selected=[])]), ('metrics', dict(confirmed=1))):
            changed = copy.deepcopy(self.base); changed[field] = value
            self.assertFalse(self.comparison(changed)['decisions_pass'])
        self.assertFalse(self.comparison(self.base, summary_changed=True)['decisions_pass'])
        for field in ('confirmation_fraction', 'selection_precision', 'selection_recall'):
            self.base['metrics'][field] = .5
            changed = copy.deepcopy(self.base); changed['metrics'][field] += 1e-12
            self.assertFalse(self.comparison(changed)['decisions_pass'])
        changed = copy.deepcopy(self.base)
        changed['attempts'][0]['result']['step']['last_grid'] = 16384
        self.assertFalse(self.comparison(changed)['decisions_pass'])

    def test_diagnostic_structure_and_nonfinite_values_still_fail(self):
        changed = copy.deepcopy(self.base)
        del changed['attempts'][0]['result']['step']['verdict']
        self.assertFalse(self.comparison(changed)['decisions_pass'])
        changed = copy.deepcopy(self.base)
        changed['attempts'][0]['result']['step']['rounding'] = float('inf')
        self.assertFalse(self.comparison(changed)['decisions_pass'])
