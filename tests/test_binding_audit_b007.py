"""Independent replay, evidence guards and corruption checks on synthetic seeds."""
import copy
import unittest
from unittest.mock import patch
from binding import audit_b007 as audit
from binding import diagnose_b007 as policy


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = policy.lab.make_series(43, 'low_moderate', episodes=4)

    def test_reconstruct_all_variants_without_policy_execution(self):
        for config in policy.P_CONFIGS:
            for name in ('D', 'R', 'N', 'K1of1', 'K2of3', 'K3of5', 'P', 'P0', 'Pstar', 'Px', 'Pxprime'):
                records = policy.simulate_series(self.series, name, 8, config)
                with patch.object(policy, 'simulate', side_effect=AssertionError('policy execution forbidden')):
                    self.assertEqual(audit.audit_series(self.series, records, name, 8, config), 4)

    def test_noise_reconstruction(self):
        records = policy.simulate_series(self.series, 'P', 8, noise=True)
        audit.audit_series(self.series, records, 'P', 8, noise=True)

    def test_rejects_changed_ledger_inference_and_stopping(self):
        records = policy.simulate_series(self.series, 'P', 8)
        index = next(i for i, r in enumerate(records) if r['trace'])
        def price(r): r['trace'][0]['cost'] += 1
        def outcome(r): r['trace'][0]['bad'] = not r['trace'][0]['bad']
        def score(r): r['trace'][0]['score'] += .01
        def posterior(r): r['diagnostics']['posterior']['observation'] += .01
        def stopping(r): r['diagnostics']['stop_reason'] = 'invented'
        def learning(r): r['state_after']['prior'][0] += 1
        def guard(r): r['diagnostics']['direct_bad'].append('fabricated')
        for mutate in (price, outcome, score, posterior, stopping, learning, guard):
            changed = copy.deepcopy(records)
            mutate(changed[index])
            with self.assertRaises(ValueError):
                audit.audit_series(self.series, changed, 'P', 8)

    def test_direct_good_and_bad_are_distinct_guards(self):
        low = {'observation': .2, 'a': .1, 'b': .7}
        high = {'observation': .01, 'a': .99}
        self.assertGreater(audit.threshold_gap(low, 'a', set(), .8), 0)
        self.assertGreater(audit.threshold_gap(low, 'a', {'a'}, .8), 0)
        self.assertEqual(audit.threshold_gap(low, 'a', set(), .8, {'a'}), 0)
        self.assertGreater(audit.threshold_gap(high, 'a', set(), .8, {'a'}), 0)
        self.assertEqual(audit.threshold_gap(high, 'a', {'a'}, .8), 0)

    def test_noninformation_does_not_depend_on_normalization_roundoff(self):
        test = dict(kind='check_source', id='a', candidate='a', cost=1)
        model = {k: dict(tp=.8, fp=.2) for k in audit.KINDS}
        self.assertFalse(audit.informative({'a': 0., 'b': .3, 'c': .7}, test, model))
        self.assertTrue(audit.informative({'a': .5, 'b': .5}, test, model))
        model['check_source'] = dict(tp=.2, fp=.2)
        self.assertFalse(audit.informative({'a': .5, 'b': .5}, test, model))

    def test_privileged_rates_are_explicit(self):
        with self.assertRaises(ValueError):
            audit.model_state(audit.starting_state('beta4_t80'), 'Pstar')
        rates = {k: dict(false_good=.1, false_bad=.2) for k in audit.KINDS}
        self.assertEqual(audit.model_state(audit.starting_state('beta4_t80'), 'Pstar', rates)['active']['check_source'], dict(tp=.9, fp=.2))

    def test_saved_intake_structure_is_protected(self):
        changed = copy.deepcopy(self.series)
        changed['intake']['seed'] += 1
        with self.assertRaises(ValueError):
            audit.verify_series_data(changed)


if __name__ == '__main__':
    unittest.main()
