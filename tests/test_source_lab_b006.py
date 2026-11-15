import copy
import unittest
from binding import source_lab_b006 as lab
from binding import diagnose_b005 as old


class LaboratoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = lab.make_series(20270501, 'low_zero', episodes=32)

    def test_same_graph_every_episode(self):
        shapes = [[(r['id'], r['parents'], r['source_ids']) for r in e['case']['records']]
                  for e in self.series['episodes']]
        self.assertTrue(all(s == shapes[0] for s in shapes))
        self.assertEqual(len(shapes[0]), 44)

    def test_factorial_pairing_isolates_error_environment(self):
        zero = lab.make_series(71, 'low_zero', episodes=8)
        moderate = lab.make_series(71, 'low_moderate', episodes=8)
        for a, b in zip(zero['episodes'], moderate['episodes']):
            self.assertEqual(lab.public_episode(a), lab.public_episode(b))
            self.assertEqual(a['evaluator']['kind'], b['evaluator']['kind'])
        high = lab.make_series(71, 'high_zero', episodes=8)
        for a, b in zip(zero['episodes'], high['episodes']):
            self.assertEqual(a['evaluator']['scope'], b['evaluator']['scope'])
            self.assertEqual(a['signal_branch'], b['signal_branch'])
            if a['evaluator']['kind'] in ('clean', 'false_objection'):
                self.assertEqual(a['evaluator']['kind'], b['evaluator']['kind'])
            if a['evaluator']['kind'] == b['evaluator']['kind']:
                self.assertEqual(lab.public_episode(a), lab.public_episode(b))

    def test_public_has_no_labels(self):
        for e in self.series['episodes']:
            public = lab.public_episode(e)
            self.assertNotIn('evaluator', public)
            for key in ('catalog', 'truth', 'culprit', 'fault', 'scope', 'inaccurate', 'affected'):
                self.assertNotIn(key, public['case'])
            self.assertTrue(all('repeat_value' not in x for x in public['case']['measurements'].values()))

    def test_clean_episode_no_signal(self):
        clean = [e for e in self.series['episodes'] if e['evaluator']['kind'] == 'clean']
        self.assertTrue(clean)
        for e in clean:
            self.assertEqual(e['signal_ids'], [])
            self.assertEqual(e['case']['inaccurate'], [])

    def test_false_objection_has_no_bad_premise(self):
        found = [e for e in self.series['episodes'] if e['evaluator']['kind'] == 'false_objection']
        self.assertTrue(found)
        for e in found:
            self.assertIsNone(e['case']['culprit'])
            self.assertEqual(e['case']['inaccurate'], [])
            self.assertTrue(e['signal_ids'])
            self.assertFalse(lab.NoisyService(e).repeat_observation(e['signal_ids']))

    def test_zero_noise_reproduces_b005_probes(self):
        for e in self.series['episodes']:
            svc, previous = lab.NoisyService(e), old.Service(e['case'])
            for sid in e['case']['sources']:
                self.assertEqual(svc.probe({'kind': 'check_source', 'id': sid}), previous.check_source(sid)[0])
                self.assertEqual(svc.source_cost(sid), previous.source_cost(sid))
            for row in e['case']['records']:
                cid = row['id']
                self.assertEqual(svc.probe({'kind': 'check_claim', 'id': cid}), previous.check_claim(cid)[0])
                self.assertEqual(svc.claim_cost(cid), previous.claim_cost(cid))

    def test_true_fault_signals_match_b005_both_branches(self):
        for e in self.series['episodes']:
            if e['evaluator']['kind'] == 'true_fault':
                for branch in ('measurement', 'conflict'):
                    shadow = dict(e, signal_branch=branch, signal_ids=old.signal(e['case'], branch))
                    self.assertTrue(shadow['signal_ids'])
                    self.assertTrue(lab.NoisyService(shadow).repeat_observation(shadow['signal_ids']))
                self.assertEqual(e['signal_ids'], old.signal(e['case'], e['signal_branch']))

    def test_error_frequencies_each_kind_and_direction(self):
        n = 16000
        for kind, rates in lab.ERRORS['moderate'].items():
            for truth in (False, True):
                wrong = sum(lab.noisy_answer(truth, rates, 415, kind, 'x', i) != truth for i in range(n)) / n
                self.assertAlmostEqual(wrong, rates['false_good' if truth else 'false_bad'], delta=.012)

    def test_repeated_attempts_independent(self):
        rates = {'false_good': .5, 'false_bad': .5}
        bits = [lab.noisy_answer(False, rates, 417, 'check_claim', 'x', i) for i in range(20000)]
        self.assertAlmostEqual(sum(bits) / len(bits), .5, delta=.015)
        self.assertAlmostEqual(sum(a and b for a, b in zip(bits[::2], bits[1::2])) / 10000, .25, delta=.015)
        other = [lab.noisy_answer(False, rates, 418, 'check_claim', 'x', i) for i in range(20000)]
        self.assertAlmostEqual(sum(a == b for a, b in zip(bits, other)) / len(bits), .5, delta=.015)

    def test_pairing_is_policy_independent(self):
        e = copy.deepcopy(self.series['episodes'][0])
        e['evaluator']['rates'] = lab.ERRORS['moderate']
        a, b = lab.NoisyService(e), lab.NoisyService(e)
        sid = next(iter(e['case']['sources']))
        test = {'kind': 'check_source', 'id': sid}
        self.assertEqual([a.probe(test) for _ in range(10)], [b.probe(test) for _ in range(10)])
        self.assertEqual([r['attempt'] for r in a.evaluate()['events']], list(range(10)))

    def test_certified_repair_and_price_match_previous(self):
        e = next(e for e in self.series['episodes'] if e['evaluator']['kind'] == 'true_fault')
        svc, previous = lab.NoisyService(e), old.Service(e['case'])
        sid = e['case']['culprit']
        with self.assertRaises(ValueError):
            svc.repair(e['case']['records'][0]['id'])
        svc.commit_repair_source(sid)
        previous.sources[sid] = copy.deepcopy(e['case']['catalog'][sid])
        for cid in e['case']['affected']:
            self.assertEqual(svc.repair_cost(cid), previous.repair_cost(cid))
            self.assertEqual(svc.repair(cid), previous.repair(cid))
        self.assertEqual(svc.repeat_cost(), 2 * e['case']['budget_unit'])

    def test_inputs_not_mutated(self):
        original = copy.deepcopy(self.series)
        e = self.series['episodes'][0]
        public = lab.public_episode(e)
        public['case']['records'][0]['value'] = -99
        svc = lab.NoisyService(e)
        svc.probe({'kind': 'check_source', 'id': next(iter(e['case']['sources']))})
        self.assertEqual(self.series, original)

    def test_d_matches_b005_noiseless_local_diagnosis(self):
        from binding import diagnose_b006 as current
        for episode in self.series['episodes']:
            if episode['evaluator']['kind'] == 'false_objection' and episode['signal_branch'] == 'conflict':
                continue
            for budget in (2, 8, 32):
                earlier = old.simulate(episode['case'], 'L', budget, episode['signal_branch'])
                later = current.simulate(episode, 'D', budget)
                self.assertEqual(earlier['protected']['signal_ids'], later['protected']['signal_ids'])
                for field in ('initial_candidates', 'checked', 'repaired', 'suspected', 'resolved', 'budget', 'cost'):
                    self.assertEqual(earlier['protected'][field], later['protected'][field], field)
                self.assertEqual(earlier['metrics']['remaining_bad'], later['metrics']['remaining_bad'])
                self.assertEqual(earlier['metrics']['false_accusations'], later['metrics']['false_accusations'])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): lab.make_series(1, 'unknown')
        with self.assertRaises(ValueError): lab.make_series(1, 'low_zero', episodes=0)
        with self.assertRaises(ValueError): lab.noisy_answer(True, {'false_good': 2}, 1, 'x', 'y', 0)
        with self.assertRaises(ValueError): lab.noisy_answer(True, {'false_good': 0}, 1, 'x', 'y', -1)


if __name__ == '__main__':
    unittest.main()
