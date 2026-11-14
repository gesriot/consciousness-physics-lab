"""Independent replay and intentional-corruption checks on synthetic inputs."""
import copy
import unittest
from unittest.mock import patch
from binding import source_lab_b006 as L
from binding import audit_b006 as A


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = L.make_series(37, 'high_moderate', episodes=8)

    def test_independent_binary_stream_and_repeated_attempts(self):
        for episode in self.series['episodes']:
            service = L.NoisyService(episode)
            evidence = A.Evidence(episode)
            rows = episode['case']['records']
            probes = [dict(kind='check_source', id=s) for s in episode['case']['sources']]
            probes += [dict(kind='check_claim', id=r['id']) for r in rows]
            for test in probes:
                for attempt in range(3):
                    self.assertEqual(service.probe(test), evidence.probe(test))
            for attempt in range(3):
                ids = episode['signal_ids']
                test = dict(kind='repeat_observation', id='|'.join(sorted(ids)))
                self.assertEqual(service.repeat_observation(ids), evidence.probe(test))
            self.assertEqual(service.evaluate()['events'], evidence.events)

    def test_independent_costs_and_repairs(self):
        episode = self.series['episodes'][0]
        service = L.NoisyService(episode)
        evidence = A.Evidence(episode)
        for sid in episode['case']['sources']:
            self.assertEqual(service.source_cost(sid), evidence.source_cost(sid))
        sid = next(iter(episode['case']['sources']))
        service.commit_repair_source(sid)
        evidence.install(sid)
        for row in episode['case']['records']:
            cid = row['id']
            self.assertEqual(service.claim_cost(cid), evidence.claim_cost(cid))
            self.assertEqual(service.repair_cost(cid), evidence.repair_cost(cid))
            self.assertEqual(service.repair(cid), evidence.repair(cid))
        self.assertEqual(service.evaluate()['values'], evidence.values)

    def test_cycle_rejected(self):
        case = copy.deepcopy(self.series['episodes'][0]['case'])
        case['records'][0]['parents'] = [case['records'][0]['id']]
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            A.inherited_labels(case)


class OutputAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from binding import diagnose_b006 as D
        cls.series = L.make_series(43, 'low_moderate', episodes=6)
        cls.episode = next(e for e in cls.series['episodes'] if e['signal_ids'])
        cls.output = D.simulate(cls.episode, 'P', 8)

    def test_no_policy_helpers_called(self):
        from binding import diagnose_b006 as D
        with patch.object(D, 'simulate', side_effect=AssertionError('policy called')), \
             patch.object(D, 'risk_score', side_effect=AssertionError('rank called')), \
             patch.object(D, 'learn', side_effect=AssertionError('learning called')):
            A.audit_output(self.episode, self.output, 'P', 8)

    def test_all_policy_synthetic_episodes(self):
        from binding import diagnose_b006 as D
        for name in ('D', 'R', 'N', 'K1of1', 'K2of3', 'K3of5', 'P', 'Px', 'Pxprime'):
            with self.subTest(policy=name):
                outputs = D.simulate_series(self.series, name, 8)
                self.assertEqual(A.audit_series(self.series, outputs, name, 8), 6)

    def test_sharp_and_noisy_posteriors(self):
        from binding import diagnose_b006 as D
        for config in A.CONFIGS:
            with self.subTest(config=config):
                outputs = D.simulate_series(self.series, 'P', 2, config, noise=True)
                A.audit_series(self.series, outputs, 'P', 2, config, noise=True)

    def test_mutations_are_detected(self):
        first = next(i for i, e in enumerate(self.output['trace']) if e['kind'] != 'repair')
        mutations = [
            lambda o: o['metrics'].__setitem__('cost', -1),
            lambda o: o['metrics'].__setitem__('false_accusations', 999),
            lambda o: o['protected'].__setitem__('decision', 'invented'),
            lambda o: o['protected'].__setitem__('resolved', not o['protected']['resolved']),
            lambda o: o['protected'].__setitem__('observable_resolved', not o['protected']['observable_resolved']),
            lambda o: o['trace'][first].__setitem__('bad', not o['trace'][first]['bad']),
            lambda o: o['trace'][first].__setitem__('cost', 0),
            lambda o: o['trace'][first].__setitem__('id', 'wrong-target'),
            lambda o: o['trace'][first].__setitem__('attempt', 99),
            lambda o: o['state_after']['prior'].__setitem__(0, 999.),
            lambda o: o['terminal'].__setitem__('probability', .1234),
            lambda o: o['evaluation']['events'][0].__setitem__('error', not o['evaluation']['events'][0]['error']),
        ]
        for index, mutate in enumerate(mutations):
            output = copy.deepcopy(self.output)
            mutate(output)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                A.audit_output(self.episode, output, 'P', 8)

    def test_state_chronology(self):
        from binding import diagnose_b006 as D
        outputs = D.simulate_series(self.series, 'P', 8)
        outputs[1]['state_before']['resolved_episodes'] += 1
        with self.assertRaisesRegex(ValueError, 'state before'):
            A.audit_series(self.series, outputs, 'P', 8)

    def test_no_evaluator_learning(self):
        from binding import diagnose_b006 as D
        out = D.simulate(self.episode, 'P', 8)
        changed = copy.deepcopy(self.episode)
        changed['case']['culprit'] = 'unrelated-evaluator-label'
        changed['case']['inaccurate'] = []
        other = D.simulate(changed, 'P', 8)
        self.assertEqual(out['protected'], other['protected'])
        self.assertEqual(out['state_after'], other['state_after'])
        A.audit_output(changed, other, 'P', 8)

    def test_prior_only_decision_cannot_teach(self):
        from binding import diagnose_b006 as D
        state = D.initial_state('beta8_t80')
        state['prior'] = [9999., 1.]
        out = D.simulate(self.episode, 'P', 8, state=state)
        self.assertEqual(out['terminal']['hypothesis'], 'observation')
        self.assertFalse(out['protected']['observable_resolved'])
        self.assertEqual(out['state_before'], out['state_after'])
        A.audit_output(self.episode, out, 'P', 8, state=state)

    def test_input_and_error_model_tampering(self):
        for mutate in (lambda s: s['episodes'][0]['evaluator']['rates']['check_source'].__setitem__('false_bad', .99),
                       lambda s: s['episodes'][0].__setitem__('seed', 999),
                       lambda s: s['intake'].__setitem__('seed', 999)):
            changed = copy.deepcopy(self.series)
            mutate(changed)
            with self.assertRaises(ValueError):
                A.verify_series_data(changed)

    def test_source_identity_tampering(self):
        from binding import run_b006 as R
        manifest = dict(R.source_metadata(), runtime=R.runtime())
        A.verify_identity(manifest)
        changed = copy.deepcopy(manifest)
        path = next(iter(changed['source_sha256']))
        changed['source_sha256'][path] = '0'*64
        with self.assertRaises(ValueError):
            A.verify_identity(changed)
        changed = copy.deepcopy(manifest)
        changed['protocol_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            A.verify_identity(changed)

    def test_independent_calibration_summary(self):
        from binding import diagnose_b006 as D, run_b006 as R
        outputs = {p+'|8': D.simulate_series(self.series, p, 8) for p in ('D', 'R', 'P')}
        outputs['P_beta8_t80|8'] = outputs.pop('P|8')
        archives = [dict(seed=43, environments={'low_moderate': dict(series=self.series, outputs=outputs)})]
        confidence = A.confidence_estimates(archives)
        self.assertEqual(confidence, R.fixed_confidences(archives))
        self.assertEqual(A.reconstructed_summary(archives, confidence), R.summarize(archives, confidence))


if __name__ == '__main__':
    unittest.main()
