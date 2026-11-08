import copy
import math
import unittest
from collections import Counter
import mpmath as mp
from attention import s3
from attention.build_a007_library import CHANNELS, V2_EDGES, TOL_EDGES, channel
from attention.calibration_samples import execute_sample, maximum_cost
from attention.io import read_json
from lab.replay import canonical_sha256


class ChannelLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=read_json('attention/sample_library_a007.json')
        cls.development=read_json('attention/results/A007/development.json')
        cls.weights=s3.load_model()['action_model']['cost_weights']

    def test_exact_channel_partition_and_provenance(self):
        self.assertEqual(len(CHANNELS),18)
        self.assertEqual(self.library['channels'],list(CHANNELS))
        self.assertEqual(set(s['channel'] for s in self.library['samples']),set(CHANNELS))
        self.assertEqual(self.library['samples_sha256'],canonical_sha256(self.library['samples']))
        self.assertEqual(self.development['requests'],60)
        self.assertEqual(set(self.development['channel_counts'].values()),{10})
        self.assertEqual(len(self.development['channel_counts']),12)
        for sample in self.library['samples']:
            self.assertEqual(sample['channel'],channel(sample['family'],sample['request']))

    def test_upper_inclusive_boundaries(self):
        for i,x in enumerate(V2_EDGES):
            self.assertEqual(channel('v2',dict(z=[x,0],mean_time=1,tolerance=1)),'v2:'+str(i))
            self.assertEqual(channel('v2_512',dict(z=[math.nextafter(x,math.inf),0],mean_time=1,tolerance=1)),'v2:'+str(i+1))
        for i,x in enumerate(TOL_EDGES):
            self.assertEqual(channel('collocation',dict(z=[1,0],mean_time=1,tolerance=x)),'collocation:'+str(i))
            self.assertEqual(channel('collocation',dict(z=[1,0],mean_time=1,tolerance=math.nextafter(x,math.inf))),'collocation:'+str(i+1))

    def test_all_standards_execute_with_exact_verdicts(self):
        self.assertEqual(len(self.library['samples']),38)
        for sample in self.library['samples']:
            with self.subTest(sample=sample['sample_id'],channel=sample['channel']):
                result=execute_sample(sample,self.weights)
                self.assertEqual(result['verdict'],sample['expected'])
                self.assertLessEqual(result['cost'],sample['maximum_cost'])
                self.assertEqual(sample['maximum_cost'],maximum_cost(sample,self.weights))
                poisoned=copy.deepcopy(sample);poisoned['expected']='invented'
                self.assertEqual(execute_sample(poisoned,self.weights)['verdict'],result['verdict'])

    def test_true_boundary_errors_and_known_fault(self):
        boundaries=[s for s in self.library['samples'] if s['standard_kind']=='boundary']
        self.assertEqual(len(boundaries),5)
        self.assertEqual({s['channel'] for s in boundaries},{'v2:'+str(i) for i in range(5)})
        self.assertEqual(self.development['channels_without_boundary'],['v2:5'])
        for sample in boundaries:
            with self.subTest(channel=sample['channel']):
                with mp.workdps(60):
                    candidate=mp.mpc(*sample['candidate']);reference=mp.mpc(*sample['reference_decimal'])
                    actual_error=abs(candidate-reference);tolerance=mp.mpf(sample['request']['tolerance'])
                    self.assertGreater(actual_error,tolerance)
                    self.assertGreater(actual_error/tolerance,mp.mpf('1.00999'))
                    self.assertLess(actual_error/tolerance,mp.mpf('1.01001'))
                self.assertFalse(sample['exact_reference'])
                self.assertNotEqual(execute_sample(sample,self.weights)['verdict'],'accept')
                broken=execute_sample(sample,self.weights,'underestimated')
                self.assertEqual(broken['verdict'],'accept')
                self.assertTrue(any(t['original']['verdict']!=t['observed']['verdict'] for t in broken['trace']))

    def test_budget_reserves_and_development_hazard(self):
        quota=self.development['quota_reference']
        self.assertEqual(quota['maximum_sample_cost'],9.)
        for package in self.development['development_packages']:
            for fraction,q in zip(quota['fractions'],quota['quotas']):
                self.assertLess(q*quota['maximum_sample_cost'],fraction*package['reference_cost'])
        vulnerable=[p for p in self.development['development_packages'] if p['permissive_collocation_vulnerabilities']]
        self.assertEqual(len(vulnerable),7)
        self.assertEqual(sum(len(p['permissive_collocation_vulnerabilities']) for p in vulnerable),9)


if __name__=='__main__':unittest.main()
