import tempfile
import unittest
from pathlib import Path
from attention.io import write_json
from attention.compare_a005 import compare_paths

class ReplayComparisonTests(unittest.TestCase):
    def fixture(self):
        return dict(experiment='A005',intake_channel=[],global_permutations=[],intake_sha256='x',self_model_sha256='y',onset_package=4,calibration={},summary={},full_pipeline_parity_failures=[],request_overlap={},packages=[{'spent':1.,'probability':.5,'verdict':'accept'}])
    def compare(self,change):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.json.gz',Path(directory)/'b.json.gz'
            first=self.fixture();second=self.fixture();change(second)
            write_json(a,first);write_json(b,second)
            return compare_paths(a,b)
    def test_float_diagnostic_is_not_a_decision(self):
        self.assertTrue(self.compare(lambda x:x['packages'][0].update(probability=.50000000001))['decisions_pass'])
    def test_changed_cost_fails(self):
        self.assertFalse(self.compare(lambda x:x['packages'][0].update(spent=2.))['decisions_pass'])
    def test_changed_verdict_fails(self):
        self.assertFalse(self.compare(lambda x:x['packages'][0].update(verdict='inconclusive'))['decisions_pass'])

    def test_expected_plan_cost_is_diagnostic(self):
        def change(x):x['packages'][0]['plan']={'expected_cost':1.0000000000000002}
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.json.gz',Path(directory)/'b.json.gz'
            first=self.fixture();first['packages'][0]['plan']={'expected_cost':1.}
            second=self.fixture();change(second)
            write_json(a,first);write_json(b,second)
            self.assertTrue(compare_paths(a,b)['decisions_pass'])
