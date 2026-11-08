import gzip
import json
from pathlib import Path
import tempfile
import unittest
from attention import a007
from attention.compare_a004 import Archive


class RunnerTests(unittest.TestCase):
    def test_durable_packages_and_canonical_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);paths=[]
            for i in range(3):
                path=base/(str(i)+'.json.gz');a007.atomic_json(path,dict(package_id=i,inputs=[],conditions={}))
                paths.append(path)
            output=base/'result.json.gz'
            a007.assemble(output,dict(experiment='synthetic',labels=['a','b']),paths,{'count':3})
            archive=Archive(output);rows=list(archive.packages())
            self.assertEqual([r['package_id'] for r in rows],[0,1,2])
            self.assertEqual(archive.metadata['summary'],{'count':3})
            self.assertEqual(json.load(gzip.open(output,'rt'))['labels'],['a','b'])
            self.assertFalse(list(base.glob('*.partial')))
            self.assertEqual(paths[0].read_bytes()[4:8],b'\0'*4)

    def test_closure_requires_two_budgets_and_both_controls(self):
        summary=a007.empty_summary()
        self.assertTrue(a007.stopping_rule(summary)['series_closed'])
        for f in ('0.2','0.4'):
            for p in ('BU','C2A'):
                summary['permissive'][f][p]['false_accepts_after']=1
        result=a007.stopping_rule(summary)
        self.assertFalse(result['series_closed'])
        self.assertEqual(result['passing_limited_budgets'],2)
        summary['permissive']['0.4']['BU']['false_accepts_after']=0
        self.assertTrue(a007.stopping_rule(summary)['series_closed'])

    def test_different_endpoints_cannot_mix_controls(self):
        summary=a007.empty_summary()
        for f in ('0.2','0.4','0.6'):
            summary['permissive'][f]['BU']['false_accepts_after']=1
            summary['underestimated'][f]['C2A']['delay_inputs_penalized']=1
        self.assertEqual(a007.stopping_rule(summary)['passing_limited_budgets'],0)

    def test_replay_rejects_another_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wrong.json.gz'
            a007.atomic_json(path,dict(experiment='OTHER',intake_channel=[],fault_specs=[],intake_sha256='bad'))
            with self.assertRaisesRegex(ValueError,'Replay experiment'):a007.intake(path)


if __name__=='__main__':unittest.main()
