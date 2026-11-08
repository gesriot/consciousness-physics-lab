"""Streaming comparison matches the original in-memory comparison contract."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from attention import a004
from attention.compare_a004 import compare_paths
from attention.io import write_json


def sample():
    return dict(experiment='A004',intake_channel=[dict(task_id=0,x=1.)],
        global_permutations=[[[0,'modal']]],intake_sha256='i',self_model_sha256='m',
        environment={'platform':'one'},protocol_sha256='p',
        packages=[dict(package_id=i,spent=1.,actions=[dict(action='modal',cost=1.,score=.1+i,
            observed={'verdict':'accept','numeric':2.},expected_cost=.5)],planning_seconds=1.) for i in range(3)],
        summary={'0.2':{'count':2,'score':.3}},calibration={'brier':.2},
        stopping_rule={'close_budget_series':True})


class StreamingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def compare(self,a,b,gzip_output=True):
        suffix='.json.gz' if gzip_output else '.json'
        left,right=self.root/('a'+suffix),self.root/('b'+suffix)
        write_json(left,a);write_json(right,b)
        actual=compare_paths(left,right)
        self.assertEqual(actual,a004.compare(a,b))
        return actual

    def test_equal_and_informational_drift_equivalence(self):
        a=sample();b=copy.deepcopy(a)
        b['environment']={'platform':'two'}
        b['replay']={'source':'different'}
        b['calibration']['brier']=.9
        for i,p in enumerate(b['packages']):
            p['planning_seconds']=100.
            p['actions'][0]['score']+=.1*(i+1)
            p['actions'][0]['observed']['numeric']+=i*.2
        result=self.compare(a,b)
        self.assertTrue(result['decisions_pass'])
        row=result['numeric_diagnostics']['$[].actions[].score']
        self.assertEqual(row['count'],3);self.assertEqual(row['changed'],3)
        self.compare(a,b,False)

    def test_authoritative_decision_and_metadata_changes(self):
        for field,value in [('action','binding'),('cost',1.1),('expected_cost',.6)]:
            a=sample();b=copy.deepcopy(a);b['packages'][1]['actions'][0][field]=value
            result=self.compare(a,b)
            self.assertFalse(result['decisions_pass']);self.assertEqual(result['differing_packages'],[1])
        for key,value in [('protocol_sha256','other'),('intake_sha256','other'),('self_model_sha256','other'),
                          ('intake_channel',[{'task_id':0,'x':1.1}]),('stopping_rule',{'close_budget_series':False})]:
            a=sample();b=copy.deepcopy(a);b[key]=value
            self.assertFalse(self.compare(a,b)['decisions_pass'])

    def test_length_empty_and_missing_fields_equivalence(self):
        a=sample();b=copy.deepcopy(a);b['packages'].pop()
        self.assertFalse(self.compare(a,b)['decisions_pass'])
        a['packages']=[];b['packages']=[]
        self.assertTrue(self.compare(a,b)['decisions_pass'])
        a=sample();b=copy.deepcopy(a);del b['packages'][0]['actions']
        self.assertFalse(self.compare(a,b)['decisions_pass'])

    def test_root_order_does_not_matter(self):
        a=sample();b=dict(reversed(list(a.items())))
        self.assertTrue(self.compare(a,b)['decisions_pass'])

    def test_invalid_and_truncated_archives_rejected(self):
        valid=self.root/'valid.json';write_json(valid,sample())
        raw=valid.read_text()
        variants=[raw[:-3],raw+'garbage',raw.replace('  "packages": [','  "packages": null,'),
                  raw.replace('  "experiment": "A004",','  "experiment": "A004",\n  "experiment": "A004",'),
                  raw.replace('"spent": 1.0','"spent": NaN'),
                  raw.replace('"spent": 1.0','"spent": 1e999'),
                  raw.replace('"numeric": 2.0','"numeric": -1e999'),json.dumps(sample()),
                  raw.replace('    },\n    {','    }\n    {',1),
                  raw.replace('  "environment": {','  "environment": {\n    "platform": "duplicate",')]
        for i,value in enumerate(variants):
            broken=self.root/f'broken{i}.json';broken.write_text(value)
            with self.subTest(i=i):
                with self.assertRaises(ValueError):compare_paths(valid,broken)
        gz=self.root/'truncated.json.gz';gz.write_bytes(gzip.compress(raw.encode())[:-8])
        with self.assertRaises(ValueError):compare_paths(valid,gz)


if __name__=='__main__':unittest.main()
