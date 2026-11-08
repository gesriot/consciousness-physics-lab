"""Compressed archive equivalence and replay integrity."""
import tempfile
import unittest
from pathlib import Path
from attention.io import read_json, write_json, load_replay
from lab.executor import request_hash
from lab.guard import Request
from lab.common_mode import encode


class CompressedArchives(unittest.TestCase):
    def test_roundtrip_determinism_and_replay(self):
        request=Request(1+2j,1,.01)
        value=dict(experiment='A002',intake_channel=[dict(task_id=0,request=encode(request),request_sha256=request_hash(request))])
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ('first.json.gz','second.json.gz','plain.json')]
            for path in paths:write_json(path,value)
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
            for path in paths:
                self.assertEqual(read_json(path),value)
                data,requests,provenance=load_replay(path,'A002',1)
                self.assertEqual(data,value)
                self.assertEqual(requests,[request])
                self.assertEqual(len(provenance['source_sha256']),64)
            value['intake_channel'][0]['request_sha256']='wrong'
            write_json(paths[0],value)
            with self.assertRaisesRegex(ValueError,'digest'):load_replay(paths[0],'A002',1)

    def test_reject_experiment_count_order_and_nonfinite(self):
        request=Request(1,1,.01)
        value=dict(experiment='A002',intake_channel=[dict(task_id=0,request=encode(request))])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.json.gz'
            write_json(path,value)
            with self.assertRaises(ValueError):load_replay(path,'A003',1)
            with self.assertRaises(ValueError):load_replay(path,'A002',2)
            value['intake_channel'][0]['task_id']=1;write_json(path,value)
            with self.assertRaises(ValueError):load_replay(path,'A002',1)
            value['intake_channel'][0]['task_id']=0
            value['intake_channel'][0]['request']['z']=[float('inf'),0]
            with self.assertRaises(ValueError):write_json(path,value)


if __name__=='__main__':unittest.main()
