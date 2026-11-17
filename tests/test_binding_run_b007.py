import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding import run_b007 as R


class B007RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=R.lab.make_series
        with patch.object(R.lab,'make_series',side_effect=lambda seed,env,noise=False,inputs=None:original(seed,env,episodes=2,noise=noise,inputs=inputs)):
            cls.archive=R.make_archive(9)

    def test_variants(self):
        self.assertEqual(len(R.variants()),26)
        self.assertEqual(R.split_variant('Pstar_beta2_t95'),('Pstar','beta2_t95'))
        self.assertEqual(len(R.variants(dict(selected_K='K2of3',selected_P='beta4_t80'))),9)

    def test_gate_quantified_and_no_small_exposure(self):
        gate=R.development_gate([self.archive])
        self.assertFalse(gate['pass_gate'])
        for config,row in gate['configurations'].items():
            self.assertEqual(set(row['reliability']),{'low_moderate','high_moderate'})
            self.assertIn('many_Px_changes',row)
            self.assertTrue(all(x['metric'] in ('false_accusations','lost_good') for x in row['low_zero_failures']))

    def test_saved_input_reused(self):
        inputs=copy.deepcopy(self.archive['inputs'])
        original=R.lab.make_series
        with patch.object(R.lab,'make_series',side_effect=lambda seed,env,noise=False,inputs=None:original(seed,env,episodes=1,noise=noise,inputs=inputs)),patch.object(R.lab.physics,'intake',side_effect=AssertionError('regeneration forbidden')):
            archive=R.make_archive(9,inputs=inputs)
        self.assertEqual(archive['inputs'],inputs)

    def _save(self,path,archive):
        R.write(path/'series-9.json.gz',archive)
        R.write(path/'manifest.json',dict(phase='development',seeds=[9],variants=R.variants(),environments=list(R.lab.ENVIRONMENTS),
                    files={'series-9.json.gz':R.file_hashes(path/'series-9.json.gz')},intakes={'9':archive['intake_sha256']}))
        R.write(path/'gate.json',dict(pass_gate=False,diagnostic=.5))
        R.write(path/'summary.json',dict(count=2,diagnostic=.5))

    def test_comparison_strict_decisions_diagnostic_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';self._save(a,self.archive);self._save(b,self.archive)
            R.write(b/'gate.json',dict(pass_gate=False,diagnostic=math.nextafter(.5,1.)))
            self.assertTrue(R.compare(a,b)['decisions_pass'])
            changed=copy.deepcopy(self.archive)
            changed['environments']['low_zero']['outputs']['D|2'][0]['protected']['cost']+=1
            self._save(b,changed)
            self.assertFalse(R.compare(a,b)['decisions_pass'])

    def test_runtime_override_not_source_override(self):
        metadata=dict(protocol_sha256='a',protocol_commit='b',source_sha256={'x':'y'})
        manifest=dict(metadata,runtime={'other':1})
        with patch.object(R,'source_metadata',return_value=metadata):
            with self.assertRaises(ValueError):R.verify_runtime_manifest(manifest)
            R.verify_runtime_manifest(manifest,True)
            manifest['protocol_sha256']='c'
            with self.assertRaises(ValueError):R.verify_runtime_manifest(manifest,True)

    def test_integer_types_are_strict(self):
        for a,b in ((1,1.),(1,True),(0,False)):
            with self.assertRaises(ValueError):R.strict_diagnostics(a,b)

    def test_summary_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';self._save(a,self.archive);self._save(b,self.archive)
            R.write(b/'summary.json',dict(count=3,diagnostic=.5))
            self.assertFalse(R.compare(a,b)['decisions_pass'])

    def test_analyzer_authenticates_embedded_inputs(self):
        from binding import analyze_b007 as A
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);changed=copy.deepcopy(self.archive)
            changed['intake_sha256']='not-the-input-digest'
            self._save(path,changed)
            with self.assertRaisesRegex(ValueError,'Intake identity'):
                A.analyze(path)
