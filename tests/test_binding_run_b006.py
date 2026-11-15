import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding import run_b006 as R


class RunnerB006Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic IDs, never development or confirmation seeds.
        original=R.lab.make_series
        with patch.object(R.lab,'make_series',side_effect=lambda seed,env,noise=False:original(seed,env,episodes=2,noise=noise)):
            cls.archive=R.make_archive(7)

    def test_summary_independent_unit_is_series(self):
        summary=R.summarize([self.archive])
        self.assertEqual(summary['independent_series'],1)
        self.assertTrue(all(c['series']==1 for c in summary['primary_cells'].values()))

    def test_variant_grid(self):
        self.assertEqual(len(R.variants()),18)
        self.assertEqual(R.split_variant('Pxprime_beta99_t95'),('Pxprime','beta99_t95'))
        self.assertEqual(len(R.variants(dict(selected_K='K2of3',selected_P='beta8_t80'))),7)

    def test_gate_no_data_cannot_pass(self):
        gate=R.development_gate([self.archive])
        self.assertFalse(gate['pass_gate'])
        self.assertIsNone(gate['selected_P'])
        self.assertIn(gate['best_constant'],('D','R',gate['selected_K']))

    def test_confidence_frozen_accuracy(self):
        conf=R.fixed_confidences([self.archive])
        self.assertEqual(set(conf),{'D','R','K1of1','K2of3','K3of5'})
        self.assertTrue(all(0<=p<=1 for p in conf.values()))
        summary=R.summarize([self.archive],conf)
        for name,cal in summary['calibration'].items():
            self.assertEqual(sum(x['count'] for x in cal['bins']),cal['decisions'])

    def _save(self,root,archive):
        R.write(root/'series-7.json.gz',archive)
        manifest=dict(phase='development',seeds=[7],variants=R.variants(),environments=list(R.lab.ENVIRONMENTS),
                      files={'series-7.json.gz':R.file_hashes(root/'series-7.json.gz')},intakes={'7':archive['intake_sha256']})
        R.write(root/'manifest.json',manifest);R.write(root/'gate.json',{})

    def test_comparison_detects_extra_episode_and_input_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';self._save(a,self.archive);self._save(b,self.archive)
            self.assertTrue(R.compare(a,b)['decisions_pass'])
            changed=copy.deepcopy(self.archive)
            changed['environments']['low_zero']['outputs']['D|2'].append(changed['environments']['low_zero']['outputs']['D|2'][0])
            self._save(b,changed);self.assertFalse(R.compare(a,b)['decisions_pass'])
            changed=copy.deepcopy(self.archive)
            changed['environments']['low_zero']['series']['episodes'][0]['evaluator']['rates']['check_source']['false_bad']=.1
            self._save(b,changed);self.assertFalse(R.compare(a,b)['decisions_pass'])

    def test_comparison_preserves_diagnostic_drift_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';self._save(a,self.archive)
            changed=copy.deepcopy(self.archive)
            row=changed['environments']['low_zero']['outputs']['D|2'][0]['diagnostics']['posterior']
            row['observation']=math.nextafter(row['observation'],math.inf)
            self._save(b,changed);result=R.compare(a,b)
            self.assertTrue(result['decisions_pass']);self.assertTrue(result['numerical_diagnostics'])
            changed['environments']['low_zero']['outputs']['D|2'][0]['protected']['cost']+=1
            self._save(b,changed);self.assertFalse(R.compare(a,b)['decisions_pass'])

    def test_replay_requires_override_and_source_identity(self):
        metadata=dict(protocol_sha256='p',protocol_commit='c',source_sha256={'s':'v'})
        manifest=dict(metadata,runtime={'different':'runtime'})
        with patch.object(R,'source_metadata',return_value=metadata):
            with self.assertRaises(ValueError):R.verify_runtime_manifest(manifest)
            R.verify_runtime_manifest(manifest,True)
            manifest['protocol_sha256']='bad'
            with self.assertRaises(ValueError):R.verify_runtime_manifest(manifest,True)

    def test_runner_refuses_failed_freeze_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);freeze=root/'freeze.json';R.write(freeze,dict(gate=dict(pass_gate=False)))
            with patch.object(R,'source_metadata',return_value={}),patch.object(R,'committed_file',return_value='commit'):
                with self.assertRaises(ValueError):R.run(root/'out','confirmation',freeze=freeze)
            R.write(root/'occupied'/'x.json',{})
            with self.assertRaises(ValueError):R.run(root/'occupied','development')
