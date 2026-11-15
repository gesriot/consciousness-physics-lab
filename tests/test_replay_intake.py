"""Saved authenticated inputs remain authoritative across arithmetic platforms."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding.replay_intake import compare_regenerated
from binding import run_b006 as R
from binding import audit_b006 as A


class IntakePortabilityTests(unittest.TestCase):
    def test_float_ulp_drift_reported(self):
        for value in (1., 1e-200, -1e100, 0.):
            result = compare_regenerated({'x':value}, {'x':math.nextafter(value, math.inf)})
            self.assertEqual(result['inputs.x']['ulps'], 1.)

    def test_integer_type_structure_and_large_float_changes_rejected(self):
        for a,b in (({'x':3},{'x':4}), ({'x':3},{'x':3.}),
                    ([1.],[1.,2.]), ({'a':1},{'b':1}),
                    (1.,1.001), (True,1), (float('nan'),float('nan'))):
            with self.assertRaises(ValueError):compare_regenerated(a,b)

    def test_canonical_manifest_detects_modified_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json.gz';R.write(path,{'x':1.})
            hashes=R.file_hashes(path)
            hashes['canonical_json_sha256']='0'*64
            with self.assertRaises(ValueError):R.verify_file(path,hashes)

    def test_replay_and_audit_use_saved_inputs(self):
        original_factory=R.lab.make_series
        original_intake=R.lab.physics.intake
        def factory(seed,env,noise=False,inputs=None):
            return original_factory(seed,env,episodes=2,noise=noise,inputs=inputs)
        def changed(seed):
            value=original_intake(seed)
            def perturb(x):
                if isinstance(x,dict):return {k:perturb(v) for k,v in x.items()}
                if isinstance(x,list):return [perturb(v) for v in x]
                if isinstance(x,float):return math.nextafter(x,math.inf)
                return x
            return perturb(value)
        with tempfile.TemporaryDirectory() as tmp, patch.object(R,'DEVELOPMENT',(7,)), \
             patch.object(R.lab,'make_series',side_effect=factory):
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            R.run(a,'development')
            with patch.object(R.lab.physics,'intake',side_effect=changed):
                R.run(b,'development',replay=a,override=True)
                archived=R.read(b/'series-7.json.gz')
                with patch.object(A.laboratory,'EPISODES',2):
                    self.assertEqual(A.audit_archive(archived),432)
            self.assertEqual(R.read(a/'series-7.json.gz'),R.read(b/'series-7.json.gz'))
            diagnostics=R.read(b/'manifest.json')['regenerated_input_diagnostics']['7']
            self.assertTrue(diagnostics)
            for mutate in (lambda x:x.__setitem__('seed',8),lambda x:x.pop('seed')):
                def malformed(seed):
                    value=original_intake(seed);mutate(value);return value
                with patch.object(R.lab.physics,'intake',side_effect=malformed):
                    with self.assertRaises(ValueError):
                        R.run(Path(tmp)/('bad'+str(len(list(Path(tmp).iterdir())))), 'development',replay=a,override=True)
                    with self.assertRaises(ValueError):A.audit_archive(archived)

    def test_legacy_identity_is_bounded(self):
        old=R.read(R.ROOT/'binding/results/B006/development/manifest.json')
        R.verify_runtime_manifest(old,True)
        bad=copy.deepcopy(old);bad['source_sha256']['binding/diagnose_b006.py']='0'*64
        with self.assertRaises(ValueError):R.verify_runtime_manifest(bad,True)
        current=R.source_metadata()['source_sha256'];current['binding/replay_intake.py']='0'*64
        self.assertFalse(R.source_compatible(old['source_sha256'],current))
