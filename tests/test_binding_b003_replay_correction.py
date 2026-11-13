"""Synthetic exact-correction replay admission, without scientific execution."""
import copy
from contextlib import ExitStack
from unittest.mock import patch
import unittest
from binding import run_b003 as runner


class ReachedSavedIntake(Exception):
    pass


class CorrectionReplayContracts(unittest.TestCase):
    def attempt(self, replay_compatibility=None, changes=None, verification_error=None):
        basic = dict(protocol_sha256='protocol', qualification_sha256='qualified', exposure_sha256='exposure')
        current = dict(basic, source_sha256={'runner': 'corrected'})
        old = dict(basic, source_sha256={'runner': 'executed'}, stage='confirmation',
                   freeze_sha256='freeze', source_compatibility_sha256='portability',
                   source_compatibility_commit='portability-commit')
        old.update(changes or {})
        provenance = dict(source_compatibility_sha256='portability',
                          source_compatibility_commit='portability-commit',
                          verifier_correction_sha256='correction',
                          verifier_correction_commit='correction-commit')
        development = dict(basic, source_sha256={'runner': 'development'})
        frozen = dict(development, status='freeze_before_confirmation', completed_series=12,
                      artifact_sha256={}, single_channel={'selected_packages': 4})
        def read(path):
            path = str(path)
            if path.endswith('freeze.json'):
                return copy.deepcopy(frozen)
            if path.endswith('manifest.json'):
                return copy.deepcopy(old if '/replay/' in path else development)
            raise ReachedSavedIntake()
        responses = [provenance, verification_error if verification_error else
                     provenance if replay_compatibility is None else replay_compatibility]
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'preflight', return_value=current))
            stack.enter_context(patch.object(runner, 'read_json', side_effect=read))
            stack.enter_context(patch.object(runner, 'sha', return_value='freeze'))
            stack.enter_context(patch.object(runner.b002, 'committed_file'))
            stack.enter_context(patch.object(runner.subprocess, 'check_output', return_value='freeze-commit'))
            stack.enter_context(patch('binding.audit_b003.verify_sources', side_effect=responses))
            runner.run('/tmp/b003-unused-correction-contract-output', 'confirmation',
                       development='binding/results/B003/synthetic-development',
                       replay='/synthetic/replay')

    def test_exact_correction_reaches_saved_intake(self):
        with self.assertRaises(ReachedSavedIntake):
            self.attempt()

    def test_confirmation_portability_alone_not_enough(self):
        with self.assertRaises(ValueError):
            self.attempt(replay_compatibility={'source_compatibility_sha256': 'portability',
                                             'source_compatibility_commit': 'portability-commit'})

    def test_unmapped_source_rejected(self):
        with self.assertRaises(ValueError):
            self.attempt(verification_error=ValueError('unmapped source'))

    def test_existing_correction_tampering_rejected(self):
        with self.assertRaises(ValueError):
            self.attempt(changes={'verifier_correction_sha256': 'tampered',
                                  'verifier_correction_commit': 'correction-commit'})

    def test_existing_portability_tampering_rejected(self):
        with self.assertRaises(ValueError):
            self.attempt(changes={'source_compatibility_sha256': 'tampered'})

    def test_changed_scientific_identity_rejected(self):
        with self.assertRaises(ValueError):
            self.attempt(changes={'protocol_sha256': 'changed'})


if __name__ == '__main__':
    unittest.main()
