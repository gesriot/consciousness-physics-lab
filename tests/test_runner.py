import unittest
from lab.runner import run_experiment

REQUEST = {"schema_version": "1.0", "mode": "analytic", "units": {"time": "s"},
           "spheres": [{"mean_time": 1.0}], "points": [{"re": 0., "im": 0.}]}

class RunnerTests(unittest.TestCase):
    def test_baseline_reports_observation_without_experience_claim(self):
        out = run_experiment(REQUEST, repeats=1)
        self.assertEqual(out["runs"][0]["exit_code"], 0)
        self.assertEqual(out["runs"][0]["response"]["results"][0]["analytic"]["re"], 1.)
        self.assertIsNone(out["consciousness_inference"])
        self.assertGreater(out["timing"]["median_seconds"], 0)
        self.assertEqual(len(out["request_sha256"]), 64)

    def test_invalid_science_input_is_not_converted_to_success(self):
        bad = {**REQUEST, "spheres": [{"mean_time": -1.}]}
        self.assertEqual(run_experiment(bad, 1)["runs"][0]["exit_code"], 2)

    def test_bounded_repeats(self):
        for repeats in [0, 11, True, 1.5]:
            with self.assertRaises(ValueError):
                run_experiment(REQUEST, repeats)
