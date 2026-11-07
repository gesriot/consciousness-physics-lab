"""Integration tests: the CLI is run as a subprocess, as an agent would run it."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "schema_version": "1.0",
    "mode": "compare",
    "units": {"time": "s"},
    "spheres": [{"mean_time": 1.0}],
    "points": [{"re": 0.0, "im": 0.0}, {"re": 0.5, "im": 3.0}],
    "modes": 500,
}


def run(request=None, args=(), raw=None):
    """Return (exit code, parsed stdout, stderr); stdout must be one JSON document."""
    if raw is None and request is not None:
        raw = json.dumps(request).encode()
    proc = subprocess.run([sys.executable, "-m", "sphere_thermal", *args], input=raw,
                          capture_output=True, cwd=ROOT, timeout=300)
    stdout = proc.stdout.decode()
    return proc.returncode, (json.loads(stdout) if stdout else None), proc.stderr.decode()


def request(**changes):
    data = copy.deepcopy(BASE)
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


class ModeTests(unittest.TestCase):
    def test_analytic_mode(self):
        code, doc, stderr = run(request(mode="analytic", modes=None))
        self.assertEqual(code, 0)
        self.assertEqual(doc["schema_version"], "1.0")
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["input"], request(mode="analytic", modes=None))
        self.assertEqual(doc["modes"], [])
        self.assertEqual(doc["results"][0]["analytic"], {"re": 1.0, "im": 0.0})
        self.assertEqual(doc["results"][0]["modal"], [])
        self.assertIsNone(doc["summary"]["all_consistent"])
        self.assertIn("status=ok", stderr)

    def test_modal_mode_does_not_compute_analytic(self):
        code, doc, _ = run(request(mode="modal"))
        self.assertEqual(code, 0)
        for result in doc["results"]:
            self.assertIsNone(result["analytic"])
            level = result["modal"][0]
            self.assertEqual(level["modes"], 500)
            self.assertIsNone(level["abs_discrepancy"])
            self.assertIsNone(level["consistent"])
            self.assertGreater(level["tail_bound"], 0)
        self.assertLess(doc["results"][0]["modal"][0]["value"]["re"], 1.0)

    def test_compare_mode_reports_discrepancy_and_tail(self):
        code, doc, _ = run(request(tolerance=1e-2))
        self.assertEqual((code, doc["status"]), (0, "ok"))
        self.assertEqual(doc["modes"], [500])
        for result in doc["results"]:
            level = result["modal"][0]
            for key in ("abs_discrepancy", "tail_bound", "rounding_allowance"):
                self.assertIsInstance(level[key], float)
            self.assertTrue(level["consistent"])
            self.assertTrue(level["tolerance_met"])
            self.assertLessEqual(level["abs_discrepancy"],
                                 level["tail_bound"] + level["rounding_allowance"])
        self.assertTrue(doc["summary"]["all_consistent"])
        self.assertTrue(any("rounding" in note for note in doc["notes"]))

    def test_convergence_report(self):
        code, doc, _ = run(request(mode="convergence", modes=None,
                                   modes_list=[10, 100, 1000, 10000], tolerance=1e-3))
        self.assertEqual((code, doc["status"]), (0, "ok"))
        levels = doc["summary"]["levels"]
        self.assertEqual([lv["modes"] for lv in levels], [10, 100, 1000, 10000])
        self.assertEqual([lv["tolerance_met"] for lv in levels], [False, False, True, True])
        self.assertEqual(doc["summary"]["smallest_modes_meeting_tolerance"], 1000)
        # 6/(pi^2 (N + 1/2)) <= 1e-3 first holds at N = 608.
        self.assertEqual(doc["summary"]["modes_required_by_bound"], 608)
        bounds = [lv["max_tail_bound"] for lv in levels]
        self.assertEqual(bounds, sorted(bounds, reverse=True))
        self.assertTrue(all(lv["all_consistent"] for lv in levels))

    def test_tolerance_not_met_is_explicit(self):
        code, doc, stderr = run(request(modes=1000, tolerance=1e-8))
        self.assertEqual((code, doc["status"]), (3, "tolerance_not_met"))
        self.assertFalse(doc["summary"]["tolerance_met"])
        self.assertIsNone(doc["summary"]["smallest_modes_meeting_tolerance"])
        self.assertGreater(doc["summary"]["modes_required_by_bound"], 10_000_000)
        self.assertIn("NOT met", doc["messages"][0])
        self.assertIn("hard limit", doc["messages"][0])
        self.assertIn("NOT met", stderr)

    def test_tolerance_below_rounding_level_is_reported(self):
        code, doc, _ = run(request(modes=10, tolerance=1e-17))
        self.assertEqual(code, 3)
        self.assertTrue(any("rounding level" in m for m in doc["messages"]))

    def test_mixture_is_weighted_sum_of_single_spheres(self):
        parts = [(0.25, 0.1), (0.75, 3.0)]
        # At z = 1e6 i the tail bounds of the two spheres differ (branch B for T = 3).
        points = BASE["points"] + [{"re": 0.0, "im": 1e6}]
        mixture = run(request(points=points,
                              spheres=[{"weight": a, "mean_time": t} for a, t in parts]))[1]
        singles = [run(request(points=points, spheres=[{"mean_time": t}]))[1] for _, t in parts]
        self.assertNotEqual(singles[0]["results"][2]["modal"][0]["tail_bound"],
                            singles[1]["results"][2]["modal"][0]["tail_bound"])
        self.assertAlmostEqual(mixture["medium"]["mean_relaxation_time"], 0.25 * 0.1 + 0.75 * 3.0,
                               delta=1e-15)
        for i, result in enumerate(mixture["results"]):
            for part in ("re", "im"):
                expected = sum(a * s["results"][i]["analytic"][part] for (a, _), s in zip(parts, singles))
                self.assertAlmostEqual(result["analytic"][part], expected, delta=1e-15)
                expected = sum(a * s["results"][i]["modal"][0]["value"][part]
                               for (a, _), s in zip(parts, singles))
                self.assertAlmostEqual(result["modal"][0]["value"][part], expected, delta=1e-15)
            expected = sum(a * s["results"][i]["modal"][0]["tail_bound"]
                           for (a, _), s in zip(parts, singles))
            self.assertAlmostEqual(result["modal"][0]["tail_bound"], expected, delta=1e-15)
            self.assertTrue(result["modal"][0]["consistent"])

    def test_radius_and_diffusivity_define_mean_time(self):
        geometric = run(request(units={"time": "s", "length": "mm"},
                                spheres=[{"radius": 2.0, "diffusivity": 0.5}]))[1]
        direct = run(request(spheres=[{"mean_time": 4.0 / 7.5}]))[1]
        self.assertAlmostEqual(geometric["medium"]["spheres"][0]["mean_time"], 4.0 / 7.5, delta=1e-16)
        self.assertEqual(geometric["results"], direct["results"])
        self.assertEqual(geometric["units"]["diffusivity"], "mm^2/s")
        self.assertEqual(geometric["units"]["z"], "1/s")

    def test_file_and_stdin_inputs_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            path.write_text(json.dumps(BASE))
            from_file = run(args=[str(path)])
        from_stdin = run(BASE)
        from_dash = run(BASE, args=["-"])
        self.assertEqual(from_file[:2], from_stdin[:2])
        self.assertEqual(from_dash[:2], from_stdin[:2])

    def test_stdout_is_json_and_diagnostics_go_to_stderr(self):
        proc = subprocess.run([sys.executable, "-m", "sphere_thermal"], input=json.dumps(BASE).encode(),
                              capture_output=True, cwd=ROOT, timeout=300)
        text = proc.stdout.decode()
        json.loads(text)
        self.assertTrue(text.startswith("{") and text.rstrip().endswith("}"))
        self.assertNotIn("sphere-thermal:", text)
        self.assertIn("sphere-thermal: mode=compare", proc.stderr.decode())


class InvalidInputTests(unittest.TestCase):
    def assert_invalid(self, data, path, kind="invalid_input", raw=None):
        code, doc, stderr = run(data, raw=raw)
        self.assertEqual(code, 2, doc)
        self.assertEqual(doc["status"], "error")
        self.assertEqual(doc["error"]["kind"], kind, doc)
        self.assertEqual(doc["error"]["path"], path, doc)
        self.assertTrue(doc["error"]["message"])
        self.assertIn("error", stderr)

    def test_invalid_values_report_their_json_path(self):
        two = [{"weight": 0.5, "mean_time": 1.0}, {"weight": 0.5, "mean_time": 2.0}]
        cases = {
            "left half-plane": (request(points=[{"re": -1e-9, "im": 1.0}]), "$.points[0].re"),
            "weights not normalized": (request(spheres=[{"weight": 0.5, "mean_time": 1.0},
                                                        {"weight": 0.4, "mean_time": 2.0}]), "$.spheres"),
            "negative weight": (request(spheres=[{"weight": 1.5, "mean_time": 1.0},
                                                 {"weight": -0.5, "mean_time": 2.0}]), "$.spheres[1].weight"),
            "missing mixture weight": (request(spheres=[two[0], {"mean_time": 2.0}]), "$.spheres[1].weight"),
            "both size forms": (request(spheres=[{"mean_time": 1.0, "radius": 1.0}]), "$.spheres[0]"),
            "radius without diffusivity": (request(units={"time": "s", "length": "m"},
                                                   spheres=[{"radius": 1.0}]), "$.spheres[0].diffusivity"),
            "radius without length unit": (request(spheres=[{"radius": 1.0, "diffusivity": 1.0}]),
                                           "$.units.length"),
            "unknown key": (request(tolerence=1e-6), "$.tolerence"),
            "missing modes": (request(modes=None), "$.modes"),
            "modes in analytic mode": (request(mode="analytic"), "$.modes"),
            "too many modes": (request(modes=10**8), "$.modes"),
            "boolean modes": (request(modes=True), "$.modes"),
            "zero modes": (request(modes=0), "$.modes"),
            "fractional modes": (request(modes=10.5), "$.modes"),
            "single convergence level": (request(mode="convergence", modes=None, modes_list=[100]),
                                         "$.modes_list"),
            "unsorted convergence levels": (request(mode="convergence", modes=None,
                                                    modes_list=[100, 10]), "$.modes_list"),
            "schema version": (request(schema_version="2.0"), "$.schema_version"),
            "unknown mode": (request(mode="fourier"), "$.mode"),
            "negative tolerance": (request(tolerance=-1e-6), "$.tolerance"),
            "missing imaginary part": (request(points=[{"re": 1.0}]), "$.points[0].im"),
            "point as string": (request(points=["1+2j"]), "$.points[0]"),
            "empty spheres": (request(spheres=[]), "$.spheres"),
            "zero mean time": (request(spheres=[{"mean_time": 0}]), "$.spheres[0].mean_time"),
            "string number": (request(spheres=[{"mean_time": "1.0"}]), "$.spheres[0].mean_time"),
            "empty time unit": (request(units={"time": " "}), "$.units.time"),
            "argument too large": (request(points=[{"re": 0.0, "im": 1e300}]), "$.points[0]"),
            "work limit": (request(modes=10**7, points=[{"re": 0.0, "im": float(i)} for i in range(200)]),
                           "$"),
            "not an object": ([1, 2, 3], "$"),
        }
        for name, (data, path) in cases.items():
            with self.subTest(name):
                self.assert_invalid(data, path)

    def test_malformed_json(self):
        good = json.dumps(BASE)
        cases = {
            "truncated": (b"{", "invalid_json"),
            "NaN literal": (good.replace('"mean_time": 1.0', '"mean_time": NaN').encode(), "invalid_json"),
            "duplicate key": (good.replace('"modes": 500', '"modes": 500, "modes": 600').encode(),
                              "invalid_json"),
            "not UTF-8": (b'{"mode": "\xff"}', "invalid_json"),
            "overflowing literal": (good.replace('"mean_time": 1.0', '"mean_time": 1e400').encode(),
                                    "invalid_input"),
        }
        for name, (raw, kind) in cases.items():
            with self.subTest(name):
                path = "$.spheres[0].mean_time" if kind == "invalid_input" else "$"
                self.assert_invalid(None, path, kind, raw=raw)

    def test_usage_errors(self):
        code, doc, _ = run(args=["a.json", "b.json"], raw=b"")
        self.assertEqual((code, doc["error"]["kind"]), (2, "usage"))
        code, doc, _ = run(args=["--pretty"], raw=b"")
        self.assertEqual((code, doc["error"]["kind"]), (2, "usage"))
        code, doc, _ = run(args=[str(ROOT / "no-such-request.json")], raw=b"")
        self.assertEqual((code, doc["error"]["kind"]), (2, "io"))
        code, doc, stderr = run(args=["--help"], raw=b"")
        self.assertEqual((code, doc), (0, None))
        self.assertIn("usage:", stderr)


class ExampleTests(unittest.TestCase):
    EXPECTED = {
        "single_analytic.json": (0, "ok"),
        "single_compare.json": (0, "ok"),
        "mixture_convergence.json": (0, "ok"),
        "tolerance_not_met.json": (3, "tolerance_not_met"),
        "invalid_left_half_plane.json": (2, "error"),
    }

    def test_examples_produce_documented_status(self):
        files = sorted(p.name for p in (ROOT / "examples").glob("*.json"))
        self.assertEqual(files, sorted(self.EXPECTED))
        for name, (exit_code, status) in self.EXPECTED.items():
            with self.subTest(name):
                code, doc, _ = run(args=[str(ROOT / "examples" / name)], raw=b"")
                self.assertEqual((code, doc["status"]), (exit_code, status))

    def test_mixture_example_keeps_mean_time_and_converges(self):
        doc = run(args=[str(ROOT / "examples" / "mixture_convergence.json")], raw=b"")[1]
        self.assertAlmostEqual(doc["medium"]["mean_relaxation_time"], 1.0, delta=1e-11)
        self.assertEqual(doc["summary"]["smallest_modes_meeting_tolerance"], 1000000)
        self.assertTrue(doc["summary"]["all_consistent"])


if __name__ == "__main__":
    unittest.main()
