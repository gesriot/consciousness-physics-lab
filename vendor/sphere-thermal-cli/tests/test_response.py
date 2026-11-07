"""Unit tests of the analytic and modal representations of g(z; T)."""
import math
import unittest

import numpy as np

from sphere_thermal import response as r

EPS = r.EPS
# Right half-plane and imaginary-axis arguments with very different scales.
POINTS = [0.37 + 2.1j, 4.0, 9j, -9j, 25 + 25j, 1e3j, 1e6j, 1e-6 + 3e-6j, 1e-3 + 50j, 0.06 + 0.03j]


def reference(z, mean_time, m=20000):
    """Independent value: direct mode sum to m plus an Euler-Maclaurin tail.

    sum_{n>m} f(n) = int_m^inf f - f(m)/2 - f'(m)/12 + O(f'''(m)) for
    f(t) = 6/(pi^2 t^2 + x). Uses neither the closed form nor response.py sums.
    """
    x = 15 * mean_time * complex(z)
    n = np.arange(1, m + 1, dtype=float)
    terms = 6 / (math.pi**2 * n * n + x)
    head = complex(math.fsum(terms.real), math.fsum(terms.imag))
    if x == 0:
        integral = 6 / (math.pi**2 * m)
    else:
        integral = 6 / (math.pi * np.sqrt(x)) * np.arctan(np.sqrt(x) / (math.pi * m))
    denominator = math.pi**2 * m * m + x
    return head + integral - 3 / denominator + math.pi**2 * m / denominator**2


class AnalyticTests(unittest.TestCase):
    def test_taylor_coefficients_match_zeta_values(self):
        # The modal form gives c_k = (-1)^k 6 zeta(2k+2) / pi^(2k+2).
        n = np.arange(1, 100001, dtype=float)
        for k, c in enumerate(r.TAYLOR[1:], start=1):
            zeta = math.fsum(n**-(2.0 * k + 2))
            expected = (-1)**k * 6 * zeta / math.pi**(2 * k + 2)
            self.assertLess(abs(c - expected), 1e-13 * abs(expected), k)
        self.assertEqual(r.TAYLOR[0], 1.0)
        self.assertAlmostEqual(r.TAYLOR[1], -1 / 15, delta=EPS)
        self.assertAlmostEqual(r.TAYLOR[2], 2 / 315, delta=EPS)

    def test_zero_limit_is_exact(self):
        for mean_time in [1e-6, 1.0, 1e6]:
            g = r.analytic([0.0, 0j, complex(0.0, -0.0)], mean_time)
            np.testing.assert_array_equal(g, [1.0, 1.0, 1.0])

    def test_small_arguments_keep_full_precision(self):
        # |x| <= 1.5e-5, so the omitted x^4 term is below 1e-23.
        for v in [1e-6, 1e-10, 1e-14, 1e-200]:
            for direction in [1, 1j, -1j, (1 + 1j) / math.sqrt(2)]:
                z = v * direction / 2.5
                x = 15 * 2.5 * z
                expected = 1 - x / 15 + 2 * x * x / 315 - x**3 / 1575
                g = r.analytic([z], 2.5)[0]
                self.assertLess(abs(g - expected), 2 * EPS)
                if direction == 1j:
                    # The part a naive q coth q - 1 would destroy by cancellation.
                    self.assertLess(abs(g.imag / expected.imag - 1), 4 * EPS)

    def test_series_and_closed_form_agree_at_switch(self):
        for radius in [0.5, 0.9, 1.0, 1.1]:
            angles = np.linspace(-math.pi / 2, math.pi / 2, 41)
            x = radius * np.exp(1j * angles)
            difference = np.abs(r._series(x) - r._closed_form(x))
            self.assertLess(difference.max(), 16 * EPS, radius)

    def test_large_arguments_do_not_overflow(self):
        z = np.array([1e295, 1e290j, 1e294 + 1e294j, 3e7j])
        g = r.analytic(z, 1.0)
        q = np.sqrt(15 * z)
        self.assertTrue(np.all(np.isfinite(g)))
        # g = 3/q - 3/q^2 up to exponentially small terms.
        np.testing.assert_allclose(g, 3 / q - 3 / q**2, rtol=4 * EPS, atol=0)
        with self.assertRaises(ValueError):
            r.analytic([1e300j], 1.0)

    def test_complex_arguments_against_independent_reference(self):
        for mean_time in [1.0, 0.37]:
            g = r.analytic(POINTS, mean_time)
            for z, value in zip(POINTS, g):
                expected = reference(z, mean_time)
                self.assertLess(abs(value - expected), 16 * EPS * abs(expected), z)

    def test_passivity_and_conjugate_symmetry(self):
        rng = np.random.default_rng(7)
        z = 10**rng.uniform(-8, 8, 400) * np.exp(1j * rng.uniform(-math.pi / 2, math.pi / 2, 400))
        g = r.analytic(z, 1.3)
        self.assertTrue(np.all(g.real > 0))
        self.assertTrue(np.all(np.abs(g) <= 1))
        np.testing.assert_allclose(r.analytic(z.conj(), 1.3), g.conj(), rtol=2 * EPS, atol=0)

    def test_mean_relaxation_time_is_minus_derivative_at_zero(self):
        for mean_time in [0.01, 1.0, 250.0]:
            step = 1e-6 / mean_time
            g = r.analytic([1j * step], mean_time)[0]
            # Im g(i e) = -T e + O(e^3) is computed without subtraction.
            self.assertLess(abs(-g.imag / step / mean_time - 1), 1e-11)

    def test_rejects_unsupported_domain(self):
        for z, mean_time in [(-1e-12 + 1j, 1.0), (complex("nan"), 1.0), (1.0, 0.0), (1.0, -2.0)]:
            with self.assertRaises(ValueError):
                r.analytic([z], mean_time)


class ModalTests(unittest.TestCase):
    def test_zero_limit_keeps_truncated_weights(self):
        for n in [1, 10, 1000, 100000]:
            g_n = r.modal_partial_sums(0.0, 1.0, [n])[0]
            missing = 1 - g_n.real
            self.assertEqual(g_n.imag, 0.0)
            self.assertGreater(missing, 0)  # weights are not renormalized
            bound = r.tail_bound(0.0, 1.0, n)
            self.assertLessEqual(missing, bound + r.rounding_allowance(g_n))
            self.assertGreater(missing, (0.99 if n >= 10 else 0.9) * bound)

    def test_mean_relaxation_time_of_truncated_sum(self):
        # sum_{n<=N} w_n tau_n = T (1 - deficit), deficit <= 30/(pi^4 (N + 1/2)^3).
        for mean_time in [0.2, 5.0]:
            step = 1e-6 / mean_time
            for n in [1, 10, 100]:
                g = r.modal_partial_sums(1j * step, mean_time, [n])[0]
                deficit = 1 + g.imag / step / mean_time
                self.assertGreaterEqual(deficit, -1e-10)
                self.assertLessEqual(deficit, 30 / (math.pi**4 * (n + 0.5)**3) + 1e-10)

    def test_partial_sums_across_chunk_boundaries(self):
        modes = [r.CHUNK - 1, r.CHUNK, r.CHUNK + 1, 3 * r.CHUNK + 5]
        z = 0.4 + 7j
        sums = r.modal_partial_sums(z, 0.8, modes)
        for n, value in zip(modes, sums):
            k = np.arange(1, n + 1, dtype=float)
            direct = np.sum(6 / (math.pi**2 * k * k + 15 * 0.8 * z))
            self.assertLess(abs(value - direct), 4 * EPS * abs(direct))
            single = r.modal_partial_sums(z, 0.8, [n])[0]
            self.assertLess(abs(value - single), 4 * EPS * abs(direct))
        with self.assertRaises(ValueError):
            r.modal_partial_sums(z, 0.8, [10, 10])

    def test_mode_sum_converges_within_tail_bound(self):
        modes = [1, 10, 100, 1000, 10000, 100000, 1000000]
        for z in POINTS:
            exact = r.analytic([z], 1.0)[0]
            sums = r.modal_partial_sums(z, 1.0, modes)
            bounds = [r.tail_bound(z, 1.0, n) for n in modes]
            for n, value, bound in zip(modes, sums, bounds):
                self.assertLessEqual(abs(exact - value), bound + r.rounding_allowance(exact, value),
                                     (z, n))
            # Convergence is asserted through the bound, not by monotone decrease
            # of the observed complex error.
            self.assertLess(abs(exact - sums[-1]), 7e-7)
            self.assertTrue(all(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])))

    def test_tail_bound_against_independent_reference(self):
        for z in POINTS + [0.0]:
            expected = reference(z, 1.0)
            for n in [1, 3, 10, 100, 1000]:
                value = r.modal_partial_sums(z, 1.0, [n])[0]
                actual_tail = abs(expected - value)
                self.assertLessEqual(actual_tail, r.tail_bound(z, 1.0, n)
                                     + r.rounding_allowance(expected, value), (z, n))

    def test_large_argument_branch_of_tail_bound(self):
        z, n = 1e6j, 10
        bound = r.tail_bound(z, 1.0, n)
        self.assertLess(bound, 6 / (math.pi**2 * (n + 0.5)) / 50)
        actual = abs(reference(z, 1.0) - r.modal_partial_sums(z, 1.0, [n])[0])
        self.assertLessEqual(actual, bound)
        self.assertGreater(actual, 0.5 * bound)


if __name__ == "__main__":
    unittest.main()
