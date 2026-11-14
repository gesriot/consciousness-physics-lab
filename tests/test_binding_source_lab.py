"""Dimensional and independent-reference tests for property/claim chains."""
import copy
import math
import unittest

import mpmath as mp
import numpy as np

from binding import source_lab_b004 as L


def source():
    return {'id': 'table-a', 'version': 2,
            'radius': {'value': 2., 'unit': 'mm'},
            'conductivity': {'value': .6, 'unit': 'W/(m K)'},
            'density': {'value': 1., 'unit': 'g/cm3'},
            'heat_capacity': {'value': 4.2, 'unit': 'kJ/(kg K)'}}


def direct():
    return {'id': 'table-b', 'version': 3,
            'radius': {'value': .002, 'unit': 'm'},
            'diffusivity': {'value': .6 / (1000 * 4200), 'unit': 'm2/s'}}


class SourceUnitsTests(unittest.TestCase):
    def test_property_table_and_explicit_diffusivity_agree(self):
        a, b = L.normalize_source(source()), L.normalize_source(direct())
        self.assertEqual(a['radius_m'], b['radius_m'])
        self.assertAlmostEqual(a['diffusivity_m2_s'], b['diffusivity_m2_s'], places=20)
        self.assertEqual(a['source_id'], 'table-a')
        self.assertEqual(a['version'], 2)
        self.assertEqual(a['declared_units']['radius'], 'mm')
        self.assertEqual(a['si_units']['diffusivity'], 'm2/s')

    def test_dimensions_derive_diffusivity_and_time(self):
        d = L.DIMENSIONS
        derived = tuple(k-r-c for k, r, c in zip(d['conductivity'], d['density'], d['heat_capacity']))
        self.assertEqual(derived, d['diffusivity'])
        self.assertEqual(tuple(2*r-k for r, k in zip(d['radius'], derived)), d['mean_time'])
        self.assertEqual(tuple(t+z for t, z in zip(d['mean_time'], d['frequency'])), d['response'])

    def test_radius_unit_conversions(self):
        values = []
        for unit, value in [('m', .002), ('cm', .2), ('mm', 2)]:
            row = direct(); row['radius'] = {'value': value, 'unit': unit}
            values.append(L.mean_time(row))
        np.testing.assert_allclose(values, [values[0]] * 3, atol=0, rtol=3e-15)

    def test_diffusivity_unit_conversions(self):
        row = direct(); kappa = row['diffusivity']['value']
        for unit, factor in [('m2/s', 1), ('cm2/s', 1e4), ('mm2/s', 1e6)]:
            other = copy.deepcopy(row)
            other['diffusivity'] = {'value': kappa * factor, 'unit': unit}
            self.assertAlmostEqual(L.mean_time(other), L.mean_time(row), places=14)

    def test_conductivity_unit_conversion(self):
        row = source(); row['conductivity'] = {'value': .006, 'unit': 'W/(cm K)'}
        self.assertEqual(L.mean_time(row), L.mean_time(source()))

    def test_bad_dimension_is_rejected(self):
        for key in ['radius', 'conductivity', 'density', 'heat_capacity']:
            row = source(); row[key]['unit'] = 's'
            with self.subTest(key=key), self.assertRaises(ValueError):
                L.normalize_source(row)

    def test_nonpositive_nonfinite_and_bool_are_rejected(self):
        for value in [0, -1, float('nan'), float('inf'), True, '1', 1j]:
            row = direct(); row['radius']['value'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                L.normalize_source(row)

    def test_required_identity_and_version(self):
        for key, value in [('id', ''), ('version', 0), ('version', True), ('version', 1.5)]:
            row = direct(); row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                L.normalize_source(row)

    def test_missing_extra_or_ambiguous_properties_rejected(self):
        a = source(); del a['density']
        b = source(); b['diffusivity'] = direct()['diffusivity']
        c = direct(); c['unknown'] = 1
        d = direct(); d['radius']['extra'] = True
        for row in [a, b, c, d]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                L.normalize_source(row)

    def test_false_unit_declaration_is_not_repaired(self):
        row = direct(); row['radius']['value'] *= 1000
        self.assertAlmostEqual(L.mean_time(row)/L.mean_time(direct()), 1e6, delta=1e-8)

    def test_mean_time(self):
        self.assertAlmostEqual(L.mean_time(direct()), .002**2/(15*(.6/4.2e6)), places=14)


class ChainTests(unittest.TestCase):
    def test_zero_response(self):
        self.assertEqual(L.response(direct(), 0), 1.)
        self.assertEqual(L.response(direct(), 0, noise=True), 1.)

    def test_exact_response_against_high_precision_formula(self):
        row = direct(); T = L.mean_time(row)
        with mp.workdps(60):
            for x in [1e-6, .01, .1, 1., 3., 10., 1e3, 1e8]:
                z = x/T
                q = mp.sqrt(15 * mp.mpf(T) * mp.mpf(z))
                reference = float(3 * (q/mp.tanh(q)-1)/(q*q))
                with self.subTest(x=x):
                    self.assertAlmostEqual(L.response(row, z)/reference, 1., delta=4e-15)

    def test_chain_three_depths_matches_correct_source(self):
        row = source(); p = L.normalize_source(row); T = L.mean_time(row)
        for x in [.01, .1, 1., 3., 10., 1e3]:
            z = x/T
            measured = L.response(row, z)
            inferred = L.infer_diffusivity(measured, p['radius_m'], z)
            self.assertAlmostEqual(inferred/p['diffusivity_m2_s'], 1., delta=3e-12)
            predicted = L.predict_response(inferred, p['radius_m'] * 1.7, z * .4)
            target = direct(); target['radius']['value'] *= 1.7
            self.assertAlmostEqual(predicted, L.response(target, z*.4), delta=3e-13)

    def test_dimensionless_scaling(self):
        row = direct(); other = copy.deepcopy(row)
        other['radius']['value'] *= 10
        for z in [.001, .1, 1, 100]:
            self.assertAlmostEqual(L.response(row, z), L.response(other, z/100), delta=3e-15)

    def test_asymptotic_is_explicit_not_silently_corrected(self):
        row = direct(); z = 1/L.mean_time(row)
        approximate = L.response(row, z, mode='asymptotic')
        self.assertAlmostEqual(approximate, math.sqrt(3/5), places=14)
        self.assertGreater(abs(approximate-L.response(row, z)), .1)
        self.assertEqual(approximate, L.response(row, z, mode='high_frequency'))
        with self.assertRaises(ValueError):
            L.response(row, 0, mode='asymptotic')

    def test_asymptotic_high_frequency_limit(self):
        row = direct(); z = 1e10/L.mean_time(row)
        self.assertLess(abs(L.response(row,z,mode='asymptotic')/L.response(row,z)-1), 3e-6)

    def test_bad_inverse_measurements_are_explicitly_unresolved(self):
        for value in [0, 1, 1.01, -1, float('nan'), 1e-300]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                L.infer_diffusivity(value, .002, 1.)

    def test_frequency_domain(self):
        for z in [-1, float('inf'), True, 1j]:
            with self.subTest(z=z), self.assertRaises(ValueError):
                L.response(direct(), z)
        with self.assertRaises(ValueError):
            L.infer_diffusivity(.5, .002, 0)
        with self.assertRaises(ValueError):
            L.response(direct(), 1, mode='invented')

    def test_noise_is_last_bit_only(self):
        row = direct(); plain = L.response(row, 1)
        self.assertEqual(L.response(row, 1, noise=True), np.nextafter(plain, np.inf))
        inverse = L.infer_diffusivity(plain, .002, 1)
        self.assertAlmostEqual(L.infer_diffusivity(plain, .002, 1, noise=True)/inverse, 1., delta=2e-14)
        prediction = L.predict_response(inverse, .003, 2)
        self.assertEqual(L.predict_response(inverse,.003,2,noise=True),np.nextafter(prediction,np.inf))

    def test_fixed_inverse_cost(self):
        row = direct(); stats = {}
        value = L.response(row, 1, stats=stats)
        self.assertEqual(stats, {'analytic_evaluations': 1})
        kappa = L.infer_diffusivity(value, .002, 1, stats=stats)
        self.assertEqual(stats['analytic_evaluations'], 1 + L.INVERSE_EVALS)
        self.assertEqual(stats['inverse_iterations'], L.INVERSE_ITERATIONS)
        L.predict_response(kappa, .003, 2, stats=stats)
        self.assertEqual(stats['analytic_evaluations'], 2 + L.INVERSE_EVALS)
        L.response(row, 1, mode='asymptotic', stats=stats)
        self.assertEqual(stats['asymptotic_evaluations'], 1)
        self.assertEqual(stats['analytic_evaluations'], 2 + L.INVERSE_EVALS)

    def test_fixed_bracket_accuracy_domain(self):
        for kappa in np.logspace(-8, -4, 9):
            for x in [.01, 1, 10, 1e4]:
                radius = .003
                z = x * 15 * kappa / radius**2
                value = L.predict_response(kappa, radius, z)
                stats = {}
                inverse = L.infer_diffusivity(value, radius, z, stats=stats)
                with self.subTest(kappa=kappa, x=x):
                    self.assertAlmostEqual(inverse/kappa, 1., delta=3e-12)
                    self.assertEqual(stats['analytic_evaluations'], 66)
                    self.assertEqual(stats['inverse_iterations'], 64)

    def test_outside_bracket_charges_endpoints(self):
        stats = {}
        with self.assertRaises(ValueError):
            L.infer_diffusivity(1e-30, .002, 1, stats=stats)
        self.assertEqual(stats, {'analytic_evaluations': 2})

    def test_source_is_not_mutated(self):
        row = source(); before = copy.deepcopy(row)
        L.normalize_source(row); L.mean_time(row); L.response(row, 1.)
        self.assertEqual(row, before)


if __name__ == '__main__':
    unittest.main()
