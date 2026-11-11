"""Independent exposure-audit regressions and negative controls."""
import copy
import unittest
from binding.audit import AuditError
from binding.audit_b002_exposure import cross_check, exposure, freeze_check, native_price, action_price, injected


class ExposureAuditTests(unittest.TestCase):
    def guarded(self):
        values=[complex(.7),complex(.8),complex(.9),complex(.91)]
        rounding=32*2.220446049250313e-16*512**2
        uncertainty=.02+rounding
        step=dict(finite=True,unguarded_verdict='accept',decision_margin=.1-uncertainty,grid=512,convergence_pass=True,decision_pass=True,verdict='accept',native_rounding=rounding,uncertainty=uncertainty,discrepancy=0.,difference=.01,convergence_margin=.04)
        return dict(values=[[z.real,z.imag] for z in values],steps=[step],verdict='accept')

    def test_native_guard_reconstructed(self):
        self.assertTrue(cross_check(self.guarded(),complex(.91),.1))

    def test_changed_guard_verdict_rejected(self):
        row=self.guarded();row['verdict']='inconclusive'
        with self.assertRaises(AuditError):cross_check(row,complex(.91),.1)

    def test_changed_guard_flag_rejected(self):
        row=self.guarded();row['steps'][0]['convergence_pass']=False
        with self.assertRaises(AuditError):cross_check(row,complex(.91),.1)

    def test_changed_finite_flag_rejected(self):
        row=self.guarded();row['steps'][0]['finite']=False
        with self.assertRaises(AuditError):cross_check(row,complex(.91),.1)

    def test_changed_raw_verdict_rejected(self):
        row=self.guarded();row['steps'][0]['unguarded_verdict']='violation_detected'
        with self.assertRaises(AuditError):cross_check(row,complex(.91),.1)

    def test_changed_decision_margin_rejected(self):
        row=self.guarded();row['steps'][0]['decision_margin']+=.01
        with self.assertRaises(AuditError):cross_check(row,complex(.91),.1)

    def test_fixed_native_price(self):
        w=dict(fv_cell=1/4096,collocation_degree_cubed=1/262144)
        self.assertEqual(native_price('collocation',w),8128/4096)
        self.assertEqual(native_price('v2',w),9.)
        self.assertEqual(action_price('v2_512',w),.125)

    def test_exposure_counts_real_dependencies(self):
        e={'a':dict(accurate=True,affected_channels=['v2:0']),
           'b':dict(accurate=False,affected_channels=[]),
           'c':dict(accurate=False,affected_channels=['v2:1'])}
        self.assertEqual(exposure(e),dict(accepted=3,initial_false=2,affected=2,affected_false=1))

    def test_clean_and_permissive_semantics(self):
        original=dict(verdict='violation_detected',difference=.2)
        self.assertEqual(injected(original,.1,'permissive',False),original)
        self.assertEqual(injected(original,.1,'permissive',True),dict(verdict='accept',difference=.2))
        self.assertEqual(original['verdict'],'violation_detected')

    def test_uncertainty_only_corruption(self):
        original=dict(verdict='inconclusive',terminal=False,step=dict(uncertainty_estimate=.2,discrepancy=.04,converging=True,verdict='inconclusive',last_difference=.01,previous_difference=.04,last_grid=512,futile=False))
        actual=injected(original,.1,'underestimated',True)
        self.assertEqual(actual['step']['uncertainty_estimate'],.002)
        self.assertEqual(actual['step']['discrepancy'],.04)
        self.assertEqual(actual['verdict'],'accept')
        self.assertTrue(actual['terminal'])
        self.assertEqual(original['step']['uncertainty_estimate'],.2)

    def fixture(self):
        rows=[{str(d):{'permissive':dict(accepted=30,initial_false=2,affected=2,affected_false=0 if d==4 else 1)} for d in (4,8,16)} for _ in range(12)]
        candidates=[dict(packages=d,accepted=360,initial_false=24,affected=24,affected_false=0 if d==4 else 12,exposure_gate=d!=4) for d in (4,8,16)]
        saved=dict(experiment='B002-exposure',development_seeds=list(range(20270101,20270113)),qualification_sha256='hash',candidates={'permissive':dict(instrument='graded_v2',candidates=candidates,selected_packages=8,exposure_gate=True)},selected={'permissive':8},confirmation_allowed=True,status='freeze_before_confirmation')
        return rows,saved

    def test_shortest_passing_duration(self):
        rows,saved=self.fixture()
        self.assertEqual(freeze_check(rows,['permissive'],'hash',saved),saved)

    def test_longer_passing_selection_rejected(self):
        rows,saved=self.fixture();saved['selected']['permissive']=16
        with self.assertRaises(AuditError):freeze_check(rows,['permissive'],'hash',saved)

    def test_changed_exposure_denominator_rejected(self):
        rows,saved=self.fixture();saved['candidates']['permissive']['candidates'][0]['affected_false']=5
        with self.assertRaises(AuditError):freeze_check(rows,['permissive'],'hash',saved)


if __name__=='__main__':unittest.main()
