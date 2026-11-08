import copy
import unittest
from binding.compare import upstream_decisions, decision_projection


class ComparatorTests(unittest.TestCase):
    def trace(self):
        return dict(budget=10., spent=4., sample_cost=2., sample_allocation={'quota':1},
                    actions=[dict(action='v2', sample_id=3, sample_reason='deadline',
                                  channel='v2:0', cost=2., counter_before=1, sample=True,
                                  observation={'verdict':'accept', 'step': {'rounding':1e-12}},
                                  original_observation={'verdict':'accept', 'shadow_C2_converging':False})],
                    inputs=[dict(verdict='accept', closure_action_index=2, observed={}, replacements={})])

    def test_decision_mutations_fail(self):
        base = self.trace()
        for key, value in [('action','collocation'), ('sample_id',4), ('sample_reason','exhausted'),
                           ('counter_before',2), ('cost',3.)]:
            changed = copy.deepcopy(base); changed['actions'][0][key] = value
            self.assertNotEqual(upstream_decisions(base), upstream_decisions(changed), key)
        for key in ('budget','spent','sample_cost'):
            changed = copy.deepcopy(base); changed[key] += 1.
            self.assertNotEqual(upstream_decisions(base), upstream_decisions(changed), key)
        changed = copy.deepcopy(base); changed['inputs'][0]['closure_action_index'] = 3
        self.assertNotEqual(upstream_decisions(base), upstream_decisions(changed))
        changed = copy.deepcopy(base); changed['actions'][0]['observation']['verdict'] = 'inconclusive'
        self.assertNotEqual(upstream_decisions(base), upstream_decisions(changed))

    def test_shadow_and_float_are_diagnostic(self):
        base = self.trace(); changed = copy.deepcopy(base)
        changed['actions'][0]['original_observation']['shadow_C2_converging'] = True
        changed['actions'][0]['observation']['step']['rounding'] *= 2
        self.assertEqual(upstream_decisions(base), upstream_decisions(changed))
        self.assertNotEqual(base, changed)

    def test_recall_guard_verdict_exact(self):
        a = dict(verdict='accept', step={'rounding':1.}, safety={'passed':True})
        b = dict(verdict='accept', step={'rounding':2.}, safety={'passed':True})
        self.assertEqual(decision_projection(a), decision_projection(b))
        b['safety']['passed'] = False
        self.assertNotEqual(decision_projection(a), decision_projection(b))
