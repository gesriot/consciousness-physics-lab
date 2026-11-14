import copy
import unittest
from binding import source_lab_b005 as L
from binding import source_lab_b004 as B


class SourceLabB005Tests(unittest.TestCase):
    def setUp(self):
        self.inputs = L.intake(20270401)
        self.clean = L.build_case(self.inputs, 'clean')

    def test_replay_inputs_are_deterministic(self):
        self.assertEqual(self.inputs, L.intake(20270401))
        self.assertNotEqual(self.inputs, L.intake(20270402))

    def test_healthy_values_agree_with_sixty_digit_observations(self):
        for seed in (1, 20270401, 20270412):
            case = L.build_case(L.intake(seed), 'clean')
            self.assertFalse(case['inaccurate'])
            for row in case['records']:
                self.assertTrue(L.close(row['value'], case['truth'][row['id']]))
                self.assertEqual(row['unit'], case['measurements'][row['id']]['unit'])

    def test_sources_are_separate_versioned_and_dimensioned(self):
        self.assertTrue(all(s['id']==sid and s['version']==2 for sid,s in self.clean['sources'].items()))
        self.assertEqual({s['kind'] for s in self.clean['sources'].values()}, {'radius', 'material', 'method'})
        for row in self.clean['records']:
            self.assertIn(row['step_id'], row['source_ids'])
            expected = [0, 2, -1, 0] if row['unit']=='m2/s' else [0, 0, 0, 0]
            self.assertEqual(row['dimensions'], expected)

    def test_dimensioned_properties(self):
        a = L._material({'conductivity': {'value': 2., 'unit': 'W/(m K)'},
                         'density': {'value': 1., 'unit': 'g/cm3'},
                         'heat_capacity': {'value': 1., 'unit': 'kJ/(kg K)'}})
        self.assertAlmostEqual(a, 2e-6)
        self.assertEqual(L._radius({'quantity': {'value': 10., 'unit':'mm'}}), .01)
        with self.assertRaises(ValueError):
            L._radius({'quantity': {'value': 10., 'unit':'m2/s'}})

    def test_invalid_parent_dimension_rejected(self):
        case = copy.deepcopy(self.clean)
        row = next(r for r in case['records'] if r['operation']=='inverse')
        L.record_map(case)[row['parents'][0]]['unit']='m2/s'
        with self.assertRaises(ValueError):
            L.rederive(row['id'],case,{r['id']:r['value'] for r in case['records']},case['sources'])

    def test_graph_is_shared_and_reconvergent(self):
        c = self.clean
        self.assertEqual({r['depth'] for r in c['records']},{1,2,3,4})
        self.assertTrue(any(len(set(r['parents']))>1 for r in c['records']))
        a,b = c['branches'][:2]
        common = L.ancestry(a['primary'][3],c) & L.ancestry(b['primary'][3],c)
        self.assertGreater(len(common),2)
        self.assertEqual(len(c['records']),44)
        self.assertEqual(len({r['timestamp'] for r in c['records']}),44)

    def test_all_deep_corruptions_have_exposure(self):
        for fault in ('units','version','regime','intermediate'):
            for scope in L.SCOPES:
                c=L.build_case(self.inputs,fault,scope)
                self.assertEqual(sum(c['sources'][k]!=v for k,v in c['catalog'].items()),1)
                self.assertIn(c['culprit'],c['sources'])
                for depth in (2,3,4):
                    self.assertTrue(any(r['depth']==depth and r['id'] in c['inaccurate'] for r in c['records']))
                self.assertTrue(set(c['inaccurate']) <= set(c['affected']))

    def test_common_culprit_has_multiple_branches_private_one(self):
        for fault in ('units','version','regime','intermediate'):
            sizes=[]
            for scope in L.SCOPES:
                c=L.build_case(self.inputs,fault,scope)
                sizes.append(sum(any(cid in c['affected'] for cid in b['primary']) for b in c['branches']))
            self.assertEqual(sizes,[2,1])

    def test_intermediate_method_is_inherited(self):
        c=L.build_case(self.inputs,'intermediate')
        step=L.record_map(c)[c['branches'][0]['primary'][1]]
        self.assertEqual(step['depth'],2)
        self.assertEqual(step['step_id'],c['culprit'])
        self.assertIn(c['culprit'],L.ancestry(c['branches'][0]['primary'][3],c))
        self.assertNotIn(c['culprit'],L.ancestry(c['branches'][0]['primary'][0],c))

    def test_methods_declare_domain_and_fault_violates_it(self):
        c=L.build_case(self.inputs,'regime')
        source=c['sources'][c['culprit']]
        self.assertEqual(source['mode'],'high_frequency')
        self.assertEqual(source['domain']['zT_min'],10000.)
        self.assertEqual(c['catalog'][c['culprit']]['mode'],'exact')

    def test_local_step_check_does_not_validate_ancestors(self):
        c=L.build_case(self.inputs,'units')
        cid=c['branches'][0]['primary'][1]
        row=L.record_map(c)[cid]
        supplied=copy.deepcopy(c['sources'])
        supplied[row['step_id']]=copy.deepcopy(c['catalog'][row['step_id']])
        val,cost=L.rederive(cid,c,{r['id']:r['value'] for r in c['records']},supplied)
        self.assertTrue(L.close(val,row['value']))
        self.assertFalse(L.close(val,c['truth'][cid]))
        self.assertEqual(cost,66)

    def test_local_step_check_detects_wrong_inverse_method(self):
        c=L.build_case(self.inputs,'intermediate')
        cid=c['branches'][0]['primary'][1]
        row=L.record_map(c)[cid]
        supplied=copy.deepcopy(c['sources'])
        supplied[row['step_id']]=copy.deepcopy(c['catalog'][row['step_id']])
        val,cost=L.rederive(cid,c,{r['id']:r['value'] for r in c['records']},supplied)
        self.assertFalse(L.close(val,row['value']))
        self.assertTrue(L.close(val,c['truth'][cid]))
        self.assertEqual(cost,66)

    def test_rederive_has_no_evaluator_dependency(self):
        c=L.build_case(self.inputs,'version')
        values={r['id']:r['value'] for r in c['records']}
        expected=[L.rederive(r['id'],c,values,c['sources']) for r in c['records']]
        for k in ('truth','culprit','affected','inaccurate','measurements','catalog'):
            del c[k]
        actual=[L.rederive(r['id'],c,values,c['sources']) for r in c['records']]
        self.assertEqual(expected,actual)

    def test_pair_signal_names_claims_not_culprit(self):
        c=L.build_case(self.inputs,'version')
        rows=L.record_map(c)
        for a,b in c['pairs']:
            self.assertEqual(rows[a]['quantity_id'],rows[b]['quantity_id'])
            self.assertNotEqual(a,b)
        self.assertTrue(any(not L.close(rows[a]['value'],rows[b]['value']) for a,b in c['pairs']))

    def test_false_measurement_does_not_corrupt_any_source(self):
        c=L.build_case(self.inputs,'false_measurement')
        self.assertEqual(c['sources'],c['catalog'])
        self.assertFalse(c['inaccurate'])
        self.assertIsNone(c['culprit'])
        self.assertEqual(len(c['false_measurement_ids']),1)
        changed=[cid for cid,m in c['measurements'].items() if m['value']!=m['repeat_value']]
        self.assertEqual(changed,c['false_measurement_ids'])
        for cid in c['false_measurement_ids']:
            m=c['measurements'][cid]
            self.assertFalse(L.close(m['value'],m['repeat_value']))
            self.assertEqual(m['repeat_value'],c['truth'][cid])

    def test_single_false_measurement_spans_registered_depths(self):
        depths=set()
        for seed in (1,2,3):
            case=L.build_case(L.intake(seed),'false_measurement')
            self.assertEqual(len(case['false_measurement_ids']),1)
            depths.add(L.record_map(case)[case['false_measurement_ids'][0]]['depth'])
        self.assertEqual(depths,{2,3,4})

    def test_cost_includes_inverse_and_shared_operations_once(self):
        c=self.clean
        self.assertEqual(c['budget_unit'],82)
        for r in c['records']:
            self.assertGreaterEqual(L.full_cost(r['id'],c),L.OPERATION_COST[r['operation']]+len(r['source_ids']))
        self.assertEqual(B.INVERSE_EVALS,L.OPERATION_COST['inverse'])

    def test_last_bit_noise_preserves_truth_classification(self):
        for fault in L.FAULTS:
            a=L.build_case(self.inputs,fault)
            b=L.build_case(self.inputs,fault,noise=True)
            self.assertEqual(a['inaccurate'],b['inaccurate'])
            self.assertEqual(a['truth'],b['truth'])
            self.assertTrue(any(x['value']!=y['value'] for x,y in zip(a['records'],b['records'])))

    def test_bad_fault_and_scope_rejected(self):
        with self.assertRaises(ValueError):
            L.build_case(self.inputs,'other')
        with self.assertRaises(ValueError):
            L.build_case(self.inputs,'units','other')


if __name__=='__main__':
    unittest.main()
