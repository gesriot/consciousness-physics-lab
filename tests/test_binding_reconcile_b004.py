import copy
import unittest
from binding import reconcile_b004 as B
from binding import run_b004 as R

class ReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=B.intake(17)
        cls.series=B.build_series(cls.inputs,'regime')

    def test_intake_deterministic_and_disjoint_ids(self):
        self.assertEqual(self.inputs,B.intake(17))
        ids=[c for g in self.inputs['groups'] for c in g['claims']]
        self.assertEqual(len(set(ids)),48)
        self.assertEqual(sorted(self.inputs['timestamps']),list(range(48)))

    def test_clean_chain(self):
        s=B.build_series(self.inputs,'clean')
        self.assertTrue(all(B.close(r['value'],s['truth'][r['id']]) for r in s['records']))
        self.assertEqual(B.conflict_pairs(s['records']),[])

    def test_corruption_inheritance_and_typed_sources(self):
        for fault in B.FAULTS[1:]:
            s=B.build_series(self.inputs,fault)
            self.assertEqual(len(s['affected']),6)
            bad=[r for r in s['records'] if not B.close(r['value'],s['truth'][r['id']])]
            self.assertEqual(len(bad),6)
            self.assertEqual(sorted(r['depth'] for r in bad),[1,1,2,2,3,3])
            self.assertEqual(len(B.conflict_pairs(s['records'])),2)
            self.assertEqual(all(x.startswith('method:') for x in s['damaged_sources']),fault=='regime')

    def test_no_hidden_truth_leak(self):
        s=copy.deepcopy(self.series)
        r=B.simulate(s,'G',8,'conflict')
        s['truth']={k:v*1.2 for k,v in s['truth'].items()}
        s['affected']=[];s['damaged_sources']=[]
        q=B.simulate(s,'G',8,'conflict')
        self.assertEqual(r['protected'],q['protected'])
        self.assertEqual(r['trace'],q['trace'])
        self.assertEqual(r['final'],q['final'])

    def test_flat_view_has_no_links(self):
        for p in ('F','T','N'):
            view,_=B.memory_view(self.series['records'],p)
            self.assertTrue(all(not r['parents'] and r['request'] is None for r in view))
            if p!='F':self.assertTrue(all(not r['direct'] for r in view))

    def test_equal_capacity(self):
        caps={B.simulate(self.series,p,2,'revision')['protected']['capacity_bytes'] for p in B.POLICIES}
        self.assertEqual(caps,{48*B.SLOT_BYTES})

    def test_closure_and_one_hop(self):
        ids={r['id']:r for r in self.series['records']}
        roots={r['id'] for r in ids.values() if any(d['source'] in self.series['notice_sources'] for d in r['direct'])}
        candidates={}
        for p in ('G','G1','F','Gx','Gxprime'):
            view,_=B.memory_view(list(ids.values()),p)
            _,a=B.candidate_order(view,p,set(),set(self.series['notice_sources']))
            candidates[p]=set(a)
        self.assertEqual(candidates['F'],roots)
        self.assertEqual({ids[x]['depth'] for x in candidates['G1']},{1,2})
        self.assertEqual({ids[x]['depth'] for x in candidates['G']},{1,2,3})
        self.assertEqual(candidates['G'],candidates['Gxprime'])
        self.assertEqual(candidates['Gx'],candidates['F'])

    def test_budget_and_full_price(self):
        for p in B.POLICIES:
            for budget in B.BUDGETS:
                for branch in B.SIGNALS:
                    r=B.simulate(self.series,p,budget,branch)
                    self.assertLessEqual(r['protected']['spent'],budget*B.TICKS_PER_UNIT)
                    self.assertEqual(r['protected']['spent'],sum(e['cost'] for e in r['trace']))
                    for e in r['trace']:
                        if e['kind']=='repair':self.assertEqual(e['cost'],{1:2,2:68,3:69}[e['depth']])

    def test_conflict_localization_is_paid_and_method_specific(self):
        r=B.simulate(self.series,'G',8,'conflict')
        ds=[e for e in r['trace'] if e['kind']=='diagnose']
        self.assertEqual(len(ds),2)
        self.assertTrue(all(e['cost']==1 for e in ds))
        self.assertEqual(set(r['protected']['diagnosed']),set(self.series['damaged_sources']))

    def test_n_never_withdraws(self):
        r=B.simulate(self.series,'N',32,'conflict')
        self.assertEqual(r['trace'],[])
        self.assertEqual(r['metrics']['remaining_bad'],6)
        self.assertEqual(r['metrics']['correct_lost'],0)

    def test_false_notice_rechecks_not_oracle_withdrawal(self):
        s=B.build_series(self.inputs,'clean')
        r=B.simulate(s,'G',32,'revision')
        self.assertEqual(r['metrics']['remaining_bad'],0)
        self.assertEqual(r['metrics']['correct_lost'],0)
        self.assertEqual(r['metrics']['unnecessary'],48)

    def test_noise_preserves_decisions(self):
        other=B.build_series(self.inputs,'regime',True)
        for p in B.POLICIES:
            r=B.simulate(self.series,p,8,'conflict')
            q=B.simulate(other,p,8,'conflict',True)
            self.assertEqual(r['protected'],q['protected'])
            self.assertEqual(r['metrics'],q['metrics'])

    def test_pair_resolution_not_gold_concordance(self):
        s=copy.deepcopy(self.series)
        r=B.simulate(s,'N',2,'conflict')
        final=copy.deepcopy(r['final'])
        for pair in r['protected']['conflicts']:
            final[pair[0]]['value']=.42;final[pair[1]]['value']=.42
        m=B.evaluate(s,final,[],r['protected'])
        self.assertEqual(m['resolved_conflicts'],2)
        self.assertGreater(m['remaining_bad'],0)

    def test_missing_notice_version_rejected(self):
        s=copy.deepcopy(self.series);s['notice_versions']={}
        with self.assertRaises(ValueError):B.simulate(s,'G',2,'revision')

    def test_bad_condition_rejected(self):
        with self.assertRaises(ValueError):B.simulate(self.series,'G',1,'revision')
        with self.assertRaises(ValueError):B.memory_view(self.series['records'],'bad')

    def test_compact_archive_identity(self):
        a=R.make_archive(self.inputs,['clean'])
        self.assertEqual(a['intake_sha256'],B.digest(a['inputs']))
        case=a['cases']['clean']
        self.assertNotIn('intake',case['series'])
        self.assertEqual(len(case['conditions']),42)
        for k,v in case['outputs'].items():self.assertEqual(k,B.digest(v))

if __name__=='__main__':unittest.main()
