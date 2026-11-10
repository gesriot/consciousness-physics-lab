"""Prospective tests for conditional B002 revision without confirmation data."""
import copy
import unittest
from binding.revision_b002 import visible, selection, recall, metrics, closure, compact, summarize


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.records = [dict(id=str(i),time=10+i,links=[dict(edge_id=str(i)+':0',channel='collocation:'+str(i%2),time=5+i),
            dict(edge_id=str(i)+':1',channel='normalization',time=4+i)]) for i in range(4)]
        self.notices = [dict(channel='collocation:'+str(i),time=20,last_success=0,source='external') for i in range(6)]

    def test_relevant_toggle_removes_family_match(self):
        for n in self.notices:
            rows, edits = visible(self.records,n,'Gx')
            self.assertEqual(selection(rows,n,'Gx'),[])
            self.assertEqual(len(edits),sum(any(e['channel']==n['channel'] for e in r['links']) for r in self.records))
            self.assertTrue(all(e['binary_distance']==1 for e in edits))
        self.assertEqual(self.records[0]['links'][0]['channel'],'collocation:0')

    def test_irrelevant_toggle_preserves_selection(self):
        for n in self.notices:
            baseline,_=visible(self.records,n,'G')
            changed,edits=visible(self.records,n,'Gxprime')
            self.assertEqual(selection(baseline,n,'G'),selection(changed,n,'Gxprime'))
            self.assertTrue(all(e['after']=='independent:normalization' for e in edits))
            self.assertEqual([r['time'] for r in baseline],[r['time'] for r in changed])

    def test_missing_normalization_rejected(self):
        self.records[0]['links'].pop()
        with self.assertRaises(ValueError):visible(self.records,self.notices[0],'Gxprime')

    def test_temporal_memory_has_no_links(self):
        for name in ('T','A','N'):
            rows,_=visible(self.records,self.notices[0],name)
            self.assertTrue(all(set(r)=={'id','time'} for r in rows))

    def test_budget_shared_and_prices_follow_notice(self):
        calls=[]
        def replace(rid,ch):
            calls.append((rid,ch));return {'verdict':'accept'},1. if ch.endswith(':0') else 3.
        result=recall(self.records,self.notices,'G',3,replace)
        self.assertEqual(len(calls),3)
        self.assertEqual(len({r for r,c in calls}),3)
        self.assertEqual(result['cost'],5.)
        self.assertEqual(result['withdrawn'],[])
        self.assertEqual(result['unused_units'],0)

    def test_abstention_and_violation_withdraw_accept_survives(self):
        verdicts={'0':'accept','1':'inconclusive','2':'violation_detected','3':'accept'}
        result=recall(self.records,self.notices,'A',32,lambda r,c:({'verdict':verdicts[r]},2.))
        self.assertEqual(result['withdrawn'],['1','2'])
        self.assertEqual(len(result['reviewed']),4)
        truth={str(i):{'accurate':i!=2,'affected_channels':['collocation:'+str(i%2)]} for i in range(4)}
        measures=metrics(result,self.records,truth,self.notices)
        self.assertEqual(measures['residual_false_accepts'],0)
        self.assertEqual(measures['unnecessary_correct_recalls'],1)
        self.assertEqual(measures['confirmed'],2)
        self.assertEqual(measures['selection_recall'],1.)

    def test_scores_round_before_stable_ties(self):
        rows,_=visible(self.records,self.notices[0],'G')
        self.assertEqual(selection(rows,self.notices[0],'G'),selection(rows,self.notices[0],'G',True))

    def test_no_notices_no_review(self):
        result=recall(self.records,[],'A',32,lambda *a: self.fail('Unexpected replacement'))
        self.assertEqual(result['cost'],0)
        self.assertEqual(result['attempts'],[])

    def test_empty_closure_not_vacuously_true(self):
        self.assertEqual(closure({},{}),{'cells':[],'links_unnecessary_here':False,'tested':False})


if __name__=='__main__':unittest.main()
