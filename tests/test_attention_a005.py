import unittest
import copy
from attention import monitor, monitor_planning as planning, agent
from lab.guard import Request

class MonitoringTests(unittest.TestCase):
    def test_threshold_and_mask(self):
        m=monitor.Monitor(1.,'BM');d=monitor.Monitor(1.,'D');dp=monitor.Monitor(1.,'Dprime')
        for _ in range(4):
            for x in (m,d,dp):x.observe('collocation','0',.1,{'verdict':'accept'},{})
        self.assertEqual(m.suspect,{'collocation'});self.assertEqual(dp.suspect,m.suspect)
        self.assertFalse(d.suspect);self.assertEqual(d.active_l1,dp.unused_l1)
        self.assertEqual(d.active_count,dp.unused_count)
    def test_clean_decline_no_alarm(self):
        m=monitor.Monitor(1.,'BM')
        for _ in range(100):m.observe('collocation','0',.99,{'verdict':'accept'},{})
        self.assertFalse(m.suspect)
    def test_score_quantization(self):
        a=planning._option('ATOMIC','binding',.5000000000000001,1.)
        b=planning._option('ATOMIC','binding',.5,1.)
        self.assertEqual(a['score'],b['score'])
    def test_extra_cannot_replace_base(self):
        self.assertEqual(monitor.combined({}, {'collocation':{'verdict':'accept'}},{'collocation'}),'inconclusive')
    def test_corruption_does_not_change_base_rule(self):
        e={a:{'verdict':'accept'} for a in ('modal','binding')+agent.FEATURES+('collocation',)}
        self.assertEqual(monitor.combined(e,{},set()),'inconclusive')
        e['v2_512']={'verdict':'accept','terminal':True}
        self.assertEqual(monitor.combined(e,{},set()),'accept')
        self.assertEqual(monitor.combined(e,{}, {'collocation'}),'inconclusive')
        self.assertEqual(monitor.combined(e,{'collocation':{'verdict':'violation_detected'}},{'collocation'}),'violation_detected')
    def test_replacement_price(self):
        self.assertEqual(monitor.replacement_cost('collocation',Request(1,1,.01),{'cost_weights':{'fv_cell':2}}),122880)

    def test_real_replacement_has_four_values(self):
        from lab.executor import request_hash
        r=Request(1,1,.01)
        e=agent.Executor(r,request_hash(r),'clean',1+0j)
        e.perform('modal')
        result=monitor.replacement(e,'collocation')
        self.assertEqual(result['grids'],[4096,8192,16384,32768])
        self.assertEqual(result['verdict'],'accept')

    def test_single_action_reuses_paid_v2_prerequisites(self):
        from lab.executor import request_hash
        from unittest.mock import patch
        r=Request(1,1,.01);e=agent.Executor(r,request_hash(r),'clean',1+0j)
        e.perform('modal')
        for n in (64,128,256,512):e.perform(f'v2_{n}')
        before=copy.deepcopy(e.observed)
        original=monitor.P.model_response
        with patch.object(monitor.P,'model_response',wraps=original) as call:
            result=monitor.replacement(e,'v2_512')
        self.assertEqual(call.call_count,1)
        self.assertEqual(e.observed,before)
        self.assertEqual(result['verdict'],before['v2_512']['verdict'])
