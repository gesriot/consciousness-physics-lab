"""Synthetic positive and negative controls for independent revision auditing."""
import copy
import unittest
from binding.audit import AuditError
from binding.audit_b002_revision import check_cell, projection, aggregate, numeric_tree
from binding.qualification import protected_step
from binding.revision_b002 import recall, metrics


class RevisionAuditTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(id='0:0',time=10,links=[dict(edge_id='a',channel='v2:0',time=5),dict(edge_id='b',channel='normalization',time=4)])]
        self.truth={'0:0':dict(accurate=True,affected_channels=['v2:0'])}
        self.notices=[dict(channel='v2:0',time=20,last_success=0,last_good_global=0,source='external')]
        self.infos={'0:0':dict(outcome={'value':[.5,0.]},tolerance=.1)}
        self.weights={'fv_cell':.01,'collocation_degree_cubed':.000001}
        step=protected_step('collocation',[.5+0j,.5+0j],.5+0j,.1)
        self.answer=dict(verdict='accept',values=[[.5,0.],[.5,0.]],steps=[step],instrument='collocation',charged_work={'degrees':[64,128]})
        self.price=(64**3+128**3)*self.weights['collocation_degree_cubed']

    def saved(self,memory='G'):
        value=recall(self.records,self.notices,memory,2,lambda *a:(copy.deepcopy(self.answer),self.price))
        value['metrics']=metrics(value,self.records,self.truth,self.notices)
        return value

    def check(self,value,memory='G'):
        return check_cell(value,self.records,self.truth,self.notices,memory,2,'cross',self.infos,self.weights)

    def test_valid_synthetic_cross_acceptance(self):
        result=self.check(self.saved())
        self.assertEqual(result['confirmed'],1)
        self.assertEqual(result['unnecessary_correct_recalls'],0)

    def test_relevant_and_irrelevant_prefix_controls(self):
        for memory in ('Gx','Gxprime','T','A','N'):
            self.check(self.saved(memory),memory)
        self.assertEqual(self.check(self.saved('Gx'),'Gx')['attempted'],0)
        self.assertEqual(self.check(self.saved('Gxprime'),'Gxprime')['attempted'],1)

    def test_modified_price_rejected(self):
        value=self.saved();value['attempts'][0]['cost']+=1e-8
        with self.assertRaises(AuditError):self.check(value)

    def test_replacement_flag_rejected(self):
        value=self.saved();value['attempts'][0]['result']['steps'][0]['finite']=False
        with self.assertRaises(AuditError):self.check(value)

    def test_wrong_withdrawal_rejected(self):
        value=self.saved();value['attempts'][0]['withdrawn']=True
        with self.assertRaises(AuditError):self.check(value)

    def test_corrupt_graph_rejected(self):
        value=self.saved('Gxprime');value['interventions'][0]['records'][0]['links'][0]['channel']='v2:1'
        with self.assertRaises(AuditError):self.check(value,'Gxprime')

    def test_false_edit_distance_rejected(self):
        value=self.saved('Gx');value['interventions'][0]['binary_distance']=2
        with self.assertRaises(AuditError):self.check(value,'Gx')

    def test_duplicate_attempt_rejected(self):
        value=self.saved();value['attempts'].append(copy.deepcopy(value['attempts'][0]))
        with self.assertRaises(AuditError):self.check(value)

    def test_metrics_mutation_rejected(self):
        value=self.saved();value['metrics']['selection_precision']=0.
        with self.assertRaises(AuditError):self.check(value)

    def test_selection_and_notice_mutations_rejected(self):
        for key in ('selected','notice'):
            value=self.saved();value['interventions'][0][key]=[]
            with self.assertRaises(AuditError):self.check(value)

    def test_missing_normalization_rejected(self):
        rows=copy.deepcopy(self.records);rows[0]['links'].pop()
        with self.assertRaises(AuditError):projection(rows,self.notices[0],'Gxprime',set())

    def test_numeric_tree_detects_summary_mutation(self):
        numeric_tree({'v':[1.,None]},{'v':[1.+1e-14,None]})
        with self.assertRaises(AuditError):numeric_tree({'v':[1.]},{'v':[2.]})


if __name__=='__main__':unittest.main()
