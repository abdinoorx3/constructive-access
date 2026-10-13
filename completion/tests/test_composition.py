from fractions import Fraction as F
from pathlib import Path
import copy
import random
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import composition as c


class CompositionTests(unittest.TestCase):
    def test_coupling_changes_required_interface(self):
        x=c.cnot_example()
        self.assertEqual(x["passive_interface_states"],2)
        self.assertEqual(x["completed_interface_states"],4)
        self.assertEqual(x["passive_tv"],0)
        self.assertEqual(x["active_tv"],1)
        self.assertTrue(x["coupling_is_permutation"])

    def test_certificate_rejects_unforced_refinement(self):
        x=c.cnot_example()
        all_states=((0,),(1,),(2,),(3,))
        result=c.complete_partition(x["passive_model"],all_states)
        result["trace"]=[x["initial_partition"],all_states]
        with self.assertRaises(ValueError):
            c.check_partition_certificate(x["passive_model"],x["initial_partition"],result)

    def test_certificate_rejects_wrong_quotient(self):
        x=c.cnot_example()
        result=copy.deepcopy(x["completion_certificate"])
        result["quotient"]["couple"]["ok"]=c.eye(4)
        with self.assertRaises(ValueError):
            c.check_partition_certificate(x["active_model"],x["initial_partition"],result)

    def test_hidden_update_not_immediate_outcome(self):
        x=c.memory_counterexample()
        self.assertEqual(x["first_action_output_tv"],0)
        self.assertEqual(x["actual_transcript_tv"],1)
        self.assertEqual(x["tv_upper_bound"],1)

    def test_correlated_preparation(self):
        x=c.correlated_preparation_example()
        self.assertEqual(x["direct_visible_tv"],0)
        self.assertEqual(x["coupled_visible_tv"],1)

    def test_exact_adaptive_result_and_tensor(self):
        x=c.adaptive_example()
        self.assertEqual(x["actual_transcript_tv"],F(3,40))
        self.assertEqual(x["tv_upper_bound"],F(1509,8000))
        self.assertEqual(x["tensor_check"]["checked_joint_rows"],16)

    def test_seeded_adaptive_bounds(self):
        rng=random.Random(20261012)
        for _ in range(100):
            model={}
            quotient={}
            for a in ["a","b"]:
                rows=[]
                for i in range(2):
                    weights=[rng.randrange(5) for _ in range(4)]
                    if not sum(weights): weights[0]=1
                    rows.append([F(x,sum(weights)) for x in weights])
                model[a]={str(y):tuple(tuple(row[2*y:2*y+2]) for row in rows) for y in range(2)}
                p=F(rng.randrange(11),10)
                quotient[a]={"0":((p,),),"1":((1-p,),)}
            def policy(depth):
                if not depth or rng.randrange(4)==0: return None
                return {"action":rng.choice(["a","b"]),
                        "children":{"0":policy(depth-1),"1":policy(depth-1)}}
            initial=F(rng.randrange(11),10)
            result=c.adaptive_defect_bound(model,quotient,((1,),(1,)),
                                           (initial,1-initial),(1,),policy(4))
            self.assertLessEqual(result["actual_transcript_tv"],result["tv_upper_bound"])

    def test_preparation_defect_not_omitted(self):
        model={"read":{"0":((1,0),(0,0)),"1":((0,0),(0,1))}}
        policy={"action":"read","children":{"0":None,"1":None}}
        result=c.adaptive_defect_bound(model,model,c.eye(2),(1,0),(0,1),policy)
        self.assertEqual(result["preparation_defect"],1)
        self.assertEqual(result["accumulated_defect"],0)
        self.assertEqual(result["actual_transcript_tv"],1)

    def test_free_postselection_and_invalid_inputs_rejected(self):
        model={"read":{"0":((F(1,2),),),"1":((F(1,2),),)}}
        with self.assertRaises(ValueError):
            c.transcript_law(model,(1,),{"action":"read","children":{"0":None}})
        with self.assertRaises(ValueError): c.instruments({"a":{"x":((2,),)}})
        with self.assertRaises(TypeError): c.instruments({"a":{"x":((1.0,),)}})
        with self.assertRaises(ValueError): c.partition_interface(((0,),(0,)),2)


if __name__=="__main__": unittest.main()
