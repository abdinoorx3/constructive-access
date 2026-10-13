from fractions import Fraction as F
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import quantum_completion as q


class QuantumPulseTests(unittest.TestCase):
    def test_rational_phase_bracket(self):
        c=q.phase_boundary_certificate()
        self.assertLess(c["q_lower"],c["q_upper"])
        self.assertLess(c["F_at_lower"][1],0)
        self.assertGreater(c["F_at_upper"][0],0)
        self.assertLess(c["q_upper"]-c["q_lower"],F(1,10**10))

    def test_declared_signs_and_gain(self):
        early=q.pulse_certificate(F(1,10),1,F(1,10),F(1,2))
        late=q.pulse_certificate(F(1,10),1,F(1,10),F(9,4))
        self.assertLess(early["response_upper"],0)
        self.assertGreater(late["response_lower"],0)
        self.assertGreater(-F(9,10)*early["response_upper"],F(3,100))
        self.assertGreater(F(9,10)*late["response_lower"],F(13,1000))
        self.assertTrue(early["all_channel_rates_strictly_positive"])
        self.assertLess(late["integral_certificate"]["radius"],F(41,10**64))

    def test_incoherent_no_reversal_grid(self):
        for j in range(1,25):
            c=q.pulse_certificate(F(1,10),0,F(1,10),F(j,4))
            self.assertLess(c["response_upper"],0)
            gamma=F(11,100)
            value=(1-math.exp(-float(4*gamma*F(j,4))))/float(4*gamma)
            self.assertAlmostEqual(float(c["integral_certificate"]["center"]),value,places=14)

    def test_general_channel_inventory_and_reverse_current(self):
        specific=q.pulse_certificate(F(1,10),1,F(1,10),F(9,4))
        general=q.general_rates_certificate(F(2,5),F(3,10),F(1,10),1,F(1,10),F(9,4))
        reversed_=q.general_rates_certificate(F(2,5),F(1,10),F(3,10),1,F(1,10),F(9,4))
        self.assertEqual(specific["response_lower"],general["response_lower"])
        self.assertEqual(specific["response_upper"],general["response_upper"])
        self.assertEqual(reversed_["response_upper"],-general["response_lower"])
        zero=q.general_rates_certificate(1,F(1,2),F(1,2),1,F(1,2),1)
        self.assertEqual(zero["response_lower"],0)
        self.assertEqual(zero["response_upper"],0)

    def test_exact_finite_menu(self):
        menu=q.timing_menu()
        self.assertEqual(len(menu["rows"]),16)
        self.assertEqual(menu["maximum_signal_time"],F(9,4))
        self.assertEqual(menu["maximum_signal_squared_per_time"],F(9,4))

    def test_dephasing_changes_this_protocol(self):
        weak=q.pulse_certificate(F(1,10),1,F(1,10),F(5,2),F(1,10))
        stronger=q.pulse_certificate(F(1,10),1,F(1,10),F(5,2),F(1,5))
        self.assertGreater(weak["response_lower"],0)
        self.assertLess(stronger["response_upper"],0)

    def test_direct_gksl_density_matrix(self):
        # Independent 4x4 complex superoperator construction, not Bloch code.
        import numpy as np
        from scipy.linalg import expm
        identity=np.eye(2,dtype=complex)
        x=np.array([[0,1],[1,0]],dtype=complex)
        z=np.diag([1,-1]).astype(complex)
        plus=np.array([[0,0],[1,0]],dtype=complex)
        minus=plus.T.conj()
        def dissipator(l):
            c=l.T.conj()@l
            return np.kron(l.conj(),l)-(np.kron(identity,c)+np.kron(c.T,identity))/2
        for kappa,r,g,d,t in [(F(1,10),F(1,10),F(1),F(0),F(9,4)),
                              (F(1,100),F(1,2),F(1),F(0),F(2)),
                              (F(1,10),F(1,10),F(1),F(1,5),F(5,2)),
                              (F(1,10),F(1,10),F(0),F(0),F(1,2))]:
            h=float(g)*x
            generator=-1j*(np.kron(identity,h)-np.kron(h.T,identity))
            generator+=float((3*r+1)*kappa)*dissipator(plus)
            generator+=float((r+3)*kappa)*dissipator(minus)
            generator+=float(d)*dissipator(z)
            rho=(expm(float(t)*generator)@(identity/2).reshape(4,order="F")).reshape((2,2),order="F")
            c=q.pulse_certificate(kappa,g,r,t,d)
            self.assertLess(abs(rho.trace()-1),1e-13)
            self.assertLess(np.max(np.abs(rho-rho.T.conj())),1e-13)
            self.assertGreater(np.min(np.linalg.eigvalsh(rho)),-1e-13)
            self.assertAlmostEqual(float(rho[1,1].real-.5),float(c["response_lower"]),places=13)

    def test_transcendental_enclosures(self):
        for x in [F(1,7),F(1,2),F(9,10)]:
            lo,hi=q.atan_interval(x,80)
            self.assertLessEqual(float(lo)-1e-15,math.atan(float(x)))
            self.assertGreaterEqual(float(hi)+1e-15,math.atan(float(x)))
        for x in [F(1),F(2),F(457,100)]:
            lo,hi=q.log_interval(x)
            self.assertLessEqual(float(lo)-1e-15,math.log(float(x)))
            self.assertGreaterEqual(float(hi)+1e-15,math.log(float(x)))
        for x in [F(0),F(2),F(81,4),F(12345,777)]:
            lo,hi=q.sqrt_interval(x)
            self.assertLessEqual(lo*lo,x)
            self.assertGreaterEqual(hi*hi,x)

    def test_invalid_and_degenerate_inputs(self):
        for args in [(0,1,F(1,2),1),(1,1,1,1),(1,1,-1,1)]:
            with self.assertRaises(ValueError): q.pulse_certificate(*args)
        with self.assertRaises(TypeError): q.pulse_certificate(.1,1,F(1,2),1)
        with self.assertRaises(ValueError): q.integral_certificate(1,100,0,100,order=10)
        with self.assertRaises(ValueError): q.general_rates_certificate(1,2,0,1,0,1)
        zero=q.pulse_certificate(F(1,10),1,F(1,10),0)
        self.assertEqual(zero["response_lower"],0)
        self.assertEqual(zero["response_upper"],0)


if __name__=="__main__": unittest.main()
