"""Exploratory solver checks, not a frozen or large simulation benchmark."""
from fractions import Fraction as F
from pathlib import Path
from itertools import combinations
from math import prod
import sys
import unittest
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import exposure_moments as em


def real(x):return mp.mpf(x.numerator)/x.denominator


class ExposureMoments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=em.Model(1,3,'1/10','1/1000',2)
        cls.moments,cls.diagnostics=em.moments(cls.model)
        cls.law=em.outcome_law(cls.model)

    def assertEncloses(self,interval,value):
        point,radius=interval
        self.assertLessEqual(abs(real(point)-value),real(radius))

    def test_finite_hierarchy_sizes_and_closure(self):
        for middle,size in ((2,60),(4,140),(6,252)):
            model=em.Model(1,3,'1/10','1/1000',middle)
            states,index=em.state_space(model)
            self.assertEqual(len(states),size)
            for active in (None,0,1,2):em.generator_rows(model,states,index,active)
            self.assertTrue(all(sum(a)+k<=middle+2 for a,k,_ in states))

    def test_second_moments_against_exact_exposure_integrals(self):
        with mp.workdps(130):
            m=self.model;g,w,h,t=map(real,(m.gamma,m.omega,m.lag,m.exposure))
            z=g-1j*w
            variance=mp.re(t/z-(1-mp.exp(-z*t))/(z*z))/(t*t)
            self.assertEncloses(self.moments[(0,2,0)],variance)
            for alpha,gap in (((1,1,0),h),((0,1,1),h),((1,0,1),2*h)):
                covariance=mp.re(mp.exp(-z*gap)*(mp.exp(z*t)+mp.exp(-z*t)-2)/(z*z*t*t))/2
                self.assertEncloses(self.moments[alpha],covariance)

    def test_exact_zero_odd_moments_and_normalization(self):
        for alpha,(point,radius) in self.moments.items():
            if sum(alpha)%2:self.assertEqual(point,0)
        self.assertEqual(self.moments[(0,0,0)][0],1)
        self.assertEqual(sum(self.law['probability_centers'],F(0)),1)
        self.assertEqual(sum(self.law['dyadic_numerators']),1<<63)
        self.assertTrue(all(p-r>0 and p+r<1 for p,r in zip(self.law['probability_centers'],self.law['probability_radii'])))

    def test_binary_law_reconstructs_the_moments(self):
        p=self.law['probability_centers'];r=self.law['outcomes']
        for selected,alpha in (((1,2),(0,2,0)),((0,1),(1,1,0)),((0,3),(1,0,1))):
            moment=sum((prob*record[selected[0]]*record[selected[1]] for prob,record in zip(p,r)),F(0))
            self.assertEqual(moment,self.moments[alpha][0])
        reversed_map={tuple(record):p for record,p in zip(r,p)}
        for record,prob,radius in zip(r,p,self.law['probability_radii']):
            self.assertEqual(reversed_map[tuple(-s for s in record)],prob)
            self.assertLessEqual(abs(reversed_map[(record[3],record[1],record[2],record[0])]-prob),2*radius)

    def test_taylor_radius_against_scalar_exponential(self):
        with mp.workdps(130):
            for coefficient,time in ((F(7,3),F(1,7)),(F(-9),F(1,3)),(F(0),F(2))):
                vector,error,growth,order=em.taylor_action((((0,coefficient),),),time,[F(1)],F(1,1<<100))
                self.assertLessEqual(abs(real(vector[0])-mp.exp(real(coefficient*time))),real(error))
                self.assertGreaterEqual(real(growth),mp.exp(abs(real(coefficient*time))))

    def test_dyadic_tv_bound_and_small_replay(self):
        law=self.law;grid=1<<law['dyadic_bits']
        radius=sum((abs(F(q,grid)-p)+e for q,p,e in zip(law['dyadic_numerators'],law['probability_centers'],law['probability_radii'])),F(0))/2
        self.assertEqual(radius,law['total_variation_upper'])
        self.assertLess(radius,F(1,10**16))
        a=em.sample_counts(law,1000,91271,batch_size=101)
        b=em.sample_counts(law,1000,91271,batch_size=1000)
        self.assertEqual(a['counts'],b['counts'])
        self.assertEqual(sum(a['counts']),1000)
        self.assertEqual(a['simulation_product_law_tv_upper'],1000*radius)

    def test_physical_resources_and_input_guards(self):
        resource=self.law['resource_per_record']
        self.assertEqual(resource['binary_detector_readouts'],4)
        self.assertEqual(resource['modeled_evolution_time'],F(201,1000))
        self.assertEqual(resource['independent_stationary_preparations'],1)
        for args in ((0,3,'1/10','1/1000'),(1,3,'1/1000','1/10'),(1,3,.1,'1/1000')):
            with self.assertRaises((TypeError,ValueError)):em.Model(*args)
        with self.assertRaises(ValueError):em.outcome_law(self.model,dyadic_bits=64)

    def test_gain_powers_and_zero_gain(self):
        gain=F(9,10)
        law=em.outcome_law(em.Model(1,3,'1/10','1/1000',2,gain))
        for chosen,alpha in (((1,2),(0,2,0)),((0,1),(1,1,0)),((0,1,2,3),(1,2,1))):
            moment=sum((p*prod(record[i] for i in chosen)
                        for p,record in zip(law['probability_centers'],law['outcomes'])),F(0))
            self.assertEqual(moment,gain**len(chosen)*self.moments[alpha][0])
        zero=em.outcome_law(em.Model(1,3,'1/10','1/1000',2,0))
        self.assertEqual(zero['probability_centers'],[F(1,16)]*16)

    def test_collapsed_law_matches_ordered_probability_sums(self):
        for replicas in (2,4,6):
            model=em.Model(1,3,'1/50','1/100000',replicas,F(9,10))
            ordered=em.outcome_law(model)
            collapsed=em.outcome_law(model,collapse_centers=True)
            grouped={record:F(0) for record in collapsed['outcomes']}
            for record,p in zip(ordered['outcomes'],ordered['probability_centers']):
                grouped[(record[0],record[1:-1].count(1),record[-1])]+=p
            self.assertEqual(list(grouped.values()),collapsed['probability_centers'])
            self.assertEqual(len(grouped),4*(replicas+1))

    def test_elementary_statistics_against_subset_enumeration(self):
        for replicas in (2,4,6):
            for plus in range(replicas+1):
                signs=(1,)*plus+(-1,)*(replicas-plus)
                for degree in range(replicas+1):
                    products=[prod(subset) for subset in combinations(signs,degree)]
                    self.assertEqual(em.elementary_symmetric(plus,replicas,degree),
                                     F(sum(products),len(products)))

    def test_symmetric_scores_have_declared_gain_degrees(self):
        h=F(1,50);b=1-h;gain=F(9,10)
        model=em.Model(1,3,h,'1/100000',6,gain)
        table,_=em.moments(model)
        law=em.outcome_law(model,collapse_centers=True)
        stats=[em.record_statistics(r,6,h) for r in law['outcomes']]
        for j in range(3):
            target=(-table[(1,2*j,1)][0]+b*(table[(1,2*j+1,0)][0]+table[(0,2*j+1,1)][0])-b*b*table[(0,2*j+2,0)][0])/(h*h)
            actual=sum((p*s['S'][j] for p,s in zip(law['probability_centers'],stats)),F(0))
            self.assertEqual(actual,gain**(2*j+2)*target)
            actual_u=sum((p*s['U'][j] for p,s in zip(law['probability_centers'],stats)),F(0))
            self.assertEqual(actual_u,gain**(2*j)*table[(0,2*j,0)][0])
        target_v=(table[(0,2,0)][0]-(table[(1,1,0)][0]+table[(0,1,1)][0])/2)/h
        actual_v=sum((p*s['V'] for p,s in zip(law['probability_centers'],stats)),F(0))
        self.assertEqual(actual_v,gain*gain*target_v)

    def test_reference_calibration_laws(self):
        gain=F(9,10)
        constant=em.reference_law(gain)
        step=em.reference_law(gain,F(1,100000),F(1,500000))
        self.assertEqual(constant[1]-constant[-1],gain)
        self.assertEqual(step[1]-step[-1],F(9,50))
        self.assertEqual(sum(constant.values()),1)
        self.assertEqual(sum(step.values()),1)
        for kwargs in ({'exposure':F(1,10)}, {'exposure':F(1,10),'step_after':F(1,10)}):
            with self.assertRaises(ValueError):em.reference_law(gain,**kwargs)

if __name__=='__main__':unittest.main()
