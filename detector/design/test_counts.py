"""Deterministic checks of exact analytical design formulas; no simulations."""
from pathlib import Path
import json
import sys
import unittest
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import counts

class DesignCounts(unittest.TestCase):
    def test_moments_against_complex_integral_formulas(self):
        with mp.workdps(100):
            for index in counts.INDICES:
                row=counts.bounds(index)
                for omega in (row['omega'],2*row['omega']):
                    z=1-1j*omega
                    exposure_response=(1-mp.exp(-z))/z
                    independent_variance=mp.re(1/z-(1-mp.exp(-z))/(z*z))
                    independent_mean=mp.re(1j*exposure_response)
                    self.assertLess(abs(independent_variance-counts.variance(omega,mp.mpf(1),mp.mpf(1))),mp.mpf('1e-90'))
                    self.assertLess(abs(independent_mean-counts.prepared_mean(omega,mp.mpf(1),mp.mpf(1))),mp.mpf('1e-90'))

    def test_information_lower_bound_constants(self):
        with mp.workdps(100):
            nu=mp.mpf('.5');eps=mp.mpf('.1');target=1-2*eps
            C=mp.cosh(1/(nu*nu))-1;H2=(3+mp.sqrt(2))**2
            for index in counts.INDICES:
                row=counts.bounds(index)
                n=row['passive_necessary_threshold']
                tv=mp.sqrt(n*C/2)*(row['variance_omega']+row['variance_double'])
                self.assertLess(abs(tv-target),mp.mpf('1e-90'))
                n=row['prepared_necessary_threshold']
                tv=3*mp.sqrt(n*H2)/(4*nu*row['omega'])
                self.assertLess(abs(tv-target),mp.mpf('1e-90'))

    def test_constructive_error_upper_bounds(self):
        with mp.workdps(100):
            nu=mp.mpf('.5');eps=mp.mpf('.1')
            for index in counts.INDICES:
                row=counts.bounds(index)
                npassive=mp.ceil(row['passive_sufficient_threshold'])
                nprepared=mp.ceil(row['prepared_sufficient_threshold'])
                passive_error=4*row['square_variance_upper']/(npassive*row['variance_gap']**2)
                prepared_error=mp.exp(-nprepared*row['prepared_mean_gap']**2/(8*(1+nu*nu)))
                self.assertLessEqual(passive_error,eps)
                self.assertLessEqual(prepared_error,eps)
                for method in ('passive','prepared'):
                    self.assertLess(row[method+'_necessary_threshold'],row[method+'_sufficient_threshold'])

    def test_quartic_and_quadratic_asymptotic_orders(self):
        with mp.workdps(100):
            first=counts.bounds(512);second=counts.bounds(1024)
            for kind in ('necessary','sufficient'):
                self.assertLess(abs(second['passive_'+kind+'_threshold']/first['passive_'+kind+'_threshold']-16),mp.mpf('.0001'))
                self.assertLess(abs(second['prepared_'+kind+'_threshold']/first['prepared_'+kind+'_threshold']-4),mp.mpf('.0001'))

    def test_numerical_ceiling_stability_not_interval_certification(self):
        low=[];high=[]
        for precision,output in ((100,low),(180,high)):
            with mp.workdps(precision):
                for index in counts.INDICES:
                    r=counts.bounds(index)
                    output.extend(int(mp.ceil(r[method+'_'+kind+'_threshold']))
                                  for method in ('passive','prepared') for kind in ('necessary','sufficient'))
        self.assertEqual(low,high)

    def test_saved_report_reproduces_and_states_scope(self):
        saved=json.loads((Path(__file__).parent/'results.json').read_text())
        self.assertEqual(saved,counts.report())
        self.assertIn('not certified',saved['status'])
        self.assertIn('known phase',saved['hypotheses'])
        self.assertIn('nu^2=kappa/E',saved['energy_scope'])
        self.assertEqual(len(saved['rows']),6)

    def test_invalid_parameters(self):
        for index in (0,-1,True,1.5):
            with self.assertRaises(ValueError):counts.bounds(index)
        for args in ({'gamma':0},{'tau':0},{'noise':0},{'error':'.5'}):
            with self.assertRaises(ValueError):counts.bounds(1,**args)

if __name__=='__main__':unittest.main()
