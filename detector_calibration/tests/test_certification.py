"""Independent algebra, interval and contract checks for the detector API."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import combinations, product
from math import comb
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import certification as c


class ExactArithmetic(unittest.TestCase):
    def test_log_outward(self):
        with localcontext() as context:
            context.prec = 90
            for q in (F(1, 1000), F(1), F(7, 4), F(1000000)):
                lo, hi = c.log_interval(q)
                target = (Decimal(q.numerator)/Decimal(q.denominator)).ln()
                self.assertLessEqual(Decimal(lo.numerator)/Decimal(lo.denominator), target)
                self.assertGreaterEqual(Decimal(hi.numerator)/Decimal(hi.denominator), target)

    def test_sqrt_outward(self):
        for q in (F(0), F(1, 9), F(2), F(1, 2**400), F(10**30)):
            upper = c.sqrt_upper(q)
            self.assertGreaterEqual(upper*upper, q)
            if upper:
                self.assertLess((upper-F(1, 2**90))**2, q)

    def test_no_floats_or_bools(self):
        for value in (0.5, True, complex(1, 0)):
            with self.assertRaises(TypeError):
                c.exact(value)


class Scores(unittest.TestCase):
    def test_elementary_against_subsets(self):
        for k in range(1, 7):
            for plus in range(k+1):
                values = (1,)*plus+(-1,)*(k-plus)
                expected = [F(1)]
                for j in range(1, k+1):
                    products = []
                    for subset in combinations(values, j):
                        p = 1
                        for value in subset:
                            p *= value
                        products.append(p)
                    expected.append(F(sum(products), comb(k, j)))
                self.assertEqual(c.elementary_vertices(k, plus), tuple(expected))
                self.assertEqual(c.elementary(values), tuple(expected))

    def test_tight_ranges_full_enumeration(self):
        for m in range(3):
            ranges = c.score_ranges(m, F(1, 7), F(2))
            data = [c.continuous_scores(bits[0], bits[1:-1], bits[-1], m, F(1, 7), 2)
                    for bits in product((-1, 1), repeat=2*m+4)]
            for key, (lo, hi) in ranges.items():
                self.assertEqual(lo, min(row[key] for row in data))
                self.assertEqual(hi, max(row[key] for row in data))

    def test_continuous_vertices_bound(self):
        ranges = c.score_ranges(0, F(2, 3), F(-1))
        for bits in product((F(-1), F(-2, 7), F(1, 3), F(1)), repeat=4):
            row = c.continuous_scores(bits[0], bits[1:-1], bits[-1], 0, F(2, 3), -1)
            for key, val in row.items():
                self.assertLessEqual(ranges[key][0], val)
                self.assertLessEqual(val, ranges[key][1])

    def test_nominal_projection_ranges_and_sums(self):
        m, gain, h = 1, F(9, 10), F(1, 20)
        ranges = c.score_ranges(m, h, 1, nominal_gain=gain)
        rows = []
        for bits in product((-1, 1), repeat=6):
            row = c.continuous_scores(bits[0], bits[1:-1], bits[-1], m, h, 1, nominal_gain=gain)
            self.assertEqual(row["R0"], row["S0"]/gain**2)
            self.assertEqual(row["R1"], row["S0"]/gain**2+row["S1"]/gain**4)
            rows.append(row)
        for key in ("R0", "R1"):
            self.assertEqual(ranges[key], (min(r[key] for r in rows), max(r[key] for r in rows)))

    def test_conditional_product_unbiasedness(self):
        m, h, lam = 2, F(1, 5), F(3, 2)
        means = [F(1, 3)]+[F(-2, 5)]*6+[F(1, 7)]
        total = {key: F(0) for key in c.score_ranges(m, h, lam)}
        for bits in product((-1, 1), repeat=8):
            probability = F(1)
            for bit, mean in zip(bits, means):
                probability *= (1+bit*mean)/2
            for key, val in c.continuous_scores(bits[0], bits[1:-1], bits[-1], m, h, lam).items():
                total[key] += probability*val
        expected = c.continuous_scores(means[0], means[1:-1], means[-1], m, h, lam)
        self.assertEqual(total, expected)

    def test_counts_preserve_sums_and_squares(self):
        m = 1
        ordered = list(range(64))
        counts = c.aggregate_ordered_counts(ordered, m)
        summary = c.summarize_counts(counts, m, F(1, 10), F(1))
        self.assertEqual(summary["records"], sum(ordered))
        for key, stat in summary["statistics"].items():
            total = square = F(0)
            for bits, n in zip(product((-1, 1), repeat=6), ordered):
                value = c.continuous_scores(bits[0], bits[1:-1], bits[-1], m, F(1, 10), 1)[key]
                total += n*value
                square += n*value*value
            self.assertEqual((stat["sum"], stat["sum_squares"]), (total, square))


class Confidence(unittest.TestCase):
    def test_empirical_variance_and_two_tails(self):
        result = c.empirical_bernstein(100, 20, 100, (-1, 1), F(1, 20))
        self.assertEqual(result["variance"], F(32, 33))
        ell = c.log_interval(F(80))[1]
        expected = c.sqrt_upper(2*F(32, 33)*ell/100)+14*ell/(3*99)
        self.assertEqual(result["radius"], expected)
        with self.assertRaises(ValueError):
            c.empirical_bernstein(100, 20, 0, (-1, 1), F(1, 20))

    def test_empirical_interval_exact_coverage_binomial(self):
        n, p, alpha = 100, F(3, 7), F(1, 5)
        failure = F(0)
        for k in range(n+1):
            ci = c.empirical_bernstein(n, 2*k-n, n, (-1, 1), alpha)["interval"]
            if not ci[0] <= 2*p-1 <= ci[1]:
                failure += comb(n, k)*p**k*(1-p)**(n-k)
        self.assertLessEqual(failure, alpha)

    def test_simultaneous_allocation(self):
        summary = c.summarize_counts(dict.fromkeys(c.category_order(2), 100), 2, F(1, 10), 1)
        result = c.confidence_statistics(summary, F(1, 50))
        self.assertEqual(len(result["statistics"]), 6)
        self.assertEqual(sum(x["alpha"] for x in result["statistics"].values()), F(1, 50))
        for option in ({"record_contract": "dependent"}, {"replica_contract": "correlated"}, {"selection": "same_data"}):
            with self.assertRaises(ValueError):
                c.confidence_statistics(summary, F(1, 50), **option)

    def test_projection_joint_allocation(self):
        summary = c.summarize_counts(dict.fromkeys(c.category_order(2), 100), 2, F(1, 10), 1, F(9, 10))
        result = c.confidence_statistics(summary, F(1, 50))
        self.assertEqual(result["nominal_gain"], F(9, 10))
        self.assertEqual(len(result["statistics"]), 9)
        self.assertEqual(result["confidence_event_count"], 8)
        self.assertEqual(result["statistics"]["R0"]["alpha"], 0)
        self.assertEqual(result["statistics"]["R0"]["interval"], tuple(x/F(9, 10)**2 for x in result["statistics"]["S0"]["interval"]))
        self.assertEqual(sum(x["alpha"] for x in result["statistics"].values()), F(1, 50))

    def test_redundant_projection_costs_no_failure_budget(self):
        counts = dict.fromkeys(c.category_order(0), 100)
        plain = c.confidence_statistics(c.summarize_counts(counts, 0, F(1, 10), 1), F(1, 50))
        extra = c.confidence_statistics(c.summarize_counts(counts, 0, F(1, 10), 1, F(9, 10)), F(1, 50))
        self.assertEqual(plain["confidence_event_count"], extra["confidence_event_count"])
        self.assertEqual(plain["statistics"]["S0"], extra["statistics"]["S0"])

    def test_binary_hoeffding_exact_coverage(self):
        n, p, alpha = 100, F(2, 3), F(1, 10)
        failure = F(0)
        for k in range(n+1):
            lo, hi = c.binary_mean_interval(n, k, alpha)
            if not lo <= 2*p-1 <= hi:
                failure += comb(n, k)*p**k*(1-p)**(n-k)
        self.assertLessEqual(failure, alpha)

    def test_reference_joint_rectangle(self):
        result = c.calibrate_references(1000000, 900000, 1000000, 700000,
                                        F(1, 100), (F(3, 200), F(1, 40)), F(1, 50))
        self.assertEqual(result["status"], "covered_region")
        self.assertLessEqual(result["gain"][0], F(4, 5))
        self.assertGreaterEqual(result["gain"][1], F(4, 5))
        self.assertLessEqual(result["width"][0], F(1, 50))
        self.assertGreaterEqual(result["width"][1], F(1, 50))
        with self.assertRaises(ValueError):
            c.calibrate_references(10, 9, 10, 7, F(1, 10), (F(1, 100), F(1, 50)), F(1, 50))

    def test_weak_and_contradictory_calibration_abstain(self):
        weak = c.calibrate_references(2, 1, 2, 1, F(1, 100), (F(1, 50), F(1, 25)), F(1, 50))
        self.assertEqual(weak["status"], "gain_not_separated_from_zero")
        incompatible = c.calibrate_references(1000000, 0, 1000000, 500000,
                                              F(1, 100), (F(1, 50), F(1, 25)), F(1, 50))
        self.assertEqual(incompatible["status"], "incompatible_reference_intervals")


class PhysicalCertificate(unittest.TestCase):
    def test_blur_low_order_and_orientation(self):
        caps, tau, h, lam = (4, 5, 16, 20), F(1, 1000), F(1, 10), F(1)
        result = c.bias_bounds(0, h, lam, tau, caps)
        b = 1-lam*h
        expected = ((1+2*abs(b))*result["delta_h"]+b*b*result["delta_0"])/h**2
        self.assertEqual(result["delta_N"], expected)
        for cluster in ((1, 3), (1, 2, 1), (2, 1, 3)):
            forward = c.monomial_blur(cluster, tau, caps)
            backward = c.monomial_blur(tuple(reversed(cluster)), tau, (5, 4, 20, 16))
            self.assertEqual(forward, backward)
            degree = sum(cluster)
            crude = tau*sum(min(k*k*5, (degree-k)**2*4) for k in range(1, degree))
            self.assertLessEqual(forward, crude)

    def test_zero_exposure_and_invalid_overlap(self):
        result = c.bias_bounds(3, F(1, 10), 1, 0, (4, 4, 16, 16))
        self.assertEqual((result["delta_N"], result["delta_cal"]), (0, 0))
        with self.assertRaises(ValueError):
            c.bias_bounds(0, F(1, 100), 1, F(1, 10), (4, 4, 16, 16))

    def test_signed_gain_extrema(self):
        gain = (F(1, 2), F(4, 5))
        for interval in ((-3, -1), (-1, 4), (2, 5)):
            lo, hi = c.normalized_interval(interval, 4, gain)
            for step in range(31):
                g = gain[0]+(gain[1]-gain[0])*step/30
                self.assertLessEqual(lo, F(interval[0])/g**4)
                self.assertGreaterEqual(hi, F(interval[1])/g**4)
        self.assertEqual(c.normalized_interval((-3, -1), 2, gain)[0], -12)

    def test_projection_gain_correction_is_outward(self):
        nominal, gain = F(9, 10), (F(4, 5), F(19, 20))
        intervals = ((F(-3), F(2)), (F(1), F(4)), (F(-2), F(-1)))
        rlo = sum(lo/nominal**(2*j+2) for j, (lo, hi) in enumerate(intervals))
        rhi = sum(hi/nominal**(2*j+2) for j, (lo, hi) in enumerate(intervals))
        bound = c.projection_lower((rlo, rhi), intervals, nominal, gain)
        for step in range(21):
            kappa = gain[0]+(gain[1]-gain[0])*step/20
            for scores in product(*intervals):
                target = sum(val/kappa**(2*j+2) for j, val in enumerate(scores))
                self.assertLessEqual(bound, target)
        self.assertEqual(c.projection_lower((F(2), F(3)), intervals, nominal, (nominal, nominal)), 2)

    def test_certificate_joint_failure_and_abstention(self):
        summary = c.summarize_counts(dict.fromkeys(c.category_order(0), 10000), 0, F(1, 10), 1)
        confidence = c.confidence_statistics(summary, F(1, 40))
        calibration = c.calibrate_references(1000000, 1000000, 1000000, 750000,
                                            F(1, 2000), (F(1, 1000), F(1, 500)), F(1, 40))
        result = c.certify(confidence, calibration, (4, 4, 16, 16))
        self.assertEqual(result["alpha"], F(1, 20))
        self.assertGreaterEqual(result["entropy_lower"], 0)
        calibration["status"] = "gain_not_separated_from_zero"
        self.assertEqual(c.certify(confidence, calibration, (4, 4, 16, 16))["status"], "calibration_abstention")
        with self.assertRaises(ValueError):
            c.certify(confidence, calibration, (4, 4, 16, 16), geometry_contract="unknown_local_diffusivity")

    def test_positive_formula_algebra_fixture(self):
        # Deterministic interval fixture for the combination algebra; this is
        # not represented as measured data or a finite-sample coverage example.
        confidence = {"alpha": F(1, 40), "order": 0, "h": F(1, 1000), "lambda": F(1),
                      "record_contract": "independent_stationary_records",
                      "replica_contract": "conditionally_independent_common_gain", "selection": "predeclared",
                      "statistics": {"S0": {"interval": (F(9, 2), F(9, 2))},
                                     "U1": {"interval": (F(1, 2), F(1, 2))},
                                     "V": {"interval": (F(1, 2), F(1, 2))}}}
        calibration = {"status": "covered_region", "alpha": F(1, 40), "gain": (F(1), F(1)),
                       "width": (F(1, 10**9), F(1, 10**9)),
                       "contract": "independent_binary_constant_one_step_zero_to_one"}
        result = c.certify(confidence, calibration, (4, 4, 16, 16))
        self.assertEqual(result["status"], "positive_certificate")
        self.assertGreater(result["entropy_lower"], 4)
        self.assertEqual(result["entropy_lower"], result["P_lower"]*result["A_lower"]/result["H_upper"])
        self.assertEqual(result["gamma_upper"], result["H_upper"]/result["A_lower"])

    def test_projection_and_prefix_max_share_one_event(self):
        # Algebra fixture: the tight interval on S0+S1 can retain cancellation
        # discarded by separate marginal intervals. No fabricated data claim.
        confidence = {"alpha": F(1, 40), "order": 1, "h": F(1, 1000), "lambda": F(1),
                      "record_contract": "independent_stationary_records", "nominal_gain": F(1),
                      "replica_contract": "conditionally_independent_common_gain", "selection": "predeclared",
                      "statistics": {"S0": {"interval": (F(-10), F(10))}, "S1": {"interval": (F(-10), F(10))},
                                     "U1": {"interval": (F(1, 2), F(1, 2))}, "V": {"interval": (F(1, 2), F(1, 2))},
                                     "R0": {"interval": (F(-10), F(10))}, "R1": {"interval": (F(5), F(6))}}}
        calibration = {"status": "covered_region", "alpha": F(1, 40), "gain": (F(1), F(1)),
                       "width": (F(1, 10**12), F(1, 10**12)),
                       "contract": "independent_binary_constant_one_step_zero_to_one"}
        result = c.certify(confidence, calibration, (4, 4, 16, 16))
        self.assertEqual(result["order_selected"], 1)
        self.assertEqual(result["alpha"], F(1, 20))
        self.assertEqual(len(result["order_results"]), 2)
        self.assertLess(result["order_results"][1]["termwise_contrast_lower"], 0)
        self.assertEqual(result["order_results"][1]["projected_contrast_lower"], 5)
        self.assertGreater(result["entropy_lower"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
