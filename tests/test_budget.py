import math
import random
import unittest

from future_gain.sampling.budget import (
    EmpiricalThreshold, ExactQuotaController, expected_calls, high_count,
    make_schedule, near_uniform_schedule,
)


class BudgetTests(unittest.TestCase):
    def test_all_small_quotas_and_extreme_online_scores_conserve_real_calls(self):
        distribution = EmpiricalThreshold([-1.0, 0.0, 0.0, 2.0])
        for length in range(1, 25):
            for high_frames in range(length + 1):
                for score in (-100.0, 0.0, 100.0):
                    controller = ExactQuotaController(length, high_frames, 8, 16,
                                                      distribution, seed=11)
                    trace = [controller.choose(score) for _ in range(length)]
                    calls = sum(d.nfe_requested for d in trace)
                    self.assertEqual(sum(d.decision == "high" for d in trace), high_frames)
                    controller.assert_complete(calls)
                    self.assertEqual(trace[-1].remaining_high_after, 0)
                    with self.assertRaises(RuntimeError):
                        controller.choose(score)

    def test_controller_checks_actual_calls_not_only_requested_budget(self):
        controller = ExactQuotaController(2, 1, 8, 16, EmpiricalThreshold([1]), seed=3)
        controller.choose(1)
        with self.assertRaises(AssertionError):
            controller.assert_complete(24)
        controller.choose(1)
        with self.assertRaises(AssertionError):
            controller.assert_complete(25)
        controller.assert_complete(24)

    def test_same_tolerance_for_calibration_and_online_ties(self):
        distribution = EmpiricalThreshold([0.0, 0.0000000000005, 1.0, 2.0], tie_tolerance=1e-12)
        rule = distribution.rule(0.75)
        self.assertEqual(rule.threshold, 0.0)
        self.assertEqual(rule.tie_probability, 0.5)
        left, right = random.Random(21), random.Random(21)
        self.assertEqual([rule.choose(0.0, left) for _ in range(20)],
                         [rule.choose(0.0000000000005, right) for _ in range(20)])

    def test_constant_scores_randomize_quota_without_global_rng_changes(self):
        before = random.getstate()
        distribution = EmpiricalThreshold([1.0] * 64)
        self.assertEqual(distribution.rule(0.25).tie_probability, 0.25)
        controller = ExactQuotaController(28, 14, 8, 16, distribution, seed=9001)
        decisions = [controller.choose(1).decision for _ in range(28)]
        self.assertEqual(decisions.count("high"), 14)
        self.assertNotEqual(decisions, ["low"] * 14 + ["high"] * 14)
        self.assertEqual(random.getstate(), before)

    def test_rounded_tolerance_boundary_preserves_calibrated_acceptance_rate(self):
        scores = [1.0, 1.0 + 1e-12, 2.0, 3.0]
        rule = EmpiricalThreshold(scores, tie_tolerance=1e-12).rule(0.75)
        self.assertEqual(rule.threshold, 1.0)
        self.assertEqual(rule.tie_probability, 0.5)
        # Test both online boundary behavior and the implied expected rate.
        class FixedRandom:
            def __init__(self, value):
                self.value = value

            def random(self):
                return self.value

        self.assertTrue(rule.choose(scores[1], FixedRandom(0.25)))
        self.assertFalse(rule.choose(scores[1], FixedRandom(0.75)))
        outcomes = [rule.choose(score, FixedRandom(value))
                    for score in scores for value in (0.25, 0.75)]
        self.assertEqual(sum(outcomes) / len(outcomes), 0.75)

    def test_all_static_schedules_preserve_total_budget(self):
        for length in range(1, 33):
            for quota in range(length + 1):
                for kind in ("near_uniform_exact_budget", "random_exact_budget",
                             "periodic_exact_budget", "front_loaded", "back_loaded"):
                    schedule = make_schedule(kind, length, 3, 8, quota, seed=32)
                    self.assertEqual(sum(schedule), expected_calls(length, 3, 8, quota))
                    if kind == "near_uniform_exact_budget":
                        self.assertLessEqual(max(schedule) - min(schedule), 1)
                    else:
                        self.assertEqual(schedule.count(8), quota)
        self.assertEqual(near_uniform_schedule(28, 336), (12,) * 28)
        self.assertEqual(high_count(2, 0.25), 1)

    def test_invalid_inputs_fail_before_a_decision(self):
        for scores in ([], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                EmpiricalThreshold(scores)
        with self.assertRaises(ValueError):
            expected_calls(2, 8, 8, 1)
        with self.assertRaises(ValueError):
            expected_calls(2, 8, 16, 3)
        controller = ExactQuotaController(2, 0, 8, 16, EmpiricalThreshold([0]), seed=0)
        with self.assertRaises(ValueError):
            controller.choose(math.nan)
        self.assertEqual(controller.frame_index, 0)


if __name__ == "__main__":
    unittest.main()
