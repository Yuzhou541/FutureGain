"""Scientific reduction tests without model weights, data, NumPy or Torch."""

import math
import sys
import unittest

from future_gain.evaluation.metrics import aggregate_frame_metric, fvd_trajectory_mean


class MetricProtocolTests(unittest.TestCase):
    def test_legacy_worst_mse_canonical_mean_and_best_are_distinct(self):
        values = [[[0.0, 0.0], [1.0, 1.0]]]
        self.assertEqual(aggregate_frame_metric(values, metric="mse", protocol="upstream_legacy")["value"], 1)
        self.assertEqual(aggregate_frame_metric(values, metric="mse")["value"], 0.5)
        best = aggregate_frame_metric(values, metric="mse", protocol="canonical_best_of_k")
        self.assertEqual(best["value"], 0)
        self.assertEqual(best["selected_trajectory"], (0,))

    def test_best_trajectory_selected_after_time_mean_not_per_frame(self):
        values = [[[0.0, 10.0], [10.0, 0.0]]]
        best = aggregate_frame_metric(values, metric="lpips", protocol="canonical_best_of_k")
        self.assertEqual(best["value"], 5.0)  # Framewise selection would incorrectly yield 0.
        self.assertEqual(best["selected_trajectory"], (0,))

    def test_legacy_metric_directions_and_context_equal_weight(self):
        values = [[[0.1, 0.3], [0.6, 0.8]], [[0.2, 0.4], [0.4, 0.6]]]
        lpips = aggregate_frame_metric(values, metric="lpips", protocol="upstream_legacy")
        ssim = aggregate_frame_metric(values, metric="ssim", protocol="upstream_legacy")
        mean = aggregate_frame_metric(values, metric="lpips")
        self.assertAlmostEqual(lpips["value"], 0.25)
        self.assertAlmostEqual(ssim["value"], 0.6)
        self.assertAlmostEqual(mean["value"], 0.425)
        self.assertEqual(lpips["selected_trajectory"], (0, 0))
        self.assertEqual(ssim["selected_trajectory"], (1, 1))

    def test_psnr_is_mean_of_frame_scores_and_keeps_perfect_infinity(self):
        per_frame = [-10 * math.log10(0.01), -10 * math.log10(1.0)]
        score = aggregate_frame_metric([[per_frame]], metric="psnr")["value"]
        self.assertEqual(score, 10)
        self.assertNotAlmostEqual(score, -10 * math.log10((0.01 + 1.0) / 2))
        self.assertEqual(aggregate_frame_metric([[[math.inf, 20.0]]], metric="psnr")["value"], math.inf)

    def test_missing_candidates_or_frames_and_nonfinite_values_are_errors(self):
        for bad in ([], [[]], [[[]]], [[[0.0]], [[0.0], [1.0]]], [[[0.0], [1.0, 2.0]]], [[[math.nan]]]):
            with self.assertRaises(ValueError):
                aggregate_frame_metric(bad, metric="lpips")
        with self.assertRaises(ValueError):
            aggregate_frame_metric([[[math.inf]]], metric="mse")

    def test_fvd_aggregates_across_contexts_separately_for_each_trajectory(self):
        predicted = [[[0.0], [10.0]], [[2.0], [14.0]]]
        reference = [[[0.0], [0.0]], [[0.0], [0.0]]]
        calls = []

        def mock_distance(fake, real):
            calls.append((fake, real))
            return (sum(row[0] for row in fake) / len(fake)
                    - sum(row[0] for row in real) / len(real)) ** 2

        result = fvd_trajectory_mean(predicted, reference, mock_distance)
        self.assertEqual(calls, [([[0.0], [2.0]], [[0.0], [0.0]]),
                                 ([[10.0], [14.0]], [[0.0], [0.0]])])
        self.assertEqual(result["per_trajectory"], (1.0, 144.0))
        self.assertEqual(result["value"], 72.5)
        self.assertNotEqual(result["value"], 6.5 ** 2)  # Incorrect pooled statistic.

    def test_fvd_rejects_one_context_or_inconsistent_features(self):
        with self.assertRaises(ValueError):
            fvd_trajectory_mean([[[1.0]]], [[[1.0]]], lambda _a, _b: 0)
        with self.assertRaises(ValueError):
            fvd_trajectory_mean([[[1.0]], [[2.0]]], [[[1.0]], [[2.0, 3.0]]], lambda _a, _b: 0)

    def test_lightweight_protocol_import_does_not_load_model_libraries(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("lpips", sys.modules)


if __name__ == "__main__":
    unittest.main()
