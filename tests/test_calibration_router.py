"""Lightweight contracts locally; tensor/linear-algebra checks only in Slurm."""

import json
import math
import os
import platform
import unittest
from unittest.mock import patch

from future_gain.calibration.features import FEATURE_NAMES, FeatureExtractor, validate_frame_position
from future_gain.calibration.ridge import (RidgeRouter, fit_ridge, video_equal_rmse,
                                          video_equal_spearman, video_weights, weighted_scaler)


def row(value):
    return [value] + [0.0] * (len(FEATURE_NAMES) - 1)


class CalibrationContractTests(unittest.TestCase):
    def test_scaler_balances_original_videos_and_ignores_row_replication(self):
        features, identities = [row(0), row(0), row(4)], ["a", "a", "b"]
        means, scales = weighted_scaler(features, identities)
        self.assertEqual(video_weights(identities), [0.5, 0.5, 1.0])
        self.assertEqual(means[0], 2.0)
        self.assertEqual(scales[0], 2.0)
        self.assertEqual(scales[1:], (1.0,) * 11)
        self.assertEqual(weighted_scaler(features * 2, identities * 2), (means, scales))

    def test_rmse_is_root_of_video_equal_mse_not_equal_row_mse(self):
        result = video_equal_rmse([0] * 4, [3, 1, 1, 1], ["a", "b", "b", "b"])
        self.assertAlmostEqual(result, math.sqrt(5))
        with self.assertRaises(ValueError):
            video_equal_rmse([0], [math.nan], ["a"])

    def test_rank_ties_and_constant_video_are_reported(self):
        report = video_equal_spearman([1, 1, 3, 1, 2, 4, 4],
                                      [2, 2, 4, 3, 2, 5, 5],
                                      ["a", "a", "a", "b", "b", "c", "c"])
        self.assertEqual(report["defined_videos"], 2)
        self.assertEqual(report["undefined_video_ids"], ["c"])
        self.assertAlmostEqual(report["mean"], 0.0)

    def test_signed_prediction_and_json_roundtrip(self):
        model = RidgeRouter(tuple(row(2)), (1.0,) * 12, tuple(row(3)), -1.0, 0.1)
        self.assertEqual(model.predict(row(1)), -4.0)
        restored = RidgeRouter.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
        self.assertEqual(restored, model)
        wrong = model.to_dict()
        wrong["feature_names"] = list(reversed(wrong["feature_names"]))
        with self.assertRaises(ValueError):
            RidgeRouter.from_dict(wrong)
        with self.assertRaises(ValueError):
            model.predict(row(math.inf))

    def test_feature_time_and_runtime_boundary(self):
        validate_frame_position(0, 1)
        for index, horizon in ((True, 28), (28, 28), (0, 0), (-1, 28)):
            with self.assertRaises(ValueError):
                validate_frame_position(index, horizon)
        with patch("future_gain.runtime.require_slurm_compute", side_effect=RuntimeError("no allocation")):
            with self.assertRaisesRegex(RuntimeError, "no allocation"):
                FeatureExtractor()
            with self.assertRaisesRegex(RuntimeError, "no allocation"):
                fit_ridge([], [], [], [], [], [])


@unittest.skipUnless(platform.system() == "Linux" and os.environ.get("SLURM_JOB_ID"),
                     "Numerical checks require a verified remote Slurm allocation")
class RemoteCalibrationNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from future_gain.runtime import require_slurm_compute
        require_slurm_compute()
        import torch
        cls.torch = torch

    def test_ridge_closed_form_and_training_only_scaler(self):
        model, report = fit_ridge([row(-1), row(1)], [-1, 1], ["a", "b"],
                                  [row(-2), row(2)], [-2, 2], ["c", "d"])
        self.assertEqual(model.alpha, 0.01)
        self.assertEqual(model.feature_mean[0], 0.0)
        self.assertEqual(model.feature_scale[0], 1.0)
        self.assertAlmostEqual(model.coefficients[0], 2 / 2.01)
        self.assertLess(model.predict(row(-1)), 0)
        self.assertFalse(report["threshold_distribution_fitted"])
        self.assertEqual(report["negative_train_targets"], 1)

    def test_unpenalized_intercept_balances_videos_and_rejects_leakage(self):
        model, _ = fit_ridge([row(0)] * 5, [-2] * 4 + [4], ["a"] * 4 + ["b"],
                             [row(0)], [1], ["c"])
        self.assertAlmostEqual(model.intercept, 1.0)
        self.assertEqual(model.alpha, 0.01)
        with self.assertRaisesRegex(ValueError, "both calibration"):
            fit_ridge([row(0)], [-1], ["a"], [row(0)], [1], ["a"])

    def test_feature_values_masks_and_single_frame_horizon(self):
        torch = self.torch
        extractor = FeatureExtractor()
        history = torch.tensor([0, 2, 1, 4, 3, 8], dtype=torch.bfloat16).reshape(1, 3, 1, 1, 2)
        field = torch.tensor([4, 3], dtype=torch.bfloat16).reshape(1, 1, 1, 1, 2)
        coarse = torch.tensor([5, 12], dtype=torch.bfloat16).reshape(1, 1, 1, 1, 2)
        result = extractor(history, field, coarse, frame_index=1, prediction_frames=3)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (12,))
        expected = [3, 3.8, 1.5, 1.9, 3.5, 3.9, 3, 3.8, 1, 1, 1, 0.5]
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-6, rtol=1e-6)
        short = extractor(history[:, :1], field, coarse, frame_index=0, prediction_frames=1).tolist()
        self.assertEqual(short[:4], [0.0] * 4)
        self.assertEqual(short[8:], [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
