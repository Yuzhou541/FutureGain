import math
import unittest

from future_gain.calibration.labels import intervention_gain


class LabelTests(unittest.TestCase):
    def test_hand_computable_current_and_future_signed_gain(self):
        result = intervention_gain([1, 4, 2, 1], [2, 1, 1, 1], frame_index=1,
                                   prediction_frames=8, lookahead=3, extra_nfe=8)
        self.assertEqual(result.delta_current, -1 / 8)
        self.assertEqual(result.delta_future_only, 4 / 8)
        self.assertEqual(result.delta_sum, 3)
        self.assertEqual(result.delta_per_video_frame, 3 / 8)
        self.assertEqual(result.delta_per_extra_nfe, 3 / 64)
        self.assertEqual(result.valid_future_mask, (True,) * 4)

    def test_zero_lookahead_includes_current_frame_and_keeps_negative_label(self):
        result = intervention_gain([1], [3], frame_index=2, prediction_frames=8,
                                   lookahead=0, extra_nfe=8)
        self.assertEqual(result.delta_per_video_frame, -2 / 8)
        self.assertEqual(result.delta_future_only, 0)
        self.assertEqual(result.effective_window_length, 1)

    def test_tail_is_truncated_and_normalized_by_full_T(self):
        result = intervention_gain([4], [2], frame_index=7, prediction_frames=8,
                                   lookahead=3, extra_nfe=8)
        self.assertEqual(result.delta_per_video_frame, 0.25)
        self.assertEqual(result.valid_future_mask, (True, False, False, False))
        self.assertEqual(result.lookahead_valid, 0)
        with self.assertRaises(ValueError):
            intervention_gain([4] * 4, [2] * 4, frame_index=7, prediction_frames=8,
                              lookahead=3, extra_nfe=8)

    def test_rejects_missing_or_nonfinite_losses(self):
        for low, high in (([], []), ([math.nan], [0]), ([0], [math.inf])):
            with self.assertRaises(ValueError):
                intervention_gain(low, high, frame_index=0, prediction_frames=8,
                                  lookahead=0, extra_nfe=8)


if __name__ == "__main__":
    unittest.main()
