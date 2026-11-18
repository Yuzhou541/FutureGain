"""Split tests use synthetic path rows, never actual data or model outputs."""

import random
import unittest

from future_gain.data.splits import build_development_splits, group_training_rows


def synthetic_rows(count=200):
    return [{"video_path": f"datasets/bair/softmotion30_44k/train/video_aux1/{i}.mp4"}
            for i in range(count)]


class DataSplitTests(unittest.TestCase):
    def test_registered_groups_disjoint_and_step_response_is_train_prefix(self):
        assignment = build_development_splits(synthetic_rows())
        groups = [{row["original_video_id"] for row in entries}
                  for entries in assignment["splits"].values()]
        self.assertEqual([len(group) for group in groups], [64, 32, 64])
        self.assertEqual(len(set.union(*groups)), 160)
        self.assertEqual(assignment["dev_step_response"], assignment["splits"]["calibration_train"][:8])
        self.assertEqual(assignment["seed"], 7001)

    def test_row_order_does_not_change_selected_ids_or_global_rng(self):
        rows = synthetic_rows()
        before = random.getstate()
        forward = build_development_splits(rows)
        reverse = build_development_splits(list(reversed(rows)))
        self.assertEqual(random.getstate(), before)
        for name in forward["splits"]:
            self.assertEqual([row["original_video_id"] for row in forward["splits"][name]],
                             [row["original_video_id"] for row in reverse["splits"][name]])

    def test_group_sequence_matches_frozen_shuffle_recipe(self):
        rows = synthetic_rows()
        expected = sorted(f"train/{i}" for i in range(200))
        random.Random(7001).shuffle(expected)
        assignment = build_development_splits(rows)
        actual = [row["original_video_id"] for entries in assignment["splits"].values() for row in entries]
        self.assertEqual(actual, expected[:160])

    def test_duplicate_missing_and_official_test_paths_fail(self):
        valid = synthetic_rows(1)[0]
        for rows in ([valid, valid], [{}], [{"video_path": None}],
                     [{"video_path": "datasets/bair/softmotion30_44k/test/video_aux1/0.mp4"}],
                     [{"video_path": "datasets/bair/softmotion30_44k/train/video_main/0.mp4"}],
                     [{"video_path": "datasets/bair/softmotion30_44k/train/video_aux1/../0.mp4"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                group_training_rows(rows)

    def test_insufficient_groups_fail_instead_of_reusing_between_splits(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            build_development_splits(synthetic_rows(159))


if __name__ == "__main__":
    unittest.main()
