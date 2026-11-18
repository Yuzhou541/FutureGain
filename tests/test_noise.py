from dataclasses import replace
import json
import os
import random
import subprocess
import sys
import unittest

from future_gain.sampling.noise import NoiseBank, NoiseIdentity


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.identity = NoiseIdentity("revision1", "video1", tuple(range(30)), 101)

    def test_frame_identity_is_independent_of_visit_order_and_global_rng(self):
        before = random.getstate()
        forward = {i: self.identity.frame_seed(i) for i in range(28)}
        backward = {i: self.identity.frame_seed(i) for i in reversed(range(28))}
        self.assertEqual(forward, backward)
        self.assertEqual(len(set(forward.values())), 28)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(set(self.identity.frame_key(0)), {
            "dataset_revision", "original_video_id", "clip_frame_indices",
            "trajectory_seed", "noise_schema_version", "future_frame_index"})

    def test_every_noise_identity_component_changes_seed(self):
        variants = [replace(self.identity, dataset_revision="revision2"),
                    replace(self.identity, original_video_id="video2"),
                    replace(self.identity, clip_frame_indices=tuple(range(1, 31))),
                    replace(self.identity, trajectory_seed=202)]
        for identity in variants:
            self.assertNotEqual(identity.frame_seed(0), self.identity.frame_seed(0))

    def test_python_hash_randomization_cannot_change_seed(self):
        code = ("from future_gain.sampling.noise import NoiseIdentity; "
                "print(NoiseIdentity('revision1','video1',tuple(range(30)),101).frame_seed(3))")
        values = [subprocess.check_output([sys.executable, "-c", code],
                  env={**os.environ, "PYTHONHASHSEED": salt}, text=True).strip()
                  for salt in ("1", "44")]
        self.assertEqual(values, [str(self.identity.frame_seed(3))] * 2)

    def test_manifest_records_shape_precision_and_device_without_importing_torch(self):
        torch_before = sys.modules.get("torch")
        bank = NoiseBank(self.identity, (1, 32, 1, 8, 8), device="cuda:0", dtype="bfloat16")
        manifest = bank.manifest(28)
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(len(manifest["frames"]), 28)
        self.assertEqual(bank.manifest(28), manifest)
        different = NoiseBank(self.identity, (1, 32, 1, 8, 8), device="cuda:0", dtype="float32")
        self.assertNotEqual(different.manifest(28)["noise_manifest_sha256"],
                            manifest["noise_manifest_sha256"])
        self.assertIs(sys.modules.get("torch"), torch_before)

    def test_invalid_keys_fail(self):
        with self.assertRaises(ValueError):
            self.identity.frame_seed(-1)
        with self.assertRaises(ValueError):
            NoiseIdentity("r", "v", [0, 1], 101)


if __name__ == "__main__":
    unittest.main()
