"""The remote-only boundary must fail before any model or dataset import."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from future_gain.runtime import require_slurm_compute


class RuntimeTests(unittest.TestCase):
    def test_mac_rejected_before_external_commands(self):
        with patch("future_gain.runtime.platform.system", return_value="Darwin"), \
             patch("future_gain.runtime.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "only on allocated Slurm"):
                require_slurm_compute()
            run.assert_not_called()

    def test_linux_without_allocation_rejected(self):
        with patch("future_gain.runtime.platform.system", return_value="Linux"), \
             patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_ID"):
                require_slurm_compute()

    def allocation(self, owner="researcher", nodes="n15\n", state="RUNNING"):
        return [SimpleNamespace(stdout=f"{owner}|{state}|n15\n"),
                SimpleNamespace(stdout=nodes)]

    def test_foreign_job_or_wrong_node_rejected(self):
        with patch("future_gain.runtime.platform.system", return_value="Linux"), \
             patch("future_gain.runtime.socket.gethostname", return_value="n15.ccvl"), \
             patch("future_gain.runtime.getpass.getuser", return_value="researcher"), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "1234"}):
            with patch("future_gain.runtime.subprocess.run", side_effect=self.allocation(owner="someone_else")):
                with self.assertRaisesRegex(RuntimeError, "current user"):
                    require_slurm_compute()
            with patch("future_gain.runtime.subprocess.run", side_effect=self.allocation(nodes="n16\n")):
                with self.assertRaisesRegex(RuntimeError, "not a member"):
                    require_slurm_compute()

    def test_owned_running_allocation_accepted(self):
        with patch("future_gain.runtime.platform.system", return_value="Linux"), \
             patch("future_gain.runtime.socket.gethostname", return_value="n15.ccvl"), \
             patch("future_gain.runtime.getpass.getuser", return_value="researcher"), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "1234"}), \
             patch("future_gain.runtime.subprocess.run", side_effect=self.allocation()):
            self.assertEqual(require_slurm_compute()["job_id"], "1234")

    def test_adapter_import_and_mac_guard_do_not_import_torch(self):
        from future_gain.adapters.far import FARAdapter
        self.assertNotIn("torch", sys.modules)
        with patch("future_gain.runtime.platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError):
                FARAdapter.from_pretrained("not-read", "not-read", "not-read")
        self.assertNotIn("torch", sys.modules)

    def test_array_master_task_uses_exact_selector(self):
        with patch("future_gain.runtime.platform.system", return_value="Linux"), \
             patch("future_gain.runtime.socket.gethostname", return_value="n15.ccvl"), \
             patch("future_gain.runtime.getpass.getuser", return_value="researcher"), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "1234", "SLURM_ARRAY_JOB_ID": "1234", "SLURM_ARRAY_TASK_ID": "3"}, clear=True), \
             patch("future_gain.runtime.subprocess.run", side_effect=self.allocation()) as run:
            self.assertEqual(require_slurm_compute()["job_selector"], "1234_3")
            self.assertEqual(run.call_args_list[0].args[0][3], "1234_3")


if __name__ == "__main__":
    unittest.main()
