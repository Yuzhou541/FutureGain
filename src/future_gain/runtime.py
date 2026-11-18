"""Reject substantial work outside the current user's allocated compute node."""

import getpass
import os
import platform
import socket
import subprocess


def require_slurm_compute() -> dict[str, str]:
    """Check allocation ownership, state and node membership before model imports."""
    if platform.system() != "Linux":
        raise RuntimeError("Model/data work is allowed only on allocated Slurm compute nodes.")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("A real SLURM_JOB_ID is required; do not run on a login node.")
    array_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if bool(array_id) != bool(array_task) or (array_id and not (array_id.isdigit() and array_task.isdigit())):
        raise RuntimeError("Incomplete or invalid Slurm array identity.")
    # One array task can share the numeric master job ID; querying that numeric
    # ID returns all tasks. The exact array selector retains unique ownership.
    selector = f"{array_id}_{array_task}" if array_id else job_id
    hostname = socket.gethostname().split(".")[0]
    if "login" in hostname.lower():
        raise RuntimeError("Substantial work is forbidden on a login node.")
    job = subprocess.run(
        ["squeue", "--noheader", "--jobs", selector, "--format=%u|%T|%N"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip().splitlines()
    if len(job) != 1:
        raise RuntimeError("Cannot uniquely verify the current Slurm allocation.")
    owner, state, nodelist = job[0].split("|", 2)
    if owner != getpass.getuser() or state != "RUNNING":
        raise RuntimeError("Slurm allocation is not running under the current user.")
    nodes = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist], check=True,
        capture_output=True, text=True, timeout=15,
    ).stdout.splitlines()
    if hostname not in [node.split(".")[0] for node in nodes]:
        raise RuntimeError("This host is not a member of the current allocation.")
    return {"job_id": job_id, "job_selector": selector, "hostname": hostname, "owner": owner, "state": state}
