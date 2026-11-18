"""Small source/configuration audits. No model or dataset imports."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

PINNED_COMMIT = "e59536ac30e5e3fa93db756fccfd422420a65ff6"
SOURCE_PATHS = (
    "far/pipelines/pipeline_far.py",
    "far/models/far_model.py",
    "far/models/autoencoder_dc_model.py",
    "far/metrics/metric.py",
    "far/metrics/fvd.py",
    "far/data/bair_dataset.py",
    "far/data/ucf_dataset.py",
    "options/model_cfg/far/scheduler_config.json",
    "options/model_cfg/dcae/model_8x_c32_config.json",
    "options/test/far/short_video_prediction/FAR_B_bair_res64_200K_bs32.yml",
    "options/test/far/short_video_prediction/FAR_B_ucf101_res64_200K_bs32.yml",
    "requirements.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def audit_upstream(root: Path) -> dict:
    root = root.resolve()
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT or dirty:
        raise ValueError(f"Expected clean FAR at {PINNED_COMMIT}; found {commit}, dirty={bool(dirty)}")
    files = {name: sha256_file(root / name) for name in SOURCE_PATHS}
    return {"commit": commit, "clean": True, "source_root": str(root),
            "file_sha256": files, "source_identity_sha256": canonical_hash(files)}


def read_config(path: Path) -> dict:
    with path.open() as stream:
        config = json.load(stream)
    task = config["task"]
    required = {"dataset": "bair", "context_frames": 2, "prediction_frames": 28,
                "image_resolution": [64, 64], "guidance_scale": 1.0,
                "condition_on_actions": False}
    if any(task.get(k) != v for k, v in required.items()):
        raise ValueError("Initial implementation is restricted to specified BAIR 2→28 protocol.")
    if config["upstream"]["commit"] != PINNED_COMMIT:
        raise ValueError("Unreviewed upstream revision.")
    runtime = config["runtime"]
    if runtime["batch_size"] != 1 or runtime["allow_backbone_training"] or runtime["use_paid_services"]:
        raise ValueError("Invalid runtime scope.")
    return config


def local_doctor(config: dict, root: Path) -> dict:
    blockers = [f"remote.{name} is unspecified" for name in
                ("project_dir", "asset_dir", "environment_dir", "partition")
                if not config["remote"].get(name)]
    return {"scope": "metadata_only", "platform": platform.platform(),
            "python": sys.version, "upstream": audit_upstream(root),
            "remote_readiness": "blocked" if blockers else "not_yet_verified",
            "blockers": blockers, "model_loaded": False,
            "gpu_verified": False, "comparison_eligible": False}
