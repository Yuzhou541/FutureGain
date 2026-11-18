"""Acquire exactly the audited MCVD source/config/checkpoint inside CCVL Slurm."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from future_gain.audit import canonical_hash, sha256_file
from future_gain.runtime import require_slurm_compute

COMMIT = "451da2eb635bad50da6a7c03b443a34c6eb08b3a"
REPOSITORY = "https://github.com/voletiv/mcvd-pytorch.git"
if not os.environ.get("FG_PROJECT_ROOT"):
    raise RuntimeError("Set FG_PROJECT_ROOT to an existing approved absolute project directory before invoking MCVD tools.")
PROJECT = Path(os.environ["FG_PROJECT_ROOT"]).expanduser()
if not PROJECT.is_absolute():
    raise ValueError("FG_PROJECT_ROOT must be an explicit absolute directory.")
PROJECT = PROJECT.resolve(strict=True)
if not PROJECT.is_dir():
    raise ValueError("FG_PROJECT_ROOT must name an existing project directory.")
EXPERIMENT = "bair64_big192_5c2_unetm"
FILES = {
    "config.yml": {"drive_id": "1p8WxZF9G3hrSNBTFSw4CG0Hd3pLb-NpT", "bytes": 1858,
                   "published_sha256": None,
                   "audited_sha256": "9fec801fb17d1c23c3147c7c47c6aace2ca9ad24fe3763861b5c51eb67e419ae"},
    "checkpoint_450000.pt": {"drive_id": "1F5yBZyd-IUpCZNU4PcdGOWzmJuhL6Rho",
                             "bytes": 4019322421, "published_sha256": None,
                             "audited_sha256": None},
}


def owned_directory(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(PROJECT) or not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError("Expected an owned directory inside the authorized FutureGain project")
    return path


def audit_source(root):
    """Hash all tracked source, and refuse a changed revision or working tree."""
    root = root.resolve(strict=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout
    commit = git("rev-parse", "HEAD").strip()
    dirty = git("status", "--porcelain", "--untracked-files=all").strip()
    if commit != COMMIT or dirty:
        raise ValueError(f"MCVD must be clean at {COMMIT}: found {commit}, dirty={bool(dirty)}")
    names = [name for name in git("ls-files", "-z").split("\0") if name]
    files = {name: sha256_file(root / name) for name in names}
    return {"repository": REPOSITORY, "commit": commit, "root": str(root), "clean": True,
            "file_sha256": files, "tree_sha256": canonical_hash(files)}


def validate_config(config):
    expected = {
        "data": {"dataset": "BAIR", "channels": 3, "image_size": 64, "num_frames": 5,
                 "num_frames_cond": 2, "rescaled": True, "logit_transform": False,
                 "uniform_dequantization": False, "gaussian_dequantization": False},
        "model": {"arch": "unetmore", "ngf": 192, "n_head_channels": 192,
                  "num_res_blocks": 2, "num_classes": 1000, "ema": True,
                  "gamma": False, "noise_in_cond": False, "cond_emb": False,
                  "version": "DDPM", "type": "v1", "sigma_dist": "linear"},
        "sampling": {"denoise": True, "clip_before": True, "data_init": False,
                     "init_prev_t": -1.0, "one_frame_at_a_time": False, "subsample": 100},
    }
    for group, fields in expected.items():
        if any(config[group].get(key) != value for key, value in fields.items()):
            raise ValueError(f"Unsupported MCVD {group} configuration")
    if config["data"].get("num_frames_future", 0) != 0 or "image_mean" in config:
        raise ValueError("Only the audited two-prefix, no-future-conditioning model is supported")


def verify_assets(manifest, asset_root):
    """The checkpoint SHA is an acquisition observation, not a published hash."""
    if manifest.get("experiment") != EXPERIMENT or set(manifest["files"]) != set(FILES):
        raise ValueError("Unexpected MCVD acquisition identity")
    verified = {}
    for name, expected in FILES.items():
        record = manifest["files"][name]
        path = Path(record["path"]).resolve(strict=True)
        if not path.is_relative_to(asset_root.resolve()) or record["drive_id"] != expected["drive_id"]:
            raise ValueError("Asset path or official Drive identity mismatch")
        actual = sha256_file(path)
        if path.stat().st_size != expected["bytes"] or actual != record["sha256"]:
            raise ValueError(f"Acquired MCVD asset changed: {name}")
        if expected["audited_sha256"] and actual != expected["audited_sha256"]:
            raise ValueError("The official config differs from the audited bytes")
        verified[name] = {**record, "path": str(path)}
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "source-dir", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    allocation = require_slurm_compute()
    assets = owned_directory(args.asset_dir)
    owned_directory(args.source_dir.parent)
    owned_directory(args.manifest.parent)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    if not args.source_dir.exists():
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", "--depth", "1",
                        REPOSITORY, str(args.source_dir)], check=True)
        subprocess.run(["git", "-C", str(args.source_dir), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(args.source_dir), "checkout", "--detach", COMMIT], check=True)
    source = audit_source(args.source_dir)
    destination = assets / "weights/MCVD" / EXPERIMENT
    destination.mkdir(parents=True, exist_ok=True)
    owned_directory(destination)
    # A preexisting final payload needs its original completed manifest. This
    # acquisition command never blesses, overwrites or deletes an orphan payload.
    if any((destination / name).exists() for name in FILES):
        raise FileExistsError("Reuse the completed acquisition manifest; existing assets are preserved")
    import gdown
    import yaml
    records = {}
    for name, metadata in FILES.items():
        target = destination / name
        partial = destination / f"{name}.partial_job{allocation['job_id']}"
        if partial.exists():
            raise FileExistsError(f"Preserved partial download: {partial}")
        result = gdown.download(id=metadata["drive_id"], output=str(partial), quiet=False,
                                use_cookies=False)
        if result is None or not partial.is_file() or partial.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Download size mismatch; retained partial: {partial}")
        actual = sha256_file(partial)
        if metadata["audited_sha256"] and actual != metadata["audited_sha256"]:
            raise ValueError(f"Downloaded config identity mismatch: {partial}")
        if name == "config.yml":
            validate_config(yaml.safe_load(partial.read_text()))
        records[name] = {**metadata, "path": str(target), "sha256": actual,
                         "sha256_provenance": "computed_from_downloaded_bytes",
                         "url": f"https://drive.google.com/file/d/{metadata['drive_id']}/view"}
    # Publish only after BOTH downloads validate; incomplete attempts retain all
    # files under job-specific partial names and cannot look like valid assets.
    for name in FILES:
        (destination / f"{name}.partial_job{allocation['job_id']}").rename(destination / name)
    manifest = {"schema_version": 1, "allocation": allocation, "experiment": EXPERIMENT,
                "source": source, "files": records, "model_loaded": False,
                "external_checkpoint_sha256_available": False}
    with args.manifest.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps({"manifest": str(args.manifest), "files": records, "model_loaded": False}))


if __name__ == "__main__":
    main()
