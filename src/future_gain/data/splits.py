"""Deterministic original-video splits using only the official BAIR train list."""

from pathlib import PurePosixPath
import random
from typing import Any, Sequence

SPLIT_SCHEMA = "bair_train_group_shuffle_v1"
TRAIN_PARENT = PurePosixPath("datasets/bair/softmotion30_44k/train/video_aux1")


def group_training_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reject ambiguous IDs instead of silently deduplicating or resampling."""
    if not rows:
        raise ValueError("Official training manifest is empty.")
    groups = {}
    seen_paths = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("video_path"), str):
            raise ValueError(f"Training row {index} lacks a video_path string.")
        raw = row["video_path"]
        path = PurePosixPath(raw)
        if (str(path) != raw or path.parent != TRAIN_PARENT or path.suffix != ".mp4"
                or ".." in path.parts or "\\" in raw or not path.stem):
            raise ValueError(f"Training row {index} is not a canonical training aux1 video.")
        group = f"train/{path.stem}"
        if raw in seen_paths or group in groups:
            raise ValueError(f"Duplicate or ambiguous original video group: {group}")
        seen_paths.add(raw)
        groups[group] = {
            "original_video_id": group,
            "source_split": "official_training",
            "manifest_index": index,
            "source_video_path": raw,
            "member": path.relative_to("datasets/bair").as_posix(),
        }
    return groups


def build_development_splits(
    rows: Sequence[dict[str, Any]], *, seed: int = 7001,
    calibration_train: int = 64, calibration_val: int = 32,
    pilot_holdout: int = 64, response_contexts: int = 8,
) -> dict[str, Any]:
    """Shuffle sorted group IDs once; all windows/seeds must retain this split.

    Small sizes are accepted for dependency-free correctness fixtures. The
    acquisition CLI exposes no size/seed overrides and uses the registered
    defaults. Official test metadata is not accepted or opened by this function.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Split seed must be an integer.")
    sizes = {"calibration_train": calibration_train,
             "calibration_val": calibration_val, "pilot_holdout": pilot_holdout}
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes.values()):
        raise ValueError("Split sizes must be positive integers.")
    if (isinstance(response_contexts, bool) or not isinstance(response_contexts, int)
            or not 1 <= response_contexts <= calibration_train):
        raise ValueError("Step-response subset must lie within calibration_train.")
    groups = group_training_rows(rows)
    if len(groups) < sum(sizes.values()):
        raise ValueError("Not enough distinct training groups for registered splits.")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    selected = {}
    offset = 0
    for name, count in sizes.items():
        selected[name] = [dict(groups[key]) for key in ordered[offset:offset + count]]
        offset += count
    return {
        "schema_version": 1,
        "split_schema": SPLIT_SCHEMA,
        "seed": seed,
        "source_split": "official_training",
        "grouping": "train/<original aux1 video filename stem>",
        "higher_level_episode_mapping_verified": False,
        "algorithm": "random.Random(seed).shuffle(sorted(original_video_ids))",
        "available_groups": len(groups),
        "sizes": sizes,
        "splits": selected,
        "dev_step_response": [dict(row) for row in selected["calibration_train"][:response_contexts]],
    }
