"""Strict UCF101 split-3 metadata checks; no video/model imports or I/O."""

import random
import re
from pathlib import PurePosixPath


VIDEO = re.compile(r"v_([A-Za-z0-9]+)_g([0-9]{2})_c([0-9]{2})\.(avi|mp4)")
MIRROR_PREFIX = "datasets/ucf101/video/"
SIZES = {"calibration_train": 64, "calibration_val": 32, "controller_holdout": 64}


def video_identity(relative: str, extension: str) -> tuple[str, int, str]:
    """Preserve official spelling; class identity comes from the directory.

    Official HandstandPushups filenames use HandStandPushups. Do not normalize
    either spelling: full mirror IDs must still equal the official list exactly.
    """
    if not isinstance(relative, str) or "\\" in relative or "\x00" in relative:
        raise ValueError("Invalid video path")
    parts = PurePosixPath(relative).parts
    if len(parts) != 2 or "/".join(parts) != relative or any(p in (".", "..") for p in parts):
        raise ValueError("Expected exactly class/video with no path traversal")
    match = VIDEO.fullmatch(parts[1])
    if not match or match[4] != extension:
        raise ValueError("Video filename or extension is invalid")
    group, clip = int(match[2]), int(match[3])
    if not 1 <= group <= 25 or clip < 1:
        raise ValueError("Invalid UCF group/clip index")
    return parts[0], group, str(PurePosixPath(parts[0]) / (parts[1].rsplit(".", 1)[0] + ".avi"))


def parse_class_index(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].isdigit() or fields[1] in result:
            raise ValueError("Invalid or duplicate official class row")
        result[fields[1]] = int(fields[0])
    if sorted(result.values()) != list(range(1, len(result) + 1)):
        raise ValueError("Official class indices must be contiguous and one-based")
    return result


def parse_official(text: str, classes: dict[str, int], training: bool) -> dict[str, dict]:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != (2 if training else 1):
            raise ValueError("Official split row has the wrong columns")
        name, group, official_id = video_identity(fields[0], "avi")
        if name not in classes or (training and fields[1] != str(classes[name])):
            raise ValueError("Official class label mismatch")
        if official_id in result:
            raise ValueError("Duplicate official video")
        result[official_id] = {"class_name": name, "group_id": group, "label": classes[name] - 1}
    return result


def verify_mirror(rows: list, official: dict[str, dict]) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("Mirror manifest must be a list")
    result, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"video_path", "label"}:
            raise ValueError("Unexpected mirror row schema")
        path = row["video_path"]
        if not isinstance(path, str) or not path.startswith(MIRROR_PREFIX):
            raise ValueError("Unreviewed mirror path prefix")
        relative = path[len(MIRROR_PREFIX):]
        name, group, official_id = video_identity(relative, "mp4")
        if official_id in seen or official_id not in official:
            raise ValueError("Duplicate or non-official mirror member")
        if type(row["label"]) is not int or row["label"] != official[official_id]["label"]:
            raise ValueError("Mirror label differs from official one-based label minus one")
        seen.add(official_id)
        result.append({"video_id": official_id, "class_name": name, "group_id": group,
                       "label": row["label"], "mirror_manifest_index": index,
                       "mirror_video_path": path, "member": "video/" + relative})
    if seen != set(official):
        raise ValueError("Mirror membership is not exactly the complete official split")
    return result


def build_assignment(train_rows: list, test_rows: list, class_text: str,
                     train_text: str, test_text: str) -> dict:
    classes = parse_class_index(class_text)
    official_train = parse_official(train_text, classes, True)
    official_test = parse_official(test_text, classes, False)
    if set(official_train) & set(official_test):
        raise ValueError("Official train/test video overlap")
    group_key = lambda row: (row["class_name"], row["group_id"])
    train_groups = {group_key(row) for row in official_train.values()}
    test_groups = {group_key(row) for row in official_test.values()}
    if train_groups & test_groups:
        raise ValueError("Official train/test group overlap")
    train = verify_mirror(train_rows, official_train)
    verify_mirror(test_rows, official_test)
    grouped = {}
    for row in train:
        grouped.setdefault(group_key(row), []).append(row)
    groups = sorted(grouped)
    random.Random(7001).shuffle(groups)
    if len(groups) < sum(SIZES.values()):
        raise ValueError("Fewer than 160 distinct training groups")
    selected = [min(grouped[key], key=lambda row: row["video_id"]) for key in groups[:160]]
    splits, offset = {}, 0
    for name, count in SIZES.items():
        splits[name] = selected[offset:offset + count]
        offset += count
    return {"schema_version": 1, "dataset": "UCF101", "source_split": "official_split3_train",
            "seed": 7001, "sizes": SIZES, "splits": splits,
            "selection": "shuffle sorted(class_name,integer_group_id) with random.Random(7001); "
                         "take first160 groups; choose lexicographically first official video ID in each",
            "membership": {"official_class_count": len(classes), "train_videos": len(train),
                           "test_videos": len(official_test), "train_groups": len(train_groups),
                           "test_groups": len(test_groups), "exact_full_membership": True,
                           "official_train_test_groups_disjoint": True},
            "bounded_gate": splits["calibration_train"][:4],
            "selected_group_count": len({group_key(row) for row in selected}),
            "pixels_match_official_avi": "not_verified; FAR mirror uses MP4",
            "test_video_decoded": False, "selection_uses_quality_or_video_bytes": False}
