#!/usr/bin/env python3
"""Fit the original signed-gain ridge from explicit, video-disjoint scalar records."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from future_gain.audit import canonical_hash, sha256_file
from future_gain.calibration.features import FEATURE_NAMES, FEATURE_VERSION
from future_gain.calibration.labels import intervention_gain
from future_gain.calibration.ridge import fit_ridge
from future_gain.runtime import require_slurm_compute


def require(value, message):
    if not value:
        raise ValueError(message)


def feature_vector(values):
    require(isinstance(values, list) and len(values) == len(FEATURE_NAMES), "Expected 12 feature values")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Features must be finite numbers")
    require(values[9] in (0, 1) and values[10] in (0, 1), "Invalid history-validity masks")
    return values


def row_identity(row):
    require(type(row.get("video_id")) is str and bool(row["video_id"]), "Original video_id is required")
    require(type(row.get("trajectory_seed")) is int and row["trajectory_seed"] >= 0, "Nonnegative integer trajectory_seed is required")
    require(type(row.get("frame_index")) is int and 0 <= row["frame_index"] < 28, "Expected frame_index in 0..27")
    return row["video_id"], row["trajectory_seed"], row["frame_index"]


def validate_input(data, lookahead):
    """Validate explicit scalar input; do not read clips or load a numerical runtime."""
    canonical_hash(data)
    require(data.get("schema_version") == 1 and data.get("feature_version") == FEATURE_VERSION,
            "Wrong calibration schema or feature version")
    require(data.get("feature_names") == list(FEATURE_NAMES), "Feature names/order differ from the extractor")
    require(data.get("protocol") == {"prediction_frames": 28, "low_nfe": 8, "high_nfe": 16,
            "label_metric": "canonical_lpips_alex", "reference_mode": "global_dense",
            "lookahead": lookahead}, "Declare the exact 2->28 label protocol")
    datasets = []
    used = []
    validation_features = {}
    for split in ("calibration_train", "calibration_val"):
        rows = data.get(split)
        require(isinstance(rows, list) and bool(rows), "Both calibration splits need labelled rows")
        features, targets, videos, identities = [], [], [], set()
        for row in rows:
            identity = row_identity(row)
            require(identity not in identities, "Repeated labelled video/seed/frame identity")
            identities.add(identity)
            features.append(feature_vector(row["features"]))
            label = intervention_gain(row["low_losses"], row["high_losses"], frame_index=identity[2],
                                      prediction_frames=28, lookahead=lookahead, extra_nfe=8)
            targets.append(label.delta_per_video_frame)
            videos.append(identity[0])
            if split == "calibration_val":
                validation_features[identity] = features[-1]
        datasets.append((features, targets, videos))
        used.append(set(videos))
    require(not used[0] & used[1], "Original videos must be disjoint between calibration splits")
    rows = data.get("validation_rollout_features")
    require(isinstance(rows, list) and bool(rows), "All-frame validation rollout features are required for the threshold CDF")
    seen, contexts = {}, {}
    for row in rows:
        identity = row_identity(row)
        require(identity[0] in used[1] and identity not in seen, "CDF rows must be unique and belong to validation videos")
        seen[identity] = feature_vector(row["features"])
        contexts.setdefault(identity[:2], set()).add(identity[2])
    require(all(frames == set(range(28)) for frames in contexts.values()), "Every CDF video/seed requires all 28 frames")
    require({video for video, seed in contexts} == used[1], "CDF video coverage differs from calibration validation")
    seed_sets = [{seed for video, seed in contexts if video == identity} for identity in sorted(used[1])]
    require(all(seeds == seed_sets[0] for seeds in seed_sets), "All validation videos require the same CDF trajectory seeds")
    require(all(identity in seen and seen[identity] == values for identity, values in validation_features.items()),
            "Labelled validation features must match their all-frame CDF records exactly")
    return datasets, [seen[key] for key in sorted(seen)], sorted(contexts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Explicit calibration JSON; see docs/workflows.md")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing artifacts are preserved")
    parser.add_argument("--lookahead", type=int, choices=(0, 1, 3, 6), default=3)
    args = parser.parse_args()
    allocation = require_slurm_compute()
    data = json.loads(args.input.read_text())
    datasets, cdf_features, cdf_contexts = validate_input(data, args.lookahead)
    require(not args.output.exists(), "Choose a new output directory")
    model, report = fit_ridge(*datasets[0], *datasets[1])
    scores = [model.predict(values) for values in cdf_features]
    payload = {"schema_version": 1, "lookahead": args.lookahead, "model": model.to_dict(),
               "validation_scores": scores, "cdf_metadata": {"weighting": "equal_video_seed_frame; all_28_frames",
               "tie_tolerance": 1e-12, "quantile": "inverse_empirical_cdf_nearest_rank"}}
    report.update({"input_sha256": sha256_file(args.input), "feature_version": FEATURE_VERSION,
                   "lookahead": args.lookahead, "threshold_distribution_fitted": True,
                   "cdf_contexts": cdf_contexts, "cdf_rows": len(scores), "allocation": allocation,
                   "generator_calls": 0, "metric_model_calls": 0})
    args.output.mkdir(parents=True)
    for name, value in (("router.json", payload), ("fit_report.json", report)):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": "completed", "output": str(args.output), "lookahead": args.lookahead,
                      "selected_alpha": model.alpha, "cdf_rows": len(scores)}))


if __name__ == "__main__":
    main()
