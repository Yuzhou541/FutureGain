"""Video-balanced signed-gain ridge calibration with fixed validation selection.

Only fit_ridge imports NumPy, after allocation validation. JSON model loading
and one-vector prediction use the standard library. The caller must attach
split/label/model/run provenance and the separate all-validation-rollout score
distribution before treating the model as a complete deployment checkpoint.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence

from future_gain.calibration.features import FEATURE_NAMES, FEATURE_VERSION

ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
WEIGHTING = "each_original_video_total_one; each_row_weight_one_over_rows_in_video"


def video_weights(video_ids: Sequence[str]) -> list[float]:
    if not video_ids or any(not isinstance(value, str) or not value for value in video_ids):
        raise ValueError("Nonempty original video IDs are required")
    counts = Counter(video_ids)
    return [1.0 / counts[value] for value in video_ids]


def _vector(values, name: str, length: int = len(FEATURE_NAMES)) -> tuple[float, ...]:
    result = tuple(map(float, values))
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain exactly {length} finite values")
    return result


def weighted_scaler(features, video_ids) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Population moments, original-video equal weight; constants get scale 1."""
    rows = tuple(_vector(row, "features") for row in features)
    if len(rows) != len(video_ids):
        raise ValueError("Feature and original-video counts differ")
    weights = video_weights(video_ids)
    total = math.fsum(weights)
    means = tuple(math.fsum(weight * row[j] for row, weight in zip(rows, weights)) / total
                  for j in range(len(FEATURE_NAMES)))
    scales = tuple(math.sqrt(math.fsum(weight * (row[j] - means[j]) ** 2
                                      for row, weight in zip(rows, weights)) / total)
                   for j in range(len(FEATURE_NAMES)))
    return means, tuple(value if value > 0 else 1.0 for value in scales)


def video_equal_rmse(targets, predictions, video_ids) -> float:
    """sqrt(mean_video(mean_rows_in_video(squared error)))."""
    if len(targets) != len(predictions) or len(targets) != len(video_ids):
        raise ValueError("Validation arrays have different lengths")
    targets = _vector(targets, "targets", len(video_ids))
    predictions = _vector(predictions, "predictions", len(video_ids))
    weights = video_weights(video_ids)
    return math.sqrt(math.fsum(weight * (actual - predicted) ** 2
                               for actual, predicted, weight in zip(targets, predictions, weights))
                     / math.fsum(weights))


def _average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def video_equal_spearman(targets, predictions, video_ids) -> dict:
    """Mean within-video rank correlation; report undefined videos explicitly."""
    # Reuse the finite/count/ID validation, without altering ranks or targets.
    video_equal_rmse(targets, predictions, video_ids)
    groups = {}
    for target, prediction, identity in zip(targets, predictions, video_ids):
        groups.setdefault(identity, []).append((target, prediction))
    correlations = {}
    undefined = []
    for identity, rows in sorted(groups.items()):
        left = _average_ranks([row[0] for row in rows])
        right = _average_ranks([row[1] for row in rows])
        center = (len(rows) - 1) / 2
        left = [value - center for value in left]
        right = [value - center for value in right]
        denominator = math.sqrt(math.fsum(value * value for value in left)
                                * math.fsum(value * value for value in right))
        if denominator == 0:
            undefined.append(identity)
        else:
            correlations[identity] = math.fsum(a * b for a, b in zip(left, right)) / denominator
    return {"mean": (math.fsum(correlations.values()) / len(correlations) if correlations else None),
            "defined_videos": len(correlations), "undefined_video_ids": undefined,
            "per_video": correlations}


@dataclass(frozen=True)
class RidgeRouter:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float

    def __post_init__(self):
        for name in ("feature_mean", "feature_scale", "coefficients"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("Feature scales must be positive")
        if not math.isfinite(self.intercept) or self.alpha not in ALPHA_CANDIDATES:
            raise ValueError("Expected finite intercept and a preregistered alpha")

    def predict(self, features) -> float:
        row = _vector(features, "features")
        return self.intercept + math.fsum(coefficient * (value - mean) / scale
                                         for value, mean, scale, coefficient in
                                         zip(row, self.feature_mean, self.feature_scale, self.coefficients))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "model_type": "ridge", "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES), "feature_mean": list(self.feature_mean),
                "feature_scale": list(self.feature_scale), "coefficients": list(self.coefficients),
                "intercept": float(self.intercept), "alpha": float(self.alpha),
                "label_scale": 1.0, "signed_targets": True, "training_weighting": WEIGHTING}

    @classmethod
    def from_dict(cls, payload: dict):
        required = {"schema_version", "model_type", "feature_version", "feature_names", "feature_mean",
                    "feature_scale", "coefficients", "intercept", "alpha", "label_scale",
                    "signed_targets", "training_weighting"}
        if (set(payload) != required or payload["schema_version"] != 1 or payload["model_type"] != "ridge"
                or payload["feature_version"] != FEATURE_VERSION or payload["feature_names"] != list(FEATURE_NAMES)
                or payload["label_scale"] != 1.0 or payload["signed_targets"] is not True
                or payload["training_weighting"] != WEIGHTING):
            raise ValueError("Router payload does not match the frozen feature/label schema")
        return cls(payload["feature_mean"], payload["feature_scale"], payload["coefficients"],
                   float(payload["intercept"]), float(payload["alpha"]))


def fit_ridge(train_features, train_targets, train_video_ids, validation_features,
              validation_targets, validation_video_ids) -> tuple[RidgeRouter, dict]:
    """Fit only calibration_train; choose alpha on disjoint calibration_val.

    Objective: sum_i (1/n_video(i)) * residual_i**2 + alpha * ||w||**2.
    Intercept is unpenalized; targets remain in their original signed units.
    Exact validation-score ties select the smaller alpha. This function does
    not calibrate thresholds from the sparsely labelled intervention positions.
    """
    from future_gain.runtime import require_slurm_compute

    require_slurm_compute()
    import numpy as np

    train_rows = tuple(_vector(row, "train features") for row in train_features)
    validation_rows = tuple(_vector(row, "validation features") for row in validation_features)
    weights = video_weights(train_video_ids)
    video_weights(validation_video_ids)
    if set(train_video_ids) & set(validation_video_ids):
        raise ValueError("An original video occurs in both calibration_train and calibration_val")
    if len(train_rows) != len(train_video_ids) or len(validation_rows) != len(validation_video_ids):
        raise ValueError("Feature and original-video counts differ")
    target = _vector(train_targets, "train targets", len(train_rows))
    validation_target = _vector(validation_targets, "validation targets", len(validation_rows))
    means, scales = weighted_scaler(train_rows, train_video_ids)
    normalized = (np.asarray(train_rows, dtype=np.float64) - means) / scales
    design = np.column_stack((normalized, np.ones(len(train_rows))))
    weight_array = np.asarray(weights, dtype=np.float64)
    gram = design.T @ (weight_array[:, None] * design)
    rhs = design.T @ (weight_array * np.asarray(target, dtype=np.float64))
    candidates = []
    models = []
    for alpha in ALPHA_CANDIDATES:
        penalty = np.eye(len(FEATURE_NAMES) + 1, dtype=np.float64) * alpha
        penalty[-1, -1] = 0  # Do not shrink the signed mean label/intercept.
        solution = np.linalg.solve(gram + penalty, rhs)
        model = RidgeRouter(means, scales, tuple(solution[:-1].tolist()), float(solution[-1]), alpha)
        predictions = [model.predict(row) for row in validation_rows]
        candidates.append({"alpha": alpha,
                           "validation_video_equal_rmse": video_equal_rmse(validation_target, predictions, validation_video_ids),
                           "validation_video_equal_spearman": video_equal_spearman(validation_target, predictions, validation_video_ids)})
        models.append(model)
    selected = min(range(len(candidates)), key=lambda i: (candidates[i]["validation_video_equal_rmse"], candidates[i]["alpha"]))
    report = {"alpha_candidates": list(ALPHA_CANDIDATES), "candidates": candidates,
              "selected_alpha": models[selected].alpha,
              "selection_metric": "sqrt(mean_video(mean_rows_in_video(squared_error)))",
              "alpha_tie_rule": "smaller_alpha_for_exact_equal_validation_rmse",
              "training_weighting": WEIGHTING, "scaler_fit_split": "calibration_train",
              "train_video_ids": sorted(set(train_video_ids)), "train_rows": len(train_rows),
              "validation_video_ids": sorted(set(validation_video_ids)), "validation_rows": len(validation_rows),
              "negative_train_targets": sum(value < 0 for value in target),
              "negative_validation_targets": sum(value < 0 for value in validation_target),
              "threshold_distribution_fitted": False}
    return models[selected], report
