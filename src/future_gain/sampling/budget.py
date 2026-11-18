"""Exact NFE allocation without looking at future frames or future scores."""

from dataclasses import dataclass
import math
import random
from typing import Sequence


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _quota(prediction_frames: int, high_frames: int) -> None:
    _integer(prediction_frames, "prediction_frames", 1)
    _integer(high_frames, "high_frames")
    if high_frames > prediction_frames:
        raise ValueError("high_frames exceeds prediction_frames")


def high_count(prediction_frames: int, fraction: float) -> int:
    """Round the quota by floor(q*T+0.5), not Python's round-to-even."""
    _integer(prediction_frames, "prediction_frames", 1)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and in [0,1]")
    return math.floor(fraction * prediction_frames + 0.5)


def expected_calls(prediction_frames: int, low_nfe: int, high_nfe: int,
                   high_frames: int) -> int:
    _quota(prediction_frames, high_frames)
    _integer(low_nfe, "low_nfe", 1)
    _integer(high_nfe, "high_nfe", 1)
    if high_nfe <= low_nfe:
        raise ValueError("high_nfe must exceed low_nfe")
    return prediction_frames * low_nfe + high_frames * (high_nfe - low_nfe)


def _spaced_positions(length: int, count: int) -> set[int]:
    # Midpoint placement is deterministic, distinct, and has no content input.
    return {(2 * i + 1) * length // (2 * count) for i in range(count)}


def near_uniform_schedule(prediction_frames: int, total_calls: int) -> tuple[int, ...]:
    _integer(prediction_frames, "prediction_frames", 1)
    _integer(total_calls, "total_calls", prediction_frames)
    base, extra = divmod(total_calls, prediction_frames)
    positions = _spaced_positions(prediction_frames, extra)
    return tuple(base + int(t in positions) for t in range(prediction_frames))


def make_schedule(kind: str, prediction_frames: int, low_nfe: int,
                  high_nfe: int, high_frames: int, *, seed: int = 0) -> tuple[int, ...]:
    """Build a content-independent policy; random uses a separate RNG."""
    budget = expected_calls(prediction_frames, low_nfe, high_nfe, high_frames)
    if kind == "near_uniform_exact_budget":
        return near_uniform_schedule(prediction_frames, budget)
    if kind == "random_exact_budget":
        positions = set(random.Random(seed).sample(range(prediction_frames), high_frames))
    elif kind == "periodic_exact_budget":
        positions = _spaced_positions(prediction_frames, high_frames)
    elif kind == "front_loaded":
        positions = set(range(high_frames))
    elif kind == "back_loaded":
        positions = set(range(prediction_frames - high_frames, prediction_frames))
    else:
        raise ValueError(f"Unknown schedule: {kind}")
    return tuple(high_nfe if t in positions else low_nfe for t in range(prediction_frames))


def fixed_schedule(prediction_frames: int, nfe: int) -> tuple[int, ...]:
    _integer(prediction_frames, "prediction_frames", 1)
    _integer(nfe, "nfe", 1)
    return (nfe,) * prediction_frames


@dataclass(frozen=True)
class ThresholdRule:
    threshold: float
    tie_probability: float
    tie_tolerance: float

    def choose(self, score: float, rng: random.Random) -> bool:
        if not math.isfinite(score):
            raise ValueError("Online score must be finite")
        if score > self.threshold + self.tie_tolerance:
            return True
        if score < self.threshold - self.tie_tolerance:
            return False
        return rng.random() < self.tie_probability


class EmpiricalThreshold:
    """Equally weighted calibration-validation scores; no test fitting.

    Quantiles use the inverse empirical CDF (nearest rank). The same absolute
    tie tolerance is used for both calibration counts and online decisions.
    """

    def __init__(self, scores: Sequence[float], *, tie_tolerance: float = 1e-12):
        self.scores = tuple(sorted(float(x) for x in scores))
        if not self.scores or not all(math.isfinite(x) for x in self.scores):
            raise ValueError("Calibration scores must be nonempty and finite")
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError("tie_tolerance must be finite and nonnegative")
        self.tie_tolerance = tie_tolerance

    def rule(self, desired_rate: float) -> ThresholdRule:
        if not math.isfinite(desired_rate) or not 0 < desired_rate < 1:
            raise ValueError("Threshold decisions require desired_rate in (0,1)")
        count = len(self.scores)
        index = max(0, math.ceil((1 - desired_rate) * count) - 1)
        threshold = self.scores[index]
        lower = threshold - self.tie_tolerance
        upper = threshold + self.tie_tolerance
        above = sum(x > upper for x in self.scores) / count
        equal = sum(lower <= x <= upper for x in self.scores) / count
        probability = min(1.0, max(0.0, (desired_rate - above) / equal))
        return ThresholdRule(threshold, probability, self.tie_tolerance)


@dataclass(frozen=True)
class BudgetDecision:
    frame_index: int
    remaining_frames_before: int
    remaining_high_before: int
    predicted_gain: float
    threshold: float | None
    tie_probability: float | None
    decision: str
    decision_reason: str
    remaining_high_after: int
    nfe_requested: int


class ExactQuotaController:
    """Consume one currently visible score per frame and preserve exact quota."""

    def __init__(self, prediction_frames: int, high_frames: int, low_nfe: int,
                 high_nfe: int, distribution: EmpiricalThreshold, *, seed: int):
        self.expected_denoiser_calls = expected_calls(
            prediction_frames, low_nfe, high_nfe, high_frames)
        self.prediction_frames = prediction_frames
        self.remaining_high = high_frames
        self.low_nfe = low_nfe
        self.high_nfe = high_nfe
        self.distribution = distribution
        self._rng = random.Random(seed)
        self.frame_index = 0
        self.requested_calls = 0

    def choose(self, score: float) -> BudgetDecision:
        if self.frame_index >= self.prediction_frames:
            raise RuntimeError("Controller has already consumed all frames")
        if not math.isfinite(score):
            raise ValueError("Online score must be finite")
        remaining = self.prediction_frames - self.frame_index
        before = self.remaining_high
        threshold = probability = None
        if before == 0:
            high, reason = False, "forced_low"
        elif before == remaining:
            high, reason = True, "forced_high"
        else:
            rule = self.distribution.rule(before / remaining)
            high, reason = rule.choose(score, self._rng), "score"
            threshold, probability = rule.threshold, rule.tie_probability
        self.remaining_high -= int(high)
        nfe = self.high_nfe if high else self.low_nfe
        result = BudgetDecision(
            self.frame_index, remaining, before, float(score), threshold, probability,
            "high" if high else "low", reason, self.remaining_high, nfe)
        self.frame_index += 1
        self.requested_calls += nfe
        return result

    def assert_complete(self, observed_calls: int) -> None:
        if (self.frame_index != self.prediction_frames or self.remaining_high != 0
                or self.requested_calls != self.expected_denoiser_calls
                or observed_calls != self.expected_denoiser_calls):
            raise AssertionError("Unfinished rollout or actual denoiser calls violate exact budget")
