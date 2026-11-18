"""Signed single-frame intervention labels; losses use one global reference."""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class InterventionGain:
    frame_index: int
    prediction_frames: int
    lookahead_nominal: int
    lookahead_valid: int
    effective_window_length: int
    valid_future_mask: tuple[bool, ...]
    loss_low_sum: float
    loss_high_sum: float
    delta_current: float
    delta_future_only: float
    delta_total_window: float
    delta_sum: float
    delta_per_video_frame: float
    delta_per_extra_nfe: float


def intervention_gain(low_losses: Sequence[float], high_losses: Sequence[float], *,
                      frame_index: int, prediction_frames: int, lookahead: int,
                      extra_nfe: int) -> InterventionGain:
    """Accept exactly the valid current-plus-H losses, without tail padding.

    delta_current/future_only/total_window are normalized by the complete video
    prediction horizon T. delta_sum retains the unnormalized effect. The per-NFE
    field divides the primary normalized label by the current-frame extra NFE.
    """
    for name, value, minimum in (("prediction_frames", prediction_frames, 1),
                                  ("frame_index", frame_index, 0),
                                  ("lookahead", lookahead, 0), ("extra_nfe", extra_nfe, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if frame_index >= prediction_frames:
        raise ValueError("frame_index must lie inside the future video")
    valid = min(lookahead + 1, prediction_frames - frame_index)
    if len(low_losses) != valid or len(high_losses) != valid:
        raise ValueError("Loss arrays must contain exactly the valid, unpadded window")
    low, high = tuple(map(float, low_losses)), tuple(map(float, high_losses))
    if not all(math.isfinite(x) for x in low + high):
        raise ValueError("Intervention losses must be finite")
    deltas = tuple(a - b for a, b in zip(low, high))
    delta_sum = math.fsum(deltas)
    total = delta_sum / prediction_frames
    return InterventionGain(
        frame_index, prediction_frames, lookahead, valid - 1, valid,
        tuple(i < valid for i in range(lookahead + 1)), math.fsum(low), math.fsum(high),
        deltas[0] / prediction_frames, math.fsum(deltas[1:]) / prediction_frames,
        total, delta_sum, total, total / extra_nfe)
