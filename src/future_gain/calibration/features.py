"""Versioned, current-information-only features; no tensor imports at import time."""

FEATURE_VERSION = "history_first_eval_v1"
FEATURE_NAMES = (
    "history_first_difference_mean", "history_first_difference_p90",
    "history_second_difference_mean", "history_second_difference_p90",
    "first_vector_field_mean", "first_vector_field_p90",
    "coarse_prediction_difference_mean", "coarse_prediction_difference_p90",
    "coarse_history_difference_cosine", "has_two_history_frames",
    "has_three_history_frames", "normalized_frame_index",
)
EPSILON = 1e-8


def validate_frame_position(frame_index: int, prediction_frames: int) -> None:
    if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
            or isinstance(prediction_frames, bool) or not isinstance(prediction_frames, int)
            or prediction_frames < 1 or not 0 <= frame_index < prediction_frames):
        raise ValueError("Expected integer 0 <= frame_index < prediction_frames")


class FeatureExtractor:
    """Initialize once in Slurm; return one float32 device vector of 12 values.

    Inputs are history [1, K, C, H, W] and the already-computed vector field and
    coarse clean prediction [1, 1, C, H, W]. No extra model call, future, label
    window, or remaining budget enters this API. The caller performs one vector
    transfer with result.cpu().tolist() and includes it in online timing.
    """

    def __init__(self):
        from future_gain.runtime import require_slurm_compute

        require_slurm_compute()
        import torch

        self.torch = torch

    def __call__(self, history, vector_field, coarse_clean, *,
                 frame_index: int, prediction_frames: int):
        validate_frame_position(frame_index, prediction_frames)
        torch = self.torch
        if not all(torch.is_tensor(value) for value in (history, vector_field, coarse_clean)):
            raise ValueError("Feature inputs must be tensors")
        if history.ndim != 5 or history.shape[0] != 1 or history.shape[1] < 1:
            raise ValueError("History must have shape [1, K>=1, C, H, W]")
        expected = (1, 1, *history.shape[2:])
        if tuple(vector_field.shape) != expected or tuple(coarse_clean.shape) != expected:
            raise ValueError("Current tensors must have shape [1, 1, C, H, W]")
        if any(size <= 0 for size in history.shape[2:]):
            raise ValueError("Latent channel and spatial axes must be nonempty")
        if any(value.device != history.device for value in (vector_field, coarse_clean)):
            raise ValueError("Feature inputs must share a device")
        if not all(value.is_floating_point() for value in (history, vector_field, coarse_clean)):
            raise ValueError("Feature inputs must be floating point")

        with torch.no_grad():
            # Convert before differences and squared norms, avoiding BF16
            # subtraction/squaring and reading only the three relevant frames.
            recent = history[0, -3:].detach().float()
            field = vector_field[0, 0].detach().float()
            coarse_difference = coarse_clean[0, 0].detach().float() - recent[-1]
            zero = recent.new_zeros(())

            def summaries(value):
                token_norm = torch.sqrt(value.square().mean(dim=0) + EPSILON).reshape(-1)
                return token_norm.mean(), torch.quantile(token_norm, 0.9)

            values = []
            if recent.shape[0] >= 2:
                first_difference = recent[-1] - recent[-2]
                values.extend(summaries(first_difference))
                numerator = (coarse_difference * first_difference).sum()
                denominator = (torch.linalg.vector_norm(coarse_difference.reshape(-1))
                               * torch.linalg.vector_norm(first_difference.reshape(-1)) + EPSILON)
                cosine = numerator / denominator
            else:
                values.extend((zero, zero))
                cosine = zero
            if recent.shape[0] >= 3:
                values.extend(summaries(recent[-1] - 2 * recent[-2] + recent[-3]))
            else:
                values.extend((zero, zero))
            values.extend(summaries(field))
            values.extend(summaries(coarse_difference))
            values.extend((cosine, recent.new_tensor(float(recent.shape[0] >= 2)),
                           recent.new_tensor(float(recent.shape[0] >= 3)),
                           recent.new_tensor(frame_index / max(prediction_frames - 1, 1))))
            return torch.stack(values)
