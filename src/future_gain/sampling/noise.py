"""Noise identities bind physical frames, never policies, NFE, or branch order."""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class NoiseIdentity:
    dataset_revision: str
    original_video_id: str
    clip_frame_indices: tuple[int, ...]
    trajectory_seed: int
    noise_schema_version: str = "stable_sha256_v1"

    def __post_init__(self) -> None:
        if not self.dataset_revision or not self.original_video_id:
            raise ValueError("Noise identity requires dataset revision and original video ID")
        if self.noise_schema_version != "stable_sha256_v1":
            raise ValueError("Unsupported noise schema")
        if (not isinstance(self.clip_frame_indices, tuple) or not self.clip_frame_indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                       for i in self.clip_frame_indices)):
            raise ValueError("clip_frame_indices must be a nonempty tuple of nonnegative integers")
        if isinstance(self.trajectory_seed, bool) or not isinstance(self.trajectory_seed, int):
            raise ValueError("trajectory_seed must be an integer")

    def frame_key(self, future_frame_index: int) -> dict[str, Any]:
        if (isinstance(future_frame_index, bool) or not isinstance(future_frame_index, int)
                or future_frame_index < 0):
            raise ValueError("future_frame_index must be a nonnegative integer")
        return {**asdict(self), "future_frame_index": future_frame_index}

    def frame_hash(self, future_frame_index: int) -> str:
        return hashlib.sha256(_canonical_json(self.frame_key(future_frame_index))).hexdigest()

    def frame_seed(self, future_frame_index: int) -> int:
        # Positive 63-bit seeds are supported by every intended torch Generator.
        return int(self.frame_hash(future_frame_index)[:16], 16) & ((1 << 63) - 1)


class NoiseBank:
    """Stateless regeneration with a fresh per-frame torch Generator.

    Actual tensors are generated only on the authorized compute node. Device,
    dtype, shape and torch version must match across compared methods. No global
    RNG state is read or changed. Hash tensors explicitly when recording a run.
    """

    def __init__(self, identity: NoiseIdentity, shape: tuple[int, ...], *,
                 device: str, dtype: str):
        if not shape or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape):
            raise ValueError("Noise shape must contain positive integers")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Noise dtype must be explicit float32/float16/bfloat16")
        self.identity, self.shape, self.device, self.dtype = identity, tuple(shape), device, dtype

    def frame(self, future_frame_index: int):
        import torch

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.identity.frame_seed(future_frame_index))
        return torch.randn(self.shape, generator=generator, device=self.device,
                           dtype=getattr(torch, self.dtype))

    def manifest(self, prediction_frames: int) -> dict[str, Any]:
        if isinstance(prediction_frames, bool) or not isinstance(prediction_frames, int) or prediction_frames < 1:
            raise ValueError("prediction_frames must be a positive integer")
        payload = {"identity": asdict(self.identity), "shape": self.shape,
                   "device": self.device, "dtype": self.dtype,
                   "frames": [{"future_frame_index": t, "seed": self.identity.frame_seed(t),
                               "noise_key_sha256": self.identity.frame_hash(t)}
                              for t in range(prediction_frames)]}
        return {**payload, "noise_manifest_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}


def tensor_sha256(tensor) -> str:
    """Hash dtype, shape and raw contiguous bytes, including BF16 tensors."""
    import torch

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    header = _canonical_json({"dtype": str(tensor.dtype), "shape": tuple(tensor.shape)})
    return hashlib.sha256(header + b"\n" + raw).hexdigest()
