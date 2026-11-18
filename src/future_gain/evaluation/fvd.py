"""FAR's I3D/FVD arithmetic with explicit lossless and paired-MP4 inputs.

Audited source: showlab/FAR@e59536ac30e5e3fa93db756fccfd422420a65ff6,
far/metrics/{metric,fvd}.py and far/trainers/trainer_far.py. FAR has no
calc_fvd.py at this revision. Imports and metadata validation are lightweight;
all decoding, model loading, feature extraction and matrix algebra need Slurm.
"""

import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

from future_gain.audit import PINNED_COMMIT, canonical_hash, sha256_file
from future_gain.evaluation.metrics import I3D_URL
from future_gain.runtime import require_slurm_compute

FEATURE_DIM = 400  # Kinetics-400 I3D logits, before softmax, not 1024 pooled units.
FVD_VERSIONS = {"torch": "2.5.0", "numpy": "1.26.4", "scipy": "1.14.1"}
DETECTOR_KWARGS = {"rescale": False, "resize": True, "return_features": True}


@dataclass(frozen=True)
class FVDProtocol:
    """Caller declares complete clips and prefix provenance; no prefix is added.

    contexts/trajectories are the intended complete evaluation population. Small
    extraction batches may contain fewer clips; final aggregation must match.
    """

    input_mode: str
    contexts: int
    trajectories: int
    context_frames: int
    prediction_frames: int
    height: int
    width: int
    prediction_prefix: str
    reference_prefix: str

    def __post_init__(self):
        for name in ("contexts", "trajectories", "prediction_frames", "height", "width"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.context_frames) is not int or self.context_frames < 0:
            raise ValueError("context_frames must be a nonnegative integer")
        if self.contexts < 2:
            raise ValueError("At least two contexts are needed for sample covariance")
        if self.total_frames < 10:
            raise ValueError("Pinned FAR evaluates I3D/FVD only for clips of at least ten frames")
        if self.input_mode not in {"canonical_lossless", "upstream_legacy_mp4"}:
            raise ValueError("Explicit canonical_lossless or upstream_legacy_mp4 mode required")
        allowed = {"dcae_reconstructed", "original_observed"} if self.context_frames else {"none"}
        if self.prediction_prefix not in allowed or self.reference_prefix not in allowed:
            raise ValueError("Prefix provenance conflicts with context_frames")
        if self.input_mode == "upstream_legacy_mp4" and self.context_frames:
            if (self.prediction_prefix, self.reference_prefix) != ("dcae_reconstructed", "original_observed"):
                raise ValueError("FAR legacy requires reconstructed prediction prefix and original GT prefix")

    @property
    def total_frames(self):
        return self.context_frames + self.prediction_frames

    def validate_clip_shape(self, shape):
        if len(shape) != 5 or shape[0] < 1 or tuple(shape[1:]) != (
                self.total_frames, 3, self.height, self.width):
            raise ValueError("Pass full clips [batch,context+future,3,height,width] at the frozen resolution")

    def validate_feature_shape(self, shape):
        if tuple(shape) != (self.contexts, self.trajectories, FEATURE_DIM):
            raise ValueError("Features must be [all contexts,all trajectories,400], without pooling or missing samples")


def validate_i3d_manifest(manifest):
    """Check metadata only. Actual file size/hash is checked inside Slurm."""
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "verified"
            or manifest.get("i3d_load_verified") is not True):
        raise ValueError("A verified I3D acquisition manifest is required")
    asset = manifest.get("assets", {}).get("i3d", {})
    if (asset.get("source_url") != I3D_URL or not Path(asset.get("path", "")).is_absolute()
            or type(asset.get("bytes")) is not int or asset["bytes"] <= 0
            or not isinstance(asset.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise ValueError("I3D asset metadata is missing its exact upstream URL/path/size/hash")
    return dict(asset)


def validate_legacy_encoding(metadata):
    """Require actual writer provenance rather than inventing MP4 defaults."""
    if metadata.get("writer") != "imageio.get_writer" or metadata.get("anno_context") is not False:
        raise ValueError("Legacy BAIR/UCF inputs require the native writer and unannotated clips")
    if type(metadata.get("fps")) not in {int, float} or metadata["fps"] != 8:
        raise ValueError("Pinned FAR paired videos are written at eight fps")
    for key in ("imageio_version", "imageio_ffmpeg_version", "ffmpeg_version", "codec", "pixel_format"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Record the actual legacy encoding {key}")
    return dict(metadata)


class FVD:
    """One verified I3D instance; outputs features so GT/reference can be cached.

    The caller must key feature caches by identity plus ordered input hashes and
    side; legacy caches also include returned decoder and encoding metadata.
    This module neither writes caches nor infers the provenance of pixels.
    """

    @classmethod
    def from_manifest(cls, manifest_path, *, protocol: FVDProtocol, device: str, batch_size: int):
        allocation = require_slurm_compute()
        if not isinstance(protocol, FVDProtocol) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Pass an FVDProtocol and a positive explicit extraction batch size")
        manifest_path = Path(manifest_path)
        asset = validate_i3d_manifest(json.loads(manifest_path.read_text()))
        weight_path = Path(asset["path"])
        if weight_path.stat().st_size != asset["bytes"] or sha256_file(weight_path) != asset["sha256"]:
            raise ValueError("I3D weight identity differs from verified acquisition")
        versions = {name: version(name).split("+")[0] for name in FVD_VERSIONS}
        if versions != FVD_VERSIONS:
            raise ValueError(f"FVD numeric environment differs from pinned versions: {versions}")
        import numpy as np
        import scipy.linalg
        import torch

        self = cls()
        self.np, self.linalg, self.torch = np, scipy.linalg, torch
        self.protocol, self.device, self.batch_size = protocol, torch.device(device), batch_size
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("I3D extraction requires a compute-node CPU or CUDA device")
        self.detector = torch.jit.load(str(weight_path), map_location=self.device).eval()
        # ScriptModule does not expose nn.Module.requires_grad_ in Torch 2.5.
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self.allocation = allocation
        self.identity = {
            "schema_version": 1, "upstream_commit": PINNED_COMMIT,
            "implementation_sha256": sha256_file(Path(__file__)),
            "metric_manifest_sha256": sha256_file(manifest_path), "i3d_sha256": asset["sha256"],
            "versions": versions, "protocol": asdict(protocol), "feature_batch_size": batch_size,
            "feature_dimension": FEATURE_DIM, "detector_kwargs": dict(DETECTOR_KWARGS),
            "preprocessing": "float32 RGB [0,1] -> 2*x-1 -> B,C,T,H,W; I3D internal bilinear224 align_corners=False",
            "device": str(self.device), "autocast": False,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "covariance": "numpy.cov rowvar=False, default sample normalization",
            "matrix_sqrt": "scipy.linalg.sqrtm(cov_fake @ cov_real,disp=False); upstream real projection",
        }
        self.identity["identity_sha256"] = canonical_hash(self.identity)
        return self

    def with_protocol(self, protocol: FVDProtocol):
        """Create an explicit protocol view using the same frozen I3D instance.

        No checkpoint reload or in-place protocol mutation occurs. Numeric
        environment, extraction batch size and device are preserved. Input-mode
        checks remain attached to each view, and identities remain independent.
        """
        require_slurm_compute()
        if not isinstance(protocol, FVDProtocol):
            raise ValueError("Pass an explicit FVDProtocol for the new view")
        view = type(self)()
        for name in ("np", "linalg", "torch", "device", "batch_size", "detector"):
            setattr(view, name, getattr(self, name))
        view.protocol = protocol
        view.allocation = copy.deepcopy(self.allocation)
        view.identity = copy.deepcopy(self.identity)
        view.identity.pop("identity_sha256")
        view.identity["protocol"] = asdict(protocol)
        view.identity["identity_sha256"] = canonical_hash(view.identity)
        return view

    def _extract(self, clips):
        torch, np = self.torch, self.np
        self.protocol.validate_clip_shape(clips.shape)
        if not torch.is_floating_point(clips) or not torch.isfinite(clips).all():
            raise ValueError("I3D full clips must be finite floating-point RGB")
        if clips.min() < 0 or clips.max() > 1:
            raise ValueError("I3D input must be [0,1]; no clipping, quantization, or implicit normalization")
        features = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(clips), self.batch_size):
                # FAR metric.py converts to float32 and scales once; its fvd.py
                # permutes to B,C,T,H,W and lets the detector resize internally.
                batch = clips[start:start + self.batch_size].detach().to(self.device, dtype=torch.float32)
                batch = (2.0 * batch - 1).permute(0, 2, 1, 3, 4).contiguous()
                result = self.detector(batch, **DETECTOR_KWARGS)
                if tuple(result.shape) != (len(batch), FEATURE_DIM) or not torch.isfinite(result).all():
                    raise ValueError("I3D returned unexpected or nonfinite features")
                features.append(result.cpu().numpy())
        return np.concatenate(features, axis=0)

    def extract_canonical(self, clips, *, side: str):
        """Extract [B,400] from lossless complete RGB clips [B,T,3,H,W]."""
        require_slurm_compute()
        if self.protocol.input_mode != "canonical_lossless" or side not in {"prediction", "reference"}:
            raise ValueError("Canonical extraction requires lossless mode and an explicit input side")
        return self._extract(clips)

    def extract_legacy_paired_mp4(self, paths, *, encoding_metadata):
        """Decode FAR's paired MP4 layout, preserving its codec and prefix effects.

        One file per context, K complete videos concatenated in time; prediction
        is the left half and GT the right half. Unlike upstream's broad except,
        any failed/missing clip stops evaluation instead of shrinking the sample.
        This consumes existing native files, and does not encode new MP4s.
        """
        require_slurm_compute()
        if self.protocol.input_mode != "upstream_legacy_mp4":
            raise ValueError("MP4 decoding cannot be labeled canonical lossless")
        encoding = validate_legacy_encoding(encoding_metadata)
        paths = [Path(path) for path in paths]
        if (len(paths) != self.protocol.contexts or len(set(paths)) != len(paths)
                or paths != sorted(paths) or any(p.suffix.lower() != ".mp4" for p in paths)):
            raise ValueError("Supply exactly the frozen contexts' unique MP4 paths in upstream sorted order")
        if version("pytorchvideo") != "0.1.5":
            raise ValueError("Legacy decode requires pinned pytorchvideo==0.1.5")
        import av
        from pytorchvideo.data.encoded_video import EncodedVideo

        predicted, reference, inputs = [], [], []
        p, torch = self.protocol, self.torch
        # Batch across contexts for a fixed trajectory, as upstream does. Only
        # one context batch's decoded clips is held, never the entire dataset.
        for start in range(0, len(paths), self.batch_size):
            decoded = []
            for path in paths[start:start + self.batch_size]:
                identity = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                video = EncodedVideo.from_path(str(path), decode_audio=False)
                try:
                    raw = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
                finally:
                    video.close()
                expected = (3, p.trajectories * p.total_frames, p.height, 2 * p.width)
                if raw is None or tuple(raw.shape) != expected:
                    raise ValueError(f"Legacy paired MP4 has unexpected frames/layout: {path}")
                if not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
                    raise ValueError(f"Legacy decoder returned invalid RGB: {path}")
                # Same b,c,(n*f),h,w -> b,n,f,c,h,w grouping as trainer_far.py.
                clips = raw.reshape(3, p.trajectories, p.total_frames, p.height, 2 * p.width)
                decoded.append(clips.permute(1, 2, 0, 3, 4).contiguous() / 255.0)
                inputs.append(identity)
            fake_batch, real_batch = [], []
            for k in range(p.trajectories):
                fake_batch.append(self._extract(torch.stack([x[k, ..., :p.width] for x in decoded])))
                real_batch.append(self._extract(torch.stack([x[k, ..., p.width:] for x in decoded])))
            predicted.append(self.np.stack(fake_batch, axis=1))
            reference.append(self.np.stack(real_batch, axis=1))
        return {"prediction": self.np.concatenate(predicted), "reference": self.np.concatenate(reference),
                "inputs": inputs, "encoding": encoding,
                "decoder": {"pytorchvideo": version("pytorchvideo"), "av": version("av"),
                            "ffmpeg_libraries": av.library_versions},
                "identity": self.identity}

    def _distance(self, fake, real):
        # Preserve FAR's arithmetic, including float32 means and real projection.
        # No epsilon covariance repair, NaN replacement, or negative-value clamp.
        np = self.np
        mu_fake, mu_real = fake.mean(axis=0), real.mean(axis=0)
        covariance_fake = np.cov(fake, rowvar=False)
        covariance_real = np.cov(real, rowvar=False)
        root, _ = self.linalg.sqrtm(np.dot(covariance_fake, covariance_real), disp=False)
        if not np.isfinite(root).all():
            raise ValueError("FVD covariance square root is nonfinite; no substituted score")
        result = np.real(np.square(mu_fake - mu_real).sum()
                         + np.trace(covariance_fake + covariance_real - 2 * root))
        if not np.isfinite(result):
            raise ValueError("FVD result is nonfinite")
        return float(result)

    def evaluate_features(self, prediction, reference):
        """Cross-context FVD for each K, then average K; never pool C*K clips.

        Both arrays are [contexts,trajectories,400]. A shared GT feature array
        may be explicitly broadcast to that shape by the caller without copying.
        """
        require_slurm_compute()
        np, torch = self.np, self.torch
        prediction, reference = np.asarray(prediction), np.asarray(reference)
        for bank in (prediction, reference):
            self.protocol.validate_feature_shape(bank.shape)
            if bank.dtype != np.float32 or not np.isfinite(bank).all():
                raise ValueError("FVD requires finite float32 I3D features from the frozen extractor")
        values = [self._distance(prediction[:, k], reference[:, k]) for k in range(self.protocol.trajectories)]
        if self.protocol.input_mode == "upstream_legacy_mp4":
            # metric.py stores the individual scipy scores in torch.zeros(K)
            # on its metric device before averaging, hence float32 rounding.
            native_values = torch.tensor(values, device=self.device, dtype=torch.float32)
            aggregate = float(native_values.mean())
            per_trajectory = native_values.cpu().tolist()
            numeric_reduction = "torch_float32_scores_then_mean"
        else:
            aggregate = math.fsum(values) / len(values)
            per_trajectory, numeric_reduction = values, "python_fsum_of_float64_scores"
        if not math.isfinite(aggregate):
            raise ValueError("FVD aggregation is nonfinite")
        return {"metric": "fvd_trajectory_mean", "aggregation": "trajectory_index_then_mean",
                "input_mode": self.protocol.input_mode, "value": aggregate,
                "per_trajectory": per_trajectory, "raw_scipy_per_trajectory": values,
                "numeric_reduction": numeric_reduction, "identity": self.identity,
                "contexts": self.protocol.contexts, "trajectories": self.protocol.trajectories,
                "total_frames": self.protocol.total_frames}
