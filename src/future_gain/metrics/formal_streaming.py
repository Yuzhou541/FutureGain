"""Bounded-memory FAR metrics with explicitly preserved native reduction axes.

No model imports occur at module import. Runtime pixel batches retain the native
global context/candidate/frame order and 256-frame boundaries. I3D retains its
32-context batches for each candidate. B1 generation is not native B200 replay.
"""

from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path

from future_gain.audit import sha256_file
from future_gain.evaluation.fvd import validate_legacy_encoding
from future_gain.evaluation.metrics import aggregate_frame_metric
from future_gain.runtime import require_slurm_compute

FRAME_BATCH = 256
I3D_BATCH = 32


def legacy_file_order_indices(contexts):
    return sorted(range(contexts), key=lambda i: f"sample_gt_{i}.mp4")


@dataclass(frozen=True)
class StreamProtocol:
    context_ids: tuple
    candidates: int
    input_mode: str
    context_frames: int = 2
    future_frames: int = 28
    height: int = 64
    width: int = 64

    def __post_init__(self):
        if (len(self.context_ids) < 2 or len(set(self.context_ids)) != len(self.context_ids)
                or any(not isinstance(x, str) or not x for x in self.context_ids)
                or type(self.candidates) is not int or self.candidates < 1):
            raise ValueError("Unique ordered contexts and positive integer candidate count required")
        if self.input_mode not in {"canonical_lossless", "upstream_legacy_mp4"}:
            raise ValueError("Explicit canonical or legacy input mode required")
        if (self.context_frames, self.future_frames, self.height, self.width) != (2, 28, 64, 64):
            raise ValueError("This entry point preserves BAIR 2→28 at64²")

    @property
    def contexts(self):
        return len(self.context_ids)

    @property
    def total_frames(self):
        return self.context_frames + self.future_frames


def candidate_ids():
    """New formal protocol proposal, fixed before any official quality: no retuning."""
    return tuple(range(41000, 41100))


def checked_pair(identity, prediction, target, protocol, index):
    import torch
    expected = (protocol.candidates, protocol.total_frames, 3, protocol.height, protocol.width)
    if identity != protocol.context_ids[index]:
        raise ValueError("Context order/membership differs from frozen evaluator order")
    for value in (prediction, target):
        if (tuple(value.shape) != expected or value.device.type != "cpu"
                or not torch.is_floating_point(value) or not torch.isfinite(value).all()
                or value.min() < 0 or value.max() > 1):
            raise ValueError("Expected complete finite CPU RGB[K,30,3,64,64] in[0,1]")
        if protocol.input_mode == "upstream_legacy_mp4" and value.dtype != torch.float32:
            raise ValueError("Native MP4 division returns float32; preserve it")
    return prediction.float(), target.float()


def iter_legacy_pairs(records, protocol, *, encoding_metadata):
    """One complete native paired MP4 per context; no missing-file recovery.

    records are already in native lexical filename order with explicit context
    identities and byte hashes. GT is decoded separately for every candidate:
    identical original GT does not imply identical MP4 codec outputs.
    """
    require_slurm_compute()
    if protocol.input_mode != "upstream_legacy_mp4":
        raise ValueError("MP4 is never canonical lossless input")
    validate_legacy_encoding(encoding_metadata)
    if version("pytorchvideo") != "0.1.5":
        raise ValueError("Require pinned pytorchvideo0.1.5 decoder")
    if len(records) != protocol.contexts:
        raise ValueError("Missing or extra contexts")
    paths = [Path(r["path"]) for r in records]
    if len(set(paths)) != len(paths) or paths != sorted(paths):
        raise ValueError("Native read_video_folder sorts unique MP4 paths")
    from pytorchvideo.data.encoded_video import EncodedVideo
    for index, (record, path) in enumerate(zip(records, paths)):
        if (path.suffix != ".mp4" or path.is_symlink() or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            raise ValueError("Paired MP4 identity changed")
        video = EncodedVideo.from_path(str(path), decode_audio=False)
        try:
            pixels = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
        finally:
            video.close()
        expected = (3, protocol.candidates * protocol.total_frames, protocol.height, 2 * protocol.width)
        if pixels is None or tuple(pixels.shape) != expected:
            raise ValueError("Native paired MP4 frame count/layout differs")
        # Exact trainer_far.py grouping and division; no clamp or re-encoding.
        clips = pixels.reshape(3, protocol.candidates, protocol.total_frames,
                               protocol.height, 2 * protocol.width).permute(1, 2, 0, 3, 4) / 255.0
        left, right = clips.split(protocol.width, dim=-1)
        checked_pair(record["context"], left, right, protocol, index)
        yield record["context"], left, right


class PixelStream:
    """Frame arrays are small; all pixels remain in bounded CPU/GPU batches.

    MSE and PSNR run on CPU as in native VideoMetric. SSIM and LPIPS use its
    exact256-frame global flattened batches, including the final short batch.
    """

    def __init__(self, protocol, frame_metrics):
        require_slurm_compute()
        import torch
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        self.protocol, self.metrics, self.torch = protocol, frame_metrics, torch
        self.psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=[1, 2, 3])
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(frame_metrics.device)
        self.values = {m: torch.empty(protocol.contexts * protocol.candidates * protocol.future_frames,
                                     dtype=torch.float32) for m in ("mse", "psnr", "ssim", "lpips")}
        self.pending_x, self.pending_y = [], []
        self.pending_frames = self.cursor = self.next_context = 0
        self.batch_sizes = []

    def _flush(self):
        torch = self.torch
        x, y = torch.cat(self.pending_x), torch.cat(self.pending_y)
        n = len(x)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            if self.protocol.input_mode == "upstream_legacy_mp4":
                # Native MSE/PSNR evaluate CPU float32 pixels; no GPU substitution.
                scores = {"mse": torch.mean((x - y) ** 2, dim=(1, 2, 3)), "psnr": self.psnr(x, y)}
                scores["ssim"] = self.ssim(x.to(self.metrics.device), y.to(self.metrics.device)).cpu()
                scores["lpips"] = self.metrics.lpips_frames(x, y, input_mode="upstream_legacy").cpu()
            else:
                scores = {k: v.cpu() for k, v in self.metrics.frame_metrics(x, y).items()}
        for name, values in scores.items():
            values = values.reshape(-1)
            if len(values) != n or not (torch.isfinite(values) | ((name == "psnr") & (values == math.inf))).all():
                raise ValueError("Nonfinite or malformed per-frame metric")
            self.values[name][self.cursor:self.cursor + n] = values
        self.cursor += n
        self.batch_sizes.append(n)
        self.pending_x, self.pending_y, self.pending_frames = [], [], 0

    def add(self, identity, prediction, target):
        if self.next_context >= self.protocol.contexts:
            raise ValueError("Extra context")
        x, y = checked_pair(identity, prediction, target, self.protocol, self.next_context)
        self.next_context += 1
        x = x[:, self.protocol.context_frames:].reshape(-1, 3, self.protocol.height, self.protocol.width)
        y = y[:, self.protocol.context_frames:].reshape_as(x)
        start = 0
        while start < len(x):
            count = min(FRAME_BATCH - self.pending_frames, len(x) - start)
            self.pending_x.append(x[start:start + count])
            self.pending_y.append(y[start:start + count])
            self.pending_frames += count
            start += count
            if self.pending_frames == FRAME_BATCH:
                self._flush()

    def finish(self):
        if self.next_context != self.protocol.contexts:
            raise ValueError("Cannot aggregate an incomplete context population")
        if self.pending_frames:
            self._flush()
        p = self.protocol
        expected = p.contexts * p.candidates * p.future_frames
        if self.cursor != expected:
            raise ValueError("Missing or duplicate candidate/frame work")
        arrays = {k: v.reshape(p.contexts, p.candidates, p.future_frames) for k, v in self.values.items()}
        if p.input_mode == "upstream_legacy_mp4":
            # Preserve native float32 mean/reduction order, not fsum approximations.
            reduced, selected = {}, {}
            for name, values in arrays.items():
                # Native LPIPS reductions stay on CUDA; other pixel reductions
                # operate on CPU. Preserve that numerical distinction as well.
                if name == "lpips":
                    values = values.to(self.metrics.device)
                trajectory = values.mean(-1)
                best = trajectory.min(-1) if name == "lpips" else trajectory.max(-1)
                reduced[name], selected[name] = float(best.values.mean()), best.indices.tolist()
            return {"mode": p.input_mode, "metrics": reduced, "selected_candidate": selected,
                    "frame_arrays": arrays, "frame_batch_sizes": self.batch_sizes,
                    "reduction": "native_float32_time_then_candidate_then_context; LPIPS CUDA, others CPU; MSE=max"}
        return {"mode": p.input_mode, "frame_arrays": arrays, "frame_batch_sizes": self.batch_sizes,
                "mean": {name: aggregate_frame_metric(values.tolist(), metric=name, protocol="canonical_mean")
                         for name, values in arrays.items()},
                "best_of_k": {name: aggregate_frame_metric(values.tolist(), metric=name, protocol="canonical_best_of_k")
                              for name, values in arrays.items()}}


def extract_feature_banks(protocol, fvd, load_clip, *, cost=None):
    """load_clip(context_index,candidate,side) returns a CPU full30 RGB tensor.

    Native candidate-major/context-batch32 order; at most32 clips on GPU.
    Legacy loaders must return already round-tripped candidate-specific GT.
    This API permits per-clip storage and needs no full RGB bank in RAM.
    """
    require_slurm_compute()
    if fvd.batch_size != I3D_BATCH or (fvd.protocol.contexts, fvd.protocol.trajectories) != (
            protocol.contexts, protocol.candidates):
        raise ValueError("Native I3D uses32-context batches and complete C×K identities")
    if (fvd.protocol.input_mode != protocol.input_mode
            or (fvd.protocol.context_frames, fvd.protocol.prediction_frames,
                fvd.protocol.height, fvd.protocol.width) != (2, 28, 64, 64)):
        raise ValueError("I3D and pixel task/mode identities differ")
    import numpy as np
    import torch
    banks = {side: np.empty((protocol.contexts, protocol.candidates, 400), dtype=np.float32)
             for side in ("prediction", "reference")}
    batches = []
    for k in range(protocol.candidates):
        for side in banks:
            for start in range(0, protocol.contexts, I3D_BATCH):
                stop = min(start + I3D_BATCH, protocol.contexts)
                rows = []
                for c in range(start, stop):
                    value = load_clip(c, k, side)
                    if (value.device.type != "cpu" or tuple(value.shape) != (30, 3, 64, 64)
                            or not torch.is_floating_point(value) or not torch.isfinite(value).all()
                            or value.min() < 0 or value.max() > 1):
                        raise ValueError("I3D loader must provide complete finite native-sized CPU clips")
                    rows.append(value.float())
                # _extract is the already-audited exact FAR preprocessing core.
                if cost is not None:
                    cost["attempted_i3d_batches"] += 1
                banks[side][start:stop, k] = fvd._extract(torch.stack(rows))
                if cost is not None:
                    cost["completed_i3d_batches"] += 1
                batches.append({"candidate": k, "side": side, "context_start": start, "contexts": stop - start})
    return {**banks, "batches": batches, "score": fvd.evaluate_features(banks["prediction"], banks["reference"])}
