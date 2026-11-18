"""Future-only frame metrics and explicit FAR aggregation semantics.

The legacy aggregation helper reproduces the reduction axes, not the whole
upstream MP4/FVD protocol. Callers must still preserve its input provenance.
"""

from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from future_gain.audit import sha256_file
from future_gain.runtime import require_slurm_compute

METRIC_VERSIONS = {"lpips": "0.1.4", "torchvision": "0.20.0", "torchmetrics": "1.6.1"}
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def aggregate_frame_metric(values: Sequence, *, metric: str,
                           protocol: str = "canonical_mean") -> dict:
    """Reduce [context][trajectory][future frame], never framewise best-of-K.

    canonical_best_of_k is an explicitly hindsight GT-selection diagnostic.
    upstream_legacy MSE intentionally uses max (worst-of-K), as pinned FAR does.
    """
    if metric not in {"mse", "lpips", "psnr", "ssim"}:
        raise ValueError("Unknown frame metric")
    if protocol not in {"canonical_mean", "canonical_best_of_k", "upstream_legacy"}:
        raise ValueError("Unknown metric reduction protocol")
    if not values or not values[0] or not values[0][0]:
        raise ValueError("Metric arrays must contain contexts, trajectories and future frames")
    k, t = len(values[0]), len(values[0][0])
    trajectories = []
    for context in values:
        if len(context) != k:
            raise ValueError("Candidate count differs between contexts")
        row = []
        for trajectory in context:
            if len(trajectory) != t:
                raise ValueError("Future frame count differs between candidates")
            numbers = tuple(map(float, trajectory))
            if any(not math.isfinite(x) and not (metric == "psnr" and x == math.inf) for x in numbers):
                raise ValueError("Nonfinite metric (except exact-match positive infinite PSNR)")
            row.append(_mean(numbers))
        trajectories.append(tuple(row))
    reduction = "mean"
    if protocol == "canonical_best_of_k":
        reduction = "min" if metric in {"mse", "lpips"} else "max"
    elif protocol == "upstream_legacy":
        reduction = "min" if metric == "lpips" else "max"
    selected = None
    if reduction == "mean":
        contexts = tuple(_mean(row) for row in trajectories)
    else:
        choose = min if reduction == "min" else max
        selected = tuple(choose(range(k), key=lambda index: row[index]) for row in trajectories)
        contexts = tuple(row[index] for row, index in zip(trajectories, selected))
    return {"metric": metric, "protocol": protocol, "candidate_reduction": reduction,
            "per_trajectory": tuple(trajectories), "per_context": contexts,
            "selected_trajectory": selected, "value": _mean(contexts),
            "contexts": len(values), "trajectories_per_context": k, "future_frames": t}


def fvd_trajectory_mean(predicted_features: Sequence, reference_features: Sequence,
                        distance_fn: Callable) -> dict:
    """Call a real FVD distance once per K across contexts; no pooled shortcut.

    Inputs are feature vectors [C][K][D]. This helper neither extracts I3D
    features nor claims that tiny fixtures constitute a usable FVD estimate.
    """
    c = len(predicted_features)
    if c < 2 or len(reference_features) != c:
        raise ValueError("FVD requires matching contexts, at least two for covariance")
    k = len(predicted_features[0])
    if k < 1:
        raise ValueError("No FVD trajectories")
    d = len(predicted_features[0][0])
    if d < 1:
        raise ValueError("Empty FVD features")
    for bank in (predicted_features, reference_features):
        for context in bank:
            if len(context) != k or any(len(feature) != d for feature in context):
                raise ValueError("FVD feature axes differ")
            if any(not math.isfinite(float(x)) for feature in context for x in feature):
                raise ValueError("Nonfinite FVD features")
    scores = tuple(float(distance_fn([predicted_features[i][j] for i in range(c)],
                                     [reference_features[i][j] for i in range(c)]))
                   for j in range(k))
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Nonfinite FVD result")
    return {"aggregation": "trajectory_index_then_mean", "per_trajectory": scores,
            "value": _mean(scores), "contexts": c, "trajectories_per_context": k}


class FrameMetrics:
    """Frozen LPIPS-Alex with supplied assets, plus exact frame MSE/PSNR/SSIM.

    Inputs are ONLY future frames [T,3,H,W] in [0,1]. The caller must separate
    the observed prefix before this interface. No resize or clamping is done.
    """

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, *, device: str = "cuda:0"):
        require_slurm_compute()
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("schema_version") != 1 or manifest.get("status") != "verified":
            raise ValueError("Metric asset manifest is not complete")
        for package, expected in METRIC_VERSIONS.items():
            installed = version(package).split("+")[0]
            if installed != expected or manifest["versions"].get(package) != expected:
                raise ValueError(f"Metric implementation requires {package}=={expected}")
        for name in ("alexnet", "lpips_alex"):
            asset = manifest["assets"][name]
            path = Path(asset["path"])
            if path.stat().st_size != asset["bytes"] or sha256_file(path) != asset["sha256"]:
                raise ValueError(f"Metric asset identity mismatch: {name}")
        import torch
        import lpips
        if Path(torch.hub.get_dir()).resolve() != Path(manifest["torch_home"]).resolve() / "hub":
            raise ValueError("Set TORCH_HOME to the verified manifest's project cache before loading LPIPS")
        expected_alexnet = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
        if expected_alexnet.resolve() != Path(manifest["assets"]["alexnet"]["path"]).resolve():
            raise ValueError("LPIPS would use a different AlexNet cache")
        self = cls()
        self.torch, self.device, self.manifest = torch, torch.device(device), manifest
        self.lpips = lpips.LPIPS(net="alex", version="0.1", spatial=False,
                                pretrained=True, pnet_rand=False, pnet_tune=False,
                                model_path=manifest["assets"]["lpips_alex"]["path"],
                                eval_mode=True, verbose=False).to(self.device).eval()
        self.lpips.requires_grad_(False)
        return self

    def _pair(self, prediction, target):
        torch = self.torch
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("Pass matching future-only RGB tensors [T,3,H,W]")
        if prediction.shape[0] < 1:
            raise ValueError("An empty future window is not a metric value")
        for tensor in (prediction, target):
            if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
                raise ValueError("Metric RGB tensors must be finite floats")
            if tensor.min() < 0 or tensor.max() > 1:
                raise ValueError("Metric RGB inputs must lie in [0,1]; no implicit clamping")
        return prediction.detach().to(self.device, dtype=torch.float32), target.detach().to(self.device, dtype=torch.float32)

    def lpips_frames(self, prediction, target, *, input_mode: str = "canonical"):
        """One normalization only; legacy deliberately preserves FAR's range."""
        if input_mode not in {"canonical", "upstream_legacy"}:
            raise ValueError("Unknown LPIPS input convention")
        x, y = self._pair(prediction, target)
        with self.torch.no_grad():
            return self.lpips(x, y, normalize=input_mode == "canonical").reshape(-1)

    def frame_metrics(self, prediction, target, *, input_mode: str = "canonical") -> dict:
        """Return unaggregated frame tensors; PSNR is computed per frame first."""
        x, y = self._pair(prediction, target)
        from torchmetrics.functional.image import structural_similarity_index_measure
        with self.torch.no_grad():
            mse = (x - y).square().mean(dim=(1, 2, 3))
            psnr = -10 * self.torch.log10(mse)  # exact matches correctly yield +inf
            ssim = structural_similarity_index_measure(
                x, y, data_range=1.0, gaussian_kernel=True, sigma=1.5,
                kernel_size=11, reduction="none", k1=0.01, k2=0.03)
            lpips_values = self.lpips_frames(x, y, input_mode=input_mode)
        return {"mse": mse, "psnr": psnr, "ssim": ssim, "lpips": lpips_values}

    def verify_lpips_normalization(self, prediction, target) -> dict:
        """Real compute-node check of two legal equivalent normalization routes."""
        x, y = self._pair(prediction, target)
        with self.torch.no_grad():
            automatic = self.lpips(x, y, normalize=True)
            explicit = self.lpips(2 * x - 1, 2 * y - 1, normalize=False)
        if not self.torch.equal(automatic, explicit):
            raise AssertionError("LPIPS normalize=True differs from exactly one explicit conversion")
        return {"normalization_equivalence": "exact", "frames": len(x),
                "max_abs_difference": float((automatic - explicit).abs().max())}
