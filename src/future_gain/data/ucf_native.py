"""Explicit native/metric RGB boundary for the frozen four-clip UCF gate.

The upper allowance is the largest excess measured in diagnostic224726: two
float32 ULP at one. It is an empirical task contract, not a general error bound.
Native values are never modified before generation; metric projection is separate.
"""
import hashlib
import json

from future_gain.sampling.noise import tensor_sha256

NATIVE_RGB_EXCESS = 2.0 ** -22
NATIVE_RGB_MAX = 1.0 + NATIVE_RGB_EXCESS


def validate_native_rgb(value, torch):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("UCF native RGB must be CPU float32")
    if not torch.isfinite(value).all() or value.numel() == 0 or value.min() < 0 or value.max() > NATIVE_RGB_MAX:
        raise ValueError("UCF native RGB exceeds the frozen [0,1+2^-22] input domain")


def diagnostic_tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                                     sort_keys=True, separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_metric_target(native, torch):
    validate_native_rgb(native, torch)
    if tuple(native.shape) != (1, 16, 3, 64, 64):
        raise ValueError("Expected complete native UCF4-to12 RGB")
    before = tensor_sha256(native)
    canonical = native.clone().clamp(0, 1)
    correction = (native - canonical).abs()
    groups = {}
    for name, values in (("all", correction), ("observed_prefix", correction[:, :4]),
                         ("future", correction[:, 4:])):
        groups[name] = {"changed_channel_values": int((values != 0).sum()),
                        "maximum_correction": float(values.max())}
    if tensor_sha256(native) != before:
        raise AssertionError("Metric target preparation changed native generation input")
    return canonical, {"native_sha256": before, "canonical_sha256": tensor_sha256(canonical),
                       "native_domain": [0, NATIVE_RGB_MAX], "groups": groups,
                       "native_unchanged": True, "projection_scope": "metric_target_only"}
