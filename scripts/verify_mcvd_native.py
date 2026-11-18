"""Slurm-only native MCVD DDIM-100 first-block repeat gate; no adapter/benchmark."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from future_gain.audit import canonical_hash, sha256_file
from future_gain.runtime import require_slurm_compute
from scripts.prepare_mcvd_assets import (
    PROJECT, audit_source, owned_directory, validate_config, verify_assets,
)


def namespace(value):
    # Same recursive mapping as upstream main.dict2namespace, without importing
    # its training runner and unrelated dataset/metric dependencies.
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()})


def tensor_hash(tensor):
    cpu = tensor.detach().cpu().contiguous()
    descriptor = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                  "bytes_sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest()}
    return canonical_hash(descriptor)


def state_identity(model):
    hashes = {name: tensor_hash(tensor) for name, tensor in model.state_dict().items()}
    return {"tensor_count": len(hashes), "sha256": canonical_hash(hashes)}


def select_context(development):
    if (development["dataset"]["repo_id"] != "guyuchao/BAIR" or
            development["dataset"]["revision"] != "776ad3b7d6591b06a01b8f91b1d89496445db77c"):
        raise ValueError("Unexpected shared development dataset")
    item = development["dev_step_response"][0]
    if (item != development["splits"]["calibration_train"][0] or
            item["source_split"] != "official_training" or
            not item["original_video_id"].startswith("train/") or
            item["requested_frame_indices"] != list(range(30)) or
            "/train/video_aux1/" not in item["source_video_path"]):
        raise ValueError("Native smoke uses only the first frozen TRAIN development video")
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream-root", "asset-dir", "asset-manifest", "development-manifest",
                 "extension-cache", "cuda-home", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    allocation = require_slurm_compute()  # Before model/data/dependency imports.
    assets = owned_directory(args.asset_dir)
    source = audit_source(owned_directory(args.upstream_root))
    owned_directory(args.output.parent)
    if args.output.exists():
        raise FileExistsError("Preserve the original smoke run; use a new output directory")
    env = Path(sys.prefix).resolve()
    if env.parent != PROJECT / "envs" or not env.name.startswith("mcvd_"):
        raise ValueError("Native MCVD must run in its separate mcvd_* environment")
    owned_directory(args.extension_cache.parent)
    args.extension_cache.mkdir(exist_ok=True)
    extension_cache = owned_directory(args.extension_cache)
    if not extension_cache.is_relative_to(assets):
        raise ValueError("CUDA extension cache must be within owned FutureGain assets")
    toolkit = args.cuda_home.resolve(strict=True)
    nvcc_report = subprocess.run([str(toolkit / "bin/nvcc"), "--version"], check=True,
                                 capture_output=True, text=True).stdout
    compiler_report = subprocess.run([os.environ.get("CXX", "c++"), "--version"], check=True,
                                     capture_output=True, text=True).stdout
    if not re.search(r"release 12\.4[, ]", nvcc_report):
        raise ValueError("This candidate environment requires CUDA 12.4 nvcc")
    os.environ["CUDA_HOME"] = str(toolkit)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(extension_cache)
    os.environ["MAX_JOBS"] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True  # Keep the pinned source clean after imports.
    manifest = json.loads(args.asset_manifest.read_text())
    verified = verify_assets(manifest, assets)
    if source["tree_sha256"] != manifest["source"]["tree_sha256"]:
        raise ValueError("Source differs from the pinned acquisition manifest")
    item = select_context(json.loads(args.development_manifest.read_text()))
    video = Path(item["path"])
    if not video.resolve().is_relative_to(assets) or sha256_file(video) != item["sha256"]:
        raise ValueError("Frozen development video identity mismatch")
    args.output.mkdir()
    result = {"schema_version": 1, "status": "running", "allocation": allocation,
              "scope": "native DDIM first-block repeat only; no adapter or benchmark",
              "source": source, "assets": verified,
              "asset_manifest_sha256": sha256_file(args.asset_manifest),
              "development_manifest_sha256": sha256_file(args.development_manifest),
              "script_sha256": sha256_file(Path(__file__)),
              "asset_script_sha256": sha256_file(Path(__file__).with_name("prepare_mcvd_assets.py")),
              "runtime_guard_sha256": sha256_file(Path(__file__).parents[1] / "src/future_gain/runtime.py"),
              "environment": {"prefix": str(env), "python": sys.version, "nvcc": nvcc_report,
                              "cxx": compiler_report, "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
                              "cuda_home": str(toolkit), "extension_cache": str(extension_cache)},
              "context": item, "context_frames": 2, "future_frames": 5,
              "requested_ddim_steps": 100, "planned_calls": 202,
              "actual_calls": 0, "runs": [], "comparison_eligible": False,
              "preprocessing": {"source": "existing FAR BAIR TRAIN MP4 auxiliary camera 1",
                                "decoder": "PyAV RGB24, first two frames only",
                                "range": "uint8 / 255, then native 2*x-1 once",
                                "test_augmentation": "none", "resize": "none",
                                "native_hdf5_benchmark_equivalence": "not established"},
              "sampler": {"function": "models.ddim_sampler", "training_config_version": "DDPM",
                          "explicit_inference_selection": "official DDIM",
                          "final_only": True, "denoise": True, "subsample_steps": 100,
                          "clip_before": True, "t_min": -1, "gamma": False,
                          "verbose": False, "log": False,
                          "selected_labels": list(range(0, 1000, 10)), "terminal_label": 99}}
    try:
        import av
        import numpy as np
        import torch
        import yaml

        for package, expected in {"torch": "2.5.0", "numpy": "1.26.4", "scipy": "1.14.1",
                                  "PyYAML": "6.0.2", "av": "13.1.0"}.items():
            actual = version(package)
            if actual.split("+")[0] != expected:
                raise ValueError(f"Unreviewed {package} version: {actual}")
            result["environment"][package] = actual
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != "12.4":
            raise RuntimeError("Exactly one allocated CUDA GPU and torch CUDA 12.4 are required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        result["environment"].update({"precision": "float32", "autocast": False,
                                     "deterministic_algorithms": True, "tf32": False,
                                     "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda})
        config_dict = yaml.safe_load(Path(verified["config.yml"]["path"]).read_text())
        validate_config(config_dict)
        config = namespace(config_dict)
        config.device = torch.device("cpu")
        if str(args.upstream_root.resolve()) not in sys.path:
            sys.path.insert(0, str(args.upstream_root.resolve()))
        from models import ddim_sampler
        from models.better.ncsnpp_more import UNetMore_DDPM
        from models.ema import EMAHelper
        for module_name in ("models", "models.better.ncsnpp_more", "models.ema"):
            if not Path(sys.modules[module_name].__file__).resolve().is_relative_to(args.upstream_root.resolve()):
                raise RuntimeError("Native MCVD import resolved to another project")
        # CPU checkpoint loading avoids materializing optimizer states on GPU.
        states = torch.load(verified["checkpoint_450000.pt"]["path"], map_location="cpu", weights_only=False)
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("Unexpected official checkpoint structure")
        result["checkpoint_structure"] = {"length": len(states),
                                          "element_types": [type(x).__name__ for x in states]}
        base = UNetMore_DDPM(config)
        # DataParallel's one-GPU constructor moves the module to CUDA. Move it
        # back before loading/EMA; optimizer/checkpoint payloads stay on CPU.
        # Strict state loading below replaces every schedule buffer as well.
        model = torch.nn.DataParallel(base, device_ids=[0]).cpu()
        expected_keys = set(model.state_dict())
        result["checkpoint_key_check"] = {"missing": sorted(expected_keys - set(states[0])),
                                          "unexpected": sorted(set(states[0]) - expected_keys)}
        if result["checkpoint_key_check"]["missing"] or result["checkpoint_key_check"]["unexpected"]:
            raise ValueError("Checkpoint keys differ; no migration or ignored keys is allowed")
        model.load_state_dict(states[0], strict=True)
        names = dict(base.named_parameters())
        if not all(parameter.requires_grad for parameter in names.values()) or set(states[-1]) != set(names):
            raise ValueError("EMA must cover every still-trainable native parameter exactly")
        ema = EMAHelper(mu=config.model.ema_rate)
        ema.register(model)
        ema.load_state_dict(states[-1])
        before_ema = {name: tensor_hash(parameter) for name, parameter in names.items()}
        ema.ema(model)  # MUST precede requires_grad_(False).
        if any(not torch.equal(parameter.detach(), states[-1][name]) for name, parameter in names.items()):
            raise ValueError("Native EMA application failed exact equality")
        after_ema = {name: tensor_hash(parameter) for name, parameter in names.items()}
        result["ema"] = {"applied_before_freeze": True, "all_parameters_equal_shadow": True,
                         "parameter_count": sum(p.numel() for p in names.values()),
                         "parameter_tensors": len(names),
                         "changed_tensors": sum(before_ema[k] != after_ema[k] for k in names),
                         "pre_ema_sha256": canonical_hash(before_ema),
                         "shadow_sha256": canonical_hash(after_ema)}
        model.requires_grad_(False).eval()
        result["state_before_sampling"] = state_identity(model)
        del states, ema, names, before_ema, after_ema
        config.device = torch.device("cuda:0")
        model.to(config.device)
        if len(base.alphas) != 1000 or any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError("Incorrect frozen model or diffusion schedule")
        # Materialize only observed RGB frames; future RGB never enters the
        # sampler API. Codec-internal buffering is not an observation tensor.
        frames = []
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == 2:
                    break
        prefix_bytes = np.stack(frames)
        if prefix_bytes.shape != (2, 64, 64, 3) or prefix_bytes.dtype != np.uint8:
            raise ValueError("Expected exactly two native-resolution RGB prefix frames")
        prefix = torch.from_numpy(prefix_bytes.copy()).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
        condition = (2 * prefix - 1.).reshape(1, 6, 64, 64).to(config.device)
        noise_identity = {"model": "mcvd", "original_video_id": item["original_video_id"],
                          "seed": 101, "trajectory": 0, "block": 0, "shape": [1, 15, 64, 64]}
        noise_seed = int(canonical_hash(noise_identity)[:16], 16) & ((1 << 63) - 1)
        noise_cpu = torch.randn((1, 15, 64, 64), generator=torch.Generator(device="cpu").manual_seed(noise_seed))
        noise = noise_cpu.to(config.device)
        result["noise"] = {**noise_identity, "derived_seed": noise_seed, "generator_device": "cpu",
                           "dtype": "float32", "sha256": tensor_hash(noise)}
        result["prefix_sha256"] = tensor_hash(prefix)
        result["condition_sha256"] = tensor_hash(condition)
        payload = {"prefix": prefix, "condition": condition.cpu(), "noise": noise_cpu}
        input_path = args.output / "native_inputs.pt"
        torch.save(payload, input_path)
        result["inputs"] = {"path": str(input_path), "sha256": sha256_file(input_path)}
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(2):
            labels, first = [], {}
            def before_forward(module, inputs, kwargs):
                result["actual_calls"] += 1
                labels.append(inputs[1].detach().cpu().tolist())
                if len(labels) == 1:
                    first.update({"input_sha256": tensor_hash(inputs[0]),
                                  "condition_sha256": tensor_hash(kwargs["cond"])})
            def after_forward(module, inputs, output):
                if len(labels) == 1:
                    first["epsilon_sha256"] = tensor_hash(output)
            pre = base.register_forward_pre_hook(before_forward, with_kwargs=True)
            post = base.register_forward_hook(after_forward)
            cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                generated = ddim_sampler(noise.clone(), model, cond=condition, **{
                    key: result["sampler"][key] for key in ("final_only", "denoise", "subsample_steps",
                        "clip_before", "t_min", "gamma", "verbose", "log")})[-1]
                torch.cuda.synchronize()
            finally:
                pre.remove()
                post.remove()
            elapsed = time.perf_counter() - start
            # Preserve even a numerically failing result before gate assertions.
            run_path = args.output / f"native_repeat_{repeat}.pt"
            torch.save({"native": generated.detach().cpu()}, run_path)
            result.setdefault("repeat_artifacts", []).append({"path": str(run_path), "sha256": sha256_file(run_path)})
            expected_labels = [[value] for value in list(range(0, 1000, 10)) + [99]]
            if labels != expected_labels or generated.shape != (1, 15, 64, 64) or not torch.isfinite(generated).all():
                raise ValueError("Native sampler call sequence, dimensions or finite-output check failed")
            if first["input_sha256"] != result["noise"]["sha256"] or first["condition_sha256"] != result["condition_sha256"]:
                raise ValueError("Native first evaluation did not receive the declared input")
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                raise ValueError("Deterministic native DDIM unexpectedly consumed global RNG")
            if tensor_hash(condition) != result["condition_sha256"] or tensor_hash(noise) != result["noise"]["sha256"]:
                raise ValueError("Native sampler modified the source noise or conditioning")
            native = generated.detach().cpu()
            rgb = ((native + 1.) / 2.).clamp(0, 1).reshape(1, 5, 3, 64, 64)
            payload[f"native_{repeat}"] = native
            payload[f"rgb_{repeat}"] = rgb
            result["runs"].append({"repeat": repeat, "actual_calls": len(labels), "labels": labels,
                                   "first_evaluation": first, "native_sha256": tensor_hash(native),
                                   "rgb_sha256": tensor_hash(rgb),
                                   "seconds": elapsed,
                                   "timing_scope": "diagnostic sampler with audit hooks; first run includes CUDA extension compilation",
                                   "global_rng_unchanged": True})
        result["native_repeat_exact"] = torch.equal(payload["native_0"], payload["native_1"])
        result["rgb_repeat_exact"] = torch.equal(payload["rgb_0"], payload["rgb_1"])
        result["state_after_sampling"] = state_identity(model)
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["peak_cpu_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["native_extension_binary_sha256"] = {
            str(path.relative_to(extension_cache)): sha256_file(path)
            for path in sorted(extension_cache.rglob("*.so"))}
        result["source_unchanged"] = audit_source(args.upstream_root)["tree_sha256"] == source["tree_sha256"]
        raw_path = args.output / "native_first_block.pt"
        torch.save(payload, raw_path)
        result["raw"] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
        if (not result["native_repeat_exact"] or not result["rgb_repeat_exact"] or
                result["state_after_sampling"] != result["state_before_sampling"] or
                not result["source_unchanged"] or result["actual_calls"] != 202):
            raise ValueError("Native repeat, frozen state, source or total-call gate failed")
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        with (args.output / "native_result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({"status": result["status"], "actual_calls": result["actual_calls"],
                          "result": str(args.output / "native_result.json")}))


if __name__ == "__main__":
    main()
