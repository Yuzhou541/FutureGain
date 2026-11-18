"""Pinned, batch-one BAIR adapter; all model work requires Slurm CUDA.

The upstream model and pipeline are imported unchanged. The adapter separates
the common first evaluation from the selected Euler grid, without an extra
cache-commit evaluation. GPU equivalence must be verified before experiments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4


UPSTREAM_COMMIT = "e59536ac30e5e3fa93db756fccfd422420a65ff6"
BAIR_TRANSFORMER_SHA256 = "1983191bec540d068440e1566231c8068a9e3a4abec49dfadb90be2cc118240b"
BAIR_CODEC_SHA256 = "1a8547fb534045fd5e7ed620707c9a9ffa3a29a00b8e7853836895ab2f6ee38f"


@dataclass
class FARState:
    history: Any
    cache: dict[str, Any]
    owner_id: str
    state_id: str
    observed_frames: int = 2
    generated_frames: int = 0
    pending_packet_id: str | None = None


@dataclass
class FirstEvalPacket:
    state_id: str
    packet_id: str
    frame_index: int
    noise_id: str
    noise: Any
    vector_field: Any
    coarse_clean: Any
    first_timestep: Any
    first_sigma: Any
    first_eval_calls: int = 1
    consumed: bool = False


@dataclass(frozen=True)
class FrameCost:
    denoiser_calls: int
    probe_extra_calls: int = 0
    cache_extra_calls: int = 0
    # Timing is deliberately measured by the outer CUDA-synchronized runner.
    # A CPU timer here would only measure launch time or add unwanted barriers.


@dataclass(frozen=True)
class NativeResult:
    latents: Any
    rgb_full: Any
    calls: int
    state: FARState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tree(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, torch) for item in value)
    return copy.deepcopy(value)


def _verify_far_module_origins(root: Path) -> None:
    """Reject cached FAR modules from any source outside the exact checkout."""
    package_root = root.resolve() / "far"
    for name, module in tuple(sys.modules.items()):
        if name != "far" and not name.startswith("far."):
            continue
        expected = package_root.joinpath(*name.split(".")[1:])
        if module is None:
            raise ValueError(f"Incomplete cached FAR module: {name}")
        source = getattr(module, "__file__", None)
        namespace = getattr(module, "__path__", None)
        if source is not None:
            allowed = {expected.with_suffix(".py"), expected / "__init__.py"}
            if Path(source).resolve() not in allowed:
                raise ValueError(f"Cached FAR module has foreign source: {name}")
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if origin is not None and Path(origin).resolve() != Path(source).resolve():
                raise ValueError(f"Cached FAR module origin disagrees with its source: {name}")
        elif namespace is None:
            raise ValueError(f"Cached FAR module has no verifiable source: {name}")
        if namespace is not None and [Path(path).resolve() for path in namespace] != [expected]:
            raise ValueError(f"Cached FAR namespace has a foreign search path: {name}")


class FARAdapter:
    """Only FAR-B BAIR 64x64, 2-frame prefix, CFG=1, BF16, one CUDA GPU.

    A state has one active packet at a time. first_evaluation mutates its clean
    history cache; finish_frame consumes the packet exactly once. Use explicit
    deep forks for offline branches. One adapter is sequential, not thread-safe.
    """

    @classmethod
    def from_pretrained(
        cls,
        upstream_root: str | Path,
        transformer_checkpoint: str | Path,
        codec_checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        mixed_precision: str = "bf16",
    ) -> FARAdapter:
        from future_gain.runtime import require_slurm_compute

        require_slurm_compute()
        if device not in {"cuda", "cuda:0"} or mixed_precision != "bf16":
            raise ValueError("Pinned BAIR protocol requires one visible CUDA GPU and BF16")
        root = Path(upstream_root).resolve(strict=True)
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
        if commit != UPSTREAM_COMMIT or dirty:
            raise ValueError("FAR must be the exact clean pinned upstream checkout")
        for dependency, expected in (("diffusers", "0.32.2"), ("accelerate", "1.4.0")):
            if version(dependency) != expected:
                raise ValueError(f"Pinned adapter requires {dependency}=={expected}")
        paths = (Path(transformer_checkpoint).resolve(strict=True), Path(codec_checkpoint).resolve(strict=True))
        hashes = tuple(_sha256(path) for path in paths)
        if hashes != (BAIR_TRANSFORMER_SHA256, BAIR_CODEC_SHA256):
            raise ValueError("Checkpoint SHA-256 does not match the pinned BAIR assets")

        # Select only the explicitly supplied checkout, never an arbitrary FAR
        # package already installed elsewhere on the machine.
        _verify_far_module_origins(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import far
        # Upstream far is a namespace package (no far/__init__.py).
        if [Path(path).resolve() for path in far.__path__] != [root / "far"]:
            raise ValueError("Imported FAR package is not the supplied pinned checkout")
        import torch
        from accelerate import Accelerator
        from diffusers import FlowMatchEulerDiscreteScheduler
        from far.models import build_model
        from far.pipelines.pipeline_far import FARPipeline
        _verify_far_module_origins(root)

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("This adapter requires exactly one Slurm-visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required; precision fallback is not allowed")
        accelerator = Accelerator(mixed_precision="bf16")
        if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
            raise RuntimeError("Only the single-process CUDA deployment protocol is supported")
        model = build_model("FAR_B")(short_term_ctx_winsize=30)
        model.load_state_dict(torch.load(paths[0], map_location="cpu", weights_only=True), strict=True)
        model.requires_grad_(False)
        model.eval()
        # Matches test.py -> accelerator.prepare -> trainer.sample -> unwrap.
        # The transformer remains FP32 with Accelerator's BF16 autocast wrapper;
        # simply casting its parameters to BF16 would change native arithmetic.
        model = accelerator.unwrap_model(accelerator.prepare(model))
        with (root / "options/model_cfg/dcae/model_8x_c32_config.json").open() as stream:
            codec_config = json.load(stream)
        codec = build_model("MyAutoencoderDC").from_config(codec_config)
        codec.load_state_dict(torch.load(paths[1], map_location="cpu", weights_only=True), strict=True)
        codec.to(accelerator.device, dtype=torch.bfloat16)
        codec.requires_grad_(False)
        codec.eval()
        codec.enable_slicing()
        with (root / "options/model_cfg/far/scheduler_config.json").open() as stream:
            scheduler_config = json.load(stream)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        pipeline = FARPipeline(transformer=model, vae=codec, scheduler=scheduler)
        pipeline.execution_device = accelerator.device
        pipeline.set_progress_bar_config(disable=True)
        self = cls()
        self.torch = torch
        self.pipeline = pipeline
        self.accelerator = accelerator
        # Accelerator can report the unindexed alias cuda; allocated tensors
        # report cuda:0. Bind the concrete device for strict tensor checks.
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        self.owner_id = uuid4().hex
        self.denoiser_calls = 0
        self.vae_encode_calls = 0
        self.vae_decode_calls = 0
        self.checkpoint_hashes = {"transformer": hashes[0], "codec": hashes[1]}
        self._scheduler_class = FlowMatchEulerDiscreteScheduler
        self._scheduler_config = dict(scheduler.config)
        self._first_grid = self._new_scheduler(50)
        if float(self._first_grid.timesteps[0]) != 1000.0 or float(self._first_grid.sigmas[0]) != 1.0:
            raise ValueError("The pinned shared first evaluation must be timestep=1000, sigma=1")

        # Ordinary PyTorch instrumentation counts actual calls in both the thin
        # adapter and untouched native pipeline. It does not alter computation.
        def count_forward(_module: Any, _inputs: Any) -> None:
            self.denoiser_calls += 1

        self._call_counter_handle = model.register_forward_pre_hook(count_forward)
        return self

    def _new_scheduler(self, nfe: int) -> Any:
        if isinstance(nfe, bool) or not isinstance(nfe, int) or not 1 <= nfe <= 1000:
            raise ValueError("NFE must be an integer in [1,1000]")
        scheduler = self._scheduler_class.from_config(self._scheduler_config)
        # Native __call__ supplies no device argument; preserve its grid dtype
        # and CPU scalar timesteps, including Euler's original rounding.
        scheduler.set_timesteps(nfe)
        return scheduler

    def _check_state(self, state: FARState, *, idle: bool) -> None:
        if state.owner_id != self.owner_id:
            raise ValueError("State belongs to another model adapter")
        if state.history.shape != (1, 2 + state.generated_frames, 32, 8, 8):
            raise ValueError("Invalid BAIR latent history shape")
        if state.history.device != self.device or state.history.dtype != self.dtype:
            raise ValueError("State history must use the bound CUDA device and BF16 dtype")
        if state.observed_frames != 2 or state.generated_frames > 28:
            raise ValueError("State exceeds the BAIR 2→28 protocol")
        if idle and state.pending_packet_id is not None:
            raise ValueError("State already has an unfinished first-evaluation packet")

    def _check_noise(self, noise: Any) -> None:
        if tuple(noise.shape) != (1, 1, 32, 8, 8):
            raise ValueError("Each frame noise must have shape [1,1,32,8,8]")
        if noise.device != self.device or noise.dtype != self.dtype:
            raise ValueError(f"Noise device/dtype {noise.device}/{noise.dtype} differs from bound {self.device}/{self.dtype}")

    def encode_observed_prefix(self, rgb_prefix: Any) -> FARState:
        torch = self.torch
        if tuple(rgb_prefix.shape) != (1, 2, 3, 64, 64):
            raise ValueError("Pass only the observed two-frame BAIR prefix [1,2,3,64,64]")
        if rgb_prefix.device.type != "cpu" or rgb_prefix.dtype != torch.float32:
            raise ValueError("Deployment input must be a prepared CPU float32 RGB tensor")
        if not torch.isfinite(rgb_prefix).all() or rgb_prefix.min() < 0 or rgb_prefix.max() > 1:
            raise ValueError("Prefix must contain finite RGB values in [0,1]")
        with torch.no_grad():
            history = self.pipeline.vae_encode(rgb_prefix.to(self.device))
        self.vae_encode_calls += 1
        return FARState(
            history=history,
            cache={"is_cache_step": True, "kv_cache": {}, "cached_seqlen": 0,
                   "multi_level_cache_init": False},
            owner_id=self.owner_id,
            state_id=uuid4().hex,
        )

    def _evaluate(self, state: FARState, latent: Any, timestep: Any) -> Any:
        torch = self.torch
        model_input = torch.cat([state.history, latent], dim=1)
        timesteps = timestep[None].to(latent.device).expand(latent.shape[0]).unsqueeze(-1)
        context_times = torch.tensor([-1], dtype=timesteps.dtype, device=timesteps.device)
        context_times = context_times.expand(latent.shape[0]).unsqueeze(-1).repeat((1, state.history.shape[1]))
        timesteps = torch.cat([context_times, timesteps], dim=-1)
        vector, state.cache = self.pipeline.transformer(
            model_input, context_cache=state.cache, timestep=timesteps,
            conditions=None, return_dict=False,
        )
        state.cache["is_cache_step"] = False
        return vector.to(model_input.dtype)

    def first_evaluation(self, state: FARState, frame_noise: Any, *, noise_id: str = "") -> FirstEvalPacket:
        self._check_state(state, idle=True)
        if state.generated_frames == 28:
            raise ValueError("BAIR rollout already contains all 28 future frames")
        self._check_noise(frame_noise)
        if not noise_id:
            raise ValueError("Provide the stable per-frame noise identity for audit")
        token = uuid4().hex
        # Claim before executing so an exception cannot silently reuse a partly
        # mutated cache. A failed state must be rebuilt from its saved lineage.
        state.pending_packet_id = token
        start_calls = self.denoiser_calls
        with self.torch.no_grad():
            noise = frame_noise.detach().clone()
            state.cache["is_cache_step"] = True
            timestep = self._first_grid.timesteps[0].clone()
            sigma = self._first_grid.sigmas[0].clone()
            vector = self._evaluate(state, noise, timestep)
            coarse = noise - float(sigma) * vector
        calls = self.denoiser_calls - start_calls
        if calls != 1:
            raise RuntimeError(f"First evaluation executed {calls} transformer calls, expected 1")
        return FirstEvalPacket(state.state_id, token, state.generated_frames, noise_id,
                               noise, vector, coarse, timestep, sigma, calls)

    def _check_packet(self, state: FARState, packet: FirstEvalPacket) -> None:
        self._check_state(state, idle=False)
        if (packet.consumed or packet.state_id != state.state_id
                or packet.packet_id != state.pending_packet_id
                or packet.frame_index != state.generated_frames):
            raise ValueError("Consumed, stale, or foreign first-evaluation packet")

    def finish_frame(self, state: FARState, packet: FirstEvalPacket, nfe: int) -> tuple[Any, FARState, FrameCost]:
        self._check_packet(state, packet)
        scheduler = self._new_scheduler(nfe)
        if (not self.torch.equal(scheduler.timesteps[0], packet.first_timestep)
                or not self.torch.equal(scheduler.sigmas[0], packet.first_sigma)):
            raise ValueError("Selected Euler grid does not share the packet's first evaluation point")
        packet.consumed = True
        start_calls = self.denoiser_calls
        with self.torch.no_grad():
            # First update is made only now, using the SELECTED NFE grid.
            latent = scheduler.step(packet.vector_field, scheduler.timesteps[0], packet.noise).prev_sample
            for timestep in scheduler.timesteps[1:]:
                vector = self._evaluate(state, latent, timestep)
                latent = scheduler.step(vector, timestep, latent).prev_sample
            calls = packet.first_eval_calls + self.denoiser_calls - start_calls
            if calls != nfe:
                raise RuntimeError(f"Frame executed {calls} transformer calls, expected {nfe}")
            state.history = self.torch.cat([state.history, latent], dim=1)
        state.generated_frames += 1
        state.pending_packet_id = None
        # The new clean frame enters KV only in the NEXT first_evaluation.
        return latent, state, FrameCost(denoiser_calls=calls)

    def fork_state(self, state: FARState) -> FARState:
        self._check_state(state, idle=True)
        return FARState(
            history=_clone_tree(state.history, self.torch),
            cache=_clone_tree(state.cache, self.torch),
            owner_id=self.owner_id, state_id=uuid4().hex,
            observed_frames=state.observed_frames, generated_frames=state.generated_frames,
        )

    def fork_pending(self, state: FARState, packet: FirstEvalPacket) -> tuple[FARState, FirstEvalPacket]:
        """Offline-only fork AFTER a shared first evaluation; no extra forward.

        Each logical branch's FrameCost includes its shared first evaluation.
        Offline physical-cost accounting instead uses adapter.denoiser_calls,
        which counts the shared evaluation once.
        """
        self._check_packet(state, packet)
        state_id, packet_id = uuid4().hex, uuid4().hex
        fork = FARState(_clone_tree(state.history, self.torch), _clone_tree(state.cache, self.torch),
                        self.owner_id, state_id, state.observed_frames, state.generated_frames, packet_id)
        fork_packet = FirstEvalPacket(
            state_id, packet_id, packet.frame_index, packet.noise_id,
            _clone_tree(packet.noise, self.torch), _clone_tree(packet.vector_field, self.torch),
            _clone_tree(packet.coarse_clean, self.torch), _clone_tree(packet.first_timestep, self.torch),
            _clone_tree(packet.first_sigma, self.torch), packet.first_eval_calls,
        )
        return fork, fork_packet

    def decode_generated_frames(self, state: FARState, *, include_prefix: bool = False) -> Any:
        self._check_state(state, idle=True)
        if state.generated_frames < 1:
            raise ValueError("No generated frame is available to decode")
        with self.torch.no_grad():
            # Decode the full sequence exactly as native; slice only afterward.
            rgb = self.pipeline.vae_decode(state.history)
        self.vae_decode_calls += 1
        return rgb if include_prefix else rgb[:, state.observed_frames:]

    def run_native_fixed(self, rgb_prefix: Any, noise_bank: Any, nfe: int) -> NativeResult:
        """Untouched upstream per-frame __call__, with explicitly paired noise.

        This independent execution route is the GPU equivalence comparator,
        not a second implementation of the adapter's Euler loop.
        """
        self._new_scheduler(nfe)  # Validate NFE before encoding.
        if noise_bank.ndim != 5 or tuple(noise_bank.shape[:1] + noise_bank.shape[2:]) != (1, 32, 8, 8):
            raise ValueError("Noise bank must have shape [1,T,32,8,8]")
        if not 1 <= noise_bank.shape[1] <= 28:
            raise ValueError("Native comparison requires 1 to 28 predicted frames")
        self._check_noise(noise_bank[:, :1])
        state = self.encode_observed_prefix(rgb_prefix)
        start_calls = self.denoiser_calls
        with self.torch.no_grad():
            for frame in range(noise_bank.shape[1]):
                latent, state.cache = self.pipeline(
                    vision_context=state.history,
                    latents=noise_bank[:, frame:frame + 1].clone(),
                    conditions=None, context_cache=state.cache,
                    context_timestep_idx=-1, guidance_scale=1.0,
                    num_inference_steps=nfe,
                )
                state.history = self.torch.cat([state.history, latent], dim=1)
                state.generated_frames += 1
        calls = self.denoiser_calls - start_calls
        if calls != noise_bank.shape[1] * nfe:
            raise RuntimeError("Native pipeline did not execute the expected number of transformer calls")
        return NativeResult(state.history, self.decode_generated_frames(state, include_prefix=True), calls, state)
