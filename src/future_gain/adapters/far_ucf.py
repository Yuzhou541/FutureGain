"""Pinned UCF101 task specialization with the unchanged FAR Euler/packet core.

The checkpoint loader, window, observed-prefix shape and rollout limit belong to
this task. Sampling, deep forks and call accounting use the audited parent
implementation without runtime changes. Both files are source-bound in UCF runs;
the separate frozen BAIR implementation and all its artifacts stay unchanged.
"""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from future_gain.adapters.far import (
    FARAdapter, FARState, FirstEvalPacket, NativeResult, UPSTREAM_COMMIT,
    _sha256, _verify_far_module_origins,
)

UCF_TRANSFORMER_SHA256 = "381d295f35f49d4185ce0306c47ccb6f60375569f45e18c13cc6dcf062ec6437"
UCF_CODEC_SHA256 = "9da18dcfb35d5a2d4804dd040887a038254f21bc726b209a260dfdb49d4bdbac"


class UCFAdapter(FARAdapter):
    """FAR-B UCF101, 64², four observed/twelve future frames, window16.

    Exactly one CUDA GPU, batch one, BF16 autocast and CFG1; no conversion of
    BAIR weights or state, and no change to the sampler arithmetic.
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
    ) -> UCFAdapter:
        from future_gain.runtime import require_slurm_compute

        require_slurm_compute()
        if device not in {"cuda", "cuda:0"} or mixed_precision != "bf16":
            raise ValueError("Pinned UCF101 protocol requires one visible CUDA GPU and BF16")
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
        if hashes != (UCF_TRANSFORMER_SHA256, UCF_CODEC_SHA256):
            raise ValueError("Checkpoint SHA-256 does not match the pinned UCF101 assets")

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
        model = build_model("FAR_B")(short_term_ctx_winsize=16)
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

    def _check_state(self, state: FARState, *, idle: bool) -> None:
        if state.owner_id != self.owner_id:
            raise ValueError("State belongs to another model adapter")
        if state.history.shape != (1, 4 + state.generated_frames, 32, 8, 8):
            raise ValueError("Invalid UCF101 latent history shape")
        if state.history.device != self.device or state.history.dtype != self.dtype:
            raise ValueError("State history must use the bound CUDA device and BF16 dtype")
        if state.observed_frames != 4 or not 0 <= state.generated_frames <= 12:
            raise ValueError("State exceeds the UCF101 4→12 protocol")
        if idle and state.pending_packet_id is not None:
            raise ValueError("State already has an unfinished first-evaluation packet")

    def encode_observed_prefix(self, rgb_prefix: Any) -> FARState:
        torch = self.torch
        if tuple(rgb_prefix.shape) != (1, 4, 3, 64, 64):
            raise ValueError("Pass only the observed four-frame UCF101 prefix [1,4,3,64,64]")
        if rgb_prefix.device.type != "cpu" or rgb_prefix.dtype != torch.float32:
            raise ValueError("Deployment input must be a prepared CPU float32 RGB tensor")
        from future_gain.data.ucf_native import validate_native_rgb
        validate_native_rgb(rgb_prefix, torch)
        with torch.no_grad():
            history = self.pipeline.vae_encode(rgb_prefix.to(self.device))
        self.vae_encode_calls += 1
        return FARState(
            history=history,
            cache={"is_cache_step": True, "kv_cache": {}, "cached_seqlen": 0,
                   "multi_level_cache_init": False},
            owner_id=self.owner_id,
            state_id=uuid4().hex,
            observed_frames=4,
        )

    def first_evaluation(self, state: FARState, frame_noise: Any, *, noise_id: str = "") -> FirstEvalPacket:
        # The parent sampling routine is task-independent below its old upper
        # bound; enforce the smaller UCF boundary before any state mutation.
        if state.generated_frames >= 12:
            raise ValueError("UCF101 rollout already contains all 12 future frames")
        return super().first_evaluation(state, frame_noise, noise_id=noise_id)

    def run_native_fixed(self, rgb_prefix: Any, noise_bank: Any, nfe: int) -> NativeResult:
        if noise_bank.ndim != 5 or not 1 <= noise_bank.shape[1] <= 12:
            raise ValueError("UCF101 native comparison requires 1 to 12 predicted frames")
        return super().run_native_fixed(rgb_prefix, noise_bank, nfe)
