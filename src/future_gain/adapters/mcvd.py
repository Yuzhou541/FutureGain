"""Thin FP32 MCVD DDIM adapter; native equivalence is a separate real GPU gate.

One decision predicts five frames jointly. Requested25/50/100 steps consume
26/51/101 calls including MCVD's native terminal-label L-1 denoise. No FAR
state, cache, codec, controller or training behavior is imported here.
"""

from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from future_gain.audit import canonical_hash, sha256_file

SUPPORTED_STEPS = (25, 50, 100)
UPSTREAM_COMMIT = "451da2eb635bad50da6a7c03b443a34c6eb08b3a"


def ddim_labels(requested_steps):
    if type(requested_steps) is not int or requested_steps not in SUPPORTED_STEPS:
        raise ValueError("Only explicitly verified candidate grids25/50/100 are implemented")
    loop = tuple(range(0, 1000, 1000 // requested_steps))
    return loop + (len(loop) - 1,)


def validate_native_result(record):
    if (record.get("status") != "passed" or record["source"]["commit"] != UPSTREAM_COMMIT
            or record["actual_calls"] != 202 or record["requested_ddim_steps"] != 100
            or record["context_frames"] != 2 or record["future_frames"] != 5
            or record["source_unchanged"] is not True or record["native_repeat_exact"] is not True
            or record["rgb_repeat_exact"] is not True or record["checkpoint_key_check"] != {"missing": [], "unexpected": []}
            or record["state_before_sampling"] != record["state_after_sampling"]
            or record["ema"]["applied_before_freeze"] is not True or record["ema"]["all_parameters_equal_shadow"] is not True):
        raise ValueError("A complete passing native MCVD repeat/EMA/source gate is required")
    expected = [[n] for n in ddim_labels(100)]
    if len(record["runs"]) != 2 or any(r["actual_calls"] != 101 or r["labels"] != expected
            or r["global_rng_unchanged"] is not True for r in record["runs"]):
        raise ValueError("Native literal call/RNG evidence differs")
    sampler = record["sampler"]
    for name, value in {"function": "models.ddim_sampler", "training_config_version": "DDPM",
                       "denoise": True, "clip_before": True, "t_min": -1, "gamma": False,
                       "subsample_steps": 100, "terminal_label": 99}.items():
        if sampler[name] != value:
            raise ValueError("Native sampler contract differs")


@dataclass
class MCVDState:
    history_native: Any
    observed_prefix: Any
    generated_blocks: int
    owner_id: str
    state_id: str


@dataclass
class PendingBlock:
    history_native: Any
    observed_prefix: Any
    condition: Any
    noise: Any
    epsilon: Any
    coarse_clean: Any
    block_index: int
    noise_id: str
    owner_id: str
    state_id: str
    packet_id: str
    binding: dict
    consumed: bool = False


@dataclass(frozen=True)
class BlockCost:
    requested_steps: int
    denoiser_calls: int  # Logical branch NFE: saved first + actual remainder.
    additional_denoiser_calls: int  # New physical forwards after first_eval.
    first_eval_calls: int = 1
    terminal_calls: int = 1
    probe_extra_calls: int = 0
    cache_extra_calls: int = 0


def _model_origins(root):
    for name, module in tuple(sys.modules.items()):
        if name == "models" or name.startswith("models."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(root / "models"):
                raise ValueError("A cached models module belongs to another source tree")


class MCVDAdapter:
    """Sequential single-GPU adapter, frozen model and independently owned packets.

    Packet hashes are correctness instrumentation, not an optimized deployment
    timing path. They bind every mutable input and saved first-forward result.
    The real native gate must pass before any research use of this adapter.
    """

    @classmethod
    def from_native_verification(cls, native_result_path, upstream_root, asset_manifest_path,
                                 asset_dir, extension_cache, cuda_home):
        from future_gain.runtime import require_slurm_compute
        allocation = require_slurm_compute()  # Before model imports, assets or CUDA.
        from scripts.prepare_mcvd_assets import PROJECT, audit_source, owned_directory, validate_config, verify_assets
        from scripts.verify_mcvd_native import namespace, state_identity, tensor_hash

        record_path = Path(native_result_path).resolve(strict=True)
        record = json.loads(record_path.read_text())
        validate_native_result(record)
        env = Path(sys.prefix).resolve()
        if env != Path(record["environment"]["prefix"]) or env.parent != PROJECT / "envs" or not env.name.startswith("mcvd_"):
            raise ValueError("Use the exact separately verified MCVD environment")
        root, assets = owned_directory(Path(upstream_root)), owned_directory(Path(asset_dir))
        cache, toolkit = owned_directory(Path(extension_cache)), owned_directory(Path(cuda_home))
        environment = record["environment"]
        if sys.version != environment["python"]:
            raise ValueError("Native Python environment differs")
        if (str(cache) != environment["extension_cache"] or str(toolkit) != environment["cuda_home"]
                or not cache.is_relative_to(assets) or not toolkit.is_relative_to(assets)):
            raise ValueError("Native toolkit/cache identities changed")
        source = audit_source(root)
        source_root = Path(__file__).resolve().parents[3]
        for relative, key in (("scripts/verify_mcvd_native.py", "script_sha256"),
                              ("scripts/prepare_mcvd_assets.py", "asset_script_sha256"),
                              ("src/future_gain/runtime.py", "runtime_guard_sha256")):
            if sha256_file(source_root / relative) != record[key]:
                raise ValueError("Native verification helper changed after the passed gate")
        if source != record["source"] or sha256_file(Path(asset_manifest_path)) != record["asset_manifest_sha256"]:
            raise ValueError("Native source/assets manifest changed")
        verified = verify_assets(json.loads(Path(asset_manifest_path).read_text()), assets)
        if verified != record["assets"]:
            raise ValueError("Native checkpoint/config asset bytes differ")
        for name in ("torch", "numpy", "scipy", "PyYAML", "av"):
            if version(name).split("+")[0] != environment[name]:
                raise ValueError("Native package environment differs")
        nvcc = subprocess.run([str(toolkit / "bin/nvcc"), "--version"], check=True, capture_output=True, text=True).stdout
        cxx = subprocess.run([os.environ.get("CXX", "c++"), "--version"], check=True, capture_output=True, text=True).stdout
        if nvcc != environment["nvcc"] or cxx != environment["cxx"] or os.environ.get("CUDAHOSTCXX") != environment["cuda_host_cxx"]:
            raise ValueError("Native compiler identity differs")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
            raise ValueError("Conflicting deterministic CUBLAS workspace")
        os.environ.update(CUDA_HOME=str(toolkit), TORCH_EXTENSIONS_DIR=str(cache), MAX_JOBS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        sys.dont_write_bytecode = True
        import torch
        import yaml
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != environment["torch_cuda"]:
            raise ValueError("Native gate requires exactly one matching CUDA GPU")
        if torch.cuda.get_device_name(0) != environment["gpu"]:
            raise ValueError("Native GPU type differs")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        config_dict = yaml.safe_load(Path(verified["config.yml"]["path"]).read_text())
        validate_config(config_dict)
        config = namespace(config_dict)
        config.device = torch.device("cpu")
        _model_origins(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from models.better.ncsnpp_more import UNetMore_DDPM
        from models.ema import EMAHelper
        _model_origins(root)
        states = torch.load(verified["checkpoint_450000.pt"]["path"], map_location="cpu", weights_only=False)
        if not isinstance(states, (list, tuple)) or len(states) != record["checkpoint_structure"]["length"]:
            raise ValueError("Native checkpoint structure changed")
        model = torch.nn.DataParallel(UNetMore_DDPM(config), device_ids=[0]).cpu()
        if set(model.state_dict()) != set(states[0]):
            raise ValueError("Strict native state keys differ; no migration is permitted")
        model.load_state_dict(states[0], strict=True)
        parameters = dict(model.module.named_parameters())
        if set(parameters) != set(states[-1]) or not all(p.requires_grad for p in parameters.values()):
            raise ValueError("EMA must cover every trainable parameter before freezing")
        ema = EMAHelper(mu=config.model.ema_rate)
        ema.register(model)
        ema.load_state_dict(states[-1])
        before = canonical_hash({name: tensor_hash(p) for name, p in parameters.items()})
        ema.ema(model)
        if any(not torch.equal(p.detach(), states[-1][name]) for name, p in parameters.items()):
            raise ValueError("Native EMA did not apply exactly")
        shadow = canonical_hash({name: tensor_hash(p) for name, p in parameters.items()})
        if before != record["ema"]["pre_ema_sha256"] or shadow != record["ema"]["shadow_sha256"]:
            raise ValueError("Native model/EMA parameter identity differs")
        model.requires_grad_(False).eval()
        state = state_identity(model)
        if state != record["state_before_sampling"]:
            raise ValueError("Loaded native parameters/schedule buffers differ")
        del states, ema, parameters
        config.device = torch.device("cuda:0")
        model.to(config.device)
        binaries = {str(p.relative_to(cache)): sha256_file(p) for p in sorted(cache.rglob("*.so"))}
        if binaries != record["native_extension_binary_sha256"] or audit_source(root) != source:
            raise ValueError("Native custom operator binary/source identity differs")
        self = cls()
        self.torch, self.model, self.device, self.dtype = torch, model, config.device, torch.float32
        self._tensor_hash = tensor_hash
        self.owner_id, self.denoiser_calls, self.call_trace = uuid4().hex, 0, []
        self._issued_packet_ids, self._consumed_packet_ids = set(), set()
        self.identity = {"native_result_path": str(record_path), "native_result_sha256": sha256_file(record_path),
            "source": source, "assets": verified, "asset_manifest_sha256": record["asset_manifest_sha256"],
            "runtime": environment, "state_identity": state, "native_extension_binary_sha256": binaries,
            "adapter_sha256": sha256_file(Path(__file__)), "supported_requested_steps": list(SUPPORTED_STEPS),
            "allocation": allocation}
        self._model_identity = canonical_hash({k: v for k, v in self.identity.items() if k != "allocation"})
        return self

    def _check_state(self, state):
        torch = self.torch
        if not isinstance(state, MCVDState) or state.owner_id != self.owner_id or type(state.generated_blocks) is not int or not 0 <= state.generated_blocks <= 6:
            raise ValueError("State belongs to another adapter or has an invalid block count")
        if (state.history_native.shape != (1, 2 + 5 * state.generated_blocks, 3, 64, 64)
                or state.history_native.device != self.device or state.history_native.dtype != torch.float32
                or state.observed_prefix.shape != (1, 2, 3, 64, 64) or state.observed_prefix.device.type != "cpu"
                or state.observed_prefix.dtype != torch.float32 or not torch.isfinite(state.history_native).all()
                or not torch.isfinite(state.observed_prefix).all() or state.observed_prefix.min() < 0 or state.observed_prefix.max() > 1):
            raise ValueError("Native state shape/device/dtype is invalid")

    def state_fingerprint(self, state):
        self._check_state(state)
        return {"history_native": self._tensor_hash(state.history_native), "observed_prefix": self._tensor_hash(state.observed_prefix),
            "generated_blocks": state.generated_blocks, "model_identity": self._model_identity}

    def packet_fingerprint(self, packet):
        return {**{name: self._tensor_hash(getattr(packet, name)) for name in
            ("history_native", "observed_prefix", "condition", "noise", "epsilon", "coarse_clean")},
            "block_index": packet.block_index, "noise_id": packet.noise_id, "state_id": packet.state_id, "model_identity": self._model_identity}

    def _check_packet(self, packet):
        if (not isinstance(packet, PendingBlock) or packet.owner_id != self.owner_id or packet.consumed
                or packet.packet_id not in self._issued_packet_ids or packet.packet_id in self._consumed_packet_ids):
            raise ValueError("Packet is consumed or belongs to another adapter")
        if packet.binding != self.packet_fingerprint(packet):
            raise ValueError("Pending packet tensor/metadata binding changed")

    def initialize(self, observed_prefix):
        torch = self.torch
        if (observed_prefix.shape != (1, 2, 3, 64, 64) or observed_prefix.device.type != "cpu"
                or observed_prefix.dtype != torch.float32 or not torch.isfinite(observed_prefix).all()
                or observed_prefix.min() < 0 or observed_prefix.max() > 1):
            raise ValueError("Supply only two observed CPU float32 RGB frames in[0,1]")
        prefix = observed_prefix.detach().clone()
        # Same arithmetic as native preprocessing; keep original RGB separately.
        history = (2 * prefix - 1.).to(self.device)
        return MCVDState(history, prefix, 0, self.owner_id, uuid4().hex)

    def _evaluate(self, value, label, condition, block_index):
        labels = (label * self.torch.ones(value.shape[0], device=value.device)).long()
        self.denoiser_calls += 1
        self.call_trace.append({"physical_call": self.denoiser_calls, "block_index": block_index, "label": label})
        return self.model(value, labels, cond=condition)

    def first_eval(self, state, noise, noise_id):
        self._check_state(state)
        torch = self.torch
        if state.generated_blocks >= 6 or not isinstance(noise_id, str) or not noise_id:
            raise ValueError("A pending block requires an available block index and explicit noise identity")
        if noise.shape != (1, 15, 64, 64) or noise.device != self.device or noise.dtype != torch.float32 or not torch.isfinite(noise).all():
            raise ValueError("MCVD noise must be finite CUDA float32[1,15,64,64]")
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            history = state.history_native.detach().clone()
            condition = history[:, -2:].reshape(1, 6, 64, 64).clone()
            initial = noise.detach().clone()
            epsilon = self._evaluate(initial, 0, condition, state.generated_blocks).detach().clone()
            alpha = self.model.module.alphas[0]
            coarse = (1 / alpha.sqrt()) * (initial - (1 - alpha).sqrt() * epsilon)
            coarse = coarse.clip_(-1, 1)
        packet = PendingBlock(history, state.observed_prefix.clone(), condition, initial, epsilon, coarse,
            state.generated_blocks, noise_id, self.owner_id, state.state_id, uuid4().hex, {})
        packet.binding = self.packet_fingerprint(packet)
        self._issued_packet_ids.add(packet.packet_id)
        return packet

    def fork_pending(self, packet):
        self._check_packet(packet)
        cloned = {name: getattr(packet, name).detach().clone() for name in
            ("history_native", "observed_prefix", "condition", "noise", "epsilon", "coarse_clean")}
        child = PendingBlock(**cloned, block_index=packet.block_index, noise_id=packet.noise_id, owner_id=self.owner_id,
            state_id=packet.state_id, packet_id=uuid4().hex, binding=dict(packet.binding))
        self._issued_packet_ids.add(child.packet_id)
        return child

    def finish_block(self, packet, requested_steps):
        labels = ddim_labels(requested_steps)
        self._check_packet(packet)
        torch = self.torch
        packet.consumed = True  # A failing partially executed packet cannot be replayed.
        self._consumed_packet_ids.add(packet.packet_id)
        before = self.denoiser_calls
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            steps = torch.tensor(labels[:-1], device=self.model.module.alphas.device)
            alphas = self.model.module.alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            value = packet.noise.detach().clone()
            for i, label in enumerate(labels[:-1]):
                epsilon = packet.epsilon if i == 0 else self._evaluate(value, label, packet.condition, packet.block_index)
                # Literal native operation order, including reciprocal multiply.
                x0 = (1 / alphas[i].sqrt()) * (value - (1 - alphas[i]).sqrt() * epsilon)
                x0 = x0.clip_(-1, 1)
                value = alphas_prev[i].sqrt() * x0 + (1 - alphas_prev[i]).sqrt() * epsilon
            # Native terminal label is L-1, not the last selected timestep.
            value = value - (1 - alphas[-1]).sqrt() * self._evaluate(value, labels[-1], packet.condition, packet.block_index)
            if not torch.isfinite(value).all():
                raise ValueError("Native block output is nonfinite")
            block = value.reshape(1, 5, 3, 64, 64)
            history = torch.cat((packet.history_native, block), dim=1)
        additional = self.denoiser_calls - before
        if additional != requested_steps:
            raise AssertionError("Physical remaining-forward count differs from selected grid")
        state = MCVDState(history, packet.observed_prefix.clone(), packet.block_index + 1, self.owner_id, uuid4().hex)
        return block, state, BlockCost(requested_steps, requested_steps + 1, additional)

    def decode_future(self, state, *, include_prefix=False, prediction_frames=None):
        self._check_state(state)
        available = state.generated_blocks * 5
        frames = available if prediction_frames is None else prediction_frames
        if type(frames) is not int or not 1 <= frames <= available or type(include_prefix) is not bool:
            raise ValueError("Decode only an explicitly available positive future length")
        rgb = ((state.history_native[:, 2:2 + frames] + 1.) / 2.).clamp(0, 1).cpu()
        return self.torch.cat((state.observed_prefix.clone(), rgb), dim=1) if include_prefix else rgb
