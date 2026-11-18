"""Explicit SenCache/Wan residual-cache port to pinned FAR-B BAIR.

No replacement of model.forward or upstream source. Torch imports are deferred
until Slurm execution. The scalar controller can be tested without model/data.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any


SENCACHE_COMMIT = "3fb96a41511ede1e38f3555c863b1cb7cddfe567"
OFFICIAL_SOURCES = {
    "Wan2.1/sencache.py": "34f769d6bb715f9a62a1f4c9952e019e06badefa061c2386647d75d1beb0da7b",
    "Wan2.1/sensitivity_calculation.py": "e20ea7ce5b4b391d8b71734a1f986b461b15df4975345eeaa2b0932bf321c6d5",
}


def nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class SenCacheConfig:
    # Wan defaults, with its cond/uncond call index expressed in solver steps.
    steps: int = 50
    threshold_start: float = 0.005
    threshold_main: float = 0.07
    max_skips: int = 10
    threshold_switch_step: int = 10
    retention_steps: int = 0
    cutoff_steps: int = 50
    latent_numel: int = 2048

    def __post_init__(self):
        for name in ("steps", "max_skips", "threshold_switch_step", "retention_steps", "cutoff_steps", "latent_numel"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if (self.steps != 50 or self.latent_numel != 2048 or self.max_skips < 0
                or not 0 <= self.retention_steps <= self.cutoff_steps <= self.steps
                or not 0 <= self.threshold_switch_step <= self.steps):
            raise ValueError("Invalid pinned FAR50 SenCache configuration")
        nonnegative(self.threshold_start, "threshold_start")
        nonnegative(self.threshold_main, "threshold_main")

    def threshold(self, step):
        raw = self.threshold_start if step < self.threshold_switch_step else self.threshold_main
        # Official Wan SCALING_FACTOR is sqrt(number of latent elements).
        # Deltas below remain L2, not RMS; do not normalize them a second time.
        return raw * math.sqrt(self.latent_numel)


@dataclass(frozen=True)
class SensitivityProfile:
    timesteps: tuple[float, ...]
    latent: tuple[float, ...]
    time: tuple[float, ...]
    provenance: str

    def __post_init__(self):
        if not self.provenance or not self.timesteps or len(self.timesteps) != len(self.latent) or len(self.latent) != len(self.time):
            raise ValueError("An identified, complete sensitivity profile is required")
        if any(not math.isfinite(v) for v in self.timesteps) or len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("Sensitivity timesteps must be finite and unique")
        for name, values in (("latent", self.latent), ("time", self.time)):
            for value in values:
                nonnegative(value, name)

    def lookup(self, timestep):
        if not math.isfinite(timestep):
            raise ValueError("Nonfinite lookup timestep")
        # np.argmin from the official implementation breaks ties by first index.
        index = min(range(len(self.timesteps)), key=lambda i: abs(self.timesteps[i] - timestep))
        return self.latent[index], self.time[index], index


class SenCachePolicy:
    """Full scalar Wan rule; reference sensitivities are locked between refreshes."""

    def __init__(self, config, *, cache_off=False):
        self.config, self.cache_off = config, cache_off
        self.next_step, self.skips, self.reference = 0, 0, None
        self.awaiting_refresh = False

    def choose(self, step, delta_z_l2, delta_t):
        if step != self.next_step or step >= self.config.steps or self.awaiting_refresh:
            raise ValueError("Out-of-order step or missing cache refresh")
        dz, dt = nonnegative(delta_z_l2, "delta_z_l2"), nonnegative(delta_t, "delta_t")
        score = None if self.reference is None else self.reference[0] * dz + self.reference[1] * dt
        threshold = self.config.threshold(step)
        in_zone = self.config.retention_steps <= step < self.config.cutoff_steps
        hit = (not self.cache_off and self.reference is not None and in_zone
               and score < threshold and self.skips < self.config.max_skips)
        self.skips = self.skips + 1 if hit else 0
        self.awaiting_refresh = not hit
        self.next_step += 1
        return {"step": step, "cache_hit": hit, "sensitivity_error": score,
                "threshold": threshold, "in_skip_zone": in_zone,
                "consecutive_skips": self.skips}

    def refresh(self, latent_sensitivity, time_sensitivity):
        if not self.awaiting_refresh:
            raise ValueError("Refresh must follow a full evaluation")
        self.reference = (nonnegative(latent_sensitivity, "J_z"), nonnegative(time_sensitivity, "J_t"))
        self.awaiting_refresh = False


def solver_step_sensitivities(evaluate, scheduler, latent, timestep, next_timestep):
    """Official two secant formulas in the native FAR arithmetic domain.

    evaluate must use one fixed clean history/KV and must not advance that
    history. It is called three times: the common baseline is shared explicitly.
    The scheduler is a private one-step state, never the generation scheduler.
    This helper does not perform calibration or read any dataset by itself.
    """
    import torch

    with torch.no_grad():
        # Wan returns float32 field outputs for secant measurements. Preserve
        # those reductions while executing the FAR Euler update in native dtype.
        baseline = evaluate(latent, timestep).float()
        next_latent = scheduler.step(baseline.to(latent.dtype), timestep, latent).prev_sample
        moved = evaluate(next_latent, timestep).float()
        shifted = evaluate(latent, next_timestep).float()
        dx = torch.norm(next_latent - latent).item()
        dt = float(next_timestep) - float(timestep)
        if dt == 0:
            raise ValueError("Time secant requires different adjacent timesteps")
        jx = 0.0 if dx < 1e-8 else torch.norm(moved - baseline).item() / dx
        jt = torch.norm((shifted - baseline) / dt).item()
    nonnegative(jx, "estimated J_z")
    nonnegative(jt, "estimated J_t")
    return {"latent": jx, "time": jt, "complete_stack_calls": 3,
            "delta_latent_l2": dx, "delta_t": dt}


class SenCacheFAR:
    """Composition over frozen native modules; no hooks alter model outputs."""

    def __init__(self, adapter, profile, config=SenCacheConfig()):
        from future_gain.runtime import require_slurm_compute
        require_slurm_compute()
        self.adapter, self.profile, self.config = adapter, profile, config
        self.model = adapter.pipeline.transformer
        if (self.model.training or self.model.config.condition_cfg is not None
                or self.model.config.patch_size != 1 or self.model.config.in_channels != 32
                or len(self.model.transformer_blocks) != 12):
            raise ValueError("Only frozen unconditional pinned FAR-B BAIR is supported")
        self.torch = adapter.torch

    def _forward(self, state, latent, timestep, residual, *, full):
        """Native FAR forward344:484, inference/CFG1 branch, with Wan residual site."""
        from accelerate.utils.operations import convert_to_fp32
        from einops import rearrange

        torch, model = self.torch, self.model
        hidden_states = torch.cat([state.history, latent], dim=1)
        input_dtype = hidden_states.dtype
        timesteps = timestep[None].to(latent.device).expand(latent.shape[0]).unsqueeze(-1)
        context_times = torch.tensor([-1], dtype=timesteps.dtype, device=timesteps.device)
        context_times = context_times.expand(latent.shape[0]).unsqueeze(-1).repeat((1, state.history.shape[1]))
        timesteps = torch.cat([context_times, timesteps], dim=-1)
        context_cache = state.cache
        if context_cache["is_cache_step"] and not full:
            raise ValueError("Clean-history KV prefill cannot be skipped")
        with torch.no_grad(), self.adapter.accelerator.autocast():
            batch_size, num_frames, _, height, width = hidden_states.shape
            token_per_frame = height * width
            hidden_states = model._pack_latent_sequence(hidden_states, patch_size=1)
            latent_seq_ids = model._prepare_latent_sequence_ids(
                batch_size, num_frames, height, width, patch_size=1,
                device=hidden_states.device, dtype=hidden_states.dtype)
            if context_cache["is_cache_step"]:
                current_seq_len = hidden_states.shape[1] - context_cache["cached_seqlen"]
                context_cache["cached_seqlen"] = hidden_states.shape[1] - token_per_frame
                hidden_states = hidden_states[:, -current_seq_len:, ...]
                timesteps = timesteps[:, -(current_seq_len // token_per_frame):]
            else:
                hidden_states = hidden_states[:, -token_per_frame:, ...]
                timesteps = timesteps[:, -1:]
            attention_mask = model._build_causal_mask(
                input_shape=(batch_size, num_frames, num_frames * token_per_frame),
                device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = model.x_embedder(hidden_states)
            seq_rotary_emb = model.pos_embed(latent_seq_ids)
            timesteps = rearrange(timesteps, "b t -> (b t)")
            timesteps_proj = model.time_proj(timesteps.to(hidden_states.dtype))
            temb = model.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
            temb = rearrange(temb, "(b t) c -> b t c", b=batch_size).repeat_interleave(token_per_frame, dim=1)
            if full:
                original = hidden_states.clone()
                for index, block in enumerate(model.transformer_blocks):
                    layer = {"is_cache_step": context_cache["is_cache_step"],
                             "kv_cache": context_cache["kv_cache"].get(index, {}),
                             "token_per_frame": token_per_frame}
                    hidden_states, layer = block(hidden_states=hidden_states, temb=temb,
                        image_rotary_emb=seq_rotary_emb, attention_mask=attention_mask, layer_kv_cache=layer)
                    context_cache["kv_cache"][index] = layer["kv_cache"]
                # First prefill includes clean tokens; future denoising contains
                # only the current frame. Never retain a residual for old tokens.
                residual = (hidden_states - original)[:, -token_per_frame:, :].detach().clone()
            else:
                if residual is None or residual.shape != hidden_states.shape or residual.dtype != hidden_states.dtype:
                    raise ValueError("Residual does not match the current frame embeddings")
                hidden_states = hidden_states + residual
            hidden_states = model.norm_out(hidden_states, temb)
            output = model.proj_out(hidden_states)
            output = output[:, -token_per_frame:, :]
            output = model._unpack_latent_sequence(output, num_frames=1, height=height, width=width)
        # Accelerator's native wrapper recursively casts BOTH output and cache.
        output, state.cache = convert_to_fp32((output, context_cache))
        state.cache["is_cache_step"] = False
        return output.to(input_dtype), residual

    def rollout(self, rgb_prefix, noises, *, cache_off=False, observer=None):
        """Only observed RGB enters this API; profile is fixed before generation."""
        adapter, torch = self.adapter, self.torch
        if tuple(noises.shape) != (1, 28, 32, 8, 8):
            raise ValueError("Expected complete BAIR per-frame noise bank")
        adapter._check_noise(noises[:, :1])
        state = adapter.encode_observed_prefix(rgb_prefix)
        trace = []
        for frame in range(28):
            adapter._check_state(state, idle=True)
            policy = SenCachePolicy(self.config, cache_off=cache_off)
            scheduler = adapter._new_scheduler(self.config.steps)
            latent = noises[:, frame:frame + 1].detach().clone()
            reference_z = reference_t = residual = None
            for step, timestep in enumerate(scheduler.timesteps):
                dz = 0.0 if reference_z is None else torch.norm(latent - reference_z).item()
                dt = 0.0 if reference_t is None else torch.abs(timestep - reference_t).item()
                decision = policy.choose(step, dz, dt)
                state.cache["is_cache_step"] = step == 0
                if observer:
                    observer("before", frame, step, state, decision)
                vector, residual = self._forward(state, latent, timestep, residual, full=not decision["cache_hit"])
                if not decision["cache_hit"]:
                    reference_z, reference_t = latent.detach().clone(), timestep.detach().clone()
                    jx, jt, index = self.profile.lookup(float(timestep))
                    policy.refresh(jx, jt)
                else:
                    index = None
                if observer:
                    observer("after", frame, step, state, decision)
                latent = scheduler.step(vector, timestep, latent).prev_sample
                trace.append({"frame": frame, **decision, "timestep": float(timestep),
                              "delta_latent_l2": dz, "delta_t": dt, "profile_refresh_index": index,
                              "cached_seqlen": state.cache["cached_seqlen"]})
            state.history = torch.cat([state.history, latent], dim=1)
            state.generated_frames += 1
        return {"state": state, "latents": state.history,
                "rgb_full": adapter.decode_generated_frames(state, include_prefix=True),
                "trace": trace, "config": asdict(self.config), "profile": asdict(self.profile),
                "solver_iterations": len(trace),
                "complete_stack_calls": sum(not row["cache_hit"] for row in trace),
                "cache_hits": sum(row["cache_hit"] for row in trace)}
