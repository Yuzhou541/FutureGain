"""Single-trajectory online allocation using current history and first evaluation."""

from dataclasses import asdict
import time

from future_gain.calibration.features import EPSILON, FeatureExtractor
from future_gain.calibration.ridge import RidgeRouter
from future_gain.sampling.budget import EmpiricalThreshold, ExactQuotaController
from future_gain.sampling.noise import tensor_sha256


class OnlineAllocation:
    """One fresh instance per video; no observations beyond its prefix are accepted."""

    def __init__(self, *, kind, validation_scores, high_frames, seed, extractor, router=None):
        started = time.perf_counter()
        if kind not in {"future_gain", "motion", "local_innovation"}:
            raise ValueError("Unknown online policy")
        if (kind == "future_gain") != isinstance(router, RidgeRouter):
            raise ValueError("Exactly the gain policy requires a fitted RidgeRouter")
        self.kind, self.router = kind, router
        if not isinstance(extractor, FeatureExtractor):
            raise ValueError("Pass the allocation-validated shared feature extractor")
        self.extractor = extractor
        self.controller = ExactQuotaController(28, high_frames, 8, 16,
                                               EmpiricalThreshold(validation_scores), seed=seed)
        self.setup_seconds = time.perf_counter() - started

    def choose(self, state, packet):
        torch = self.extractor.torch
        if self.kind == "future_gain":
            vector = self.extractor(state.history, packet.vector_field, packet.coarse_clean,
                                    frame_index=state.generated_frames, prediction_frames=28)
            score = self.router.predict(vector.cpu().tolist())
        else:
            with torch.no_grad():
                last, previous = state.history[0, -1].float(), state.history[0, -2].float()
                difference = (last - previous if self.kind == "motion" else
                              packet.coarse_clean[0, 0].float() - (2 * last - previous))
                score = float(torch.sqrt(difference.square().mean(dim=0) + EPSILON).mean())
        return self.controller.choose(score)


def generate_video(adapter, observed_prefix, noise_bank, *, schedule=None, policy=None):
    """Prepared CPU prefix to decoded CPU RGB, with every online cost included.

    Exactly one fixed schedule or fresh OnlineAllocation is required. Artifact
    hashing, serialization and quality metrics belong outside this timing scope.
    No extra branches, hidden teacher/GT or forward probes are performed.
    """
    if (schedule is None) == (policy is None):
        raise ValueError("Supply exactly one schedule or online policy")
    if schedule is not None:
        if (len(schedule) != 28 or any(isinstance(n, bool) or not isinstance(n, int)
                                     or not 1 <= n <= 1000 for n in schedule)):
            raise ValueError("Expected 28 valid fixed per-frame NFE values")
    elif not isinstance(policy, OnlineAllocation) or policy.controller.frame_index != 0:
        raise ValueError("Online policy must be fresh for this trajectory")
    torch = adapter.torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    before = adapter.denoiser_calls
    encodes, decodes = adapter.vae_encode_calls, adapter.vae_decode_calls
    state = adapter.encode_observed_prefix(observed_prefix)
    trace, noises = [], []
    for frame in range(28):
        noise = noise_bank.frame(frame)
        noises.append(noise)
        packet = adapter.first_evaluation(state, noise,
                                          noise_id=noise_bank.identity.frame_hash(frame))
        if policy is None:
            nfe, decision = schedule[frame], {"frame_index": frame, "decision_reason": "fixed_schedule"}
        else:
            choice = policy.choose(state, packet)
            nfe, decision = choice.nfe_requested, asdict(choice)
        _, state, cost = adapter.finish_frame(state, packet, nfe)
        trace.append({**decision, "nfe": nfe, "actual_calls": cost.denoiser_calls,
                      "probe_extra_calls": cost.probe_extra_calls, "cache_extra_calls": cost.cache_extra_calls})
    rgb = adapter.decode_generated_frames(state, include_prefix=True).cpu()
    torch.cuda.synchronize()
    rollout_seconds = time.perf_counter() - started
    setup_seconds = policy.setup_seconds if policy is not None else 0.0
    seconds = rollout_seconds + setup_seconds
    calls = adapter.denoiser_calls - before
    if policy is not None:
        policy.controller.assert_complete(calls)
    if calls != sum(row["nfe"] for row in trace) or any(row["nfe"] != row["actual_calls"]
            or row["probe_extra_calls"] or row["cache_extra_calls"] for row in trace):
        raise AssertionError("Actual NFE differs from selected allocation")
    if not torch.isfinite(rgb).all():
        raise AssertionError("Generated video contains nonfinite values")
    return {"rgb_full": rgb, "latents": state.history.cpu(), "trace": trace,
            "noise_hashes": [tensor_sha256(noise) for noise in noises],
            "calls": calls, "seconds": seconds, "rollout_seconds": rollout_seconds,
            "per_video_controller_setup_seconds": setup_seconds,
            "vae_encode_calls": adapter.vae_encode_calls - encodes,
            "vae_decode_calls": adapter.vae_decode_calls - decodes,
            "controller_calls": 28 if policy is not None else 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "latency_scope": "prepared CPU prefix through CPU decoded video, includes noise and online controller"}
