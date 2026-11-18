"""Only expose implemented commands; experiment stages require real GPU gates."""

import argparse
import json
from pathlib import Path

from .audit import audit_upstream, canonical_hash, local_doctor, read_config
from .sampling.budget import expected_calls, high_count, near_uniform_schedule


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "audit-upstream", "plan"])
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="These metadata-only commands never execute experiments.")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    project = args.config.resolve().parent.parent
    source = project / config["upstream"]["source_dir"]
    if args.command == "audit-upstream":
        result = audit_upstream(source)
    elif args.command == "doctor":
        result = local_doctor(config, source)
    else:
        t = config["task"]["prediction_frames"]
        low = config["sampler"]["low_nfe_candidate"]
        high = config["sampler"]["high_nfe_candidate"]
        budgets = []
        for q in config["method"]["high_fractions"]:
            m = high_count(t, q)
            calls = expected_calls(t, low, high, m)
            budgets.append({"q": q, "high_frames": m, "calls_per_trajectory": calls,
                            "matched_uniform_schedule": near_uniform_schedule(t, calls)})
        result = {
            "stage": "S0_AUDIT", "dry_run": True,
            "config_sha256": canonical_hash(config),
            "native_smoke": {"contexts": 1, "trajectories": 1, "future_frames": t,
                             "nfe": 50, "denoiser_calls": 50 * t},
            "budgets_per_trajectory": budgets,
            "estimated_gpu_hours": None,
            "estimate_blocker": "No measured throughput is bound to this configuration.",
            "verified_reusable_generation_artifacts": 0,
            "next": "Audit the allocation, acquire assets on compute, and verify the native adapter.",
            "comparison_eligible": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
