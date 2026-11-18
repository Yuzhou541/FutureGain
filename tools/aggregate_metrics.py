#!/usr/bin/env python3
"""Aggregate precomputed frame metrics with the original explicit reduction axes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from future_gain.audit import canonical_hash
from future_gain.evaluation.metrics import aggregate_frame_metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON with metric and values[context][trajectory][future_frame]")
    parser.add_argument("--output", type=Path, required=True, help="New output JSON file")
    parser.add_argument("--protocol", choices=("canonical_mean", "canonical_best_of_k", "upstream_legacy"),
                        default="canonical_mean")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    canonical_hash(data)  # Reject non-finite values before creating output.
    if data.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    result = aggregate_frame_metric(data["values"], metric=data["metric"], protocol=args.protocol)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "output": str(args.output), "protocol": args.protocol}))


if __name__ == "__main__":
    main()
