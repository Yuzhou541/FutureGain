# Calibration and evaluation APIs

The source package contains the feature extractor, paired-rollout state handling,
label function, ridge regression, exact-quota controller, and frame/FVD metric
implementations. The two tools in this document expose the scalar fitting and
aggregation stages without depending on the original experiment storage layout.
They are new packaging entry points around the original numerical routines;
the published frozen `policy.json` remains the checkpoint used by `infer.py`.

## Offline paired-rollout collection

Run model/data work only in your own Linux Slurm compute allocation. Use
`FARAdapter.first_evaluation(state, frame_noise, noise_id=...)` to obtain the
common first model evaluation, then `FeatureExtractor()(state.history,
packet.vector_field, packet.coarse_clean, frame_index=t, prediction_frames=28)`
to compute the 12 current-information features. The input history includes the
observed prefix and already completed predicted frames.

Use `FARAdapter.fork_pending(state, packet)` to create independent low/high branch
states and packets after this common evaluation. Complete the current frame with
8 and 16 evaluations respectively, then continue both branches under the same
predeclared sampling policy and matched future per-frame noise. Do not alter
which observations are available or construct an online policy from future data.
Record both branches' canonical LPIPS losses against the same complete 50-step
reference rollout. `calibration.labels.intervention_gain` accepts the current
frame and up to H subsequent frames, truncates the valid tail, and divides the
signed gain by the full 28-frame prediction length. Negative gains are retained.

The low/high branches must each own their history, cache and packet state.
Their histories diverge after the intervention. Common noise does not mean
reusing cache tensors across branches. Label horizons H0/1/3/6 use the same
feature map. A collected calibration row stores an original video identity,
trajectory seed, intervention position, the 12 feature values, and the low/high
loss arrays; each array has exactly `min(H + 1, 28 - frame_index)` entries.

## Fit a signed-gain router

`tools/fit_router.py` accepts one JSON document with these top-level fields:

```json
{
  "schema_version": 1,
  "feature_version": "history_first_eval_v1",
  "feature_names": ["the exact 12 names from FEATURE_NAMES"],
  "protocol": {
    "prediction_frames": 28,
    "low_nfe": 8,
    "high_nfe": 16,
    "label_metric": "canonical_lpips_alex",
    "reference_mode": "global_dense",
    "lookahead": 3
  },
  "calibration_train": [],
  "calibration_val": [],
  "validation_rollout_features": []
}
```

The abbreviated `feature_names` field above must be replaced by
`list(future_gain.calibration.features.FEATURE_NAMES)` in its exact order.
Both labelled split arrays contain records of this form (the feature array is
abbreviated here and must contain all 12 finite numbers):

```json
{
  "video_id": "train/123",
  "trajectory_seed": 101,
  "frame_index": 7,
  "features": [0.0],
  "low_losses": [0.2, 0.3, 0.4, 0.5],
  "high_losses": [0.1, 0.2, 0.3, 0.4]
}
```

`validation_rollout_features` contains `video_id`, `trajectory_seed`,
`frame_index`, and `features` for **all 28 generated frames** of every validation
video/seed, rather than only intervention positions. Every validation video must
use the same set of trajectory seeds, and the labelled validation rows must
exactly match their all-frame features. This preserves the original distinction
between sparse regression labels and the full-rollout threshold distribution.
Videos are disjoint between train and validation. The caller supplies source
and asset provenance and retains the original split; this tool cannot certify
that external losses or feature values were computed correctly.

```sh
# Run inside an authorized CPU Slurm allocation with NumPy available.
python -B tools/fit_router.py \
  --input "$FG_CALIBRATION_ROWS" \
  --lookahead 3 \
  --output "$FG_NEW_ROUTER_DIRECTORY"
```

The tool calls the unchanged `intervention_gain` and `fit_ridge` routines. It
fits the scaler on training videos, gives every original video equal total
weight, selects regularization from 0.01/0.1/1/10/100 by video-equal validation
RMSE, and preserves signed targets. The output `router.json` contains the fitted
`model` and the scores of all validation-frame features; `fit_report.json`
records selection, input hash, counts and the allocation identity. No generator
or metric-model call is made by fitting.

For a new research rollout, load this model through
`RidgeRouter.from_dict(payload["model"])` and pass its `validation_scores` to
`OnlineAllocation` with the declared exact quota. The released inference
entrypoint deliberately verifies the inherited frozen policy and does not load
this new research router automatically. Keep new fits and their evaluation
results separately identified.

## Aggregate precomputed frame metrics

Canonical frame scores use only future frames. Per-frame values must be computed
with the intended canonical range and backend first; aggregation does not run
LPIPS, PSNR or SSIM models. Input JSON is:

```json
{
  "schema_version": 1,
  "metric": "lpips",
  "values": [[[0.1, 0.2], [0.2, 0.3]], [[0.3, 0.4], [0.4, 0.5]]]
}
```

The axes are `[original_video][candidate_trajectory][future_frame]`. This small
example illustrates the schema only and is not a dataset benchmark.

```sh
python -B tools/aggregate_metrics.py \
  --input "$FG_FRAME_METRICS" \
  --protocol canonical_mean \
  --output "$FG_NEW_METRIC_SUMMARY"
```

The default averages frames, then trajectories, then original videos. The
explicit `canonical_best_of_k` mode chooses a complete trajectory after temporal
averaging. `upstream_legacy` preserves the original metric-specific candidate
reductions; it does not by itself reproduce decoding, encoding or FVD details.
JSON inputs and outputs must contain finite values. Perfect-match PSNR can be
positive infinity in the core API, but such a result must be handled explicitly
before writing standard JSON with this scalar tool.

`future_gain.evaluation.metrics.FrameMetrics` implements canonical frame metric
calls, `future_gain.evaluation.fvd.FVDProtocol` declares full-clip I3D input
provenance, and `future_gain.metrics.formal_streaming` preserves bounded-memory
aggregation axes. Their actual model loading and tensor work remain Slurm-only.
Do not pool candidate trajectories into contexts when computing FVD, and do not
compare a small paired study with a full many-candidate benchmark.

## Auxiliary MCVD native gate

The MCVD constructor depends on `scripts.prepare_mcvd_assets` and
`scripts.verify_mcvd_native`, both included as auxiliary native verification
tools. Before invoking either tool or the MCVD constructor, set
`FG_PROJECT_ROOT` to an explicit existing approved absolute project directory.
It defines the owned storage boundary and contains the separate
`envs/mcvd_*` environment, external source, assets, and run records; it is not
inferred from a machine-specific shared path. Asset acquisition and the native
repeat gate run only within an owned Slurm compute allocation. Invoke these
modules from the repository root with `python -m scripts.prepare_mcvd_assets`
and `python -m scripts.verify_mcvd_native`; their `--help` lists the required
explicit paths.

MCVD uses its own pinned source, checkpoint, configuration, Python environment,
CUDA toolkit and extension cache. Preserve those identities and run a new native
repeat gate before constructing the adapter; the gate binds the actual helper
file hashes, so historical verification files from another project layout are
not accepted automatically. This source packaging generalizes only the storage
root; it does not establish a new MCVD numerical reproduction or a learned
MCVD FutureGain policy. The frozen FAR inference quickstart is separate.
