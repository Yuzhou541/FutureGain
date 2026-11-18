# FutureGain: Future-Aware Compute Allocation for Autoregressive Video Prediction

FutureGain allocates the denoising computation of a frozen autoregressive video
predictor according to its estimated benefit for subsequent predictions. A small
ridge regressor predicts the signed effect of refining the current frame. An
online controller then selects low or high denoising effort while enforcing an
exact total budget.

This repository contains the inference and calibration core, frozen lightweight
H0/H3 policies, evaluation utilities, auxiliary adapters, and tests. Model
weights and datasets are obtained separately from their original providers.

## Method

1. **Measure computation value offline.** Starting from the same generated
   history and matched per-frame noise, increase only the current frame's
   denoising budget. Continue both branches with the same fixed policy and
   compare their short-horizon deviations from a shared high-compute reference.
2. **Fit a small gain predictor.** Train a ridge regressor on features of the
   available history and a reused initial denoiser evaluation. The video model
   and encoder remain frozen.
3. **Allocate computation online.** Select the current frame's denoising budget
   using the predicted gain, a frozen validation-score distribution, and the
   remaining upgrade quota. Generate one trajectory without future observations.

The supplied policies use a frozen FAR-B predictor on BAIR: two observed native
64 x 64 frames and 28 predicted frames. `future_gain` uses a three-frame lookahead
label (H3); `current_gain` uses a current-frame label (H0). Both use the same
12-dimensional feature map. Exactly 14 predicted frames receive 8 evaluations
and 14 receive 16, totaling **336 denoiser evaluations** per trajectory.

Agreement with a high-compute model reference and accuracy against the realized
future are distinct objectives. The existing experiments do not establish a
stable quality-latency advantage or a consistent advantage from future-window
supervision. A single inference run provides an output and diagnostic timing;
it is not a controlled benchmark.

## Reproduction status

The frozen H0/H3 core has passed a bounded relocated inference check on one
BAIR context and seed. Four trajectories covered both policies on the original
clip and on a clip with an identical observed prefix and replaced future frames.
Native latent/RGB tensors, all 28 noise hashes, and complete allocation traces
matched exactly. A separately rebuilt virtual environment on the same existing
infrastructure also reproduced both policies exactly on that context and seed.
These checks establish the stated numerical smoke tests; they do not establish
a full benchmark reproduction or installation on an arbitrary new machine.
The GitHub packaging adds documentation and scalar workflow tools; it has not
received a new GPU inference run.

## Requirements

The frozen inference entry point preserves the runtime identity used to export
these policies:

| Component | Required value |
| --- | --- |
| Operating environment | Linux, an owned running Slurm compute allocation |
| Python | 3.10.12 |
| PyTorch | 2.5.0+cu124 |
| CUDA reported by PyTorch | 12.4 |
| GPU | NVIDIA RTX A5000, one visible device |
| Diffusers | 0.32.2 |
| Accelerate | 1.4.0 |
| Decord | 0.6.0 |
| Predictor arithmetic | FP32 transformer parameters with BF16 autocast |
| Codec arithmetic | BF16 |

The main frozen entry point checks this runtime identity. The Slurm `squeue`
and `scontrol` commands must be available to verify job ownership and compute
node membership. Other adapters have separate requirements and validation gates.

## Installation

Use Python 3.10.12 in your Linux compute environment:

```sh
git clone https://github.com/Yuzhou541/FutureGain.git
cd FutureGain
python -m pip install torch==2.5.0 torchvision==0.20.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install exposes the `future_gain` package and its metadata-only
`future-gain` CLI. Run `python -B infer.py --help` for inference arguments.
No model or dataset is loaded by that help command.

## External predictor and assets

Use the original [FAR repository](https://github.com/showlab/FAR) at commit
`e59536ac30e5e3fa93db756fccfd422420a65ff6`. Keep this checkout outside FutureGain
and keep it clean. Download model files into a separate asset directory.

The commands below use paths supplied explicitly by the person running the
experiment. Set these variables to existing approved storage locations:

```sh
: "${FG_FAR_ROOT:?Set an external destination for the pinned FAR checkout}"
: "${FG_ASSET_ROOT:?Set an external model-asset directory}"

git clone https://github.com/showlab/FAR.git "$FG_FAR_ROOT"
git -C "$FG_FAR_ROOT" checkout e59536ac30e5e3fa93db756fccfd422420a65ff6

```

Obtain these files from [the original FAR model repository](https://huggingface.co/guyuchao/FAR_Models):

| File | SHA-256 |
| --- | --- |
| `short_video_prediction/FAR_B_BAIR_Uncond64-1983191b.pth` | `1983191bec540d068440e1566231c8068a9e3a4abec49dfadb90be2cc118240b` |
| `dcae/DCAE_BAIR_Res64-1a8547fb.pth` | `1a8547fb534045fd5e7ed620707c9a9ffa3a29a00b8e7853836895ab2f6ee38f` |

For example, with `huggingface_hub` available:

```sh
python - <<'PYDOWNLOAD'
import os
from huggingface_hub import hf_hub_download
for filename in (
    "short_video_prediction/FAR_B_BAIR_Uncond64-1983191b.pth",
    "dcae/DCAE_BAIR_Res64-1a8547fb.pth",
):
    hf_hub_download(
        repo_id="guyuchao/FAR_Models",
        filename=filename,
        local_dir=os.environ["FG_ASSET_ROOT"],
    )
PYDOWNLOAD

export FG_TRANSFORMER="$FG_ASSET_ROOT/short_video_prediction/FAR_B_BAIR_Uncond64-1983191b.pth"
export FG_CODEC="$FG_ASSET_ROOT/dcae/DCAE_BAIR_Res64-1a8547fb.pth"
```

The dataset identity is [guyuchao/BAIR](https://huggingface.co/datasets/guyuchao/BAIR)
at revision `776ad3b7d6591b06a01b8f91b1d89496445db77c`. Prepare an original native
64 x 64 RGB clip and retain its original logical ID, such as `train/10147`.
The entry point requests only frames 0 and 1 from Decord. It does not resize,
crop, or decode the future frames for conditioning.

## Run one trajectory

Run from the FutureGain repository in the required allocation. `FG_NATIVE_CLIP`
must point to the prepared clip. `FG_ORIGINAL_VIDEO_ID` must name that clip's
original dataset identity. Create an approved cache directory and an output
parent outside both source checkouts. `FG_NEW_OUTPUT` must not already exist.

```sh
: "${FG_NATIVE_CLIP:?Set the original native BAIR clip path}"
: "${FG_ORIGINAL_VIDEO_ID:?Set the original train/<id> or test/<id>}"
: "${FG_APPROVED_CACHE:?Set an existing owned external cache directory}"
: "${FG_NEW_OUTPUT:?Set a new external output directory}"

FG_NATIVE_CLIP_SHA256=$(sha256sum "$FG_NATIVE_CLIP" | cut -d ' ' -f 1)

python -B infer.py \
  --far-root "$FG_FAR_ROOT" \
  --transformer-checkpoint "$FG_TRANSFORMER" \
  --codec-checkpoint "$FG_CODEC" \
  --input-clip "$FG_NATIVE_CLIP" \
  --clip-sha256 "$FG_NATIVE_CLIP_SHA256" \
  --cache-dir "$FG_APPROVED_CACHE" \
  --output "$FG_NEW_OUTPUT" \
  --dataset-revision 776ad3b7d6591b06a01b8f91b1d89496445db77c \
  --original-video-id "$FG_ORIGINAL_VIDEO_ID" \
  --trajectory-seed 303 \
  --method future_gain
```

Select `--method current_gain` for the inherited H0 policy. Use the same original
video identity and seed to match per-frame noise between policies. File paths,
file checksums, and the selected method do not enter that noise identity.

The entry point configures its supplied cache and offline settings before model
imports, so all external assets must already be available. It verifies the FAR
checkout and checkpoint hashes, consumes the initial denoiser evaluation once,
and checks the actual frame count and 336-call budget.

### Outputs

| File | Contents |
| --- | --- |
| `native.pt` | Native BF16 RGB and latent tensors, complete trace, noise hashes, measured costs |
| `trace.json` | All 28 frame decisions and their actual budgets |
| `metadata.json` | Runtime, source, checkpoint, policy, input, and noise identities |
| `completion.json` | Output hashes and successful call-count closure |
| `execution.started.json` / `execution.finished.json` | Execution state, counters, elapsed time, and any recorded failure |

The full RGB tensor has shape `[1, 30, 3, 64, 64]`; the latent tensor has shape
`[1, 30, 32, 8, 8]`. The first two displayed frames are codec reconstructions of
the input prefix. The remaining 28 frames are predictions. `native.pt` is the
lossless tensor artifact; this entry point does not export a display video.

## Features

The 12 features comprise mean and 90th-percentile token norms for four tensors
(first history difference, second history difference, initial vector field,
and coarse prediction minus latest latent), one global direction cosine,
two history-validity masks, and normalized prediction position.

For the cosine, both the coarse-prediction displacement and the last observed
history displacement are flattened over all channel and spatial dimensions.
Their inner product is divided by the product of their L2 norms plus `1e-8`.
The calculation uses FP32, missing history yields zero, and a zero displacement
also yields zero. The regressor receives neither true future frames nor its
supervision horizon or remaining budget. The controller handles the budget.

## Code organization

- `infer.py`: frozen H0/H3 inference entry point.
- `release_manifest.json`: hashes of the released files, checked by inference.
- `configs/`: metadata configuration examples.
- `policy.json`: inherited lightweight regressors and validation distributions.
- `src/future_gain/adapters/`: predictor interfaces and sampling state.
- `src/future_gain/calibration/`: features, intervention labels, and ridge fitting.
- `src/future_gain/sampling/`: exact-quota control and matched noise identities.
- `src/future_gain/evaluation/` and `metrics/`: frame and video metric utilities.
- `tools/fit_router.py` and `tools/aggregate_metrics.py`: scalar workflow entry points.
- `docs/workflows.md`: collection, fitting, aggregation, and auxiliary-adapter details.
- `tools/export_policy.py`: checked projection of original fitting artifacts.
- `tests/`: lightweight source and algorithm checks.

## Calibration and exported policies

`policy.json` stores the existing H0/H3 ridge coefficients, training-set feature
standardization, and 1,792 validation-score values per policy. Running inference
does not fit a head, update a threshold distribution, or train the video model.

`tools/export_policy.py` reproduces this exact policy projection from the
original small fitting artifacts. It checks each artifact hash before export:

```sh
python -B tools/export_policy.py \
  --fit-dir "$FG_ORIGINAL_SMALL_FIT_JSON_DIR" \
  --output "$FG_NEW_EXPORT_JSON"
```

The required files are `fit_manifest.json`, `protocol.json`, `routers.json`, and
`validation_scores.json`. This exporter requires the original frozen artifacts;
it is not a trainer for arbitrary new calibration data. The supplied
`policy.json` is sufficient for the inference command above.

For a new calibration experiment, keep video identities disjoint across
fitting, validation, and testing; bind the generator, precision, per-frame noise,
continuation policy, and reference trajectory in the experiment protocol. The
core calibration modules implement signed intervention labels and ridge
fitting. A new fitted head requires its own completed calibration and validation
artifacts before deployment.

### Research workflow tools

The tools below expose the original scalar fitting and metric-aggregation
routines without requiring the original experiment storage layout. Input
schemas and paired-rollout collection APIs are documented in
[docs/workflows.md](docs/workflows.md).

```sh
# Fit an H3 regressor from completed, video-disjoint calibration records.
python -B tools/fit_router.py \
  --input "$FG_CALIBRATION_ROWS" --lookahead 3 \
  --output "$FG_NEW_ROUTER_DIRECTORY"

# Aggregate already computed frame metrics by video, trajectory, and frame.
python -B tools/aggregate_metrics.py \
  --input "$FG_FRAME_METRICS" --protocol canonical_mean \
  --output "$FG_NEW_METRIC_SUMMARY"
```

Fitting runs inside an owned CPU Slurm allocation with NumPy available.
Aggregation does not generate videos or run metric models. Newly fitted research
routers are kept separate from the frozen policy verified by `infer.py`.
The source also includes an FP32 FAR adapter with a separate V100 identity;
this does not change the BF16/A5000 inference command above.

### Auxiliary MCVD code

The MCVD adapter and its asset/native-verification helpers use a separate
environment described by `requirements-mcvd.txt`. Set `FG_PROJECT_ROOT` to an
explicit approved project-storage directory before using those helpers. Run a
new native verification gate with the required external source, checkpoint,
configuration, and CUDA environment; see [docs/workflows.md](docs/workflows.md).
This auxiliary path does not provide a calibrated MCVD FutureGain policy.

## Lightweight checks

Run the test suite from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s tests -p 'test*.py' -v
```

Lightweight tests cover source and policy identities, prefix-only frame
requests, logical noise keys, quota conservation, calibration and aggregation
schemas, and allocation rejection conditions. Tests requiring real tensor or
model execution retain their compute-environment gates. Source-level tests
alone do not establish numerical inference equality.

## Attribution

FutureGain uses the frozen predictor and codec from
[FAR](https://github.com/showlab/FAR). Model code, checkpoints, datasets, and any
other external components retain their original licenses and access terms.
See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for attribution and licensing
scope.
