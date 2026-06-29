# Spec 077 User Guide - Minimap Deconstruction Engine

This guide is the operator path for the current Spec 077 implementation. It is Windows PowerShell first and assumes commands are run from the repo root `I:\parp\parp-tools` unless a command sets its own working directory.

## Current Proof Level

- Phases 1-6 are code-complete for contracts, synthetic tests, and smoke-testable CLIs.
- Real-data proofs are still pending for T021, T029, T034, and T038.
- The normal lane is analytic only for MVP. Do not train a normal model unless T043/T044 are explicitly reopened.
- The C# one-object capture lane is still deferred. The first object-library proof can use staged capture artifacts or the synthetic pytest e2e.

## One-Time Setup

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv sync
```

## Validation Commands

Run these after changing Spec 077 code.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools"
dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug
dotnet test "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug --filter ObjectLibraryContractsTests
```

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run pytest tests/test_object_library.py tests/test_object_library_e2e.py tests/test_teacher_prior.py tests/test_height_only_prior.py tests/test_terrain_augment.py tests/test_inference_object.py tests/test_height_to_normal.py -q
```

## Stage A - Object Library

Enumerate capture jobs from an existing V18 dataset store.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/enumerate_object_capture_jobs.py --dataset-dir "..\output\datasets\v18" --build "3_3_5_12340" --include-modf --output "..\output\datasets\object-library\jobs_3_3_5_12340.jsonl"
```

Build a library from staged capture artifacts. Until the C# capture lane lands, `--captures-dir` can point at a manual or synthetic flat directory containing `<variant_id>_image.png`, `<variant_id>_mask.png`, and optional `<variant_id>_pose.json`.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/build_object_library.py --jobs "..\output\datasets\object-library\jobs_3_3_5_12340.jsonl" --captures-dir "..\output\datasets\object-library\captures_3_3_5_12340" --output-root "..\output\datasets\object-library" --run-name "smoke_3_3_5_12340" --target-size 128
```

Review the library.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/review_object_library.py --library "..\output\datasets\object-library\smoke_3_3_5_12340.zarr" --output-dir "..\output\analysis\object-library\smoke_3_3_5_12340"
```

Open `wow-viewer/output/analysis/object-library/smoke_3_3_5_12340/index.html` and check entry counts, capture statuses, object previews, and missing-artifact `not_attempted` rows.

## Stage B - Teacher Prior

Build or reuse the V18 curation manifest first. This writes
`wow-viewer/output/datasets/v18/curation/v18_focus_terrain_v1/kept_tiles.parquet`.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/build_v18_curation_manifest.py
```

Build ADT-backed teacher priors from the 0.5.3 and 3.3.5 V18 stores, filtered by the curation manifest. The default teacher mask priority is now `object_precise_mask`, then `object_filtered_mask`, then `object_mask`; pass `--mask-priority` only for ablation/comparison runs.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/build_teacher_prior_dataset.py --v18-path "..\output\datasets\v18\0_5_3_3368.zarr" --output-root "..\output\datasets\teacher-prior" --curation-manifest "..\output\datasets\v18\curation\v18_focus_terrain_v1"
uv run python scripts/build_teacher_prior_dataset.py --v18-path "..\output\datasets\v18\3_3_5_12340.zarr" --output-root "..\output\datasets\teacher-prior" --curation-manifest "..\output\datasets\v18\curation\v18_focus_terrain_v1"
```

Review the priors.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/review_teacher_prior_dataset.py --library "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --output-dir "..\output\analysis\teacher-prior\3_3_5_12340" --max-tiles 16 --prefer-mask-source object_precise_mask
```

Open `wow-viewer/output/analysis/teacher-prior/3_3_5_12340/index.html` and check raw minimap, teacher mask, suppressed prior, and mask-source counts.

To diagnose a specific original tile ID and compare the teacher mask against the source V18 masks, include `--tile-id` and `--v18-path`:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/review_teacher_prior_dataset.py --library "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --output-dir "..\output\analysis\teacher-prior\3_3_5_12340_tile54" --v18-path "..\output\datasets\v18\3_3_5_12340.zarr" --tile-id 54 --max-tiles 1
```

The targeted contact sheet renders raw minimap, teacher mask, `object_precise_mask`, `object_filtered_mask`, `object_mask`, raw+mask overlay, suppressed prior, and changed-pixel diff.

Audit whether ADT-derived teacher masks are actually visible in the baked minimap. This writes `visibility_audit.parquet`, `summary.json`, and a second-stage `kept_tiles.parquet` that can be used as the trainer curation manifest.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/audit_teacher_prior_visibility.py --library "..\output\datasets\teacher-prior\0_5_3_3368.zarr" --output-dir "..\output\analysis\teacher-prior\visibility-audit\0_5_3_3368"
uv run python scripts/audit_teacher_prior_visibility.py --library "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --output-dir "..\output\analysis\teacher-prior\visibility-audit\3_3_5_12340"
```

For the two-build trainer, write one combined visibility manifest:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/audit_teacher_prior_visibility.py --library "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --output-dir "..\output\analysis\teacher-prior\visibility-audit\two_build"
```

Tiles bucketed as `weak` or `tiny` are rejected by the generated `kept_tiles.parquet`. These are candidates where ADT placement masks do not appear strongly represented in the minimap and should not silently train the height model.

## Stage C - Height-Only Training

CPU smoke proof after rebuilding priors and writing the combined visibility-audit manifest.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\smoke_visibility_audited_two_build" --run-name "smoke_visibility_audited_two_build" --steps 4 --val-steps 1 --batch-size 1 --device cpu --max-tiles 32 --normal-guidance-weight 0.10 --no-amp --no-compile
```

For the cleaner second-stage route, train against visibility-audited rows by passing the combined visibility-audit directory as `--curation-manifest`.

CUDA training with the V18 performance stack enabled.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\cuda_visibility_audited_two_build" --run-name "cuda_visibility_audited_two_build" --epochs 40 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --autotune-batch-size --target-vram-gb 12 --num-workers 0 --no-persistent-workers
```

If validation plateaus while train loss keeps falling, resume with a lower LR and validation-driven plateau scheduling:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\cuda_visibility_audited_two_build" --run-name "cuda_visibility_audited_two_build" --resume-checkpoint "models\spec077\height-only\cuda_visibility_audited_two_build\cuda_visibility_audited_two_build_latest.pt" --epochs 260 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --hard-error-weight 0.05 --hard-error-power 1.0 --hard-error-max-multiplier 4.0 --resume-learning-rate 3e-5 --lr-plateau-patience 6 --lr-plateau-factor 0.5 --min-learning-rate 1e-6 --num-workers 0 --no-persistent-workers
```

### Validation plateau diagnosis and shadow-safe training

If validation loss plateaus around 0.54-0.56 while train loss keeps falling, first separate two failure modes: spatial leakage in the split, and missing signal in the input. Do **not** use D4 rotation/flip augmentation as the default fix for baked minimap RGB. Terrain height is a scalar field, but the minimap appearance is not: baked lighting and terrain shadows have a fixed world direction, so rotated/flipped minimaps create physically inconsistent examples.

- `--augment` now uses `--augment-policy shadow-safe` by default, which is identity-only. It preserves the train/validation guard behavior but does not rotate or flip minimap RGB.
- `--augment --augment-policy d4` remains available only as an explicit ablation for orientation-free inputs or geometry-only tests.
- `--split-mode map` holds out entire maps so val tiles are never spatial neighbors of train tiles. This is a **diagnostic**: if val loss jumps well above the random-split plateau, the old flat split was masking spatial leakage. If it stays near the plateau, the ceiling is genuinely shape-difficulty, not leakage. Falls back to random if no map metadata is present.

D4 ablation command (not canonical for shadow-bearing minimaps):

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\diag_d4_aug_two_build" --run-name "diag_d4_aug_two_build" --epochs 40 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --hard-error-weight 0.05 --hard-error-power 1.0 --hard-error-max-multiplier 4.0 --augment --augment-policy d4 --augment-seed 0 --autotune-batch-size --target-vram-gb 12 --num-workers 0 --no-persistent-workers
```

One-off map-grouped split diagnostic (short run, just to measure the leakage contribution):

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\diag_map_split" --run-name "diag_map_split" --epochs 40 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --split-mode map --autotune-batch-size --target-vram-gb 12 --num-workers 0 --no-persistent-workers
```

Compare `diag_map_split` val loss against the random-split plateau. A large jump confirms spatial leakage; a small change confirms the ceiling is shape-difficulty and the next lever is better signal/refinement, not D4 minimap augmentation.

### Albedo guidance channel

When train and validation loss are close but validation still plateaus, the ceiling is **signal-limited**: the model has capacity to learn but the suppressed-minimap RGB alone does not carry enough information to disambiguate terrain shapes. The `--albedo` flag addresses this by appending a 3-channel texture-identity guidance signal to the model input, widening it from 3 to 6 channels:

- **What it is**: `albedo_rgb` is generated from V18 `alpha_256` plus `mcly_texture_ids` / `mcly_layer_mask` via `compositor.composite_texture_identity_albedo`. It encodes *which terrain texture identity* each pixel belongs to using stable pseudo-colours per MCLY texture ID. It is not decoded BLP colour, but it should not collapse every tile to the same cyan placeholder.
- **Why it helps**: different terrain textures (grass, rock, sand, snow) have different typical height profiles. The albedo channel gives the model a direct hint about texture boundaries, which correlate with height discontinuities and ridge lines that are hard to infer from RGB alone.
- **No teacher-prior rebuild needed**: albedo comes from existing V18 `alpha_256`, but generate a sidecar store before a real CUDA run so the training loop consumes fixed reviewed inputs instead of lazily compositing every sample.
- **Shadow-safe by default**: the albedo is a plain image and can be transformed consistently, but the concatenated minimap RGB still carries fixed-direction lighting/shadows. The canonical run therefore leaves geometric augmentation off; use `--augment-policy d4` only for explicit ablations.
- **Backward-compatible**: without `--albedo`, the model uses 3 channels as before and existing checkpoints remain valid. The `V18HeightModel` now accepts an `in_channels` parameter (default 3).

Build the albedo sidecars first:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/build_albedo_dataset.py --v18-path "..\output\datasets\v18\0_5_3_3368.zarr" --output-root "..\output\datasets\albedo" --overwrite
uv run python scripts/build_albedo_dataset.py --v18-path "..\output\datasets\v18\3_3_5_12340.zarr" --output-root "..\output\datasets\albedo" --overwrite
```

Each output is `wow-viewer/output/datasets/albedo/<build>.zarr` with `albedo_rgb_256`, `tiles.parquet`, and `metadata.json`. The trainer can still fall back to lazy V18 `alpha_256`/MCLY compositing if `--albedo-path` is omitted, but full runs should pass the sidecars so previews and training use exactly the same precomputed signal. If the albedo preview is a flat cyan block, the sidecar is stale or was built by the old placeholder compositor; rebuild it with `--overwrite`.

Recommended albedo run after the `cuda_albedo_shadow_safe` plateau (start fresh; do not resume that checkpoint):

- `--albedo` still widens the model to 6 channels with precomputed texture-identity guidance.
- `--model-norm group` removes BatchNorm running-stat drift between train and eval previews.
- `--decoder-upsample nearest` avoids the legacy bilinear decoder path that produced grid-like artifacts.
- The run is still height-only: one input tensor, one `height_257` output, no normals/object/liquid heads.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --albedo-path "..\output\datasets\albedo\0_5_3_3368.zarr" "..\output\datasets\albedo\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\cuda_albedo_group_nearest" --run-name "cuda_albedo_group_nearest" --epochs 240 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --hard-error-weight 0.05 --hard-error-power 1.0 --hard-error-max-multiplier 4.0 --albedo --model-norm group --decoder-upsample nearest --autotune-batch-size --target-vram-gb 12 --num-workers 0 --no-persistent-workers
```

Review `*_metrics.json`, `*_latest.pt`, `*_best.pt`, `*_model.pt`, `*_preview.png`, and per-epoch `*_validation_previews/epoch_####.png` in the output directory. With `--albedo`, both final and validation preview grids include an `albedo input` panel so you can verify the exact 6-channel signal being fed. The model predicts only `height_257`; it does not predict normals, liquids, or objects. `--normal-guidance-weight` is an auxiliary training loss: it derives normals from predicted height and compares them to V18 `normal_xyz` for sharper/faster height convergence, without adding a normal output head. `--hard-error-weight` is training-only online hard-pixel mining from detached absolute height residuals; validation abs-error remains a held-out diagnostic map and is not backpropagated. `--augment`, `--augment-policy`, `--split-mode`, `--albedo`, `--model-norm`, and `--decoder-upsample` are recorded in `*_metrics.json` under `augment`, `augment_policy`, `augment_transforms`, `augment_seed`, `split_mode`, `albedo`, `model_in_channels`, `model_norm`, and `decoder_upsample`. `--albedo-path` is recorded under `albedo_paths`.

### MCLY-guided small-detail refinement (follow-up, not in the base run)

If the albedo run improves broad terrain shape but still misses small detail, do not add extra heads to the base height model. The next bounded lane should be a separate residual refinement pass:

- Freeze the accepted base height checkpoint.
- Generate base predictions for the curated V18/teacher-prior rows.
- Derive low/medium/high detail masks from MCLY layer activity, active-layer counts, and `alpha_256` transition gradients. These are curation/loss masks, not new prediction targets.
- Train a tiny model that predicts one signal only: `height_delta_257 = height_truth - base_height_pred`, weighted toward the MCLY high/transition-detail mask.
- Compose inference as `height_refined = base_height + detail_mask * height_delta`; derive normals analytically afterward.

This mirrors the older low/medium/high-detail idea but lets MCLY choose the detail regions naturally. It also preserves the repo rule that every model predicts one residual signal and trains independently.

For full training, leave `--max-tiles` unset or set it to `0`. Use `--max-tiles` only for smoke runs. `--steps` is only a smoke/resume cap; use `--epochs` for real runs. `--val-steps 0` means validate the full deterministic validation split each epoch.

### RunPod cloud training package

For cloud training, do not copy staged game clients, MPQs, CASC data, or asset trees. Package only the derived training artifacts and Python training code:

- teacher-prior Zarr stores for `0_5_3_3368` and `3_3_5_12340`
- slim V18 target Zarr stores containing only `height_257`, `object_filtered_mask`, `normal_xyz`, and `normal_mask`
- precomputed albedo sidecar Zarr stores
- visibility-audit `kept_tiles.parquet` manifest
- `data-harvester/src`, focused scripts, RunPod helper scripts, and manifest

Validate required inputs without copying data:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/package_spec077_runpod.py --validate-only
```

Build the default RunPod bundle under `wow-viewer/output/cloud-packages/`:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/package_spec077_runpod.py --archive-format tar --overwrite
```

To create the RunPod training Pod from the same local command, set `RUNPOD_API_KEY` and run the setup helper:

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
$env:RUNPOD_API_KEY = "<your-runpod-api-key>"
uv run python scripts/setup_spec077_runpod.py --overwrite-package
```

Default setup choices:

- GPU: `NVIDIA RTX 4000 Ada Generation` only by default; use `--gpu-fallback` or `--gpu-types` only if you intentionally choose another cost target
- Datacenter: `auto`, resolved from RunPod availability candidates before creating a volume or Pod
- RAM request: `--min-ram-gb 50`
- vCPU request: `--min-vcpu 8`
- Storage: RunPod network volume `--network-volume-gb 150` mounted at `/workspace`
- Container disk: `--container-disk-gb 50`
- Run name: `cuda_albedo_group_nearest`
- Transfer: Pod bootstrap waits in `/workspace` with `runpodctl receive <code>` and the local setup helper starts `runpodctl send <bundle.tar> --code <same-code>` when `runpodctl` is on `PATH`

Dry-run the API payload without creating a Pod:

```powershell
uv run python scripts/setup_spec077_runpod.py --dry-run --overwrite-package
```

The helper creates the network volume in the selected concrete datacenter, then creates the Pod against the same datacenter. If RunPod returns `There are no instances currently available`, the helper deletes that unused newly-created volume and tries the next candidate. Existing volumes can be attached with `--network-volume-id`, but then pass the matching concrete `--data-center`; `auto` is rejected for network-volume Pod payloads.

Use a Pod-local persistent volume only for throwaway runs:

```powershell
uv run python scripts/setup_spec077_runpod.py --overwrite-package --no-network-volume --volume-gb 150
```

To disable the bootstrap transfer command and do all upload/setup manually, pass `--no-auto-transfer`. The Pod payload then omits `dockerStartCmd`.

Important RunPod API limitation: `RUNPOD_API_KEY` can create Pods and network volumes, but direct network-volume file upload uses RunPod's separate S3-compatible API credentials. The default setup avoids separate S3 credentials by pairing Pod-side `runpodctl receive` with local `runpodctl send`. If `runpodctl` is not installed locally, the setup manifest and final console output print the exact manual send command. `rsync` over SSH remains an alternate transfer path after SSH is configured.

The generated bundle includes `README_RunPod.md`, `manifest.json`, `requirements-runpod.txt`, and these pod-side helpers:

- `runpod/install_deps.sh`
- `runpod/verify_bundle.sh`
- `runpod/smoke_spec077.sh`
- `runpod/train_spec077.sh`

Recommended RunPod route for the first cloud proof:

- Use a PyTorch CUDA Pod template with Python 3.11+ and persistent `/workspace` storage.
- Use the default network volume so the dataset and checkpoints survive Pod termination and can be reused across Pods.
- Let the setup helper perform the `runpodctl send`/`receive` handoff when possible.
- If `--no-auto-start-training` is used, SSH into the Pod after transfer/unpack and run `bash runpod/smoke_spec077.sh`, then `bash runpod/train_spec077.sh` manually.

RunPod Flash and Serverless notes from the docs:

- Flash is convenient for launching local Python functions on RunPod Serverless GPUs, and it can attach a network volume mounted at `/runpod-volume/`.
- Serverless workers can also attach network volumes, but require a worker image/handler or GitHub integration.
- For this first week-long training run, a Pod + network volume is simpler than a custom Serverless worker. Flash/Serverless can be added later once the bundle is resident on a network volume and the training command is stable.
- RunPod API MCP can manage Pods, templates, endpoints, and volumes if configured with a `RUNPOD_API_KEY`; keep that key outside the repo and outside the bundle.

## Stage D - ADT-Free Prior

This stage consumes a predicted object mask. T034, the learned object-mask producer, is not implemented yet, so use a synthetic or external predicted-mask NPZ/Zarr only for pipeline proof.

```powershell
Set-Location -LiteralPath "I:\parp\parp-tools\wow-viewer\data-harvester"
uv run python scripts/build_adt_free_prior.py --v18-path "..\output\datasets\v18\3_3_5_12340.zarr" --predicted-mask "..\output\datasets\object-masks\predicted_3_3_5_12340.npz" --output-root "..\output\datasets\adt-free-prior" --max-tiles 64
```

Use the resulting `wow-viewer/output/datasets/adt-free-prior/3_3_5_12340.zarr` as the processed-prior input for downstream height inference once an inference entrypoint exists.

## Stage E - Analytic Normals

Normals are derived from predicted height through `harvester.height_to_normal.analytic_normals_from_height`. This is deterministic and separate from the height model. If visual quality later requires refinement, add a new normal-only dataset/trainer instead of adding a normal head to the height model.

## Expected Outputs

- Object library: `wow-viewer/output/datasets/object-library/<run>.zarr`, `assets.parquet`, `index.parquet`, `capture_rgb`, `capture_mask`.
- Object review: `wow-viewer/output/analysis/object-library/<run>/index.html`.
- Teacher prior: `wow-viewer/output/datasets/teacher-prior/<build>.zarr`, `tiles.parquet`, `raw_minimap_rgb_256`, `teacher_object_mask_256`, `teacher_object_confidence_256`, `processed_minimap_prior_256`.
- Teacher review: `wow-viewer/output/analysis/teacher-prior/<build>/index.html`.
- Height training: `models/spec077/height-only/<run>/*_metrics.json`, `*_latest.pt`, `*_best.pt`, `*_model.pt`, `*_preview.png`.
- Albedo guidance: `wow-viewer/output/datasets/albedo/<build>.zarr` with `albedo_rgb_256`, `tiles.parquet`, and `metadata.json`.
- ADT-free prior: `wow-viewer/output/datasets/adt-free-prior/<build>.zarr`.

## Troubleshooting

- If `No index.parquet` appears, the input is not a full V18 store.
- If `No minimap_rgb array` appears, build or point to the V18 tensor-pack store before running teacher-prior commands.
- If object-library entries are all `not_attempted`, the capture artifact names or `--captures-dir` do not match the generated variant IDs.
- If CUDA runs out of memory, use `--autotune-batch-size --target-vram-gb <gb>` or lower `--batch-size`.
- If a command needs real client data, use only staged data under `I:\parp\parp-tools\output\tmp\wowarchive-clients\`; do not use legacy raw-client roots.
- If checkpoint save fails on Windows with error code `1224`, the trainer now retries atomic replacement and falls back to a timestamped `*_epoch####_step#######.pt` checkpoint. Resume from the fallback path if `*_latest.pt` did not update.
