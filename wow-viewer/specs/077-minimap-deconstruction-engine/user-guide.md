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
uv run pytest tests/test_object_library.py tests/test_object_library_e2e.py tests/test_teacher_prior.py tests/test_height_only_prior.py tests/test_inference_object.py tests/test_height_to_normal.py -q
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
uv run python scripts/train_height_only_prior.py --prior "..\output\datasets\teacher-prior\0_5_3_3368.zarr" "..\output\datasets\teacher-prior\3_3_5_12340.zarr" --v18 "..\output\datasets\v18\0_5_3_3368.zarr" "..\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build" --output-dir "models\spec077\height-only\cuda_visibility_audited_two_build" --run-name "cuda_visibility_audited_two_build" --resume-checkpoint "models\spec077\height-only\cuda_visibility_audited_two_build\cuda_visibility_audited_two_build_latest.pt" --epochs 260 --val-steps 0 --batch-size 8 --device cuda --normal-guidance-weight 0.10 --resume-learning-rate 3e-5 --lr-plateau-patience 6 --lr-plateau-factor 0.5 --min-learning-rate 1e-6 --num-workers 0 --no-persistent-workers
```

Review `*_metrics.json`, `*_latest.pt`, `*_best.pt`, `*_model.pt`, and `*_preview.png` in the output directory. The model predicts only `height_257`; it does not predict normals, liquids, or objects. `--normal-guidance-weight` is an auxiliary training loss: it derives normals from predicted height and compares them to V18 `normal_xyz` for sharper/faster height convergence, without adding a normal output head.

For full training, leave `--max-tiles` unset or set it to `0`. Use `--max-tiles` only for smoke runs. `--steps` is only a smoke/resume cap; use `--epochs` for real runs. `--val-steps 0` means validate the full deterministic validation split each epoch.

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
- ADT-free prior: `wow-viewer/output/datasets/adt-free-prior/<build>.zarr`.

## Troubleshooting

- If `No index.parquet` appears, the input is not a full V18 store.
- If `No minimap_rgb array` appears, build or point to the V18 tensor-pack store before running teacher-prior commands.
- If object-library entries are all `not_attempted`, the capture artifact names or `--captures-dir` do not match the generated variant IDs.
- If CUDA runs out of memory, use `--autotune-batch-size --target-vram-gb <gb>` or lower `--batch-size`.
- If a command needs real client data, use only staged data under `I:\parp\parp-tools\output\tmp\wowarchive-clients\`; do not use legacy raw-client roots.
- If checkpoint save fails on Windows with error code `1224`, the trainer now retries atomic replacement and falls back to a timestamped `*_epoch####_step#######.pt` checkpoint. Resume from the fallback path if `*_latest.pt` did not update.
