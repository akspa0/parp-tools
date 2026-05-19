# ACTIVE CONTEXT — V16 + Multi-Model Architecture

## Branch: `v0.5.0-dev`

## V16 Consolidated Zarr Dataset (2026-05-16)

Single Zarr store per client build. Data flows from C# harvester via binary pipe → Python Zarr writer. No intermediate NPZ files on disk.

### Pipeline
- `WowViewer.Tool.Harvest harvest-stream` → NPZB length-prefixed binary blobs on stdout
- `build_v16_dataset.py build --build <key>` → reads pipe, writes Zarr + Parquet index + placements Parquet
- `train_v16.py --builds <keys>` → V16Dataset reads Zarr, trains V15Model arch (~27.4M params)

### New Direction: Paired Inference Output Store (2026-05-18)
- V16 spec now defines a required one-to-one inference companion store:
  - input: `wow-viewer/output/datasets/v16/<build>.zarr`
  - output: `wow-viewer/output/datasets/v16_inference/<run_name>/<build>.pred.zarr`
- Output stores must preserve `tile_id` order from `index.parquet` and emit deterministic terrain predictions per tile (`height_pred_257`, `normal_pred_xyz`, `alpha_pred_256`, `holes_pred_16`, `liquid_pred_mask_256`, `mcly_pred_logits_16x16x4x16`).
- Reconstruction routing is now explicit in spec and mapped to existing commands:
  - patch existing ADTs from inference summaries via `terrain-patch-adt`
  - route patched LK outputs into alphaWDT via `convert-lk-to-alpha`
  - convert alphaWDT outputs back to LK where needed via `convert-alpha-to-lk`

### New: V16 inference bridge to existing LK/Alpha tooling (2026-05-18)
- `wow-viewer/data-harvester/scripts/infer_v16.py` now exists and emits:
  - deterministic paired prediction stores (`<build>.pred.zarr`)
  - patch-ready per-tile `inference_summary.json` + `predicted_height_257.npy`
- The spec now reflects current repo truth: `terrain-patch-adt`, `convert-lk-to-alpha`, and `convert-alpha-to-lk` are already implemented and should be the immediate V16 inference post-processing path.

### New: V16 training contract sync + focused readiness proof (2026-05-18)
- `v16-terrain-model-spec-2026-05-16.md` now includes a training contract matrix mapping expected surfaces to concrete files (`v16_dataset.py`, `train_v16.py`, `v15_model.py`, validator, infer bridge).
- Spec drift was corrected: `src/harvester/v16_model.py` is not present; current V16 training architecture is `V15Model` in `src/harvester/v15_model.py`.
- Focused readiness validation was run with small sample sizes on staged `3_3_5_12340`:
  - command: `uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340 --train-samples 8 --val-samples 4 --batch-size 2`
  - report: `wow-viewer/output/datasets/v16/validation/3_3_5_12340.training_readiness.json`
  - result: `overall_ok=true`, `issues=0`, model forward shapes match expected V16 heads on CPU.

### New: liquid-height supervision and inference signal wiring (2026-05-18)
- V16 dataset/trainer now use liquid height directly:
  - `V16Dataset` now returns `liquid_height` tensor from Zarr.
  - `V15Model` now includes a dedicated liquid-height head (in addition to liquid-mask head).
  - `train_v16.py` now supervises both liquid mask and liquid height (liquid-height loss masked to liquid-present pixels).
- Validator and contracts were updated:
  - `validate_v16_training_ready.py` now checks `liquid_height` tensor contract and model output shape for liquid height.
  - `v16-terrain-model-spec-2026-05-16.md` now lists liquid height as an actively supervised target.
- Inference surface now carries liquid signals for downstream ADT liquid writing work:
  - `infer_v16.py` now writes `liquid_pred_height_256` into `.pred.zarr`.
  - patch-ready per-tile summaries now include `predicted_liquid_mask_256.npy` and `predicted_liquid_height_256.npy` sidecars.
- Focused proof:
  - command: `uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340 --train-samples 4 --val-samples 2 --batch-size 2`
  - result: `overall_ok=true`, `issues=0`, report confirms `trainer_uses_liquid_height=true`.

### New: trainer-side curation evidence + labeled validation overview (2026-05-18)
- `train_v16.py` now supports deterministic subset curation directly from available V16 Zarr splits:
  - `--train-max-tiles <n>`
  - `--val-max-tiles <n>`
  - `--curation-seed <seed>`
- Every run now writes dataset chain-of-evidence artifacts:
  - `models/v16/runs/<run>/evidence/curation_manifest.json`
  - `models/v16/runs/<run>/evidence/train_selection.jsonl`
  - `models/v16/runs/<run>/evidence/val_selection.jsonl`
  - `models/v16/runs/<run>/evidence/train_epoch_orders.jsonl` (deterministic per-epoch sample order)
- Validation export now writes one labeled composite:
  - `models/v16/runs/<run>/validation/epoch_XXXX/validation_overview.png`
  - includes labeled input/output panels per tile so samples are operator-readable without extra context.
- Training compile behavior is now guarded:
  - `torch.compile` is used only on CUDA by default, and can be disabled with `--no-compile`.
  - CPU-only smoke no longer fails on missing `cl.exe`.

### New: strict per-sample `has_*` loss gating fix (2026-05-18)
- `train_v16.py` no longer scales optional-head losses by batch-mean `has_*`.
- Optional heads now compute per-sample loss first, then apply masked reduction with:
  - `has_normals`, `has_alpha`, `has_holes`, `has_liquid`, `has_mcly`
- This prevents tiles missing a signal from still influencing that head via zero-filled targets.
- Focused smoke proof:
  - `uv run python scripts/train_v16.py --builds 3_3_5_12340 --epochs 1 --batch-size 2 --device cpu --train-max-tiles 8 --val-max-tiles 4 --val-interval 1 --val-snapshots 1 --run-name smoke_v16_per_sample_maskfix`
  - completed successfully with checkpoints/validation output.

### New: durable train resume for long sporadic runs (2026-05-18)
- `train_v16.py` now supports explicit resume routing:
  - `--resume-from none|auto|last|best`
  - legacy explicit `--resume-checkpoint <path>` still works and takes priority.
- Checkpoints now persist full training state, not just weights:
  - model, optimizer, scheduler, AMP scaler, `best_val_height`, and `training_log`.
- Per-run checkpoint policy is now:
  - `checkpoints/v16_last.pt` written every epoch
  - `checkpoints/v16_best.pt` updated on best `val_h`
  - `checkpoints/v16_final.pt` written at run completion
- Focused restart proof:
  - run 1: `smoke_v16_resume` with `--epochs 1`
  - run 2: same `--run-name smoke_v16_resume --epochs 2 --resume-from auto`
  - second run resumed from `v16_last.pt` at `start_epoch=2` and finished cleanly.

### New: liquid-height deferred from core V16 model (2026-05-18)
- Per user direction, current V16 terrain model no longer predicts or supervises `liquid_height`.
- `V15Model` now outputs 6 heads for V16 terrain lane:
  - `height`, `normals`, `alpha`, `holes`, `liquid_mask`, `mcly_logits`
- `train_v16.py` no longer consumes `batch["liquid_height"]` and no longer computes `loss_lqh`.
- `infer_v16.py` no longer emits `liquid_pred_height_256` or `predicted_liquid_height_256.npy`; only liquid mask remains in core path.
- `validate_v16_training_ready.py` now reports:
  - `trainer_uses_liquid_height=false`
  - consumed targets exclude `liquid_height`
- `liquid_height` stays in the V16 dataset contract for a later dedicated liquid-refinement model.
- Focused proofs:
  - readiness: `overall_ok=true`, `issues=0` on staged `3_3_5_12340`
  - smoke train run: `smoke_v16_no_liquid_height` completed (`Parameters: 27,396,026`).

### New: V16 liquid presence-mask contract hardening (2026-05-19)
- Root cause for missing Alpha liquid GT confirmed: Python derivation treated `type_mask > 0` as liquid presence, which drops valid type `0` water.
- `TerrainTileTensorPack`/NPZ now carry explicit `mh2o_presence_mask` and `mclq_presence_mask`.
- `build_v16_dataset.py` liquid derivation now prefers explicit presence masks, uses Alpha-safe `mclq_type_mask >= 0` fallback when legacy shards use `-1` sentinel, and keeps WL* only as last-resort fallback.

### New: V16 post-build signal gate + faster default write profile (2026-05-19)
- `build_v16_dataset.py build` now runs signal validation after promotion by default and writes `signal_validation.json` into each finalized `<build>.zarr`.
- Strict mode is enabled by default (`--signal-validation --signal-validation-strict`) and fails the build if required has-signal coverage or era-specific liquid-source expectations regress.
- Dataset-build defaults now prioritize speed: `--codec none` / `--clevel 0` / `--shuffle noshuffle` (no compression); optional compression remains available via `--codec lz4 --clevel 1 --shuffle shuffle`.

### New: V16 trainer snapshot/curation randomness fix (2026-05-19)
- `train_v16.py` no longer pins every run to `seed=42` unless explicitly requested; omitted `--seed` now generates a fresh run seed, while resume routes reuse existing run seed from `config.json`.
- Curation order for selected train/val subsets is now randomized by seed (no post-sample sort back to dataset index order).
- Validation snapshot export no longer uses first-N from ordered `val_loader`; it now samples positions per epoch from curated val data, with build-balanced selection enabled by default and a per-epoch `snapshot_selection.json` evidence artifact.
- Overview row titles now include source build id, making cross-build validation mixes visible at a glance.

### New: V16 in-place datastore liquid patch command (2026-05-19)
- `build_v16_dataset.py patch-liquids` now patches existing finalized stores in place without full rebuild.
- It re-streams tiles from staged clients, recomputes liquid supervision (`liquid_mask`, `liquid_height`, and liquid-source `has_*` flags), rewrites only those two Zarr arrays plus `index.parquet` liquid flags, and emits `liquid_patch_report.json`.
- Default behavior also runs post-patch signal validation; `index.parquet` is backed up to `index.parquet.bak.liquids` unless `--no-backup` is used.

### New: V16 human-eye dataset QA sampling lane (2026-05-19)
- `inspect_v16_dataset.py` now supports seeded random sample selection (`--sample-seed`, `--sample-mode=random|linspace|liquid_focus`) instead of fixed linspace-only picks.
- The inspector now emits a labeled visual artifact per build (`<build>.validation_audit_overview.png`) with minimap/height/liquid/object panels and tile metadata in each row header.
- JSON evidence remains paired with visuals (`<build>.summary.json`, `<build>.samples.json`) so dataset readiness can be reviewed by humans before training runs.

### New: data-harvester README compression + advanced guide split (2026-05-19)
- `wow-viewer/data-harvester/README.md` is now a concise operator runbook (setup, core commands, outputs).
- Dense command variants and extended options were moved to `wow-viewer/data-harvester/docs/advanced-v16-workflows.md`.
- Intent: keep the default README scannable and execution-focused; keep deep detail in one advanced companion doc.

### New: explicit liquid-refinement model note (2026-05-18)
- Docs now explicitly define a separate planned liquid model lane:
  - inputs centered on `minimap_rgb` plus optional liquid priors
  - targets/outputs for `liquid_mask` and `liquid_height`
  - strict boundary: terrain lane owns terrain geometry/material channels, liquid lane owns liquid placement/height fidelity.
- Terrain lane guidance now explicitly states liquid/object masks remain loss-gating signals for terrain supervision.

### New: Alpha map-name placeholder fix for `"memory"` labels (2026-05-18)
- Root cause confirmed: Alpha archive byte-path reads used `AlphaWdtReader.TryReadTile(byte[],...)`, which previously tagged tile source path as `"memory"`, and `AlphaTensorPackBuilder` derived `map_name` from that source path.
- Code changes:
  - `AlphaWdtReader` now has a public `TryReadTile(byte[] wdtData, int tileX, int tileY, string sourcePath, out AlphaTileData? data)` overload.
  - Harvest MPQ alpha call-sites now pass `World\\Maps\\<map>\\<map>.wdt` as source path.
  - Python builder now normalizes placeholder map names (`memory`, `<memory>`, empty, unknown) back to the requested stream map.
  - `repair-index` now has a fallback relabel mode when existing index map labels are placeholder-only.
- Dataset evidence check before fix:
  - `0_5_3_3368`: `1729/1729` rows had `map=memory`
  - `0_5_5_3494`: `1820/1820` rows had `map=memory`
  - later builds (`0_7.0+`) had `0` placeholder rows.
- Trainer-side guard now also blocks bad metadata rows from curation by default:
  - `train_v16.py` drops placeholder-map rows during subset selection unless
    `--include-placeholder-map-tiles` is explicitly passed.
- Legacy Alpha coordinate recovery hardening:
  - `build_v16_dataset.py` now parses `#alpha-tile(x,y)` markers from
    `tile_name` / `source_adt_path` when explicit metadata `tile_x` / `tile_y`
    are absent, so Alpha quilt coordinates no longer silently default to `(0,0)`.

### Zarr Arrays (per tile)
height_257, normal_xyz, normal_mask, alpha_256, holes_16, liquid_mask, liquid_height, **object_mask**, **object_precise_mask**, **object_instance_mask** (NEW), minimap_rgb, shadow_mask, mcly_texture_ids, mcly_layer_mask

### New: `object_instance_mask_257`
Per-pixel instance label: 0=terrain, 1+=placement index (MDDF first, then MODF). Each placement's footprint is stamped with its unique instance ID. This enables per-object segmentation training.

### New: `placements.parquet`
Companion table mapping tile_id → per-placement rows with columns: nameId, uniqueId, posX-Y-Z, rotX-Y-Z, scale, bbMin-Max, instance_type, instance_idx, asset_path. Links instance mask IDs to real model paths.

### Data Flow (full)
```
C# harvester → NPZB pipe → build_v16_dataset.py
  ├── Zarr arrays (14 fixed-shape arrays per tile)
  ├── index.parquet (tile_id, build, map, tile_x/y, height_mean/std, has_* flags, n_mddf, n_modf)
  └── placements.parquet (per-placement rows with asset_path linkage)
```

### Key Files
| File | Purpose |
|------|---------|
| `scripts/build_v16_dataset.py` | V16 build pipeline (streaming → Zarr + placements) |
| `scripts/train_v16.py` | V16 training script |
| `src/harvester/v16_dataset.py` | PyTorch Dataset from Zarr |
| `src/harvester/v15_model.py` | V15Model (= V16 model arch) |
| `WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs` | C# instance mask generation |
| `WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs` | Alpha instance mask generation |
| `WowViewer.Core/Maps/TerrainTileTensorPack.cs` | ObjectInstanceMask257 property |
| `WowViewer.Core.IO/Maps/NpzTileSerializer.cs` | Writes `object_instance_mask_257` as int32 |
| `docs/architecture/v16-terrain-model-spec-2026-05-16.md` | V16 full spec |
| `docs/architecture/multi-model-terrain-reconstruction-2026-05-16.md` | Multi-model architecture |

## Multi-Model Architecture (2026-05-16)

Six independent models, each training on ground truth only:

1. **V16 Terrain** (current): minimap → height/normals/alpha/holes/liquid. Uses object_mask for downweighting.
2. **Model A (Object Seg)**: minimap → per-pixel object mask + instance IDs. Ground truth: individually projected `placement_mddf_data`/`placement_modf_data`. **Unblocked now** — instance mask C# code landed.
3. **Model B (Liquid Seg)**: De facto V16 liquid head. Already training.
4. **Model D (Asset Attr)**: instance crop → asset path classification. Ground truth: metadata name tables. **Needs**: global asset vocabulary scan (Gap 2).
5. **Model F (Terrain V2)**: Clean minimap → terrain. Retrains with inpainted objects. **Needs**: Model A + clean minimap pipeline.
6. **PM4 Cross-Ref**: Use Model D predictions on PM4-only tiles to build CK24 → asset mapping. **Needs**: trained Model D.

### Data Gaps
- Gap 1: ✅ Per-instance object mask — **LANDED** in C# harvester
- Gap 2: Global asset vocabulary — needs scan of all tiles, not yet built
- Gap 3: PM4-to-object mapping — deferred, needs Model D
- Gap 4: Clean minimap generation — needs object inpainting pipeline

## C# Changes (This Session)

- `WowViewer.Tool.Harvest` archive-backed extraction now stages `_obj0.adt` beside the root ADT and `_tex0.adt`, so V16/archive harvest output no longer drops placements, object masks, or instance masks on split ADT builds like `3_3_5_12340`
- `AdtTensorPackBuilder.BuildUnifiedLiquid` now uses explicit liquid-presence masks from `MH2O`/`MCLQ` instead of treating `height == 0` as "no liquid", fixing sea-level water loss in unified liquid masks
- `AdtTensorPackBuilder.ExtractMapName` now falls back to the staged tile stem so archive temp extraction still records the real map name in NPZ metadata
- `WowViewer.Tool.Harvest` WL fallback no longer synthesizes fake `World\Maps\<map>\<map>.wl*` paths. It now enumerates actual `*.wlw/*.wlm/*.wlq/*.wll` virtual files from the loaded MPQ listfiles under `World\Maps\<map>\`, caches and parses them once per staged client/map, and reuses them across tiles. Focused `harvest-stream --limit 1` smoke on staged `3_3_5_12340 / Azeroth` now reports `no WL* files found in loaded archives for Azeroth`, so the fallback is currently a real archive-backed no-op for that build/map rather than a naming bug
- `AdtTensorPackBuilder.BuildObjectMasks` → returns `(float[,], float[,], int[,])` tuple with instance mask
- `AlphaTensorPackBuilder.BuildObjectMasks` → same pattern, added `PaintIntCircle`/`PaintIntRect` overloads
- `TerrainTileTensorPack.ObjectInstanceMask257` → new `int[,]?` property
- `NpzTileSerializer` → writes `object_instance_mask_257` as `<i4`
- Both builders assign instance IDs starting at 1 (0=terrain), MDDF first then MODF

## Python Changes (This Session)

- `wow-viewer/data-harvester/scripts/run-data-harvester-python.ps1` remains available as a repo-local fallback when sandboxed agent sessions cannot reach the uv-managed AppData paths, but elevated proof on 2026-05-16 showed both `.venv\Scripts\python.exe` and `uv run` work correctly in a real shell and remain the canonical operator path
- `build_v16_dataset.py` now forwards `harvest-stream` stderr live, prints per-map progress early enough for small maps, and raises explicit errors on truncated headers, bad magic, invalid blob lengths, NPZ decode failures, non-zero harvester exit codes, missing `ENDS`, and zero-tile maps instead of silently `break`ing
- V16 builds now stage into `wow-viewer/output/datasets/v16/<build>.zarr.partial` and only replace the final `.zarr` store after successful finalization; failed runs preserve the partial store and no longer silently leave a poisoned final dataset path with preallocated `50000`-tile arrays
- `build_v16_dataset.py stats` now warns when `index.parquet` is missing or when array length does not match finalized index rows, uses `pyarrow.compute.sum` for `has_*` counts, and suppresses the harmless Zarr sidecar warnings from `index.parquet` / `placements.parquet`
- `build_v16_dataset.py` now writes dropped missing-required tiles to `wow-viewer/output/datasets/v16/<build>.rejected_tiles.jsonl` and surfaces `dropped_missing_required=<n>` in each per-map summary so rejected coordinates/keys survive the live console scrollback
- `WowViewer.Tool.Harvest` now exposes `discover-maps --client-root <staged client>` and filters map candidates using the real V16 contract instead of a bootstrap hard-coded map list: pure WMO-only maps (`MWMO/MONM` present, no terrain tiles), zero-tile maps, missing-WDT transport entries, and "terrain but no V16-usable probe tile" maps are skipped, where "usable" currently means the archive probe path can produce both `height_257` and `minimap_rgb_256`
- `build_v16_dataset.py` no longer aborts the whole build when one discovered map produces zero usable V16 tiles at stream time; it now warns and skips that map, while still failing loud if the entire requested build produces zero usable tiles
- `build_v16_dataset.py` → now carries 14 Zarr arrays (was 12), adds `object_precise_mask` and `object_instance_mask`, writes `placements.parquet` companion table with per-placement rows + asset_path linkage, index includes `n_mddf`/`n_modf` counts
- `v16_dataset.py` → reads `object_instance_mask` from Zarr, returns int64 `instance_mask` tensor and `has_instance` flag
- `train_v16.py` → unchanged (V16 model doesn't use instance mask yet; will be used by future Model A)

## Active Recovery Plan (2026-05-17)

- The archive-backed V16 lane was still staging root/`_tex0`/`_obj0` ADTs to `%TEMP%` before tensor-pack extraction. That seam is the primary performance regression this recovery slice targets.
- Recovery plan doc: `wow-viewer/docs/architecture/v16-harvest-recovery-plan-2026-05-17.md`
- Recovery implementation order:
  1. move archive-backed harvest/discovery to an in-memory ADT family builder path
  2. add real map-level resume for `<build>.zarr.partial`
  3. switch future V16 builder defaults to a faster Zarr codec profile while keeping the current schema and reader contract
- Existing finished `.zarr` stores remain valid; the recovery slice is aimed at future rebuild speed and at resuming incomplete client builds without redoing finished maps.
- Implementation is now partially landed but unvalidated in-chat because the user explicitly blocked agent-run builds:
  - `WowViewer.Tool.Harvest` archive-backed ADT families now route through `AdtTensorPackBuilder.BuildFromBytes(...)` instead of temp-file staging
  - `build_v16_dataset.py` now carries map-level `_resume_state.json` state for `<build>.zarr.partial`
  - `--resume` now bootstraps cleanly when no resume state exists yet instead of tripping on a just-created staged directory
  - `build_v16_dataset.py` now skips already-complete final `<build>.zarr` stores by default unless `--rebuild-existing` is passed
  - successful final stores now retain `_resume_state.json` as completion metadata instead of deleting it at finalization
  - `scripts/backfill_v16_resume_state.py` can backfill `_resume_state.json` into older completed final stores
  - `scripts/inspect_v16_dataset.py` can backfill `_dataset_summary.json`, emit human-friendly JSON summaries, and generate sample image sheets from existing V16 stores
  - `scripts/validate_v16_training_ready.py` now provides a separate trainer-readiness proof path: it opens finalized V16 stores, reads real samples through `V16Dataset`, validates a real `DataLoader` batch, and can run one `V15Model` forward pass so "dataset built" and "trainer can consume it" stop being conflated
  - the Python Zarr writer now retries transient Windows `WinError 5` / `WinError 32` chunk-write failures instead of aborting immediately on the first `LocalStore` atomic-replace race
  - the Python Zarr writer now buffers tiles in memory and flushes them in small slice batches, reducing one-row-at-a-time chunk rewrites on the filesystem
  - incoming fixed-shape signals are now coerced to canonical Zarr shapes before batching so variable layer-count payloads do not fail `np.stack(...)` during resume/build runs
  - the C# builder no longer reparses the same placement catalog twice per tile for object masks and placement-array export, which was wasted work on placement-heavy tiles
  - `build_v16_dataset.py stats` now reports logical raw array bytes versus on-disk Zarr bytes so compression savings are visible per array and per store
  - future V16 builds now default to `lz4` / level `1` / `shuffle`
  - `V16Dataset` now exposes `mcly_ids` / `mcly_mask` from Zarr and `train_v16.py` now uses the existing V15-style masked cross-entropy path for MCLY supervision; `instance_mask` remains readable but still is not used by the current terrain trainer
  - V16 coordinate bookkeeping is now patched in two places: future streamed NPZ metadata carries explicit `tile_x` / `tile_y`, and `build_v16_dataset.py repair-index --build <key>` can rewrite existing `index.parquet` files in place from a metadata-only re-stream without touching the stored tensor arrays
- Repo truth still needs operator proof from a user-run rebuild before this recovery slice can be treated as validated.
- `wow-viewer/README.md` now leads with the V16 dataset/training lane, including repo-level links and command surfaces for `build`, `repair-index`, validator, and `train_v16.py`, so new chats no longer have to infer that terrain-AI dataset generation is a primary repo goal.

## NOT YET (Blocked on User)
- Full V16 builds for all client builds (rebuild harvester binary first)
- V16 training run
- Object segmentation Model A training script
- Asset vocabulary build
- PM4 cross-reference analysis
- Rebuild the remaining staged client builds after `3_3_5_12340`
- Full V16 rebuilds can use canonical `uv run` again; the remaining environment caveat is sandbox/AppData access during agent-run validation, not a broken repo-local `.venv`
