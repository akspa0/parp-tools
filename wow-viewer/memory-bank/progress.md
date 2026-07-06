# Progress — wow-viewer

Keep this file to last-week truth. Older history moved to `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## 2026-07-06

### Spec 094 WDL prior + lattice detailer (V24) implemented end-to-end

- Read and fleshed out the existing spec/plan/data-model/research/tasks Spec Kit artifacts, then verified every architectural guess against real code and real data before implementing: added an "Implementation Amendments" section to `spec.md` (A1-A8) recording the audited C# WDL grid shape (17×17 outer + 16×16 inner int16, MAHO not read), the MPQ-based (not loose-file) WDL resolution path, the actual V18 store schema (`minimap_rgb` 256² uint8, `object_precise_mask` 257² float32), the paired-array V24 store schema, the exact quincunx 33→257 upsample math, and the added V22 dataset audit lane.
- Built `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead` (new C# CLI, added to `WowViewer.slnx`, 0 build errors): `read` wraps `WdlSummaryReader` via `NativeMpqService`; `synth` wraps `WdlWriter.ExtractTileHeightsFromAlpha` with shim-local nearest-non-liquid resampling. Neither existing C# reader nor writer was modified.
- Verified the "99% match" claim directly: synthetic-vs-real WDL convergence on 8 real Azeroth tiles hit 100% cell agreement within 1.0 world unit on both `3_3_5_12340` and `0_5_3_3368`.
- Implemented the full Python pipeline under `data-harvester/src/harvester/v24/`: `shim.py` (subprocess bridge), `wdl_reader.py`, `synth_wdl.py`, `merged_wdl_prior.py`, `clean_minimap.py`, `lattice.py` (quincunx geometry), `tiles.py` (V24+V18 joined tile loader), `stage_a.py` (337,485-param residual U-Net), `stage_b.py` (827,681-param conv-deconv), `train_common.py`. Plus `scripts/build_wdl_prior.py`, `inspect_v24_dataset.py`, `clean_minimap.py`, `train_v24_stage_a.py`, `infer_v24_stage_a.py`, `train_v24_stage_b.py`, `infer_v24_stage_b.py`, `validate_v24.py`.
- Full v24 pytest suite passed: `uv run python -m pytest tests/v24 -m v24 -q` → `30 passed`.
- Ran two bounded real-data builds (both pass SC-001 coverage ≥ 95%): Northrend on `3_3_5_12340` (100% real WDL coverage) and Azeroth on `0_5_3_3368` (85% real / 15% synthetic).
- Trained and validated Stage A + Stage B on a rough-terrain (`--min-height-std 15`) 50-tile Northrend set: Stage A real-cell L1 (0.479) < synth-cell L1 (0.736) < `block_reduce` baseline (0.603); Stage B final L1 (1.783) < upsampled-prior L1 (3.563) < `block_reduce+bilinear` baseline (3.247). SC-004 determinism (bit-identical across seeds) and SC-005 hardware envelope (peak VRAM 0.19 GB, max wall-time 0.05 s/tile) both pass cleanly.
- Also validated on a flat-terrain 50-tile set to characterize a real edge case: SC-001's confidence bound passes at 100%, but SC-003 fails against `block_reduce+bilinear` because flat terrain gives that trivial baseline near-zero error — nothing to beat. Reported honestly rather than hidden; see `docs/architecture/v24-validation-2026-07-06.md`.
- Found and worked around a real V18 dataset defect during this work: `holes_16` is inverted at the C# source (`AdtTensorPackBuilder.ReadMcrfAndHoles`, flags-based derivation wrong for LK-era MCNKs). Workaround lives in `harvester/v24/tiles.py::_normalize_holes`; the real fix is out of scope for Spec 094 (would touch `WowViewer.Core.IO`).
- User-directed scope addition: built `scripts/audit_v22_dataset.py`, a C#-grounded V22/V18 signal audit (re-extracts reference signals via the existing `WowViewer.Tool.Harvest extract-unified`, Python only compares). Confirmed V24's actual input signals are sound; the object-mask family's divergence from naive re-extraction is by design (V22's enriched projection beats the reference heuristic). Report: `docs/architecture/v22-dataset-audit-2026-07-06.md`.

## 2026-07-04

### Documentation audit and rewrite

- Rewrote `wow-viewer/AGENTS.md` to current repo truth.
- Rewrote root `README.md`, `docs/PLANS-OVERVIEW.md`, `docs/WoWViewer/README.md`, `docs/WoWViewer/USERGUIDE.md`, and `data-harvester/README.md`.
- Added `docs/DOCUMENTATION-STATUS.md` as canonical doc map.
- Removed dead links and stale path guidance from high-traffic docs.

### Spec 089 local 12 GB pivot

- `train_v23_height.py` now applies real memory profiles, honors `grad_accum_steps`, records `peak_vram.json`, and retries OOM by shrinking batch size, then GPCT-K, then AMP mode.
- Default target VRAM is now 12 GB, not 22 GB.
- Focused validation passed with `3 passed`: `uv run python -m pytest tests/v23/test_train_profiles.py tests/v23/test_train_smoke.py -m v23 -q`.
- T035 local CUDA proof passed on RTX 4070 Ti SUPER: `t035_local_12gb_20260704`, 16 real V22 train tiles, 4 val tiles, zero CUDA OOM, `peak_vram.json` max allocated `0.408541184 GB`; caveat: HF DA-V2-Small weights were unavailable locally, so this is an envelope proof rather than pretrained quality proof.
- Added V23 `--maps` training filter and reran a Northrend-specific local CUDA smoke: `t035_northrend_local_12gb_20260705`, `3_3_5_12340`, `--maps Northrend`, checkpoint config records `maps = ["Northrend"]`, zero CUDA OOM, max allocated `0.408541184 GB`.
- Fixed the bigger V23 route bug: trainer now accepts the V18 curation manifest, filters V22 samples through the same keep/threshold rules, selects validation from high-mismatch curated rows first, and writes labeled validation preview PNGs. Curated local proof: `v23_curated_northrend_labeled_smoke_20260705`, `--curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet`, `--maps Northrend`, zero CUDA OOM, max allocated `0.3959296 GB`.
- Fixed V23 trainer console silence and weak loss evidence. `train_v23_height.py` now prints startup configuration, train/val batch `loss=...`, component breakdowns, epoch `train_loss`/`val_loss`/`best_val_loss`, preview/checkpoint/metrics paths, `loss_history.jsonl`, and CUDA peak VRAM; `--log-interval` controls batch cadence. `peak_vram.json` is capacity proof only. Focused smoke and full V23 suite passed with `35 passed, 14 warnings`: `uv run python -m pytest tests/v23 -m v23 -q`.
- Added V23 startup batch autotune: `--autotune-batch-size`, `--autotune-batch-candidates`, `--autotune-safety-factor`, and `batch_autotune.json`. It probes CUDA candidates before epoch 1 and rebuilds loaders with the selected batch size. Focused profile/smoke tests passed with `6 passed, 14 warnings`; full V23 suite passed with `36 passed, 14 warnings`.
- Fixed V23 validation cadence. `--val-interval` now skips unscheduled validation epochs, records `validation_skipped=true` in `loss_history.jsonl`, keeps saving `v23_height_last.pt`, and validates on the final epoch when enabled. Focused profile/smoke tests passed with `8 passed, 14 warnings`; full V23 suite passed with `38 passed, 14 warnings`.
- Added visible per-step heartbeat lines for V23 training. `--log-interval 1` now shows `status=start` and `status=done` per batch with step/batch/sample progress, elapsed/ETA, optimizer-step status, loss breakdown, and CUDA memory.
- Read the first 2K key-map artifacts. Loss moved in the right direction, but the run selected the last batch candidate (`24`) while using only about `6.21 GB` reserved, and `sdc_loss` stayed dead-zero. Recommended autotune candidates now extend to `32 40 48`; SDC sparse-mask handling now uses fractional patch weights.
- RunPod packaging now carries the V18 curation manifest into `config/curation_manifest.parquet`, and no-arg `runpod/v23/train.sh` runs the curated 2K key-map path with startup autotune, per-step logging, GPCT-K 2, SDC, bias-free masking, and `--val-interval 2`. Next proof owner = T046 Pod smoke plus curated 2K key-map Pod training; no more local training runs unless explicitly reopened.

### Spec 080 compatibility slice in `MdxViewer`

- Bottom display bar now owns terrain/world toggles.
- Top toolbar now acts as launcher strip for minimap, terrain workbench, PM4 workbench, and capture automation.
- `DrawPm4ObjectMatchWindow()` and `DrawPm4WmoCorrelationWindow()` are wired back into `DrawUI()` and exposed from `Tools`.
- Legacy build still fails on broad pre-existing missing refs outside touched slice. Status = source-complete only.

## 2026-07-05

### Spec 080 wow-viewer UI audit and bottom-bar slice

- Added missing Spec Kit execution artifacts for `specs/080-wow-ui-consolidation`: `plan.md` and `tasks.md`.
- Audited the current right-sidebar/workbench state: WMO group boxes existed but were buried in model info, standalone WMO labels only drew for selected/highlighted groups, world wireframe was monolithic, Model LOD is placeholder text, and World LOD is missing from `WorkbenchNavigator`.
- Implemented the bounded `wow-viewer` Phase 1 slice: bottom bar now has split world wireframe controls (`Terrain WF`, `M2/WMO WF`), standalone model/WMO wireframe, standalone WMO group bounding boxes, all-group WMO labels, and a Settings launcher.
- Standalone WMO group labels default to visible for every render group when loading a single WMO object; the existing selected/highlighted label workflow remains.
- Implemented the bounded Phase 2 Settings follow-up: File -> Settings now opens the same Settings window as the bottom bar, Tools menu, utility popup, and workspace card; Settings has persisted Camera Speed and FOV defaults through `viewer_settings.json`.
- Added the first World -> LOD destination: WDL visibility, world bounding boxes, PM4 overlay/status tooltip, loaded tile/chunk counts, and ADT detail-tile budget controls now live under a real `LOD` bottom tab.
- Validation: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed three times with 0 errors and existing warning noise. Manual viewer checks remain open.

### Spec 090 viewer memory profiler

- Added `specs/090-viewer-memory-profiler` for the 4.0.0 Stormwind memory blow-up lane.
- Runtime Stats now shows process working set/private bytes, managed heap/live allocated bytes, GC counts, MPQ raw-cache bytes, and world asset raw-cache bytes.
- `WorldAssetManager` raw file-data cache now tracks byte residency and evicts by both entry count and a 512 MiB byte cap. Live renderer eviction was intentionally not changed.
- Validation: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with 0 errors and existing warning noise. Manual staged 4.0.0 Stormwind measurement remains open.

### Spec 093 render performance and WMO liquid audit

- Added `specs/093-render-performance-liquid-audit` for the slow-frame and WMO liquid lane.
- Audited the source path: WMO rendering is per visible placement and per group/material batch; MDX "batched" means shared-shader submission rather than true GPU instancing; WMO transparent/liquid work was being timed under the MDX transparent bucket.
- Added diagnostic WMO counters: total WMO draw calls, batch draws, fallback group draws, liquid draws, doodad submissions, and visible group submissions.
- Runtime Stats now separates WMO opaque and WMO transparent submission timing and displays WMO draw composition. WMO liquid visuals were not changed in this slice.
- Validation: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with 0 errors and existing warning noise; focused `WorldRenderOptimizationAdvisorTests` passed with `3 passed`.

### V23 remote selector preference cleanup

- `setup_v23_runpod.py` now prefers `3090 -> 4090 -> 5090` when no explicit GPU list is given.
- Focused test passed with `1 passed`: `tests/test_setup_v23_runpod.py`.

### Spec 091 raw audio unswizzle probe

- Added `specs/091-raw-audio-unswizzle` for the map-derived WAV/raw pattern investigation.
- Added `data-harvester/scripts/unswizzle_audio_raw_patterns.py` to strip WAV payloads or read raw bytes, sweep grayscale byte views, deltas, bitplanes, byte phases, RGB triplets, 16-bit sample interpretations, and float32 probes.
- The script writes ranked `summary.json`, candidate PNGs, and `contact_sheet.png`; results are explicitly layout hypotheses, not proof of hidden payloads.
- Validation: `uv run python -m py_compile scripts/unswizzle_audio_raw_patterns.py`, `uv run python scripts/unswizzle_audio_raw_patterns.py --help`, and a bounded smoke run wrote 60 candidates under `C:\tmp\wow-unswizzle-smoke`.
- Reran against `output/azeroth_audio/Azeroth_all_tiles_0_5_3_3368_11025Hz.wav`; payload was 41,082,478 mono 16-bit samples = 622 complete `257x257` tiles with zero remainder. Added stream-order and V18 `index.parquet` coordinate-order tile mosaics under `wow-viewer/output/analysis/raw-audio-unswizzle/azeroth_0_5_3_3368/tile_unswizzle`.

### Spec 092 heightmap pattern miner

- Added `specs/092-heightmap-pattern-miner` and `data-harvester/scripts/mine_heightmap_patterns.py`.
- The miner reads Zarr `height_257`, samples configurable patches, hashes locally normalized low-resolution signatures, filters low-variance and saturated artifacts, then writes ranked `summary.json` and `pattern_atlas.png`.
- Validation: py_compile/help passed; bounded real-data run on `0_5_3_3368` Azeroth 128 tiles kept 19,727 patches, found 15,702 buckets, and produced filtered output at `wow-viewer/output/analysis/heightmap-patterns/azeroth_0_5_3_3368_filtered`.
- Corrected the miner to reject tiny patch matching. Default mode now uses terrain-cell spans `32 64`, minimum span `32`, chunk-aligned 16-cell starts, and coarse `4x4/q4` signatures. Corrected proof kept 16,390 patches and produced `wow-viewer/output/analysis/heightmap-patterns/azeroth_0_5_3_3368_chunkcells_coarse`.
- V23 trainer was intentionally unchanged; next step is motif-to-validation-error joining.

## 2026-07-03

### Spec 089 local stack reached bundle boundary

- V23 encoder, head, model, losses, trainer, inference, checkpoint, and RunPod bundle surfaces all landed.
- Local proof suite passed with `28 passed, 14 warnings`: `uv run python -m pytest tests/v23 -m v23 -q`.
- Real Pod creation happened, but upload and remote smoke remain open. Not proof owner.

### Spec 088 real-data V22 path repaired

- `WowViewer.Tool.V22Enrich` builds again and the Python writer now emits coherent `index.parquet`, `placements.parquet`, `asset_inventory.parquet`, and `finalization.json`.
- Canonical V22 stores now exist for `0_5_3_3368` and `3_3_5_12340`.
- Contract is `paths_only`, with provenance sidecars and no embedded asset payload blobs.
- Remaining bounded gate: run same proof for `4_0_0_11927`.

### Environment repair

- Stale `.venv` moved aside.
- Fresh env rebuilt on `C:\Python314\python.exe`.
- `pyproject.toml` now exposes `src/` package metadata and missing `setuptools`.

## 2026-06-30

### Spec 088 replaced broken V22 payload plans

- Specs 086 and 087 were superseded by Spec 088.
- New route = `V22Enrich` + paths-only V22 store built from V18 substrate.
- This remains live background because Spec 089 depends on it.

## 2026-06-29

### Spec 077 masking correction

- `HeightOnlyPriorDataset` weight gating now prefers `object_precise_mask`, then filtered, then coarse fallback.
- RunPod slim bundles must carry `object_precise_mask` before more trust in that lane.

### Viewer animation/UI source fixes

- Model animation controls resurfaced in default info surfaces.
- Save-dialog-backed animation state export landed.
- Shell-wrapper blocker was later replaced by real legacy build failures outside the UI slice.
