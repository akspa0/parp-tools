# Tasks: V25 Terrain Convergence Model (Spec 102)

This document contains the step-by-step tasks for the V25 SegFormer Decompiler and Terrain-Texture Convergence model.

---

## Phase 0: Lean V25 Dataset (added 2026-07-11 re-implementation pass)

- [x] **T025 [US1/US2] Author `data-harvester/src/harvester/v25/dataset.py`**:
  - Lean 8-array Zarr schema (Blosc LZ4 level 1, per-tile chunks) + index/placements/tileset_vocab parquet sidecars.
  - `build_v25_dataset` sources V18 substrate, V22 tileset ids/placements/model paths, V24 pre-computed cleaned minimaps (`v18_row` join); curation-manifest/map/limit filters; contiguous slice reads.
  - `wdl_height_33` derived at build with the exact WdlDownsampler stride-8 math (SC-102-004 by construction).
  - `V25TileSource` with `preload(rows)` (FR-102-502) and per-row training records.
  - `attach_pm4_segments`/`load_pm4_segment_records` round-trip pre-parsed `Pm4SegmentSignalRecord`s (FR-102-402 — no PM4 binary parsing).
- [x] **T026 [US1/US2] Rewrite `data-harvester/scripts/build_v25_dataset.py`** around the lean builder; retire `harvester/v25_zarr_io.py` (V22-clone writer) and its test.
- [x] **T027 [US1/US2] Author `data-harvester/tests/v25/test_dataset.py`**:
  - Builder round-trip (shapes/dtypes/WDL math), lean-signal-only assertion, vocab + MCLY mapping, placements table, row selection (maps/manifest/bucket), preload==random-access equivalence, PM4 record round-trip, prediction-store writer.
- [x] **T028 Checkpoint Phase 0**: `uv run python -m pytest tests/v25` green (30 passed); bounded real build from `3_3_5_12340` V18+V22+V24 verified (24 curated Azeroth tiles, all `v24_precomputed` clean sources, 56 placements, exact WDL alignment).
- [x] **T029 [US1/US2] Multi-era corpus + liquid/flag loss signals** (user-directed 2026-07-11):
  - Schema gained `liquid_mask_256`, `liquid_type_256`, `liquid_height_256`, `mcnk_flags_16` (FR-102-002 amendment); `holes_16` stays excluded (inverted at C# source per V24 audit).
  - Builder accepts index-paired multi-build source triples (`--v18-store A B --v22-store A B --v24-store - B`); tileset vocabulary shared across builds keyed by normalized tileset path (FR-102-008), so 0.5.3 grass and 3.3.5 grass share one vocab index.
  - Combined curated two-era store built: `output/datasets/v25/0_5_3+3_3_5_v25_curated_v1.zarr` (777 kept 0.5.3 tiles + 2,804 kept 3.3.5 tiles from the V18 curation manifest), `--vocab-size 1024` so **all 769 distinct tilesets across both eras are in-vocab** (user-directed — no frequency truncation; OOV bucket only for unseen textures at inference).
  - Suite: 32/32 v25 tests green (multi-build shared-vocab + liquid round-trip tests added).
- [x] **T030 [US1/US2] Curation bake-in + full-signal completeness (v25.2, user-directed 2026-07-11)**:
  - All 40 curation-manifest columns joined per tile into `index.parquet` (buckets, quality/usefulness/difficulty scores, coverage stats, profiles); mismatch-audit `mismatch_severity`/`mismatch_reason` joined; `V25TileSource.rows_for_buckets` + trainer `--difficulty-buckets` filter make the bucketing actionable at train time.
  - Mismatch-repair store support: `--height-repair-root` prefers per-build `height_corrected_257` over raw heights (3 corrected tiles in 3.3.5); `height_repaired` flagged per row and in store attrs.
  - Schema v25.2 adds `normal_xyz_257` (int8 MCNR-native), `shadow_mask_256`, `object_visibility_256`, `ground_intent_height_257`, `object_instance_mask`. Liquid-masked height/prior loss landed in the trainer (`--no-liquid-height-mask` opt-out) with a zero-gradient-into-liquid test.
  - Excluded and documented: `holes_16` (corrupt at source), derivable masks (normal/mcnr validity, mddf/modf footprints), deprecated diagnostics (roof/filtered/focus/above-terrain).
  - Suite: 37/37 v25 tests green.
- [x] **T031 [US1/US2] True hole bitmasks end-to-end (user-directed 2026-07-11)**:
  - C# source fix: `AdtTensorPackBuilder.ReadMcrfAndHoles` reads the MCNK `holes` uint16 @0x3C (renderer-truth field) instead of flags bits 8-15; `AlphaWdtReader` now threads the full per-chunk ushort masks (@0x40, already parsed) into `AlphaTileData.HoleFullMasks`. Full solution builds, 0 errors.
  - New `WowViewer.Tool.Harvest extract-holes` command (era-aware alpha WDT + LK/split ADT via new public `AdtTensorPackBuilder.ReadHoleBitmasks`), run against both staged clients: 0.5.3 = 1,736 tiles (228 holed), 3.3.5 = 5,501 tiles (630 holed).
  - Python `attach_holes_bits` joins exports into the store as `holes_bits_16` (int32, -1 = unknown); `--attach-holes` on the builder CLI; TileSource exposes `holes_bits`. 100% join coverage on the two-era corpus (3,581/3,581, 767 holed tiles).
  - **Bug caught during verification**: the mismatch-repair store is a sparse NaN overlay (5,131/5,134 rows NaN); the first repaired build consumed it wholesale, poisoning ~2,800 tiles with NaN heights. Builder now merges per cell (corrected where finite) and hard-counts `nonfinite_height_tiles` (must be 0). Test models the sparse overlay explicitly. 38/38 v25 tests green.
  - Latent follow-up flagged, not changed: `ReadMcrfAndHoles` reads MCRF `wmoCount` from payload offset 0x3C (the holes field) — nMapObjRefs is 0x38 per `Lk/Mcnk.cs`; needs its own audit since object-mask signals were previously validated as sound.
- [x] **T032 [US1/US2] Era-scoped tileset vocabulary + tileset images (user-directed 2026-07-11)**:
  - Vocabulary re-keyed to (build, normalized path): same-named tilesets in different eras are distinct entries (their pixels differ). `tileset_vocab.parquet` gains `build` + `key` columns. Vocab 2048 covers 0.5.3 (322) + 3.3.5 (748) without truncation.
  - New `WowViewer.Tool.Harvest extract-tilesets` command decodes era-specific BLPs (with normalized-name → BLP virtual-path fallback): 316/322 (0.5.3) + 747/748 (3.3.5) decoded to PNGs + manifests under `output/tmp/tilesets_<build>/`.
  - `attach_tileset_images` writes the vocab-aligned `tilesets` group (`tileset_rgb_256`, `tileset_present`); 7 unresolvable textures stay `present=0`.
  - Suite: 39/39 v25 tests green (era-scoped vocab test replaces the shared-path test; tileset attachment round-trip added).

---

## Phase 1: SegFormer Frontend & Decompiler Modules

- [x] **T001 [US2] Author `data-harvester/src/harvester/v25/segformer.py` - Custom Segformer**:
  - Load `nvidia/mit-b0` or `nvidia/mit-b1` model using Hugging Face `transformers` library.
  - Implement semantic segmentation logits mapping.
  - Implement `TerrainInpaintHead` to map the raw RGB minimap + predicted object masks to $3\times256\times256$ clean terrain-shadow maps.
- [x] **T002 [US1] Author `data-harvester/src/harvester/v25/segformer.py` - Placements**:
  - Implement `ObjectPlacementHead` to regress object classes, translations, and 3D rotations from the encoder feature layers.
- [x] **T003 [US1/US2] Author `data-harvester/tests/v25/test_segformer.py`**:
  - Test SegFormer loading (offline-compatible), output shape checks, and classifier head forward paths.
- [x] **T004 [US1/US2] Checkpoint Phase 1**:
  - Run `uv run python -m pytest tests/v25/test_segformer.py` and verify all pass.

---

## Phase 2: Height Solver & Sylvester Math

- [x] **T005 [US2] Author `data-harvester/src/harvester/v25/solver.py`**:
  - Implement `BatchedSylvesterSolver` solving $(I + \gamma_c L_c) X + X (\gamma_r L_r) = Y$ via GPU eigendecomposition (`torch.linalg.eigh`).
- [x] **T006 [US2] Author `data-harvester/src/harvester/v25/lapnet.py`**:
  - Implement progressive `V25StageBPredictor` scaling heights progressively ($33 \rightarrow 65 \rightarrow 129 \rightarrow 257$) using the Sylvester solver guided by features from the clean minimap, zeroing out edge affinities inside the predicted object mask.
- [x] **T007 [US2] Author `data-harvester/tests/v25/test_solver.py` and `test_lapnet.py`**:
  - Test Sylvester math correctness vs. CPU Scipy solver and progressive output shapes.
- [x] **T008 [US2] Checkpoint Phase 2**:
  - Run `uv run python -m pytest tests/v25/test_solver.py tests/v25/test_lapnet.py` and verify all pass.

---

## Phase 3: WDL Downsampler

- [x] **T009 [US1] Author `data-harvester/src/harvester/v25/prior.py`**:
  - Implement `WdlDownsampler` mapping $(257, 257) \rightarrow (33, 33)$ prior coordinates via mathematical average pooling.
- [x] **T010 [US1] Author `data-harvester/tests/v25/test_prior.py`**:
  - Test WDL prior downsampler output coordinates and shape alignment.
- [x] **T011 [US1] Checkpoint Phase 3**:
  - Run `uv run python -m pytest tests/v25/test_prior.py` and verify all pass.

---

## Phase 4: Decoupled PM4 Post-Processing Handler

- [x] **T012 [US1] Author `data-harvester/src/harvester/v25/pm4_guide.py`**:
  - Implement `V25Pm4GuideHandler` as a standalone post-processing class.
  - Implement snapping logic to align predicted object coordinates to pre-parsed PM4 segment centroids loaded from the database.
  - Integrate `harvester.pm4_asset_matching.scorer` to resolve WMO/M2 counterparts from segment bounds.
- [x] **T013 [US1] Author `data-harvester/tests/v25/test_pm4_guide.py`**:
  - Test coordinate snapping, asset match verification, and out-of-bounds predicted mask rejection.
- [x] **T014 [US1] Checkpoint Phase 4**:
  - Run `uv run python -m pytest tests/v25/test_pm4_guide.py` and verify all pass.

---

## Phase 5: Differentiable Fractal Generator and Parameter Head

- [x] **T015 [US1] Author `data-harvester/src/harvester/v25/fractal.py` - Generator**:
  - Implement `DifferentiableFractalGenerator` in PyTorch generating multi-octave Perlin/Simplex noise on a $256\times256$ grid.
- [x] **T016 [US1] Author `data-harvester/src/harvester/v25/fractal.py` - Parameter Head**:
  - Implement `FractalParameterHead` predicting translation seed $(S_x, S_y)$, frequency $f$, amplitude $A$, persistence $p$, and soft paint boundary mask $M$ ($256\times256$) per active layer.
- [x] **T017 [US1] Author `data-harvester/tests/v25/test_fractal.py`**:
  - Test fractal parameter extraction, output gradients, and numerical stability.
- [x] **T018 [US1] Checkpoint Phase 5**:
  - Run `uv run python -m pytest tests/v25/test_fractal.py` and verify all pass.

---

## Phase 6: Terrain Texture Decoders (MCLY/MTEX) & Losses

- [x] **T019 [US1] Author `data-harvester/src/harvester/v25/texture.py`**:
  - Implement `MtexPredictor` mapping visual features to multi-hot texture index probability vectors.
  - Implement `MclyDecoder` predicting active layers over MCNK grids (`mcly_tileset_ids` of shape `16x16x4`).
- [x] **T020 [US1/US2] Author `data-harvester/src/harvester/v25/losses.py`**:
  - Implement `V25UnifiedLoss` combining SegFormer CE, height L1, progressive height L1/SiLog, MCAL fractal parameter MSE, MCLY CE, MTEX CE, and object placement losses.
  - Write tests under `tests/v25/test_losses.py`.
- [x] **T021 [US1/US2] Checkpoint Phase 6**:
  - Verify model routing and loss gradients propagate: `uv run python -m pytest tests/v25/`

---

## Phase 7: Training and Zarr Dataset Integration

- [x] **T022 [US1] Author `data-harvester/scripts/train_v25_decompiler.py`** (rewritten 2026-07-11):
  - Trains the full universal pipeline from the lean V25 store: SegFormer mask/placements, TerrainInpaintHead (clean-minimap target), `V25StageAPredictor` (WDL prior target), progressive Sylvester Stage B (GT-prior teacher forcing by default, `--student-prior` opt-in), MTEX/MCLY decoders (vocab-sized), fractal alpha heads.
  - `V25TileSource.preload()` before the epoch loop; real held-out validation on the universal (student-prior) path; best/last checkpoints with config (vocab, class map, normalization constants); `peak_vram.json` for SC-102-001.
  - `--gradient-checkpointing`, `--8bit-optimizer`, `--amp-dtype {fp16,bf16,none}`; no hardcoded absolute paths.
  - The earlier draft trainer was replaced: it fed the ground-truth 33x33 prior into the solver without ever training Stage A or the inpaint head (single-image inference was impossible), used per-row random Zarr reads, and reported the train loss as a placeholder val loss.
- [x] **T023 [US1] Author `data-harvester/scripts/validate_v25.py` and `infer_v25_decompiler.py`**:
  - `validate_v25.py` scores SC-102-003 (alpha SSIM), SC-102-004 (WDL/GT alignment), SC-102-005 (mask IoU), plus height L1 numbers; writes `report.json`.
  - `infer_v25_decompiler.py` runs a bare PNG or store row through the universal path and writes the structured Zarr prediction store (Blosc LZ4 level 1: heights 257+33, masks, clean minimap, alpha, MCLY/MTEX, placements parquet).
  - `--pm4-records` snaps placements against pre-parsed `pm4_segments.parquet` records via `V25Pm4GuideHandler` (separate post-processing step, never in training).
- [x] **T024 [US1] Checkpoint Phase 7**:
  - CPU smoke run 2026-07-11 on the bounded real store (`3_3_5_12340_v25_smoke24.zarr`): 1-epoch train (all loss heads live), inference export verified (`wdl_height_33 == height_257[::8,::8]` in the prediction store), validation report written with honest FAIL gates for the untrained model and PASS for dataset alignment.
  - GPU-scale training on the full curated store remains open (needs explicit go-ahead per resource rules).
