# Progress — wow-viewer

Keep this file to last-week truth. Older history moved to `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## 2026-07-11

### Depth Anything blacklisted (2026-07-12, user-directed)

- All DA-family models (DA-V2, PromptDA) are off the table permanently: non-repeatable outputs even with fixed seeds, "too random in its decisions as to what is the right way to map depth." ~Half a month spent on the V23 (Spec 089) and V24.1 (Spec 101) DA lanes is written off; do not cite those runs to reopen the avenue. Spec 102 FR-102-101 ("No Depth Anything models allowed") is the standing rule. Recorded in activeContext Boundaries.

### V25 Terrain Convergence Model (Spec 102)

- **Spec & Plan Created**: Created spec, plan, and task lists under `specs/102-v25-terrain-convergence/` detailing the Visual Segformer Decompiler, progressive Sylvester upsampler, WDL downsampler, differentiable fractal generator, PM4 placement alignment handler, trainer/VRAM optimizations, and structured Zarr output dataset stores.
- **SegFormer Frontend & Decompiler**: Implemented `V25SegformerDecompiler` wrapping `nvidia/mit-b0`, `TerrainInpaintHead` for gated object removal, and `ObjectPlacementHead` for anonymous bounding box regressing. Tests pass.
- **Progressive Height Solver**: Implemented `BatchedSylvesterSolver` solving $(I + \gamma_c L_c) X + X (\gamma_r L_r) = Y$ via GPU eigendecomposition, and progressive `V25StageBPredictor` upscaling heights progressively ($33 \rightarrow 65 \rightarrow 129 \rightarrow 257$). Checked math accuracy against Scipy CPU solvers. Tests pass.
- **WDL Downsampler**: Implemented `WdlDownsampler` mapping $(257, 257) \rightarrow (33, 33)$ via strided coordinate average pooling, and visual height prior predictor `V25StageAPredictor`. Tests pass.
- **PM4 Guided Handler**: Implemented `V25Pm4GuideHandler` which snaps ML predictions to PM4 segment centroids, rejects predicted placements outside PM4 regions, and matches WMO/M2 counterparts in our reference library using the `pm4_asset_matching.scorer` library. Tests pass.
- **Differentiable Fractal Noise Generator**: Implemented `DifferentiableFractalGenerator` in PyTorch generating multi-octave noise maps by sampling a static continuous sine wave canvas via bilinear coordinate maps, and `FractalParameterHead` predicting translation seed, frequency, and soft paint masks. Tests pass.
- **Multi-Task Loss & CVPR 2026 Integrations**: Implemented CVPR 2026 additions: TexADiff texture-density weighting for alpha loss, FRAMER frequency-decomposed height loss (LF structure vs HF detail), and ReMD progressive residual correction blocks between upscaling solver stages. Tests pass.
- **Real-Data Handoff & Discovery**: Validated the C# `Harvest` tool against the staged `3.3.5` game client files, discovered maps using WDT summaries, successfully extracted a real `.npz` data shard for Zul'Aman tile (27, 29), and verified end-to-end model/loss execution on real client arrays.
- **Test Coverage**: 399 unit tests now pass successfully (0 failures, 39 skipped) across the entire harvester project, with robust local unit mocks and corrected RunPod packaging script assertions.

### V25 dataset + Phase 7 re-implementation (Spec 102, evening pass)

- **Lean V25 dataset built for real.** New `harvester/v25/dataset.py`: 8 per-tile arrays only (`minimap_rgb`, `clean_minimap_256`, `object_mask_256`, `height_257`, `wdl_height_33`, `alpha_256`, `mcly_layer_mask`, `mcly_vocab_ids`; Blosc LZ4 clevel 1, 1-tile chunks) + `index/placements/tileset_vocab` parquet. Sources: V18 substrate + V22 (tileset vocab from `mcly_tileset_ids` frequencies, MDDF/MODF placements with resolved asset paths) + V24 (`cleaned_minimap_256` via `v18_row` join, `clean_minimap()` compute fallback). Liquids/normals/holes/MCNK flags/roof-visibility-instance masks deliberately excluded (FR-102-002). Curation-manifest/map/limit filters; contiguous slice reads at build; `V25TileSource.preload()` for training (FR-102-502).
- **Old path retired.** `harvester/v25_zarr_io.py` (V22-writer subclass dragging every V22 signal) and its test deleted; `build_v25_dataset.py` rewritten around the lean builder. The old on-disk store `output/datasets/v25/3_3_5_12340.zarr` (1.3 GB) turned out to be **corrupt** — arrays + finalization.json present but no root Zarr group, `zarr.open_group` fails — flagged for manual deletion.
- **Unified decompiler per FR-102-102.** `TerrainInpaintHead` is now wired into `V25SegformerDecompiler.forward` (dict output incl. `clean_rgb`); `V25UnifiedLoss` gained optional `clean_rgb`/`h_33` terms.
- **Trainer rewritten honestly.** Old draft fed the GT-downsampled 33x33 prior to the solver and never trained Stage A or the inpaint head (single-image inference impossible), did per-row random Zarr reads, and logged train loss as val loss. New `train_v25_decompiler.py` trains all heads, teacher-forces the solver with the GT prior (`--student-prior` opt-in), validates on the universal path, preloads, records config + `peak_vram.json`, keeps `--gradient-checkpointing`/`--8bit-optimizer`/`--amp-dtype`.
- **New CLIs**: `validate_v25.py` (SC-102-003/004/005 gates + report.json) and `infer_v25_decompiler.py` (PNG/store-row → structured Zarr prediction store; optional `--pm4-records` snapping through pre-parsed `pm4_segments.parquet` records).
- **PM4 records**: `attach_pm4_segments`/`load_pm4_segment_records` round-trip `Pm4SegmentSignalRecord` through parquet — dataset builder loads the C# export JSON, Python never parses `.pm4`.
- **Proofs**: tests/v25 30/30 (10 new dataset tests); v24 77/77, v22-io 9/9 unaffected. Bounded curated build (24 Azeroth tiles, 7.1 MB, exact WDL alignment, 56 real placements) + 1-epoch CPU trainer smoke + inference export + validation report (honest FAIL on quality gates for the untrained model, PASS on alignment). Full curated store `3_3_5_12340_v25_curated_v1.zarr` (2,804 kept tiles) built from the curation manifest.
- **Open**: GPU-scale training run (explicit go required); real PM4 segment-export JSON to attach for dev maps.
- **v25.2 curation + completeness follow-up (user-directed, same evening)**: "bake in our dataset curation bucketing... ensure we have every signal we will ever need." All 40 curation-manifest columns now join per tile into the v25 `index.parquet`; mismatch-audit severity/reason columns ride along; `--height-repair-root` prefers `height_corrected_257` from `v18_mismatch_repair.zarr` over raw heights (3 corrected 3.3.5 tiles, ≤0.24 wu); trainer gained `--difficulty-buckets` (via `V25TileSource.rows_for_buckets`) + a bucket histogram at startup. Schema v25.2 adds `normal_xyz_257` (int8), `shadow_mask_256`, `object_visibility_256`, `ground_intent_height_257` (object-inpainted intended ground — likely future height target), `object_instance_mask`. Liquid-masked height/prior supervision landed (masked residual pre-FFT keeps FRAMER valid; opt-out flag; zero-gradient-into-liquid test). 37/37 v25 tests. Note: kept corpus contains a `pathological` bucket (2,150 tiles) — inclusion is a user decision, default trains on all kept tiles.
- **Era-scoped tilesets + images (user-directed, same evening)**: "tilesets are missing, and we need to handle both 0.5.3 and 3.3.5 separately... the images are literally different between them, even though they will have the same names." Vocab re-keyed to (build, normalized path) — the earlier path-shared keying was wrong; `tileset_vocab.parquet` gains build/key columns; vocab 2048 holds all 1,070 era-scoped entries. New `Harvest extract-tilesets` (BLP→PNG + manifest, with normalized-name→.blp lookup fallback since v22 paths are stored as .png/forward-slash): 316/322 (0.5.3) + 747/748 (3.3.5) decoded. `attach_tileset_images` writes vocab-aligned `tilesets/tileset_rgb_256` + `tileset_present` into the store. 39/39 v25 tests.
- **True holes follow-up (user-directed, same evening)**: "holes are like mcnk flags for liquids... WoWViewer flips them perfectly for every build." Fixed the Spec 094 holes defect at the C# source: `AdtTensorPackBuilder.ReadMcrfAndHoles` now reads MCNK `holes` uint16 @0x3C (the field `WorldTerrainHoleMask` renders) instead of flags bits 8-15; `AlphaWdtReader` threads its full ushort masks (@0x40) into `AlphaTileData.HoleFullMasks`. New `Harvest extract-holes` command (era-aware) + `attach_holes_bits` → `holes_bits_16` (int32, -1 unknown) in the v25 store; 100% join coverage (3,581 tiles, 767 with holes). **Second bug caught in verification**: `v18_mismatch_repair.zarr` is a sparse NaN overlay (5,131/5,134 rows NaN) — wholesale use had poisoned ~2,800 tiles with NaN heights; builder now merges per cell and asserts `nonfinite_height_tiles == 0`. Full C# solution builds 0 errors; 38/38 v25 tests. Latent flag for a future audit: MCRF `wmoCount` is read from 0x3C (holes) instead of 0x38 in the same function.
- **Multi-era follow-up (user-directed, same evening)**: training corpus must span eras — start 0.5.3 + 3.3.5, later re-target the image side to any era. Builder now takes index-paired multi-build source triples; tileset vocab is shared across builds keyed by **normalized tileset path** (per-build tileset ids never collide; grass in 0.5.3 == grass in 3.3.5). Schema gained liquid/flag loss signals: `liquid_mask_256` (uint8 coverage), `liquid_type_256`, `liquid_height_256`, `mcnk_flags_16` — liquid areas must be maskable out of height supervision and era restoration needs MH2O/MCLQ facts. `holes_16` stays excluded (inverted at C# source). Combined store: `output/datasets/v25/0_5_3+3_3_5_v25_curated_v1.zarr` (777 + 2,804 curated tiles; 0.5.3 cleans ride `no_object_minimap`, the viewer-rendered composite). v25 suite 32/32.

### UI release convergence plan

- Consolidated competing viewer UI plans under Spec 080: `specs/080-wow-ui-consolidation/ui-release-convergence-plan.md` is canonical; the earlier 080 plan/tasks now point to it as historical partial context.
- Added companion research, UI surface-inventory data model, and release quickstart artifacts. The plan requires every visible control to be classified and manually proven in both tabbed and legacy modes before release.
- Read-only audit confirmed a tabbed-mode Settings regression: menu routes set `_showSettingsWindow`, but `DrawSettingsWindow()` is dispatched only in the non-tabbed branch. Source fix remains pending.
- 073b converter integration is completely open and part of the release inventory. No UI source behavior changed in this planning slice.
- Extended the Spec 080 convergence plan with a UI/minimap/overlay performance workstream. It requires fixed-camera A/B Runtime Stats captures, explicit minimap cache/load activity, and one measured optimization at a time; Specs 090/093 remain the memory and renderer-counter evidence owners.

## 2026-07-10

### Spec 101 — V24.1 DA-V2 pretrained convergence model

- **Research**: Surveyed HuggingFace + GitHub for existing models that help with image-to-terrain-height convergence. Report at `docs/architecture/v24-convergence-research-2026-07-10.md`.
- **Key finding**: V24 Stage A's 190 L1 (minimap-only) is a model capacity + pretrained features problem. The 335K-param from-scratch U-Net cannot learn minimap → heightmap from ~2,000 tiles. The V23 codebase already has DA-V2-Small (24.8M params, pretrained on 62M images) with LoRA in `harvester/v23/encoder.py`.
- **Most relevant models found**:
  - Depth Anything V2 (8,430 stars) — DINOv2 encoder + DPT head, metric depth with SiLogLoss
  - Prompt Depth Anything (1,135 stars, CVPR 2025) — RGB + low-res depth prompt → high-res depth (exactly our Stage B)
  - Marigold (3,178 stars) — diffusion-based depth, sharp outputs
  - pix2pix/PatchGAN (25,184 stars) — auxiliary adversarial loss (Spec 100 approach)
- **Bugs found in training code**: OneCycleLR `total_steps=args.epochs` (should be `n_batches * epochs`); `scheduler.step()` called per-epoch instead of per-batch for OneCycleLR; plain L1 loss (should use SiLogLoss for metric depth).
- **Spec 101 created** at `specs/101-v241-dav2-model/spec.md` with 7 slices, checklist at `specs/101-v241-dav2-model/checklists/requirements.md`.
- **Slice 1**: `StageADAV2` model class in `stage_a.py` — wraps `DepthAnythingV2SmallEncoder` from V23 with a DPT-style head outputting 33×33 quincunx → 17×17 outer + 16×16 inner. Total ~25M params, ~1-2M trainable (LoRA + patch proj + head). Backbone frozen.
- **Slice 2**: `SiLogLoss` class in `stage_a.py` — scale-invariant log loss with shift parameter for negative heights. `hybrid_loss` = 0.7 SiLogLoss + 0.3 L1.
- **Slice 3**: Scheduler fix in `train_v24_stage_a.py` — OneCycleLR `total_steps` corrected to `n_batches * epochs`; per-batch stepping for OneCycleLR, per-epoch for CosineAnnealingLR. New `--scheduler` flag.
- **Slice 4**: `--dav2` flag on `train_v24_stage_a.py` — loads pretrained DA-V2-Small, hybrid loss, lr=5e-6, batch_size=8. Checkpoint records `model_type`, `loss_type`, `scheduler_type`, `dav2`, `guided`.
- **Slice 5**: 15 new tests in `test_stage_a_dav2.py` — model shape (3ch + 9ch), param count (≤26M total, ≤2M trainable), backbone frozen, offline load, SiLogLoss (positive, negative, gradient, perfect, zero-weight), hybrid loss (positive, gradient), `build_dav2_input` (3ch, 9ch, no-normal). All pass.
- **Full v24 suite**: 67 passed, 1 deselected, 0 failed (was 46 before Spec 101). No regressions.
- **Slice 6**: `StageBPromptDA` model class in `stage_b.py` — DA-V2-Small encoder (4ch: 3 RGB + 1 depth prompt) + LoRA + DPT head → 257×257 heightmap. `build_promptda_input` assembles (4, 256, 256) from minimap + WDL prior. 4 tests pass.
- **Slice 7**: `WDLDiscriminator` PatchGAN in new `discriminator.py` — Conv(stride=2)→Conv(stride=2)→Conv→Conv with LeakyReLU, ~693K params (base=32). `gan_step` helper does D step + G step with BCEWithLogits + L1. `_render_quincunx_33` interleaves outer+inner into 33×33. 6 tests pass (shape 33+257, param count, gradient, quincunx render, GAN step updates both models).
- **Full v24 suite**: 77 passed, 1 deselected, 0 failed (was 46 before Spec 101). Zero regressions.
- **All 7 slices implemented.**
- **VRAM optimizations**: `--8bit-optimizer` (bitsandbytes 8-bit AdamW), `--gradient-checkpointing` (recompute activations), `--lora-rank` (configurable), `--weight-decay` (regularization), V18 preload cache freed after tensor extraction.
- **Training results**:
  - Run 1 (lr=5e-6, 40ep, 500 tiles): val_l1=91.11. LR too low for LoRA.
  - Run 2 (lr=1e-4, 200ep, 500 tiles, LoRA 16): best val_l1=48.13 @ ep 143, overtraining (train 17/val 49).
  - Run 3 (lr=1e-4, 200ep, 2,011 tiles, LoRA 32, wd=1e-3, bs=8): cuDNN OOM at ep 132.
  - Run 4 (in progress): lr=1e-4, 200ep, 2,011 tiles, LoRA 32, wd=1e-3, bs=8, bf16, 8-bit opt, grad ckpt. 26.3M total, 1.74M trainable.
- **Progress**: val_l1 went 190 (old U-Net) → 91 → 48 (DA-V2). Still improving with more data + higher LoRA rank.
- **Spec 101 plan.md and tasks.md created** at `specs/101-v241-dav2-model/`.

## 2026-07-09

### Spec 094 V24 full-scale curated training

- User-directed full-scale curated open-world training (50 epochs, `openworld_curated.zarr`, curated 2,011 tiles).
- **Data-loading bug squashed:** Two independent bottlenecks caused ~1 s/tile load times:
  1. `clean_minimap()` median-fill loop (512 passes) ran on every tile load. **Fix:** replaced with raw-minimap normalization. Pre-computed cleaning should be a dataset-build step.
  2. Per-tile random Zarr reads from V18 store (read amplification via large chunks) caused ~25 min sequential load. **Fix:** added `TileSource.preload(rows)` that reads V18 in one contiguous Zarr slice, then caches in memory. Load time dropped to < 10 s.
- Both `train_v24_stage_a.py` and `train_v24_stage_b.py` updated to call `source.preload()` before iteration.
- **Full 50-epoch curated run completed.** Stage A val_l1_real_cells=0.397 (better than previous 0.412). Stage B final L1=0.857 (5x better than baselines 4.30/4.20). 7/10 SC checks pass (SC-001 confidence bound is the known rough-terrain sampling-phase disagreement).
- **Pre-computed cleaned minimaps:** Wrote `scripts/precompute_v24_cleaned_minimaps.py` — one-time build that stores `clean_minimap()` output as `cleaned_minimap_256` array in the V24 Zarr store. Modified `TileSource.load()` in `tiles.py` to prefer the stored array when available (fast Zarr read, no per-load computation).
- **Cleaned-minimap retrain running now** — 2011 tiles, 50+ epochs, target final L1 < 0.5.

### Spec 096 V24 minimap-only deployment wiring

- Closed the deployment gap from Spec 094 FR-013 / US3 scenario 5: there was no way to drop a bare PNG minimap into a CLI and get a WDL prior NPZ out. Built it. Four small slices, each validated before the next.
- **Slice 1 — train the minimap-only Stage A.** Trainer `--minimap-only` flag existed but had never been run on real data. 50 epochs on `3_3_5_12340_v24_all_v1.zarr` (2,241 train / 560 val, autotune batch=512, peak_vram=0.005 GB). `stage_a.pt` 334,965 params, `minimap_only: true` in config, best_val_l1=190.31, overtraining detected. Patched the trainer's save code to record `minimap_only: bool` and the right `in_channels` (was hard-coded to 13 before).
- **Slice 2 — `scripts/infer_v24_stage_a_png.py`.** ~200 lines, PIL+numpy+torch+harvester.v24, no matplotlib, no V24 store, no V18 store, no staged client. Loads PNG → resize 256×256 → mean-pool 64×64 → StageAMinimapOnly forward → bilinear upsample (1,1,64,64)→(1,1,33,33) → outer (17,17) + inner (16,16) → `* HEIGHT_SCALE=100` → world units. Writes NPZ + optional 4-up preview PNG (input | outer | inner | quincunx). Strict-checkpoint refusal of 13-channel cheat checkpoints. Smoke on a real PNG: 212 ms wall, 0.005 GB peak VRAM, world_min=51.95, world_max=319.02.
- **Slice 3 — `validate_v24.py --minimap-only-checkpoint`.** New `stage_a_minimap_only` block + `SC-002-MINIMAP` gate. Ran full validation. Result: minimap-only val_l1=190.31 vs cheat val_l1=1.21 vs `block_reduce` baseline 1.31. The minimap-only regime is **158× worse than the cheat regime** on the same held-out tiles. `SC-002-MINIMAP` gate is **FAIL** (recorded honestly per Risk 1 in the spec). All other checks pass except the existing SC-001 confidence bound (known data-quality issue). Report: `output/v24_validation/v24_minimap_only_validation_20260709/report.json`.
- **Slice 4 — docs.** Architecture doc at `docs/architecture/v24-minimap-deploy-2026-07-09.md`. Memory bank + this progress entry. Also fixed the dangling `>>>>> REPLACE` marker from the prior session.
- **Honest deployment result.** The CLI works. The model it loads is not yet accurate enough to be useful — the bare RGB minimap does not carry enough signal to predict the WDL prior at the precision the WDL grid requires. This is the documented Risk 1 from the spec; the slice still ships because the CLI, the trained checkpoint, and the test suite are all real.
- **Next step: Spec 095 (learned minimap cleaner).** A small U-Net that takes a raw minimap + V18 `object_precise_mask` and outputs a "terrain-only" minimap. Run as a pre-step to Stage A. Most likely path to closing the 158× gap. If 095 doesn't get us there, Spec 097 is "send the PNG to a server that has the staged client" — the honest fallback.
- Test suite: 36/36 v24 tests pass (up from 31 before Spec 096).

### Spec 096 follow-on — one-shot wrapper + OBJ mesh export

- Wrote `v24_run_on_png.py` and `v24_prior_to_obj.py`. One command, any PNG, get back a WDL prior NPZ + 4-up preview PNG + 257×257 textured OBJ mesh. `--batch-dir <dir>` runs the wrapper on every PNG in a folder and stitches the tiles into a single grid OBJ with an atlas.
- **X-flip fix:** the OBJ was opening mirrored along the X axis because the source PNG's image-X runs opposite to the world-X. The prior is now `np.fliplr`'d at load time so the mesh opens correctly in any 3D viewer. Discovered + fixed by the user testing a real PNG and reporting the bug.
- Test suite: 36/36 v24 tests pass.

### Spec 097 Slice 1 — per-map V18 Zarr → stitched OBJ + baked atlas

- Wrote `scripts/v24_export_map.py`. CLI: `--v18-store --map [--build] [--curation-manifest] [--output] [--device] [--seed]`. Reads a per-map V18 Zarr, runs the V24 minimap-only Stage A prior on each tile, upsamples the (17,17)+(16,16) prior to 257×257, applies edge alignment across tile boundaries, and writes a single OBJ + atlas.
- **Edge alignment:** the 16-pixel band on each side of every shared border is averaged; corner cells (4-way) inherit the average. The OBJ opens with continuous height across the seams — no visible hard step at the 256-pixel-tile borders.
- **Northrend smoke:** 29 rows × 39 cols = 1,131 tiles, 7,453 × 10,023 heightmap, 74.7M vertices, world -786.9..409.3, **6.3 min** wall time on the 12 GB GPU. Output: `Northrend.obj` + `Northrend.atlas.png` + `Northrend_manifest.json` + `tiles/<tx>_<ty>.prior.npz` (1,131 NPZs). Reproduce: `uv run python scripts/v24_export_map.py --v18-store output/datasets/v18/3_3_5_12340.zarr --map Northrend`.
- Wrote `tests/v24/test_export_map.py` — 4 tests (3 seam-alignment, 1 V18 loader smoke). All green.
- Test suite: **40/40 v24 tests pass** (was 36 before Spec 097 Slice 1).

### Spec 097 Slices 2/3/4 — NOT shipped this session (documented handoff)

- WDL file writer and ADT file writer are substantial binary format work. The proper round-trip path is a small `write` subcommand on the existing C# `WowViewer.Tool.WdlRead` shim — a multi-step C# change that does not belong in a quick chat session.
- Faking the writers in Python would produce files the C# readers cannot open. Worse than no output.
- Spec/plan/tasks/checklist are all written under `wow-viewer/specs/097-v18-to-wdl-adt/`. Next session: pick up at Slice 2.

## 2026-07-07

### Spec 080 world object wireframe correction

- Fixed the active `wow-viewer` world `M2/WMO WF` path so it no longer toggles M2/WMO renderers into line-only mode or implicitly enables hover-only wireframe reveal.
- World object wireframe now renders a visible overlay for every currently visible WMO/M2 instance over the normal solid object pass; the previous hover reveal path is kept separate.
- Validation: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with 0 errors and existing warning noise. Manual world-map viewport proof remains open.

### Spec 094 V24 curated open-world training run

- Wired V24 into the existing V18 curation manifest: `build_wdl_prior.py` gained `--curation-manifest` + `--difficulty-bucket` (join on `(build, tile_id)`, keep only `keep==True`), replacing the naive `--min-height-std` heuristic with the curated, mismatch-omitted corpus the rest of the model stack already uses. Committed `d6544f1a`. 30/30 v24 pytest still green.
- Ran the full pipeline on the curated open-world corpus for `3_3_5_12340`: 2,011 kept tiles across Azeroth (488), Kalimdor (741), Northrend (423), Expansion01 (359), all `hard`/`pathological` buckets, 76% real / 24% synthetic WDL coverage, 1,609 train / 402 val, 30 epochs/stage.
- Stage A: real-cell L1 0.412 < synth-cell L1 6.54 < `block_reduce` baseline 1.76 (SC-002 PASS). Stage B: final L1 **0.649** < upsampled-prior 4.31 < `block_reduce+bilinear` 4.20 — beats both no-learning baselines by ~6.5× across four terrain-distinct continents (SC-003 PASS). SC-004 determinism + SC-005 hardware envelope (peak VRAM 0.187 GB, max 0.111 s/tile) PASS. SC-001 confidence bound FAIL (75.9% vs 80%) — the same documented rough-terrain sampling-phase disagreement, a data-quality signal not a pipeline defect.
- This is the reliable, terrain-generalizable proof (vs. the 50-tile Northrend bounded pipeline proof from 2026-07-06). Report: `output/v24_validation/v24_openworld_curated_20260706/report.json`; updated `docs/architecture/v24-validation-2026-07-06.md` with the curated run section + reproduce commands.

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
