# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## 2026-06-19 — V21 TERRAIN-MESH-ONLY PIVOT (V19/V20 abandoned)
**V19 and V20 are abandoned.** Both over-engineered terrain mesh reconstruction away from the proven V18 path. V19 added a dead `bounds` head + duplicate `local_height` head fighting the global head. V20 bolted on 4 models (segmentor/brush-classifier/inpainter/placement-restorer) — none produce a usable terrain mesh.

**Root cause of bad validation:** three stacked bugs ignored the `object_filtered_mask` (WMOs + player objects, NO trees) that the user explicitly wanted:
1. `v16_1_dataset.py` mask precedence put `object_precise_mask` (includes trees) BEFORE `object_filtered_mask` — filtered was unreachable.
2. The filtered result fed `weight_257`, but `terrain_valid_mask_257` (built from coarse `object_roof_mask` rectangles) was a separate path.
3. `_height_loss` in `train_v16_1_common.py` read `terrain_valid_mask_257` (coarse), never `weight_257` (filtered).

**Fix landed (uncommitted):**
- `v16_1_dataset.py`: flipped precedence — `object_filtered_mask` now wins over `object_precise_mask`.
- `train_v16_1_common.py` `_height_loss`: now uses `batch["weight_257"]` (filtered) instead of `batch["terrain_valid_mask_257"]` (coarse).
- Verified on tile 600: `weight_257 == 1 - object_filtered_mask` (exact). Differs from precise by 10.7% (the trees precise wrongly eliminated).

**V21 = V18 height model + filtered-mask loss. No new model. No new dataset.**
- Model: `V161HeightModel` (3.5M params, 3ch minimap in → 1ch height_257 out, single head).
- Trainer: `train_v18.py height` (stable, efficient, proven).
- Normals: OFF for now (`--normal-consistency-weight` defaults 0). Separate minimap→normals model later.
- Liquids: ignored. Water = flat plane later. wl* files place water; human fixes in map editor. `height_257` target already includes terrain under water — correct for mesh reconstruction.
- Smoke validated: 1 epoch, val loss 0.7395, 2.2GB VRAM, preview at `models/v18/height/runs/v21_smoke/`.

**Next:** full training run on V18 dataset (0_5_3_3368 + 3_3_5_12340) with filtered-mask loss + curation manifest. Then terrain mesh export.

**V21 DOWNSTREAM REFINEMENT MODEL PLAN (Phase 2+, NOT NOW):** Once the main height model (Phase 1) is validated against real terrain mesh output, two optional refinement models may be built. These are NOT required — they're opt-in tools based on whether the main model's outputs need refinement. Each is a separate small model, separate training script, separate checkpoint, predicting ONE signal. RULE 7 applies.

- **Liquid model** (optional, later): minimap → liquid mask/type. Learns where water exists so a human map-editor pass can place wl* files correctly. NOT for height reconstruction — water = flat plane placed separately. Independent checkpoint, trained independently.
- **Object/masking model** (optional, later): minimap → object masks (the `object_filtered_mask` we now use for loss). Useful if you want to predict the filtered mask at inference time when no placement metadata exists. Independent checkpoint, trained independently. This is the V18 object-roof-identifier concept (spec 025) but bounded to filtered-mask prediction only — NOT the V20 four-model pile.
- **Normals model** (optional, later): minimap → normals_257. Separate from height. Already exists as `V161NormalModel` + `train_v18.py normal`. Independent checkpoint.
- **Residual chaining (eventually):** each downstream model's output becomes an OPTIONAL input to refinement passes on the main height model. Not shared weights, not joint training — chain via outputs only. Build this only if Phase 1 outputs prove it's needed.

**DO NOT START PHASE 2 UNTIL PHASE 1 IS VALIDATED.** Phase 1 = main height model trained + terrain mesh exported + visually inspected against real game terrain. RULE 8.

**CURATION IS MANDATORY.** Always use `--curation-manifest` pointing at `output/datasets/v18/curation/v19_terrain_1800/kept_tiles.parquet` (or a newer curated manifest) with `--curation-min-terrain-validity 0.20 --curation-min-minimap-usefulness 0.10 --curation-reject-what-plate`. Without curation, whiteplate tiles with mismatched heightmaps poison training (4108 → 834 real terrain tiles on 3_3_5_12340). NOTE: the CLI flag is still spelled `--curation-reject-what-plate` (code identifier — rename is a separate refactor), but the concept is "whiteplate" — blank genesis tiles with no real terrain data.

## 2026-06-19 — V19 Minimal-Signal Height Regressor and Dataset Refactoring [ABANDONED — see V21 pivot above]
- Refactored `V19Dataset` to inherit from `V161Dataset` directly, ensuring full compatibility with the unified training pipeline (difficulty bucket sampling, epoch rotation, build balancing).
- Created a specialized, fully optimized standalone `train_v19.py` script based on `train_v16_1_common.py`. It retains the advanced training optimizations (VRAM autotuning, mixed precision, deterministic epoch sampler, early stopping, compile) but configured specifically for the V19 height regression task.
- Added CLI parameters: `--input-channels` (3 or 6) and `--edge-weight` (Sobel edge loss).
- Training loop, model compilation, dataset loading, and device routing validated in dry run.
- **Abandoned**: V19 added a dead `bounds` head (no gradient) + duplicate `local_height` head fighting `global_height`. Both predict identical target. Over-engineering detour. V21 uses the proven V18 `V161HeightModel` single-head instead.

## 2026-06-17 — Surface correlation matcher — PIVOT from hull to per-triangle matching

**Hull footprint matching ABANDONED.** Produced false positives (Ironforge/Darnassis at 0.999 overlap despite NOT being in dev map). User confirmed: "how in the fuck are we still using footprints to figure out what objects are?! use the fucking correlation of surfaces in pm4's to real wmo objects!"

**Surface correlation implemented (commit 21aa0064).** PM4 MSUR surfaces triangulated → per-triangle sorted edge lengths binned to integers (transform-invariant hash) → histogram intersection matching against WMO MOVI/MOVT collision triangles.

**Results (1604 PM4 vs 2790 WMO surface fingerprints):**
- 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible
- P@1=1.3%, P@3=10.3% (2.3x improvement over hull P@3=4.5%)
- NO false positives — Ironforge/Darnassis eliminated
- 12 correct top-1: GoldshireInn (0.86 PM4 coverage), classicalelfruins, arathistonebridge, orchut
- GoldshireInn matches tiles 0_2/1_1 at 0.86 coverage but ADT doesn't list it — likely ADT gap, not matcher error

**Code**: `Pm4SurfaceCorrelationExtractor` (triangulate + histogram), `Pm4SurfaceCorrelationMatcher` (histogram intersection + F1), CLI: `build-wmo-surface-db`, `extract-pm4-surfaces`, `match-surfaces`.

**Remaining gaps**: WMO DB coverage (503/1985), dev map ADT unreliability, edge bin size (1.0 may be too coarse), no triangle area in histogram key.

**Code**: `Pm4FingerprintExtractor` (PCA + hull), `Pm4FingerprintMatcher` (prefilter + EvaluateMetrics + flip), `Pm4FingerprintBuildSupport` (WMO DB builder), CLI: `build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`. 15 unit tests pass.

**Next**: Phase 5 (validate against ADT ground truth), reduce ambiguous count (add surface/vertex count signals, per-group matching, tune thresholds). WMO enumeration still 506/1985 (listfile needed).

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
1. WMO DB coverage: 503/1985 WMOs — archive enumeration misses 75%. Need listfile for full coverage. This is the #1 driver of low ADT validation precision (1.8% P@1).
2. Dev map ADT unreliability: dev map mixes WMOs from all zones, ADT placements are sparse. Need validation on a real game map (Elwynn, Darnassus, etc.).
3. Remaining ambiguity: 502/1604 — mostly Stormwind vs StormwindHarbor (genuinely identical architecture). May need CK24 ObjectId mapping or tile context to resolve.

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- WMO archive enumeration misses ~75% of WMOs (506/1985) — need listfile-based enumeration (spec 065 Task 2.2)
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)