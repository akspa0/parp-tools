# Progress

## 2026-06-19 — V21 TERRAIN-MESH-ONLY PIVOT (V19/V20 abandoned)

### What landed (uncommitted)
- Diagnosed why validation looked wrong: three stacked bugs ignored `object_filtered_mask` (the mask the user explicitly wanted — WMOs + player objects, NO trees).
  1. `v16_1_dataset.py` mask precedence: `object_precise_mask` (includes trees) shadowed `object_filtered_mask` — filtered unreachable.
  2. Filtered result → `weight_257`; but `terrain_valid_mask_257` built separately from coarse `object_roof_mask` rectangles.
  3. `_height_loss` in `train_v16_1_common.py` read `terrain_valid_mask_257` (coarse), never `weight_257` (filtered).
- Fix 1: flipped precedence in `v16_1_dataset.py:269` — `object_filtered_mask` now wins over `object_precise_mask`.
- Fix 2: `_height_loss` in `train_v16_1_common.py:1167` now uses `batch["weight_257"]` (filtered) instead of `batch["terrain_valid_mask_257"]` (coarse).
- Verified on tile 600: `weight_257 == 1 - object_filtered_mask` (exact). Differs from precise by 10.7% (trees kept as training signal — correct, terrain under trees is valid).
- Smoke run: `train_v18.py height` on 3_3_5_12340, 64 train/16 val, 1 epoch, val loss 0.7395, 2.2GB VRAM. Preview at `models/v18/height/runs/v21_smoke/best_epoch_0001.png`.

### V21 contract (simple, what user wanted all along)
- Model: `V161HeightModel` (3.5M params, 3ch minimap → 1ch height_257, single head). NO new model.
- Trainer: `train_v18.py height` (stable, efficient, proven). NO new trainer.
- Loss: masked L1 on height, gated by `weight_257` = `1 - object_filtered_mask`.
- Normals: OFF (separate minimap→normals model later).
- Liquids: ignored. Water = flat plane later. `height_257` target includes terrain under water.
- V19 (dead bounds head + duplicate height head) and V20 (4-model pile) both abandoned.

### Next
- Full training run on V18 dataset (0_5_3_3368 + 3_3_5_12340) with filtered-mask loss.
- Terrain mesh export from trained checkpoint.
- Commit the two-file fix once full run validates.

## 2026-06-19 — V20 Precise Object Mask & Inpainting Target Correction [ABANDONED — see V21 pivot]

### What landed
- Modified [v20_dataset.py](file:///i:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/v20_dataset.py) to load `object_precise_mask` (contains precise 3D silhouettes/polygons) directly from Zarr. Added edge padding crop/pad logic to correctly map it to the `object_precise_mask_256` and `object_precise_mask_257` loader targets instead of falling back to bounding-box rectangles (`object_roof_mask`).
- Modified [patch_v20_signals.py](file:///i:/parp/parp-tools/wow-viewer/data-harvester/scripts/patch_v20_signals.py) to prioritize `object_precise_mask` for inpainting, and dynamically pad 256-width mask arrays to 257.
- Simplified loss calculations and Autocast blocks in `train_v20_segmentor.py`.
- Verified pathway executing dry runs on both patching and training components.
- **Abandoned**: V20's 4 models (segmentor/brush-classifier/inpainter/placement-restorer) are not terrain mesh reconstruction. User wants simple terrain meshes, not years-of-honing asset recovery.

## 2026-06-19 — V19 Minimal-Signal Height Regressor and Dataset Refactoring [ABANDONED — see V21 pivot]

### What landed
- Refactored `V19Dataset` (in `v19_dataset.py`) to subclass `V161Dataset`, gaining direct compatibility with all of the unified trainer's data pipeline features (difficulty bucket sampling, rotating determinism, etc.).
- Created a specialized standalone training script `train_v19.py` containing the advanced training loop (mixed precision, autotuning, early stopping, and compiles).
- Custom Sobel edge loss and input channel toggling (3 or 6) implemented and validated in a local dry run.

## 2026-06-17 — Fingerprint-database PM4→WMO matching implemented (Phases 1-4)

### What landed
- Phase 1: `Pm4FingerprintExtractor` (PCA normalization + convex hull), `Pm4FingerprintContracts` (serializable records). 9 unit tests.
- Phase 2: `Pm4FingerprintBuildSupport` (WMO collision → fingerprint DB). CLI `build-wmo-fingerprint-db`. 503 root + 2287 group = 2790 fingerprints from 506 staged 3.3.5 WMOs.
- Phase 3: CLI `extract-pm4-fingerprints`. 1604 CK24 group fingerprints from 616 dev PM4s. Type distribution matches: 0x42=584, 0x43=466, 0x41=161, 0xC1=100.
- Phase 4: `Pm4FingerprintMatcher` (sorted-dim prefilter + EvaluateMetrics + 4-flip PCA). CLI `match-fingerprints`. 6 unit tests. Real-data: 50 matched, 1203 ambiguous, 78 unresolved, 273 ineligible. Top: Ironforge 0.94/0.999, Stormwind Harbor 0.92/0.98.

### Phase 5: ADT validation — DONE (commit 4974a5cc)
- CLI `pm4 validate-matches`: reads ADT obj0 placements, checks fingerprint-DB top-1/top-3 against ADT WMO placement list.
- Results: 138/616 tiles have ADT, 913 CK24 groups with ground truth. P@1=1.8%, P@3=4.5%.
- Correct matches prove approach works: nightelfmoonwell, stormwindharbor, magetower, arathistonebridge.
- Low precision driven by: WMO DB coverage (503/1985), dev map ADT unreliability, geometric similarity between city WMO groups.
- Next: expand WMO coverage with listfile, validate on real game map (not dev).

### Commits
- fe7a304e: spec 065 pivot
- e8d4f1d5: Phase 1 fingerprint extraction library
- 4db79689: Phase 2-3 WMO DB + PM4 extraction CLI
- c7239549: Phase 4 fingerprint matcher

### Next
- Phase 5: validate against ADT ground truth (precision@1/@3)
- Reduce ambiguous count: add surface/vertex count ratio, TypeFlags profile, per-group matching
- WMO enumeration: 506/1985 — need listfile for full coverage

## 2026-06-17 — Spec 065 revised: fingerprint-database approach (route change)

### What changed
- ADT-based PM4→WMO matching ABANDONED. `correlate-models`/`sweep-correlate` need ADT anchors (222 PM4-only tiles have none). `identify-models` is bounding-box-only (too coarse). `match-assets` has ADT-dependent `sameTileBonus` (dead on PM4-only tiles).
- New approach: fingerprint database from WMO collision geometry (MOVT/MOVI) via `Pm4CorrelationMath` convex-hull footprint + PCA normalization. Match PM4 CK24 fingerprints against WMO DB. No ADT for matching.
- Spec 065 rewritten: spec.md, plan.md, tasks.md all revised. 6 phases: fingerprint extraction library → WMO DB → PM4 fingerprints → matching → ADT validation → generator (downstream).
- Legacy commands kept for validation ground truth, not as primary matchers.

### Key insight
- The right correlation approach (`Pm4CorrelationMath`: convex hull footprint overlap, symmetric footprint distance, planar gap) already exists but was never used to build a fingerprint database. Instead, we relied on ADT placements for position info. The fix: use the correlation math to extract rotation-invariant fingerprints from WMO collision geometry directly, store to DB, match PM4 against it.

## 2026-06-16 — PM4 matcher is broken; spec 065 written

### Root cause diagnosis
- `pm4 match-report` produces "Candidate Count: 0" for ALL PM4 objects on ALL tiles because it compares PM4 raw-ADT coordinates against WoW world coordinates without any conversion. `ConvertPm4VertexToWorld` produces `(tileY*533+mappedU, tileX*533+mappedV, localUp)` but `AdtPlacementReader` produces `(17066-rawY, 17066-rawX, rawZ)`. Gap is ~24000 units — all spatial matching is dead.
- `pm4 match-assets` (the shape scorer) is architecturally correct but produces sub-threshold scores (~0.42 vs 0.45 minimum) because PM4 segments are individual surfaces, not whole-object groupings. On tile 22_18, 40543 PM4 objects become only 92 segments, each scored against 1985 WMO models.
- Tile 22_18 is NOT the oil platform (2 WMO placements in ADT). It's a "snowball fort" — multiple WMOs stacked together including Ulduar titan structures. CK24 0x0042084C type 66 is the dominant object.
- The `pm4 match-report` command is architecturally wrong for data recovery — it's placement-centered (needs existing ADT placements as anchors) instead of PM4-object-centered (match PM4 shapes against model shapes from archive).

### Fix attempts
- Attempted coordinate conversion in `Pm4MatchSupport` (RawAdtToWowWorld) — reverted. The match-report approach itself is wrong, not just the coordinates.

### Spec written
- `wow-viewer/specs/065-pm4-object-identity/spec.md` — 5-phase plan:
  - P1: Coordinate fix + known-tile correlation on 24_35
  - P1: CK24 identity table (CK24 → model path mapping)
  - P1: Synthetic PM4 signal generation from WMO collision
  - P2: CK24-grouped segment scoring (merge surfaces per CK24)
  - P2: Unknown-tile resolution using identity table + shape fallback

### Key insight
- The right approach is: generate what PM4 data WOULD look like for a given WMO/M2 model, then compare that synthetic data against real PM4 data. Not position-matching against broken ADT placements.

## 2026-06-16 — PM4 ADT writing reverted, replaced with match-report
- Deleted Pm4AdtWriter, Pm4BinaryAdtPatcher — ADT patching was corrupting output
- Replaced `pm4 write-adt` with `pm4 match-report` (human-readable markdown)
- LkAdtWriter untouched — not part of PM4 matcher work
- Checkpoint commit: 5133bfe3

## 2026-06-14 — Spec consolidation + tool fixes
- Replaced engine-program plan with viewer-first + UE bridge (509→35 lines)
- Archived 005, 020, 026, 033, 036, 059 (done/dead)
- Fixed stale status: 025→Complete, 060→Complete, 043→stale noted
- Marked research specs 030/031/032/038/040 consumed by 056
- Fixed 044 T006: removed dead MK Dataset GUI
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ
- Ran weak signal on 0.5.3/0.5.5/1.12.1/3.0.1 maps — proven on real data
- Current focus: **046 PM4 asset matching** (C# done, Python lane needed)

## 2026-06-15 — Session polluted by hallucinations and wrong assumptions
- Implemented `pm4 dump-collision` command and WMO validation (works, 40 OIDs)
- Spent too long on tangents, wrong assumptions about M2/MD20, and coordinate systems
- Key deliverables: collision dumper tool, serialization fixes, Python scorer validation
- Memory bank updated. Needs fresh session with clear direction.
