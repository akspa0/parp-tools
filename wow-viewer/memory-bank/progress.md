# Progress — wow-viewer

Last updated: 2026-08-08

**This file is a dated ledger of what shipped, newest first.** One entry per session, a few lines
each. Findings and how-it-works go in the workstream file; this only records *that* it happened and
what the evidence was. See "Memory bank layout" in `coding_standards.md`.

Current state and open work: [activeContext.md](activeContext.md).

## 2026-08-08 — Spec 134 control-corpus and object-sieve implementation

Branch `134-v60-unified-dataset-model`.

- Reframed the initial v60 experiment around a small synthetic control corpus; the abandoned
  multi-client v50 harvest is not evidence of a working v60 dataset.
- Added deterministic fBm/ridged-fractal, dendritic lightning-burn, global 2×2 cross-tile
  lightning/burn, mountainous, arbitrary-angle sheer-dropoff, and zone-style-blend families to the
  C# control generator. The default taxonomy is 27 families × 4 variants = 108 rows across four
  complexity buckets. Non-grid fields carry deterministic sub-cell offsets; `chunk_grid` is the
  explicit aligned diagnostic.
- Added fail-closed cross-tile metadata validation, shared-pattern-ID checks, duplicate-position
  checks, alignment metadata validation, stitched height/shadow visual atlas output, and visual
  coverage contract fields.
- Added a sibling `object-sieve-v1` writer with 540 deterministic rows: four object families across
  none/sparse/dense/overlap/boundary-crossing regimes, plus clean terrain and contamination-mask
  targets. Added object manifest/hash validation, input/mask atlases, and clean-only,
  auxiliary-mask-loss, and predicted-mask-guided model variants.
- Evidence: harvest tool build 0 errors; focused Python checks remain the final local gate. No corpus
  generation, client harvest, or training was run by Codex.

## 2026-08-08 — Spec 134 object-sieve design extension

- Added the next bounded signal lane to the Spec 134 plan: synthetic objects over canonical terrain,
  exact clean terrain-shadow targets, and a separate screen-space contamination mask.
- Defined three ablations: clean-output-only, auxiliary mask loss, and predicted-mask guidance. The
  ground-truth mask remains loss-side supervision only; it is never an inference input.
- Added object-control data-model and experiment contracts, placement regimes (none/sparse/dense/
  overlap/boundary-crossing), task decomposition, and continuity notes. Implementation waits for the
  user-run terrain control corpus gate.

## 2026-08-08 — PM4/PD4 versioning & Specs 135, 136, 137 landed

- **PM4/PD4 Versioning**: Created `Pm4VersionFormatter.cs` for correct MVER version string parsing (`0x10` Cataclysm = v16, `0x30` WoD = v48). Wired to viewer status bar and CLI inspection output.
- **Spec 135 (Phased Terrain Dual-Map Overlay)**: `ITerrainAdapter`, `StandardTerrainAdapter`, `TerrainManager`, `WorldScene` support `SecondaryOverlayMap` / `OverlayMapName`. Real-time ADT tile replacement from overlay directories with live tile eviction & streaming. Searchable map dropdown picker added to UI.
- **Spec 136 (M2 Doodad Performance Optimization)**: Enabled batched instancing (`BeginBatch` + `RenderInstance`) for M2 adapter models without particles/ribbons by updating `ModelRenderer.RequiresUnbatchedWorldRender`. Deduplicated `UpdateAnimation` calls in `WorldScene.cs` so shared models advance at most once per frame. Restored smooth framerates on dense doodad maps (>60 FPS).
- **Spec 137 (Phased Minimap Overlay & Consistent Teleport)**: Updated `MinimapRenderer` & `MinimapHelpers` to render active secondary overlay tile BLPs on the minimap surface. Unified fullscreen minimap to use 3-click armed teleport (`MinimapTeleportMode.Armed`) matching small minimap panel.

## 2026-08-07 — Spec 134 route reset (not a working v60 dataset)

Branch `134-v60-unified-dataset-model`.

- The earlier datastore/harvest consolidation was not accepted as a working v60 dataset and is no
  longer the first experiment.
- Spec 134 now gates on the synthetic control corpus, followed only later by a tiny explicit 0.x/1.x
  albedo-normalized transfer sample. Later client builds remain out of the initial scope.

## 2026-08-05 — Spec 132 Phase 1: three-tier brush-signature classification

Branch `132-terrain-brush-signature-classification`, commit `f19fc774`.

- **`classify.py`** — new library module with `compute_signal_tier()`: strong/normal/weak/na with
  published criteria (weak < 5 range, normal 5-50 range or 8-64 levels or low correlation).
- **`v50_tile_classify.py`** — CLI: reads V50 Zarr store or NPZ dir -> per-tier CSV/JSON + summary.
- **`tile_inventory.py`** — gains `signal_class` / `signal_class_evidence` per row + `by_signal_class`
  summary. `tile_composite.py` gains green normal-tier outline. Both archaeology orchestrators run
  the classifier.
- **13 new tests pass**; 22 existing inventory/composite tests still pass (no regressions).
- **tasks.md** covers all 6 phases; Phase 2 (nested weak signal detection) is next.

## 2026-08-04 — Archaeology pipeline built, spec 132 drafted

- **Single-command archaeology pipeline** (`run-archaeology.ps1`): harvest MPQ → V50 Zarr store → tile inventory → synthesis → composites. Proven on TBC 2.0.0.5610 (Expansion01, 741 tiles).
- **Batch archaeology** (`run-batch-archaeology.ps1`): discovers all 15 1.x Windows clients in H:\CLIENTS, runs pipeline on each.
- **V50 store builder from NPZ** (`build_v50_store_from_npz.py`): reads NPZ shards, builds proper V50-format Zarr store with index.parquet, then runs full archaeology suite.
- **Spec 132** (`terrain-brush-signature-classification`): 6 user stories for three-tier classification, nested signals, brush-texture correlation, cross-map alignment, rescale boundary detection, and predictive model. Plan written.
- **Harvested data on disk**: `output/archaeology/2_0_0_5610/` with NPZ, Zarr store, and archaeology results for Expansion01.

## 2026-08-04 — PM4 scene graph: tree view restored, MSLK linking summary added

Spec 131, branch `131-pm4-scene-graph-doodads`. Detail:
[workstream-pm4-decode.md](workstream-pm4-decode.md).

- **PM4 Scene Graph panel restored as a full scene outliner** (Blender-style). Two modes:
  "Full Scene" (tile → CK24 → Part hierarchy with MSLK group/MPRL refs at each level, search
  filter, right-click select/frame-all context menu) and "Selected Object" (existing detailed
  decomposition, now with MSLK linking info). Clicking any row selects the object and frames the
  camera.
- **MSLK Linking Summary** section added to the outliner: anchor-only vs path-window link counts,
  component link coverage, RefIndex mismatch counts, and research leads — computed live from all
  loaded PM4 research contexts via `WorldScene.GetPm4MslkLinkingStats`.
- New public API on `WorldScene`: `GetPm4TileObjectSummaries()`, `SelectPm4ObjectByKey()`,
  `GetPm4MslkLinkingStats()` and the `Pm4MslkLinkingStats` struct. Build passes.

## 2026-08-04 — PM4 scene graph: CK24 decoded, doodad identity found

Spec 131, branch `131-pm4-scene-graph-doodads`. Detail:
[workstream-pm4-decode.md](workstream-pm4-decode.md).

- **PM4 placement confirmed correct in the viewer.** Tiles aligned, tents identified, previously
  rotated walls and buildings correct. Committed by the user as "finally right".
- **`pm4 doodad-split`** — a keyed (non-zero) CK24 is one placed WMO. 47 WMO-free tiles carry zero
  keyed objects (47 chances to falsify, none taken); keyed count matches WMO count exactly on
  136/179 tiles, within ±1 on 163. CK24 0 is the per-tile remainder — exactly one per tile.
- **`pm4 component-identity`** — the CK24 0 remainder splits into per-doodad components by mesh
  connectivity: 19,124 of 20,113 (95.1%) land on an MDDF placement, closest at distance 0.00.
  **`MSLK.GroupObjectId` is the per-doodad identity** — 3,343 of 3,345 pure components unique on
  their tile. `MSUR.GroupKey` scores 100% purity but 0% distinctness and would have been a false
  positive without the distinctness column.
- **New ground truth**: Blizzard WoW Editor 1.9.0 screenshots of this data, with Karazhan Crypts
  loaded for comparison.
- Memory bank restructured into per-workstream files (this change).

## 2026-08-03 — PM4 placement solved

Spec 130. Detail: [workstream-pm4-decode.md](workstream-pm4-decode.md).

- **The coordinate transform**, verified against ADT ground truth at 92.4% vs 0.7% for the
  alternative. MSVT is ADT placement space; the map tile index is `31 - band`.
- **Region-scoped frames refuted** via `pm4 bounds-audit --by-region` (1,877 of 1,895 objects on one
  frame).
- **The MPRL-scored placement fitter disproven and removed from the render path.**
  `pm4 yaw-evidence`: containment 93.3% canonical vs 88.2% with the fitted yaw vs 79.0% for a
  known-wrong 45° control; hurt 96, helped 3. `Pm4PlacementMath` left untouched — all 16
  `PlacementMath_*` tests still pass.
- **Both PM4 disk caches bumped to version 9.** They store post-transform geometry, so a stale cache
  replayed the old placement and made a correct fix look broken.
- `Pm4CoordinateService.TryGetObj0PathForPm4` replaces a padded-name lookup that matched none of the
  616 corpus files.
- Earlier the same day: MSPV/MSPI decoded as a vertical planar quad mesh (the walls), walls rendered
  in the viewer, `pm4 mprr` structural hypothesis eliminated.

## 2026-08-03 — terrain ML: stacked-height trainer unblocked

Specs 114/125/126. Detail: [workstream-terrain-ml.md](workstream-terrain-ml.md).

- Residual→height feed-forward **proven dead** (r = 0.0024; three approaches agree).
- Forward-model-as-referee built; it establishes the information limit of single-view shading.
- Stacked height model (4-channel `direct_cnn_v112`) implemented; the channel-count crash is fixed
  via one shared `build_model_input_channels`. **Not yet trained** — user-run gate.
- Full data-harvester suite: ~1150 passed / ~45 skipped / 3 pre-existing unrelated failures.

## Before 2026-08-01

Condensed into [archive/2026-08-01-progress-detail.md](archive/2026-08-01-progress-detail.md) at the
feature-complete declaration and spec audit. Older session history is in
[archive/](archive/README.md).
