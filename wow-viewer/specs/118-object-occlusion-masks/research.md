# Research: Per-Object Occlusion-Aware Masks for Object-Deconfounded Terrain Height

**Spec**: [spec.md](spec.md) | **Date**: 2026-07-22

## D-01: Visibility source is the existing strict geometry rasterizer, not a renderer diff

**Decision**: The visible-portion-only mask comes from the **already-built strict object-geometry
target path** in the C# harvester: `TerrainVisibleObjectMaskRasterizer`
(`src/core/WowViewer.Core.IO/Maps/TerrainVisibleObjectMaskRasterizer.cs`) driven by
`AdtTensorPackBuilder.BuildStrictTerrainVisibleObjectMask`. It rasterizes transformed M2/WMO mesh
triangles and retains a pixel only where the interpolated object elevation exceeds the exact raw
MCVT quincunx terrain surface by a fixed clearance (`DefaultTerrainClearance = 0.25f`), with raw
liquid evidence independently hiding submerged fragments. Overlapping objects resolve to the
front-most (highest top elevation) fragment.

**Rationale**: The spec's Assumptions section says "renderer truth (a with-objects vs
without-objects render difference)." The geometry rasterizer is a strictly better satisfaction of
the same intent — visibility decided by occlusion structure, never by color — and it is (a) already
implemented, tested, and wired into the harvest stream, (b) CPU-side and deterministic (no GPU
render, no framebuffer readback, no camera policy), (c) exactly the "pokes through the terrain"
semantic the spec demands, including the flush-object tolerance (the 0.25 clearance) and the
front-most overlap rule the spec's Edge Cases call for. The ValidationCapture renderer-diff mask
remains a viewer-side validation surface, not the dataset signal.

**Alternatives considered**: (1) Renderer with/without-objects diff — rejected: needs a GPU capture
pass per tile, camera-policy dependent, and would be a new harvest pipeline; the geometry path
already exists. (2) Full-footprint MDDF/MODF projection — rejected explicitly by the spec (the
80–90% over-masking failure). (3) Old `object_precise_mask_257` — rejected: historical semantics,
explicitly never reused by the strict path.

## D-02: Class taxonomy at harvest is the pixel-source enum; no harvest-side "unknown"

**Decision**: The per-pixel class label is `ObjectGeometryPixelSource` (byte): `0 = none`,
`1 = M2Triangle` (doodad-type), `2 = WmoTriangle` (building-type), already streamed as
`object_geometry_visible_source_257`. This satisfies FR-003's minimum (building vs doodad).

**Rationale**: The enum is total over painted pixels — every visible fragment came from a resolved
M2 or WMO asset, so a harvest-side "unknown" class cannot occur. The spec's "unknown" edge case
("roof/model class unavailable") is about *class-label* gaps, not geometry gaps; when asset
**geometry** is unreadable the strict path makes the whole tile ineligible (excluded and counted by
the generic store builder), which is the v50 "never fabricate" policy and is honest, not a mask
error. Finer model-family classes remain an extensibility point (the per-tile asset table records
the normalized asset path per `assetIndex`).

**Alternatives considered**: per-family fine classes at harvest (deferred — needs a taxonomy
decision like Spec 115's; coarse two-class is enough for US2's loss and US3's first model).

## D-03: Per-object identity needs ONE new dense C# array — the only new C# code

**Decision**: Add `object_geometry_visible_instance_257` (int32, 257×257): `0 = no object`,
`1..K` = per-tile compact instance ids assigned to resolved placements in deterministic iteration
order, painted by the same front-most rule as `visibleSource`. Per-instance provenance
(compact id → placement `UniqueId`, `assetIndex`, class, visible pixel count) is recorded in the
existing per-tile metadata JSON alongside `object_geometry_target_assets`.

**Rationale**: FR-002 requires per-object distinguishability. The strict path already tracks
`placementUniqueId` per fragment, but only inside the variable-length fragment trace
(`object_geometry_fragment_*`), which does not fit the v50 catalog's fixed-shape-per-tile contract
and is far heavier than downstream needs. Painting compact ids in the same raster pass that already
decides visibility costs one extra `int[,]` store in the existing `if` block — a bounded addition,
not a parser rewrite (Rule 3 safe). The legacy `object_instance_mask_257` is footprint-based and
stays deferred/uncataloged.

**Alternatives considered**: (1) Reconstruct instances in Python from the fragment trace — rejected:
variable-length trace is not catalog-shaped, and the trace deliberately includes rejected fragments,
so Python would have to re-implement the union rule. (2) Intersect legacy instance mask with the
visible mask — rejected: different id spaces, footprint-derived ids would merge/split wrongly.

## D-04: US1 is catalog wiring + regeneration, mirroring Spec 117 US1 exactly

**Decision**: Add three rows to the frozen v50 signal catalog
(`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`) and regenerate
`v50_configs/v50-manifest-template-0_5_3_3368.json` + `v50-signals-0_5_3_3368.json` with the
existing `v50_generate_manifest_template.py` generator. Zero hand-editing; the 1:1-name-matched
store builder then selects the arrays with no new ingestion code, and the drift-guard test
(`test_committed_053_template_matches_the_frozen_catalog`) must pass unmodified.

**Rationale**: Identical gap shape to Spec 117 US1 — the harvester already computes and streams
`object_geometry_visible_mask_257` / `object_geometry_visible_source_257` (Full and V16 profiles;
**not V22** — the V22 writer omits the strict object-geometry arrays, noted in data-model.md), and
will stream the new instance array after D-03. The catalog is the only selection gate.

**Alternatives considered**: catalog the two elevation arrays
(`object_geometry_visible_top/terrain_elevation_257`) for a soft clearance-weighted loss —
deferred: the binary mask already satisfies FR-006 (exclusion scales with visible coverage); the
rows can be added later without invalidating anything.

## D-05: US2 loss is a `--liquid-mask-weight`-pattern flag on both existing trainers

**Decision**: Add `--object-mask-weight` (float, default `0.0` = parity) to
`direct_geometry_train.py` and `geometry_detailer_train.py`: per-point loss weight
`1 - w * object_mask` with the visible mask loaded from the store and cropped to the target shape
(mirroring `point_weight = 1.0 - args.liquid_mask_weight * liq_d` at
`geometry_detailer_train.py:740`). Object-touched-tile subset metrics (tiles with any visible
object pixel) are reported alongside aggregate and relief-stratified metrics (Spec 116 machinery),
so a flat-tile aggregate cannot hide the effect (FR-008). Ground-truth masks are loss-side only
(FR-014); the flag warns-and-continues as no-op when the store lacks the array, exactly like the
liquid flag's missing-signal warning.

**Rationale**: The liquid-mask flag is the proven, reviewed precedent for confound down-weighting
in these trainers; mirroring it keeps the trainer diff minimal and auditable (Rule 6: one focused
trainer change, separately committed, validated by paired dry-run + user-run comparison).

**Alternatives considered**: hard-dropping object tiles — rejected by the spec (FR-007, the
full-footprint failure). Curriculum exclusion — rejected: object-touched tiles are ~52–54% of the
corpus; dropping them destroys the corpus (the Root-cause-2 finding in the v50 audit doc).

## D-06: US3 is a small from-scratch semantic segmenter + a feature-store bridge

**Decision**: New `harvester/spec118/` package, mirroring the Spec 116/117 package conventions:

- `object_contract.py` — stage constants (`STAGE = "object_segmentation"`), class table
  (`none/doodad/building`), `architecture_identity`, `build_object_stage_run` assembling and
  self-validating a `v50-model-stage-run-v1` document (STAGES enum widened by one value — the one
  schema change, same shape as Spec 117's `"lattice_prior"` addition).
- `object_segment_model.py` — `ObjectSegmentNet`: U-Net-lite (Spec 117 v2 pattern), RGB 256×256 →
  3-class logits at 256×256, from scratch, single-digit-hundred-K params (SC-005).
- `object_segment_train.py` — dry-run-first, `--held-out-split` REQUIRED (no fallback), masked
  class-weighted CE on the 3-class target derived from `object_geometry_visible_source_257`
  (cropped to 256), per-class IoU/recall gate, `promotion_verdict = "pending"`.
- `object_segment_infer.py` + thin CLI — two mutually exclusive modes: `--inputs` loose images
  (no store, runs on hand-painted OOD tiles, FR-009) and `--store` batch; audit record
  `v118-object-infer-v1` mirroring `v50-structure-infer-v1`.
- `object_feature_bridge.py` + thin CLI — frozen checkpoint → `v115-feature-map-v1` store with
  `class_count = 2` (doodad/building softmax channels; the `none` channel is redundant),
  checkpoint path+sha256 bound, source store immutable — consumed by the existing
  `--feature-store` contract with **zero trainer changes** (FR-011), proven by dry-running both
  trainers against a fixture bridge output (Spec 117 T018 pattern).

**Rationale**: Every piece is a direct analogue of an already-validated Spec 116/117 component, so
the risk is confined to the signal itself, not new infrastructure.

**Alternatives considered**: predicting separated instances from the minimap — stretch goal only
(spec Assumptions); semantic class segmentation is the first target. Pretrained segmentation
backbone — rejected (FR-010, project constitution).

## D-07: Evaluation reuses the Spec 116 spatially-isolated split; thresholds fixed here

**Decision**: US2 and US3 are both judged only on the existing `v50-held-out-split-v1`
(spatially-isolated) with relief stratification. US3 gate thresholds (SC-004): held-out
visible-object pixel IoU (union of doodad+building) ≥ 0.40 median across object-touched tiles,
per-class recall ≥ 0.50, plus one hand-painted OOD tile with a human-verified object region.
US2 success (SC-003): object-masked run's relief-stratified MAE on object-touched held-out tiles
strictly lower than the paired no-mask run, or an honest null report.

**Rationale**: spec.md leaves the SC-004 thresholds "to be fixed in planning" — fixed here so
tasks.md and the trainer gate can reference constants. The thresholds match the Spec 116 D-08
gate's order of magnitude for a first-cut small model on 679–1,629 tiles.

**Alternatives considered**: stricter thresholds (IoU ≥ 0.6) — rejected for a first cut on a small
corpus; the gate exists to catch non-learning, not to set a ceiling.
