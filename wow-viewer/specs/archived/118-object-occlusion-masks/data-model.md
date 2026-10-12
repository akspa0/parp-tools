# Data Model: Per-Object Occlusion-Aware Masks

**Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md) | **Date**: 2026-07-22

## Store Signal Additions (v50 catalog rows — US1)

Three new per-tile arrays, added to the frozen catalog and regenerated configs (research D-04).
All are emitted by the C# harvest stream (Full and V16 profiles — **the V22 profile omits the
strict object-geometry arrays**; the canonical v50 build uses Full, so no profile change is needed).

| Array | dtype | shape | policy | has-flag | Meaning |
|---|---|---|---|---|---|
| `object_geometry_visible_mask_257` | float32 | (257,257) | copy-if-verified | no | Binary {0.0, 1.0}: 1.0 where a transformed object triangle is visible above the raw MCVT surface (+0.25 clearance) and not liquid-hidden. Never the full footprint. |
| `object_geometry_visible_source_257` | uint8 | (257,257) | copy-if-verified | no | Per-pixel class of the front-most visible fragment: 0 = none, 1 = doodad (M2Triangle), 2 = building (WmoTriangle). |
| `object_geometry_visible_instance_257` | int32 | (257,257) | copy-if-verified | no | Per-tile compact instance id of the front-most visible fragment: 0 = none, 1..K = resolved placements in deterministic iteration order. NEW C# array (research D-03). |

**Eligibility semantics** (inherited from the strict path + generic store builder):

- Placement catalog unavailable / terrain surface unavailable / any placement's geometry
  unreadable / liquid visibility unknown → the three arrays are null for the tile → the row is
  **excluded and counted** by the store builder (never fabricated, never zero-filled).
- Zero placements → `CompleteEmpty`: all-zero arrays, valid row (synthetic terrain-only rows land
  here and are valid negatives).
- Placements present, all fully underground/occluded → `CompleteVisible` with ≈0 marked pixels:
  valid row, mask all-zero (FR-004).

**Provenance** (FR-005): the existing per-tile metadata JSON already carries
`object_geometry_target_status`, fragment counts, `object_geometry_fragment_sha256`, and
`object_geometry_target_assets` (asset index → source class + normalized path). D-03 adds
`object_geometry_visible_instances`: one record per compact id `{instance_id, placement_unique_id,
asset_index, source, visible_pixel_count}`.

## Per-Object Instance Table (metadata, per tile)

| Field | Type | Notes |
|---|---|---|
| `instance_id` | int (1..K) | compact, per-tile, deterministic assignment order |
| `placement_unique_id` | int | MDDF/MODF unique id; links to placement data |
| `asset_index` | int | foreign key into `object_geometry_target_assets` |
| `source` | byte | 1 = doodad, 2 = building (class label per FR-003) |
| `visible_pixel_count` | int | may be 0 (fully occluded/underground placement) |

Class-per-instance in Python is derivable without this table (mode of
`object_geometry_visible_source_257` over the instance's pixels); the table is provenance, not a
load-bearing consumer input.

## Loss-Weight Entity (US2)

`object-mask-weight w ∈ [0,1]` (trainer flag, default 0.0): per-point loss weight
`weight[p] = 1 - w * mask[p]`, where `mask` is `object_geometry_visible_mask_257` cropped to the
trainer's target shape (256 or 257 — same `Crop257To256` convention as the liquid arrays). At
`w = 1.0` visible object pixels contribute zero loss; at `w = 0.0` the run is bit-identical to
today's (parity default, Rule 6). Object-touched tile = tile with ≥1 visible pixel; subset metrics
are reported separately from aggregate and relief-stratified metrics (FR-008).

## Segmentation Target (US3)

Per-pixel 3-class target at 256×256 derived from `object_geometry_visible_source_257`:

| class id | name | source values |
|---|---|---|
| 0 | `none` | 0 |
| 1 | `doodad` | 1 |
| 2 | `building` | 2 |

Model output: 3 logits/pixel (`ObjectSegmentNet`). Bridge output: 2 softmax channels
(doodad, building) — the `none` channel is redundant (1 − sum) and is dropped at bridge time.

## Run-Record Schema

Reuses `v50-model-stage-run-v1` verbatim (research D-06), with the `STAGES` enum widened by one
value: `"object_segmentation"`. `output_signal = "object_class_3"`, `upstream_models = []` for the
standalone segmenter; `promotion_verdict = "pending"` until the user promotes.

## Infer Audit Record

`v118-object-infer-v1` (mirrors `v50-structure-infer-v1`): checkpoint path+sha256, input identity
(store row key or loose-file sha256), predicted class histogram, marked fraction, and — store mode
only, when ground truth is present — per-class IoU/recall. OOD/loose-image runs record
`ground_truth: "unavailable"`; they never fabricate reference data.

## Mask Audit Record

`v118-object-mask-audit-v1` (US1 verification artifact): per-map + corpus totals of marked
fraction (p05/p50/p95), tiles excluded + reason counts, per-instance visible-pixel-count
distribution, class-per-instance consistency violations (must be 0 beyond a documented mixed-pixel
tolerance), and — where `object_mask_257` is also present in the store — the visible-vs-footprint
reduction factor (SC-001's ≥3× target on underground-heavy tiles).

## Validation Rules (from FRs)

- Instance ids appear only where mask = 1.0 (FR-002 consistency; audited, tolerance 0).
- Class values only {0,1,2}; source = 0 exactly where mask = 0 (FR-003 consistency).
- Marked fraction on a no-object tile is exactly 0 (US1 acceptance 3).
- Byte-identical arrays on re-harvest of the same tile + build (US1 acceptance 4 — the rasterizer
  is deterministic; provenance records the contract version).
- Bridge store: `schema = "v115-feature-map-v1"`, `class_count = 2`, row-aligned `index.parquet`,
  checkpoint sha256 in attrs, source stores never mutated (FR-011/FR-013).
