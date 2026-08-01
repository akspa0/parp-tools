# Data Model: Relational Terrain Layer Reconstruction

**Feature**: 116-relational-terrain-layers | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This document defines the entities, the on-disk store/array schemas, and the JSON run-record
schemas for Spec 116. It references decisions from [research.md](research.md) by id (D-0n). Array
shapes use the v50 store's existing channel-last convention. All new artifacts are written under
`wow-viewer/output/datasets/spec116/` (or a feature-scoped subroot the operator configures); source
v50 stores are read-only.

---

## Entities

### Surface Family
A canonical category of terrain surface derived from a texture's identity. The visually-determined
unit. **Reused verbatim from Spec 115** (`harvester.v50.terrain_feature_labels`), revision `v115.1`.

- Ordinals: `0=unknown, 1=terrain, 2=road, 3=water, 4=structure` (`FAMILY_NAMES`).
- `CLASS_COUNT = 5`. Contract-stable; ordinal IS the channel index of every predicted map.
- Derivation: leaf-only texture-name rule match (first match wins), no match → `unknown`.

### Layer Entry (the "row")
One ordered row of a chunk's layer table. The central relational object (spec Key Entity).

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `tile_row` | int | curriculum row index | identifies the tile in the v50 store |
| `chunk_y`, `chunk_x` | int 0..15 | `mcly_*` axes | 16×16 chunks per tile |
| `slot` | int 0..3 | `mcly_*` channel | 0 = base (always opaque, no alpha), 1..3 = detail |
| `local_texture_id` | int | `mcly_texture_ids[cy,cx,slot]` | foreign key into this tile's MTEX table |
| `coverage` | float 0..1 | `mcly_layer_mask[cy,cx,slot]` | absent/1.0 for base slot |
| `family` | int 0..4 | join id→name→`classify_texture_name` | the predicted/predictable unit |

**Validation rules** (from requirements):
- `slot == 0` MUST be treated as opaque; it is excluded from any alpha stack (FR-008) and never
  predicted (D-04).
- A `local_texture_id` is meaningless outside its own `tile_row` (spec Key Entity: Texture Table).
- Rows whose tile has no texture-name dump entry, or an empty MTEX table, are **excluded
  wholesale and counted**, never emitted as all-`unknown` (mirrors Spec 115 label builder).

### Texture Table (per-tile, local)
The per-tile list of texture names that layer entries reference. Local to a tile; a reference is
meaningless outside its own tile. Materialized only as the texture-name dump
(`WowViewer.Tool.Harvest dump-texture-names`), keyed by `(map, tile_x, tile_y)` → `list[str]`. The
list's ORDER is the contract: position equals the value stored in `mcly_texture_ids`.

### Coverage Map
Per-location strength of a layer entry. Absent for the base layer (always fully opaque). Stored as
`mcly_layer_mask` at chunk resolution and `alpha_256` at pixel resolution.

### Dominant Structure
Per-location resolution of which layer entry is visible, following paint order — the topmost entry
whose coverage clears `DOMINANT_ALPHA_THRESHOLD = 0.5` (mirrors
`TerrainMinimapCompositor.BlendLayers`). Reused from Spec 115 `resolve_dominant_layer`.

### Reused Piece
A region of coverage recurring elsewhere in the corpus under rotation or mirroring. Measured for
the FR-013/SC-008 train/held-out overlap report via the existing Spec 113 dihedral block-matcher
(`minimap_alignment.py`), restricted to cross-set pairs.

### Held-Out Set
A spatially isolated group of tiles with **no edge or corner (8-neighbour) contact** with training
tiles (D-06). Materialized as a split manifest (see Store Schemas below).

### Relief Stratum
A partition of locations by how much height variation they contain: `flat` vs `relief-bearing`,
by chunk-level height std above a reported constant threshold (D-07). Used to report error where a
trivial predictor cannot already succeed.

---

## Store / array schemas

### Input (read-only): v50 curriculum Zarr store
Consumed as-is. Required arrays for this feature: `mcly_texture_ids (16,16,4) int32`,
`mcly_layer_mask (16,16,4) float32`, `alpha_256 (256,256,4) uint8`, `height_257 (257,257) float32`,
`minimap_rgb (256,256,3) uint8`, `mcnk_flags_16 (16,16) int32`, plus `index.parquet`
(`map, tile_x, tile_y, split, source, ...`). No writes to this store.

### Derived: spatially-isolated held-out split (US4)
**Path**: `<output>/spec116-held-out-<build>-v1/`

| Artifact | Format | Content |
|----------|--------|---------|
| `split.parquet` | Parquet | one row per curriculum tile: `tile_row, map, tile_x, tile_y, split ∈ {train, held_out}` |
| `split.json` | JSON | identity block (see `held-out-split.schema.json`): schema, build id, store sha256, taxonomy revision, adjacency rule (`8-neighbour`), buffer rings, train/held_out counts, **verified_violation_count** (must be 0), created_utc |

**Invariants**: `verified_violation_count == 0` (SC-005); deterministic given the same store +
seed; rebuilding invalidates absolute comparison with all prior results (FR-017).

### Derived: analysis reports (US1, US2)
**Path**: `<output>/spec116-reports/`

| Artifact | Format | Content |
|----------|--------|---------|
| `family-slot-consistency-<build>-v1.json` | JSON | per-family slot distribution, summary consistency score, threshold, recommendation (`slot_keyed` or `family_keyed`), store/taxonomy identity (see `analysis-report.schema.json`) |
| `shape-coverage-coupling-<build>-v1.json` | JSON | per-(tile,layer) explained variance, dip-test p-value, mixture BIC (1 vs 2 components), high-coupling tile share, linear-vs-nonlinear note |

Both are **durable decision artifacts** (FR-002): US3 consumes the US1 recommendation verbatim.

### Derived: predicted-structure store (US5 materialization)
**Path**: `<output>/spec116-structure-<checkpoint-hash>-v1/`

| Array | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| `structure_family` | `(N, 3, 16, 16)` | uint8 | predicted family per detail slot (1..3), per chunk; base slot omitted |
| `structure_confidence` | `(N, 3, 16, 16)` | float16 | max softmax probability per chunk/slot |
| `structure_legal` | `(N, 3, 16, 16)` | bool | whether a legal same-family local id was resolved (SC-004) |
| `index.parquet` | — | — | row-aligned to the source curriculum; `upstream_checkpoint_sha256`, `taxonomy_revision` in attrs |

Source stores are NEVER mutated; the derived store is immutable once written and bound to the
frozen checkpoint hash (D-09, mirroring `feature_map_materialize.py`).

---

## JSON run-record schemas

### `v50-structure-run-v1` (US3 training run)
Schema-validated by `structure_contract.py` (reuses the `model_stage_contract.py` validator
pattern). Required fields:

```jsonc
{
  "schema": "v50-structure-run-v1",
  "created_utc": "2026-07-21T00:00:00Z",
  "feature": "116-relational-terrain-layers",
  "slot": 1,                       // which detail slot this independent model predicts
  "vocabulary_decision": "family_keyed",  // verbatim from US1 artifact (D-02)
  "identity": { "path": "...", "sha256": "<64hex>" },
  "inputs": {
    "store": { "path": "...", "sha256": "<64hex>" },
    "held_out_split": { "path": "...", "sha256": "<64hex>", "verified_violation_count": 0 },
    "texture_name_dumps": [ { "path": "...", "sha256": "<64hex>" } ],
    "taxonomy_revision": "v115.1",
    "rule_set_sha256": "<64hex>"
  },
  "architecture": { "class": "StructureSlotNet", "base": 32, "slot": 1, "num_classes": 5, "param_count": 0 },
  "config": { "batch_size": 0, "epochs": 0, "lr": 0.0, "max_class_weight": 15.0, "device": "cpu" },
  "split_counts": { "train": 0, "held_out": 0 },
  "baselines": { "majority_class": { "family": "terrain", "per_class_iou": {}, "per_class_recall": {} } },
  "best_epoch": 0,
  "metrics": {
    "per_class": { "unknown": {"iou":0,"recall":0}, "terrain": {...}, "road": {...}, "water": {...}, "structure": {...} },
    "macro_iou": 0.0,
    "rarest_class_iou": 0.0,
    "rarest_class_recall": 0.0
  },
  "promotion_verdict": "pending",   // pending | promoted | refused
  "gate": { "rule": "per_class_iou_recall", "rarest_class": "structure", "sc003": false }
}
```

**Invariants enforced**: `inputs.held_out_split.verified_violation_count == 0`; `vocabulary_decision`
matches the US1 artifact; `promotion_verdict` starts `pending` and is set only after the user-run
evaluation; aggregate accuracy is recorded for reporting only and is **not** referenced by `gate`
(D-08).

### `v50-structure-infer-v1` (US3 deployment inference audit)
```jsonc
{
  "schema": "v50-structure-infer-v1",
  "created_utc": "...",
  "checkpoint": { "path": "...", "sha256": "<64hex>", "taxonomy_revision": "v115.1" },
  "inputs": [ { "path": "...", "sha256": "<64hex>" } ],
  "legal_table_available": true,   // false for OOD hand-painted images (D-05)
  "sc004_all_references_legal": true,
  "per_tile": [ { "input_sha256": "...", "class_fractions": {}, "low_confidence_chunks": 0 } ]
}
```

### `v50-structure-geometry-comparison-v1` (US5)
```jsonc
{
  "schema": "v50-structure-geometry-comparison-v1",
  "held_out_split": { "path": "...", "sha256": "<64hex>" },
  "without_structure": { "checkpoint_sha256": "...", "relief_mae": 0.0, "flat_mae": 0.0, "trivial_baseline_relief_mae": 0.0 },
  "with_structure":    { "checkpoint_sha256": "...", "relief_mae": 0.0, "flat_mae": 0.0, "trivial_baseline_relief_mae": 0.0 },
  "sc007_beats_trivial_on_relief": false,
  "absolute_comparison_to_prior_runs_invalid": true   // FR-017
}
```

---

## State transitions

- **Vocabulary decision**: `undecided` → (`US1 measurement`) → `slot_keyed | family_keyed`. Once
  set, US3 head design is fixed; the decision artifact is immutable.
- **Derivability decision**: `undecided` → (`US2 measurement`) → `coverage_derivable | coverage_independent`. Determines whether US3 needs coverage regressors (D-04).
- **Held-out set**: `absent` → (`US4 build`) → `verified (violation_count==0)`. Any rebuild moves
  to a new identity and invalidates prior comparisons.
- **Structure model (per slot)**: `pending` → (`user-run train`) → `evaluated` →
  `promoted | refused` (gate = per-class IoU/recall, D-08).
- **Geometry comparison**: `pending` → (`user-run paired train`) → `reported` (SC-007 true or
  honest negative).