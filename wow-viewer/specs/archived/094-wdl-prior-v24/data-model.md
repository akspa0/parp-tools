# Data Model: 094-wdl-prior-v24

**Purpose**: Define the V24 Zarr store schema, the per-tile data structures, and the entity relationships for the WDL Prior + Lattice Detailer (V24) spec.
**Created**: 2026-07-06
**Source**: [`./spec.md`](./spec.md), [`./research.md`](./research.md)

> **Amended 2026-07-06** (audited; see `spec.md` "Implementation Amendments" A1/A4/A5): the C# reader returns 17×17 outer + 16×16 inner int16 per MARE and does **not** read MAHO, so `wdl_prior_holes` is dropped. The V24 store uses paired arrays — `wdl_prior_outer` (N,17,17), `wdl_prior_inner` (N,16,16), with `_source_*` (uint8) and `_confidence_*` (float32) mates — plus a copied `index.parquet`; V18 arrays are referenced from the V18 store by `tile_id`, not copied. V18 actual names/shapes: `minimap_rgb` (256,256,3) uint8, `object_precise_mask` (257,257) float32, `liquid_mask` (256,256) float32, `holes_16` (16,16) bool, `no_object_minimap` present on 0_5_3 only.

## Storage Layout

V24 lives at `wow-viewer/output/datasets/v24/<build>.zarr/`. It is a Zarr v2 store, group-rooted, extending a V18 Zarr store with 3 mandatory + 1 optional new top-level arrays.

```
v24.zarr/
├── [V18 arrays: minimap_rgb, height_257, alpha_256, mcnr_mask_257, normal_xyz, object_precise_mask, liquid_mask_256, ...]
├── wdl_prior              # per-tile float32, shape per build (C# reader's output per MARE)
├── wdl_prior_source       # per-tile uint8, same shape, values 0=real | 1=synthetic | 2=learned-fill
├── wdl_prior_confidence   # per-tile float32, same shape, values in [0, 1]
├── wdl_prior_holes        # (optional) per-tile bool, present only if C# reader exposes MAHO
└── .zattrs                # build-time metadata
```

## Per-Tile Shape

The per-tile shape of the WDL prior arrays is **whatever the C# WDL reader returns per MARE for the build being processed**. The spec does not hard-code a shape.

- For 3.3.5a (per [`gillijimproject_refactor/reference_data/wowdev.wiki/WDL_v18.md:131`](gillijimproject_refactor/reference_data/wowdev.wiki/WDL_v18.md:131)), the documented MARE layout is "17*17 + 16*16 = 545 signed 16-bit integers." The C# reader's actual output for 3.3.5a is likely a (17, 17) outer grid + a (16, 16) inner grid, but the C# reader is the source of truth.
- For 0.5.3 (Alpha), the C# reader may return a different layout. V24 reads whatever the C# reader returns.
- The synthetic-WDL builder, the merged-coverage builder, and Stage A all read the shape from `.zattrs` on load. No hard-coded shape constants.

The shape is recorded in `wdl_prior.attrs` (per-array attributes) and in `.zattrs` (store-level metadata):

```python
wdl_prior.attrs = {
    "shape_per_tile": [17, 17, 16, 16],   # outer H, outer W, inner H, inner W (example)
    "shape_source": "csharp_wdl_reader",
    "build_id": "3_3_5_12340",
    "dtype_per_cell": "float32",
    "scale": "same as ADT heightmap (int16 -> float32 at read boundary, no scale conversion)"
}

.zattrs = {
    "spec": "094-wdl-prior-v24",
    "created": "2026-07-XX",
    "v18_store_path": "<path to source V18 store>",
    "staged_client_path": "<path to source staged client>",
    "coverage_real_ratio": 0.65,         # example
    "coverage_synthetic_ratio": 0.30,
    "coverage_learned_fill_ratio": 0.05,
}
```

## Entities

### E1: WDL Prior (`wdl_prior`)

**Type**: Per-tile float32 array.
**Shape**: `(*, outer_h, outer_w)` if the C# reader returns an outer grid; or `(*, outer_h, outer_w, inner_h, inner_w)` if the reader returns a tuple. The exact shape is the C# reader's output.
**Source**: Merged coverage of real WDL reads (where available and consistent with V18 `height_257`) + synthetic WDL (built from V18 `height_257` via the C# terrain→WDL path) + learned-fill placeholder (per-tile mean height on audit-empty tiles).
**Use**: Stage A's training target. Stage B's LR input (after bilinear upsample to 257×257).
**Validation rule**: All cells in non-empty tiles must be finite floats. Empty tiles have a flat per-tile mean (audit-empty).

### E2: Prior Source (`wdl_prior_source`)

**Type**: Per-tile uint8 array.
**Shape**: Same as `wdl_prior`.
**Values**:
- `0` = real (cell sourced from a real staged-client WDL, agrees with V18 `height_257` within `disagree_threshold`).
- `1` = synthetic (cell sourced from the synthetic-WDL builder, because real was missing or disagreed).
- `2` = learned-fill (cell sourced from per-tile mean height, because V18 `height_257` was audit-empty).
**Use**: Stage A's per-cell sample selection (real weighted 1.0, synthetic weighted 0.7, learned-fill excluded from loss). The validation report's coverage stats.
**Validation rule**: All cells in non-empty tiles must be 0, 1, or 2. Empty tiles have source=2 for all cells.

### E3: Prior Confidence (`wdl_prior_confidence`)

**Type**: Per-tile float32 array.
**Shape**: Same as `wdl_prior`.
**Values**: In [0, 1].
- `1.0` = real-and-agreeing cell (`source=0` and the real WDL matched the synthetic within `disagree_threshold`).
- `0.7` = synthetic-only cell (`source=1` and no real WDL was available).
- `0.4` = synthetic-disagreeing cell (`source=1` and the synthetic disagreed with the available real WDL).
- `0.0` = learned-fill cell (`source=2`).
**Use**: Stage A's per-cell sample weight. Cells with confidence=0 are excluded from loss.
**Validation rule**: All values must be in [0, 1]. Cells where `source=0` must have `confidence=1.0`. Cells where `source=2` must have `confidence=0.0`. Cells where `source=1` must have `confidence` in {0.4, 0.7}.

### E4: Prior Holes (`wdl_prior_holes`) — Optional

**Type**: Per-tile bool array.
**Shape**: Same as `wdl_prior` (or a coarser shape if MAHO is 16-uint16, which is 16×16 = 256 cells vs the MARE grid's 545 cells).
**Values**: `True` if the cell is a MAHO hole (no terrain at this position), `False` otherwise.
**Source**: MAHO chunk in the real staged-client WDL, if exposed by the C# reader. All-`False` if MAHO is missing.
**Use**: Stage B's loss gate. Cells with `holes=True` are excluded from the residual loss.
**Validation rule**: All values are bool. The array may be absent from the store if the C# reader didn't expose MAHO. The V24 store's `.zattrs` records whether `wdl_prior_holes` is present.

### E5: V18 Substrate (Re-Used, Not Modified)

**Type**: Existing V18 Zarr arrays.
**Source**: V18 dataset build (Spec 001).
**Use**: V24 reads V18 arrays but does not modify them. The V18 store's `minimap_rgb`, `height_257`, `alpha_256`, `mcnr_mask_257`, `normal_xyz`, `object_precise_mask`, `liquid_mask_256` are V24's inputs.

| V18 array | V24 use |
|---|---|
| `minimap_rgb` | Stage A input (cleaned via `object_precise_mask` first) |
| `height_257` | Synthetic WDL builder input. Stage B target (as `height_257 - upsampled_prior`). |
| `alpha_256` | Stage A input (down-sampled). |
| `mcnr_mask_257` | Stage A input (down-sampled). |
| `normal_xyz` | Stage A input (down-sampled). |
| `object_precise_mask` | Minimap cleaner input. Stage A and Stage B loss gate. |
| `liquid_mask_256` | Synthetic WDL builder input (liquid exclusion). Stage B loss gate. |

### E6: Cleaned Minimap (Per-Tile, In-Memory, Not Stored)

**Type**: Per-tile 257×257×3 float32 image.
**Source**: `clean_minimap(minimap_rgb, object_precise_mask)` — pure NumPy function, no model.
**Use**: Stage A input (replaces raw `minimap_rgb`). Stage B input.
**Storage**: Not stored in the V24 Zarr store. Computed on-the-fly during training and inference.
**Validation rule**: Same shape as input minimap. Object pixels replaced by 8-connected median of non-object neighbours, or global mean if no non-object neighbour exists.

### E7: C# WDL Reader Output (Per-MARE, In-Memory, Not Stored)

**Type**: Per-MARE float32 grid (and optional bool hole bitmask).
**Shape**: Whatever the C# reader returns.
**Source**: `WowViewer.Tool.WdlRead` CLI shim → `WowViewer.Core.IO` C# WDL reader.
**Use**: Real WDL source for the merged-coverage builder.
**Storage**: Not stored in the V24 Zarr store. Read on demand from staged-client `.wdl` files.
**Validation rule**: Same shape across all MARE reads in a single build (the C# reader is consistent per build).

### E8: Stage A Model (In-Memory, Persisted to Checkpoint)

**Type**: PyTorch module. Small U-Net. ≤ 1M trainable params.
**Inputs**: `[cleaned_minimap (down-sampled to WDL prior grid size), alpha_256 (down-sampled), normal_xyz (down-sampled), mcnr_mask_257 (down-sampled), downsampled_synthetic_wdl]`.
**Output**: Per-tile WDL prior, same shape as the C# reader's per-MARE output.
**Loss**: L1 with `wdl_prior_confidence` as sample weight and `wdl_prior_source != 2` as sample selection.
**Storage**: Persisted to `wow-viewer/output/v24_validation/<run_id>/stage_a.pt`.

### E9: Stage B Model (In-Memory, Persisted to Checkpoint)

**Type**: PyTorch module. Small conv-deconv. ≤ 2M trainable params.
**Inputs**: `[bilinear_upsample(stage_a_prior, 257), cleaned_minimap, alpha_256, normal_xyz, mcnr_mask_257, object_precise_mask]`.
**Output**: Per-tile 257×257 residual over the upsampled prior.
**Loss**: L1 gated to non-liquid, non-object, non-MAHO-hole pixels.
**Storage**: Persisted to `wow-viewer/output/v24_validation/<run_id>/stage_b.pt`.

### E10: V24 Final Height (Per-Tile, In-Memory or NPZ)

**Type**: Per-tile 257×257 float32.
**Source**: `bilinear_upsample(stage_a_prior, 257) + stage_b_residual`.
**Use**: V24 inference output. Compared against V18 `height_257` for validation.
**Storage**: Persisted to NPZ during validation. Not in the V24 Zarr store.

## Entity Relationships

```
[V18 Zarr store] ──reads──> [V24 builder]
                              │
                              ├── reads ──> [staged-client .wdl files via C# WDL reader shim]
                              ├── reads ──> [C# terrain→WDL path via the same shim]
                              └── emits ──> [V24 Zarr store: wdl_prior + wdl_prior_source + wdl_prior_confidence (+ wdl_prior_holes)]

[V18 Zarr store] ──reads──> [clean_minimap] ──emits──> [cleaned minimap (in-memory)]
[V24 Zarr store] ──reads──> [cleaned minimap + V18 arrays] ──input──> [Stage A model] ──emits──> [Stage A prior (in-memory)]
[V24 Zarr store + Stage A prior + cleaned minimap + V18 arrays] ──input──> [Stage B model] ──emits──> [V24 final height]
```

## Validation Rules Summary

| Entity | Validation rule | Where enforced |
|---|---|---|
| `wdl_prior` | All cells finite floats; flat per-tile mean on audit-empty tiles | `merged_wdl_prior.py` |
| `wdl_prior_source` | All values in {0, 1, 2}; empty tiles have source=2 | `merged_wdl_prior.py` |
| `wdl_prior_confidence` | All values in [0, 1]; source=0 → confidence=1.0; source=2 → confidence=0.0; source=1 → confidence in {0.4, 0.7} | `merged_wdl_prior.py` |
| `wdl_prior_holes` | All values bool; optional | `wdl_reader.py` shim |
| `cleaned_minimap` | Same shape as input minimap; object pixels replaced by 8-connected median | `clean_minimap.py` |
| Stage A model | ≤ 1M trainable params | `stage_a.py` |
| Stage B model | ≤ 2M trainable params | `stage_b.py` |
| V24 Zarr store | `.zattrs` has spec, build_id, coverage stats; wdl_prior.attrs has shape_per_tile | `build_wdl_prior.py` |
| Coverage stats | `coverage_real_ratio + coverage_synthetic_ratio ≥ 0.95` | `inspect_v24_dataset.py` |

## Out of Data-Model Scope

- Per-tile provenance (which V18 tile this came from, which build, which map). Recorded in `index.parquet` and `tile_id` arrays, but those are V18's responsibility, not V24's.
- The C# WDL reader's actual shape. Audited in Phase 0; documented in `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-XX.md`.
- The synthetic-WDL builder's internal algorithm. Wraps the existing C# path; no new algorithm.
- The model architectures (U-Net depth/width for Stage A, conv-deconv depth/width for Stage B). Bite-sized implementation choices in `tasks.md`. Bounded by ≤ 1M / ≤ 2M total params.
