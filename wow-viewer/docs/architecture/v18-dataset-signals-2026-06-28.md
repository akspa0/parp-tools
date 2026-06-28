# V18 Dataset Signal Reference

> Canonical list of every Zarr array, Parquet table, and build-time artifact
> in a V18 dataset store, including which are base-streamed, post-build patched,
> and sidecar metadata.

## Store Layout

```
output/datasets/v18/<build>.zarr/
├── zarr.json
├── .zgroup / .zattrs
├── index.parquet              # tile index with has_* flags
├── placements.parquet         # MDDF + MODF placement rows
├── decoded_metadata.parquet   # per-tile raw-chunk metadata
├── signal_validation.json     # signal coverage + validation results
├── harvest_metrics.json       # build-run metrics
├── finalization.json          # store completeness report
├── merge_manifest.json        # (only for merged_all stores)
├── _resume_state.json         # build-resume state
│
├── height_257/                # float32  (N, 257, 257)
├── normal_xyz/                # float32  (N, 257, 257, 3)
├── normal_mask/               # bool     (N, 257, 257)
├── alpha_256/                 # float32  (N, 256, 256, 4)
├── holes_16/                  # bool     (N, 16, 16)
├── liquid_mask/               # float32  (N, 256, 256)
├── liquid_height/             # float32  (N, 256, 256)
├── object_mask/               # bool     (N, 257, 257)
├── object_precise_mask/       # float32  (N, 257, 257)
├── object_instance_mask/      # int32    (N, 257, 257)
├── mcnk_flags_16/             # int32    (N, 16, 16)
├── mddf_mask/                 # float32  (N, 257, 257)
├── modf_mask/                 # float32  (N, 257, 257)
├── object_filtered_mask/      # float32  (N, 257, 257)
├── object_roof_mask/          # float32  (N, 256, 256)
├── object_roof_confidence/    # float32  (N, 256, 256)
├── minimap_rgb/               # uint8    (N, 256, 256, 3)
├── shadow_mask/               # float32  (N, 256, 256)
├── mcly_texture_ids/          # int32    (N, 16, 16, 4)
├── mcly_layer_mask/           # float32  (N, 16, 16, 4)
│
├── object_visibility_mask/    # float32  (N, 256, 256)    [promoted]
├── no_object_minimap/         # uint8    (N, 256, 256, 3) [promoted]
│
├── mcnr_mask_257/             # bool     (N, 257, 257)    [patched]
├── liquid_type_256/           # uint8    (N, 256, 256)    [patched]
├── ground_intent_height_257/  # float32  (N, 257, 257)    [patched]
```

## Base Arrays (Streamed from C# Harvester)

These 20 arrays are written during `build_v18_dataset.py build`. Every tile has
all arrays; missing-source tiles get fill values (zero, false, or -1).

| Array | Shape | Dtype | Range | Source | Meaning |
|-------|-------|-------|-------|--------|---------|
| `height_257` | 257×257 | float32 | world-space | ADT MCNK heights | 1-vertex-per-cell terrain height; 257 = 16×16 cells + 1 boundary |
| `normal_xyz` | 257×257×3 | float32 | [-1, 1] | ADT MCNR | Per-vertex normals; **checkerboard gaps zeroed**, see `mcnr_mask_257` |
| `normal_mask` | 257×257 | bool | {0,1} | derived | True where `normal_xyz` has nonzero magnitude (checkerboard pattern ~50%) |
| `alpha_256` | 256×256×4 | float32 | [0, 1] | MCAL/MCLY | Per-pixel alpha blend weights for up to 4 texture layers |
| `holes_16` | 16×16 | bool | {0,1} | ADT MCNK holes | Terrain hole mask (1 = hole, terrain not rendered) |
| `liquid_mask` | 256×256 | float32 | [0, 1] | MCNK/MH2O/MCLQ/WL | Unified liquid presence; computed with priority: MCNK→MH2O→MCLQ→unified→WL |
| `liquid_height` | 256×256 | float32 | world-space | MH2O/MCLQ/WL | Unified liquid surface height |
| `object_mask` | 257×257 | bool | {0,1} | MDDF+MODF bounds | Binary footprint of all placed objects |
| `object_precise_mask` | 257×257 | float32 | [0, 1] | MDDF+MODF | Soft object footprint (fractional coverage via rasterised bounds) |
| `object_instance_mask` | 257×257 | int32 | [0, N) | MDDF+MODF | Per-pixel instance ID of the dominant object |
| `mcnk_flags_16` | 16×16 | int32 | raw | ADT MCNK flags | Per-chunk MCNK header flags; liquid type encoded in bits 2-5 |
| `mddf_mask` | 257×257 | float32 | [0, 1] | MDDF bounds | Doodad-only binary footprint |
| `modf_mask` | 257×257 | float32 | [0, 1] | MODF bounds | WMO-only binary footprint; archive build prefers geometry-derived footprints |
| `object_filtered_mask` | 257×257 | float32 | [0, 1] | MDDF+MODF | Terrain-loss gating mask; includes WMO + filtered doodad footprints |
| `object_roof_mask` | 256×256 | float32 | [0, 1] | MDDF geometry | Roof/structure overhead mask from doodad bounding-box analysis |
| `object_roof_confidence` | 256×256 | float32 | [0, 1] | MDDF geometry | Confidence score for `object_roof_mask` |
| `minimap_rgb` | 256×256×3 | uint8 | [0, 255] | baked ADT minimap | Client-rendered minimap texture as RGB |
| `shadow_mask` | 256×256 | float32 | [0, 1] | MCSH | Archived MCSH shadow signal |
| `mcly_texture_ids` | 16×16×4 | int32 | [0, ∞) | MCLY | Texture-layer file data IDs per chunk; -1 fill for unused layers |
| `mcly_layer_mask` | 16×16×4 | float32 | [0, 1] | MCLY | Layer visibility/effect mask per chunk |

### Normal Derivations

`normal_xyz` is the raw MCNR per-vertex normal data renormalised to unit
length, with **checkerboard gap positions zeroed** at build time. The
checkerboard pattern means only positions where `x%2 == y%2` carry real data.
The `normal_mask` array records which positions are valid (~50% coverage).

If you need dense normals at every pixel, use `_interpolate_checkerboard_normals`
from `v16_1_dataset.py` at load time to average cardinal neighbours.

## Promoted Arrays (Renderer-Truth Pipeline)

These arrays are patched into an existing store by `build_v18_dataset.py`'s
`--experimental-renderer-truth-promotion` path. They capture live GPU-rendered
views from WowViewer/MdxViewer.

| Array | Shape | Dtype | Range | Meaning |
|-------|-------|-------|-------|---------|
| `object_visibility_mask` | 256×256 | float32 | [0, 1] | Renderer-truth per-pixel object visibility; white=object visible |
| `no_object_minimap` | 256×256×3 | uint8 | [0, 255] | Minimap RGB rendered without any objects (terrain-only) |

Coverage is partial — only tiles with completed validation-capture stubs.

## Patched Arrays (Post-Build Scripts)

These are added to existing stores by standalone patch scripts.

| Array | Patch script | Shape | Dtype | Range | Meaning |
|-------|-------------|-------|-------|-------|---------|
| `mcnr_mask_257` | `patch_v18_mcnr_mask.py` | 257×257 | bool | {0,1} | MCNR checkerboard pattern; True where `x%2 == y%2` (valid vertex) |
| `liquid_type_256` | `patch_v20_signals.py` | 256×256 | uint8 | [0, 4] | Liquid class: 0=none, 1=water, 2=ocean, 3=magma, 4=slime |
| `ground_intent_height_257` | `patch_v20_signals.py` | 257×257 | float32 | world-space | Heightmap with structure footprints inpainted via scipy griddata |

### `mcnr_mask_257`

The old C# `AssembleNormals` interpolated checkerboard gaps before writing
`normal_xyz`. For V18, `patch_v18_mcnr_mask.py` zeros out gap positions and
adds `mcnr_mask_257` so consumers know which pixels carry real MCNR data.
Without this patch, `normal_mask` has 100% coverage from the pre-interpolated
V16 pipeline and `mcnr_mask_257` does not exist.

### `liquid_type_256`

Derived from `mcnk_flags_16` (bits 2-5) block-broadcasted from 16×16 to 256×256,
then masked by `liquid_mask` > 0.1. Channels: water (1), ocean (2), magma (3),
slime (4).

### `ground_intent_height_257`

`height_257` with structure footprints inpainted using scipy griddata (cubic
spline → linear → nearest fallback). Masked regions are pixels where
`object_precise_mask` (or fallback object mask) ≥ 0.05.

## Sidecar Metadata (Parquet Tables)

### `index.parquet`

| Column | Dtype | Meaning |
|--------|-------|---------|
| `tile_id` | int64 | row index into all fixed-shape arrays |
| `build` | string | build key |
| `map` | string | map name |
| `tile_x` | int32 | ADT tile X |
| `tile_y` | int32 | ADT tile Y |
| `height_mean` | float32 | per-tile mean height |
| `height_std` | float32 | per-tile height stddev |
| `n_mddf` | int32 | doodad placement count |
| `n_modf` | int32 | WMO placement count |
| `object_roof_mask_source` | string | roof-mask derivation label |

Plus `has_<array>` booleans for every array name and liquid-source provenance
flag (`has_liquid_source_mcnk`, `has_liquid_source_mh2o`, `has_liquid_source_mclq`,
`has_liquid_source_unified`, `has_liquid_source_wl`).

### `placements.parquet`

One row per MDDF or MODF placement. Columns: `tile_id`, `instance_type`,
`instance_idx`, `asset_path`, `nameId`, `uniqueId`, `posX/Y/Z`, `rotX/Y/Z`,
`scale` (MDDF), `bbMin/Max X/Y/Z` (MODF).

### `decoded_metadata.parquet`

Per-tile decoded ADT chunk metadata: `tile_id`, `build`, `map`, `tile_x`,
`tile_y`, `tile_name`, `source_adt_path`, `source_wdt_path`,
`raw_chunks_count`, `decoded_metadata_json` (full JSON payload),
`decoded_metadata_keys_json`.

## Build Artifacts

| File | Format | Meaning |
|------|--------|---------|
| `signal_validation.json` | JSON | Per-build signal coverage, failures, warnings |
| `harvest_metrics.json` | JSON | Tile counts, signal fractions, map breakdown, throughput |
| `finalization.json` | JSON | Store completeness check after build |
| `merge_manifest.json` | JSON | Source builds and dedup config for merged stores |
| `_resume_state.json` | JSON | Build progress for resumable pipeline |
| `liquid_patch_report.json` | JSON | (optional) liquid patch results |
| `object_patch_report.json` | JSON | (optional) object patch results |
| `renderer_truth_patch_report.json` | JSON | (optional) renderer-truth promotion results |

## Current Corpus

Finalised V18 stores exist for six builds:

- `0_5_3_3368` (Alpha)
- `0_5_5_3494` (Alpha)
- `0_7_0_3694` (Alpha)
- `3_0_1_8303` (Wrath)
- `3_3_5_12340` (Wrath)
- `4_0_0_11927` (Cata)

## Consumer Datasets

| Dataset | Base | Extra signals | Defined in |
|---------|------|---------------|------------|
| V18Dataset | V161Dataset | (alias, no extras) | `v18_dataset.py` |
| V19Dataset | V161Dataset | `normals` as optional input channel; liquid-override height target | `v19_dataset.py` |
| V20Dataset | V161Dataset | `liquid_type_256`, `ground_intent_height`, `object_precise_mask` | `v20_dataset.py` |
| V21ScarMaskDataset | standalone | alpha-derived scar masks | `v21_scar_dataset.py` |
