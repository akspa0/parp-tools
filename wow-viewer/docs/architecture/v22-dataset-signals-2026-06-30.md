# V22 Dataset Signal And Store Contract

> Canonical schema freeze for Spec 086. V22 replaces V18 plus patch scripts, placement side paths, live MPQ asset reparsing, and decoded-texture sidecars with one Zarr-backed dataset. The C# harvester pre-decodes signals into a binary V22 stream; the Python Zarr package writes and reads the canonical dataset. Consumers load decoded data from Zarr instead of the client.

## Scope

V22 owns four training surfaces in one store:

1. fixed per-tile signal arrays,
2. native placement arrays with per-tile offsets,
3. a per-build decoded model library,
4. a per-build decoded tileset library.

Index, placement, decoded metadata, and asset inventory outputs are audit mirrors only. A downstream consumer must be able to read core semantics from the Zarr dataset without opening MPQs or relying on sidecar-only data.

Client scope is intentionally limited to `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927`. `4_0_0_11927` is included because the development map references Cata-only assets and existing object decode/render support covers that era. Other staged clients are not V22 targets unless Spec 086 is explicitly reopened.

Implementation status: Phase 2 has a C# `RawArraySerializer.StreamProfile.V22` tile-record profile. It emits final V22 tile key names from `TerrainTileTensorPack`, derives `liquid_type_256` and `ground_intent_height_257`, converts `mcly_layer_mask` to float32, emits MDDF/MODF placement data under V22 names, emits explicit per-placement asset path metadata, emits tile-local MTEX texture path metadata, and expands legacy 14-column MODF rows to the V22 17-column layout with zero-filled fields when the source format does not carry flags/doodadSet/nameSet. The Python Zarr writer and reader are the next phase.

## Record Layout

```text
output/datasets/v22/<build>.zarr/        # Python Zarr store (canonical cache/load surface)
|-- index report                  # audit mirror and tile metadata
|-- placements report             # audit mirror of placement arrays
|-- decoded_metadata report       # audit mirror of decoded ADT metadata
|-- asset_inventory report        # audit mirror of model/tileset coverage
|-- signal_validation.json
|-- harvest_metrics.json
|-- finalization.json
|-- _resume_state.json
|
|-- height_257/                   # float32  (N, 257, 257)
|-- normal_xyz/                   # float32  (N, 257, 257, 3)
|-- normal_mask/                  # bool     (N, 257, 257)
|-- alpha_256/                    # float32  (N, 256, 256, 4)
|-- holes_16/                     # bool     (N, 16, 16)
|-- liquid_mask/                  # float32  (N, 256, 256)
|-- liquid_height/                # float32  (N, 256, 256)
|-- object_mask/                  # bool     (N, 257, 257)
|-- object_precise_mask/          # float32  (N, 257, 257)
|-- object_instance_mask/         # int32    (N, 257, 257)
|-- mcnk_flags_16/                # int32    (N, 16, 16)
|-- mddf_mask/                    # float32  (N, 257, 257)
|-- modf_mask/                    # float32  (N, 257, 257)
|-- object_filtered_mask/         # float32  (N, 257, 257)
|-- model_focus_mask/            # float32  (N, 257, 257)
|-- model_above_terrain_mask/    # float32  (N, 257, 257)
|-- object_roof_mask/             # float32  (N, 256, 256)
|-- object_roof_confidence/       # float32  (N, 256, 256)
|-- minimap_rgb/                  # uint8    (N, 256, 256, 3)
|-- shadow_mask/                  # float32  (N, 256, 256)
|-- mcly_texture_ids/             # int32    (N, 16, 16, 4)
|-- mcly_layer_mask/              # float32  (N, 16, 16, 4)
|
|-- object_visibility_mask/       # float32  (N, 256, 256)
|-- no_object_minimap/            # uint8    (N, 256, 256, 3)
|-- mcnr_mask_257/                # bool     (N, 257, 257)
|-- liquid_type_256/              # uint8    (N, 256, 256)
|-- ground_intent_height_257/     # float32  (N, 257, 257)
|
|-- mddf_placement_offset/        # int64    (N)
|-- mddf_count/                   # int32    (N)
|-- mddf_placement_data/          # float32  (total_mddf, 9)
|-- mddf_unique_ids/              # int32    (total_mddf)
|-- mddf_model_ids/               # int32    (total_mddf)
|-- modf_placement_offset/        # int64    (N)
|-- modf_count/                   # int32    (N)
|-- modf_placement_data/          # float32  (total_modf, 17)
|-- modf_unique_ids/              # int32    (total_modf)
|-- modf_model_ids/               # int32    (total_modf)
|-- mcly_tileset_ids/             # int32    (N, 16, 16, 4)
|
|-- models/
`-- tilesets/
```

## Root Signal Arrays

The 20 V18 base arrays are still present and keep their existing shapes, dtypes, and meanings. V22 changes their build contract, not their semantic names: every V22 record writes them during C# preprocessing, and missing source data is represented by documented fill values rather than absent arrays. The Python Zarr writer is responsible for putting them on disk.

| Array | Shape | Dtype | Fill | Source |
|-------|-------|-------|------|--------|
| `height_257` | 257x257 | float32 | 0 | ADT MCNK heights |
| `normal_xyz` | 257x257x3 | float32 | 0 | ADT MCNR, checkerboard gaps zeroed |
| `normal_mask` | 257x257 | bool | false | derived normal coverage |
| `alpha_256` | 256x256x4 | float32 | 0 | MCAL/MCLY alpha weights |
| `holes_16` | 16x16 | bool | false | ADT/WDT terrain holes |
| `liquid_mask` | 256x256 | float32 | 0 | MCNK/MH2O/MCLQ/WL liquid presence |
| `liquid_height` | 256x256 | float32 | 0 | liquid surface height |
| `object_mask` | 257x257 | bool | false | MDDF+MODF object footprint |
| `object_precise_mask` | 257x257 | float32 | 0 | spec-001 triangle-fill M2 plus WMO footprint path |
| `object_instance_mask` | 257x257 | int32 | -1 | dominant object instance id |
| `mcnk_flags_16` | 16x16 | int32 | 0 | MCNK header flags |
| `mddf_mask` | 257x257 | float32 | 0 | doodad footprint |
| `modf_mask` | 257x257 | float32 | 0 | WMO footprint |
| `object_filtered_mask` | 257x257 | float32 | 0 | deprecated — centroid-based object footprint pointer |
| `model_focus_mask` | 257x257 | float32 | 0 | renamed from object_filtered_mask; centroid-based footprint pointer |
| `model_above_terrain_mask` | 257x257 | float32 | 0 | placements with Z above terrain height; 0 = underground (not on minimap) |
| `object_roof_mask` | 256x256 | float32 | 0 | roof/structure overhead mask |
| `object_roof_confidence` | 256x256 | float32 | 0 | roof confidence |
| `minimap_rgb` | 256x256x3 | uint8 | 0 | baked minimap RGB |
| `shadow_mask` | 256x256 | float32 | 0 | MCSH shadow signal |
| `mcly_texture_ids` | 16x16x4 | int32 | -1 | per-tile MCLY texture ids |
| `mcly_layer_mask` | 16x16x4 | float32 | 0 | layer active/coverage mask |

`mcly_texture_ids` values are tile-local MTEX indices. V22 records must preserve the tile-local MTEX table as `mtex_texture_paths`, aligned by index, before build-wide tileset remapping. Without this table, a texture id is not an asset identity.

V22 also promotes former patch or side outputs into the base build:

| Array | Shape | Dtype | Fill | Source |
|-------|-------|-------|------|--------|
| `mcnr_mask_257` | 257x257 | bool | false | MCNR checkerboard validity, true where `x % 2 == y % 2` |
| `liquid_type_256` | 256x256 | uint8 | 0 | MCNK/MH2O/MCLQ liquid class broadcast and masked by liquid presence |
| `ground_intent_height_257` | 257x257 | float32 | `height_257` | height with object footprints inpainted |
| `object_visibility_mask` | 256x256 | float32 | 0 | renderer-truth object visibility when available |
| `no_object_minimap` | 256x256x3 | uint8 | 0 | renderer-truth terrain-only minimap when available |

`object_visibility_mask` and `no_object_minimap` may be all-zero for tiles without renderer-truth capture. The arrays still exist so consumers do not need `has_*` branches.

## Placement Arrays

Placement data is stored as flat root arrays with per-tile offsets. For tile row `i`, a consumer reads `offset = *_placement_offset[i]` and `count = *_count[i]`, then slices `offset:offset + count`.

### MDDF

`mddf_placement_data` has shape `(total_mddf, 9)` and float32 columns:

| Column | Meaning |
|--------|---------|
| 0 | `nameId` |
| 1 | `uniqueId` |
| 2 | `posX` |
| 3 | `posY` |
| 4 | `posZ` |
| 5 | `rotX` |
| 6 | `rotY` |
| 7 | `rotZ` |
| 8 | `scale` |

Companion arrays:

- `mddf_unique_ids`: int32, one per placement.
- `mddf_model_ids`: int32, one per placement, indexing `models/model_paths`.
- `mddf_count`: int32, one per tile.
- `mddf_placement_offset`: int64, one per tile.
- `placement_mddf_asset_paths`: string metadata, one canonical asset path per placement row.

### MODF

`modf_placement_data` has shape `(total_modf, 17)` and float32 columns:

| Column | Meaning |
|--------|---------|
| 0 | `nameId` |
| 1 | `uniqueId` |
| 2 | `posX` |
| 3 | `posY` |
| 4 | `posZ` |
| 5 | `rotX` |
| 6 | `rotY` |
| 7 | `rotZ` |
| 8 | `flags` |
| 9 | `doodadSet` |
| 10 | `nameSet` |
| 11 | `boundsMinX` |
| 12 | `boundsMinY` |
| 13 | `boundsMinZ` |
| 14 | `boundsMaxX` |
| 15 | `boundsMaxY` |
| 16 | `boundsMaxZ` |

Companion arrays:

- `modf_unique_ids`: int32, one per placement.
- `modf_model_ids`: int32, one per placement, indexing `models/model_paths`.
- `modf_count`: int32, one per tile.
- `modf_placement_offset`: int64, one per tile.
- `placement_modf_asset_paths`: string metadata, one canonical asset path per placement row.

Asset paths are required, not optional. `nameId` alone is not enough for V22 consumers because it only becomes meaningful when joined against the correct per-tile name table. Every placement row must be resolvable to its canonical M2/WMO path without rereading ADT chunks or MPQs.

## Model Library

The model library is a per-build group. Entries are keyed by integer model id, and `model_paths` maps id to canonical normalized path. Model payloads are emitted as separate C# V22 stream messages and stored once per build by the Python Zarr writer, not duplicated per tile.

```text
models/
|-- model_paths/                  # string   (num_models)
|-- model_kind/                   # uint8    (num_models) 0=unknown, 1=M2, 2=WMO
|-- load_error/                   # uint8    (num_models)
|-- load_error_message/           # string   (num_models)
|-- m2/<model_id>/...
`-- wmo/<model_id>/...
```

### M2 Entry

```text
models/m2/<model_id>/
|-- vertices/                     # float32  (V, 3)
|-- normals/                      # float32  (V, 3)
|-- texcoords_0/                  # float32  (V, 2)
|-- texcoords_1/                  # float32  (V, 2)
|-- bone_indices/                 # uint8    (V, 4)
|-- bone_weights/                 # float32  (V, 4)
|-- triangles/                    # int32    (T, 3)
|-- render_flags/                 # uint32   (R)
|-- blend_modes/                  # uint8    (R)
|-- texture_lookup/               # uint16   (R)
|-- texture_paths/                # string   (P)
|-- texture_replaceable_ids/      # uint32   (P)
|-- texture_flags/                # uint32   (P)
|-- transparency_lookup/          # uint16   (R)
|-- bone_lookup/                  # uint16   (B)
`-- bounds/                       # float32  (2, 3)
```

### WMO Entry

```text
models/wmo/<model_id>/
|-- vertices/                     # float32  (V, 3)
|-- triangles/                    # int32    (T, 3)
|-- normals/                      # float32  (V, 3)
|-- group_counts/                 # int32    (G)
|-- group_indices/                # int32    (G)
|-- materials/                    # int32    (K, 8)
|-- material_texture_paths/       # string   (P)
|-- bounds/                       # float32  (2, 3)
|-- portal_vertices/              # float32  (PV, 3)
|-- portal_indices/               # int32    (PI, 3)
|-- doodad_set_paths/             # string   (DS)
|-- flags/                        # uint32   scalar
`-- version/                      # uint32   scalar
```

Unloadable models still get a `model_paths` entry with `load_error=1` and zero-length payload arrays. The build must not crash because one asset is corrupt or missing.

## Tileset Library

The tileset library is a per-build group. Entries are keyed by integer tileset id, and `tileset_paths` maps id to canonical normalized BLP path. Tileset payloads are separate C# V22 stream messages and stored once per build by the Python Zarr writer.

```text
tilesets/
|-- tileset_paths/                # string   (num_tilesets)
|-- load_error/                   # uint8    (num_tilesets)
|-- load_error_message/           # string   (num_tilesets)
|-- texture_shape/                # int32    (num_tilesets, 2)
`-- texture_rgb/<tileset_id>/      # uint8    (H, W, 3)
```

Root array `mcly_tileset_ids` remaps tile-local `mcly_texture_ids` into per-build `tileset_paths` indices. Unused layers are `-1`. Unloadable textures still get a `tileset_paths` entry with `load_error=1`, `texture_shape=(0, 0)`, and a zero-sized or zero-filled texture payload.

Phase 2 tile records emit `mtex_texture_paths` in metadata and `tileset_texture_rgb_<index>` arrays for decoded tile-local textures. Phase 3 must promote these tile-local texture paths into stable build-wide `tileset_paths` ids and write `mcly_tileset_ids`.

## Stream Boundary

V22 has three message classes:

- **Tile messages**: fixed-shape tile signals, placement rows, per-placement canonical paths or ids, and per-tile metadata. These stay regular and cheap to buffer.
- **Model-library messages**: one per unique canonical M2/WMO path per build session. These contain the full parsed model payload and load-error status.
- **Tileset-library messages**: one per unique canonical terrain texture path per build session. These contain decoded BLP RGB and load-error status.

The C# harvester owns id assignment in the V22 stream. A path table is complete only after all stream messages for a build are consumed. A placement id is valid if it indexes `models/model_paths`; `-1` is allowed only when the placement source path is absent and must be counted in the asset inventory report.

## Dataset Read Contract

The V22 Zarr reader returns a fixed-key tile record for every tile. Required tile keys are all root signal arrays listed above plus:

- `mddf_placement_data`: float32 `(tile_mddf_count, 9)`
- `modf_placement_data`: float32 `(tile_modf_count, 17)`
- `mddf_unique_ids`: int32 `(tile_mddf_count,)`
- `modf_unique_ids`: int32 `(tile_modf_count,)`
- `mddf_model_ids`: int32 `(tile_mddf_count,)`
- `modf_model_ids`: int32 `(tile_modf_count,)`
- `mddf_count`: int32 scalar
- `modf_count`: int32 scalar
- `mcly_tileset_ids`: int32 `(16, 16, 4)`

Empty placement tiles return zero-length arrays with the correct second dimension, never `None`. Missing optional source data returns the documented fill array. Missing model or tileset payloads are represented by valid ids whose library entry has `load_error=1`.

The V22 Zarr reader exposes a cached mapping from model id to the decoded entry summary and payload arrays, plus a cached mapping from tileset id to decoded texture payloads. The cache is read-only after first load.

## Validation Gates

Phase 2 and later tests must pin this schema through:

- root-array existence, dtype, and shape checks,
- placement offset/count round-trip checks,
- model id and tileset id bounds checks,
- `load_error` coverage summaries,
- asset inventory parity against placement references,
- fixed-key `V22Dataset` tests on tiles with and without placements,
- no MPQ or sidecar-only reads in core downstream consumers.

Phase 1 exit criteria are satisfied when every V18 base, patched, promoted, placement, model, and tileset surface has a documented V22 home and no unresolved store-location question remains.
