# Feature Specification: V22 Asset Library Payloads

**Feature Branch**: `087-v22-asset-library-payloads`

**Date**: 2026-06-30

**Status**: Draft

**Input**: The V22 tile stream (Spec 086) emits per-tile signals, placement rows, asset paths, and decoded `tileset_texture_rgb_*` arrays. But the `models/` and `tilesets/` Zarr groups remain empty because:
- M2/WMO geometry is loaded internally by `AdtTensorPackBuilder` for mask rasterization but is not stored in `TerrainTileTensorPack` for serialization.
- `tileset_texture_rgb_*` arrays are per-tile rather than accumulated per-build.
- Model IDs are placeholder `nameId` values, not build-wide path indices.

This spec adds three bounded additions to the pack and serializer to close those gaps.

**Client Scope**: `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`. Same scope as Spec 086.

---

## User Stories

### User Story 1 — M2 Model Payloads in the Stream (P1)

As a dataset builder, I want every unique M2 model referenced by placements to have its full geometry + skin data emitted by the C# V22 stream, so the Python Zarr writer can populate the `models/` group without reparsing the game client.

**Why this priority**: Without model payloads, the `models/` group is empty and mask-reconstruction models cannot work.

**Independent Test**: A V22 stream for a tile with 10 MDDF placements contains exactly the same arrays as without this feature, plus additional `m2_model_*_vertices`, `m2_model_*_triangles`, etc. arrays identifying each unique M2 path.

**Acceptance Scenarios**:

1. **Given** a `TerrainTileTensorPack` with 3 MDDF placements referencing 2 unique M2 paths,
   **When** the V22 stream profile serializes it,
   **Then** the stream contains exactly 2 unique model payloads (deduplicated by path), each with `vertices`, `triangles`, `normals`, `texcoords`, `render_flags`, `blend_modes`, `texture_paths`, `bounds`, and `load_error`.

2. **Given** a tile with no MDDF placements,
   **When** the V22 stream profile serializes it,
   **Then** no model payloads are emitted (stream is not polluted with empty model blocks).

### User Story 2 — WMO Model Payloads in the Stream (P1)

As a dataset builder, I want every unique WMO model referenced by placements to have its merged geometry + materials + portals emitted by the C# V22 stream.

**Why this priority**: WMO models are needed for mask reconstruction and tileset reference resolution, same as M2 models.

**Independent Test**: A V22 stream for a tile with 1 MODF placement contains `wmo_model_0_vertices`, `wmo_model_0_triangles`, etc.

**Acceptance Scenarios**:

1. **Given** a pack with 2 MODF placements referencing the same WMO path,
   **When** serialized,
   **Then** exactly 1 WMO model payload is emitted (deduplicated).

2. **Given** a corrupt WMO that cannot be parsed,
   **When** serialized,
   **Then** the model payload has `load_error=1` and zero-length geometry arrays. The build does not crash.

### User Story 3 — Tileset Dedup and `mcly_tileset_ids` (P1)

As a dataset builder, I want the Python Zarr writer to deduplicate per-tile `tileset_texture_rgb_*` arrays by path into the build-wide `tilesets/` group, and emit `mcly_tileset_ids` mapping chunk-layer tex IDs to build-wide tileset indices.

**Why this priority**: Without dedup, each tile carries redundant decoded BLP data and consumers cannot resolve texture paths to stored RGB.

**Independent Test**: A V22 store for 2 tiles sharing the same texture has exactly 1 entry in `tilesets/tileset_paths`.

**Acceptance Scenarios**:

1. **Given** a V22 store with 100 tiles referencing 50 unique terrain textures,
   **When** `tilesets/tileset_paths` is read,
   **Then** length is 50 (not 100+).

2. **Given** any tile,
   **When** `mcly_tileset_ids` is read,
   **Then** every non-negative ID indexes a valid entry in `tilesets/tileset_paths`, and unused layers are `-1`.

---

## Requirements

### Model cache in TerrainTileTensorPack

- **FR-001**: `TerrainTileTensorPack` MUST carry a per-tile model cache dictionary mapping canonical M2/WMO path → `M2GeometryDocument` (M2) or `WmoRenderDocument` (WMO). This is populated by `AdtTensorPackBuilder` during mask rasterization and consumed by the V22 serializer.
- **FR-002**: The model cache MUST be populated with the full `M2GeometryDocument` when `TryLoadDoodadModelMetadata` successfully loads an M2. Fields: `vertices`, `normals`, `texcoords_0`, `texcoords_1`, `bone_indices`, `bone_weights`, `triangles` (from companion `.skin`), `render_flags`, `blend_modes`, `texture_lookup`, `texture_paths`, `texture_replaceable_ids`, `texture_flags`, `bounds`.
- **FR-003**: For WMO models, the cache MUST carry `WmoRenderDocument` data: `vertices`, `triangles`, `normals`, `group_counts`, `group_indices`, `materials`, `material_texture_paths`, `bounds`, `portal_vertices`, `portal_indices`, `doodad_set_paths`, `flags`, `version`.
- **FR-004**: De-duplication by canonical path MUST happen in the C# cache — same path referenced by 100 placements on one tile produces one cache entry.
- **FR-005**: Models with unloadable data (corrupt M2, missing `.skin`, unreadable WMO) MUST have a cache entry with `load_error = 1` and zero-length payload arrays.

### V22 stream serialization

- **FR-006**: The V22 stream profile MUST serialize the model cache as per-tile named arrays. Each unique model on the tile gets arrays named `m2_model_{hash}_vertices`, `m2_model_{hash}_triangles`, etc. (or `wmo_model_{hash}_*` for WMOs). The hash is a stable 8-char hex digest of the canonical model path.
- **FR-007**: The V22 stream metadata JSON MUST include `tile_model_paths` (list of canonical paths for models emitted in this tile's blob) and `tile_model_kinds` (matching `m2`/`wmo`/`unknown` labels).
- **FR-008**: Model payloads in the stream MUST include `load_error` per model. Python writer uses `load_error > 0` to skip the model's geometry arrays.

### Python Zarr writer accumulation

- **FR-009**: `V22ZarrWriter` MUST accumulate per-tile model payloads into the build-wide `models/` group, keyed by canonical path. Same path from different tiles → one Zarr entry.
- **FR-010**: `V22ZarrWriter` MUST accumulate per-tile decoded texture RGB arrays into the build-wide `tilesets/` group, keyed by MTEX path. Same path from different tiles → one Zarr entry.
- **FR-011**: `V22ZarrWriter` MUST write `mcly_tileset_ids` for each tile, mapping per-chunk MCLY texture ids to indices into the build-wide `tileset_paths` array. Unused layers remain `-1`.
- **FR-012**: `V22ZarrWriter` MUST remap `mddf_model_ids` and `modf_model_ids` from per-tile nameId values to build-wide indices into `models/model_paths`.
- **FR-013**: All accumulation is done once per build session. After `finalize()` is called, subsequent tiles cannot be added. The writer MUST validate this and throw on double-finalize or post-finalize add.

### Error handling

- **FR-014**: Missing `TerrainTileTensorPack` model cache dictionary → zero model payloads emitted. Build must not crash.
- **FR-015**: Missing model paths in metadata → `mddf_model_ids` stays as placeholder values (unchanged). Writer remaps only models it has in the library.
- **FR-016**: Detached radar screen. If anything blows, we keep the array data and report it.

---

## Success Criteria

- **SC-001**: V22 stream for a single tile with known M2/WMO placements contains model payload arrays, not just placeholder IDs.
- **SC-002**: Python `wrap_strike_writer()` writes the model cache. Build-wide `models/model_paths` has the correct count.
- **SC-003**: Build-wide `tilesets/tileset_paths` has the correct count.
- **SC-004**: `mcly_tileset_ids` and `mcly_texture_ids` have the correct mapping for a known tile.
- **SC-005**: V22 stays within the same stream format, no new C# tool needed.

---

## Assumptions

- `AdtTensorPackBuilder.BuildObjectMasks` already loads `M2GeometryDocument` + `M2SkinDocument` per M2 and `WmoRenderDocument` per WMO. Adding them to the pack is a capture step on already-loaded data, not new parsing.
- Per-tile model payload size is small (typical tile has 1-20 unique models; M2 assets average ~50KB each). Blose LZ4 + per-tile dedup in Python keeps the stream manageable.
- Build model dedup (`Dictionary<string, (path, kind, payload)>`) fits in memory for a single build (< 10K unique models total).