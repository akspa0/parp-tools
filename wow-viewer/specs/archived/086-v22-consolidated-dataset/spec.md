# Feature Specification: V22 Consolidated Dataset

**Feature Branch**: `086-v22-consolidated-dataset`

**Created**: 2026-06-30

**Status**: Draft

**Input**: V18 has 20 base arrays built from the stream + 3 post-build patches (`mcnr_mask_257`, `liquid_type_256`, `ground_intent_height_257`) + renderer-truth promotion (`object_visibility_mask`, `no_object_minimap`) + `placements.parquet` sidecar. Every new signal means a new patch script. The actual 3D model data (vertices, triangles, normals) for every M2/WMO referenced by placements is only inside game client MPQs. V22 ends all of this: the C# harvester pre-decodes every signal and every referenced M2/WMO geometry into the binary V22 stream, and the Python Zarr package writes/reads the canonical dataset cache. Downstream consumers load decoded data from Zarr instead of the client.

**Client Scope**: V22 is scoped only to `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927`. The Cata `4_0_0_11927` build is included because the development map references assets that are not present in the Alpha/LK builds and the viewer already has appropriate object decoding/rendering support for that era. Do not add other clients simply because staged copies exist. Any expansion beyond these three builds requires an explicit spec update.

## User Scenarios & Testing

### User Story 1 — Zero-Patch V22 Build (Priority: P1)

As a dataset builder, I want the C# harvest command to pre-decode every signal in one pass and the Python Zarr writer to flush the final cache, so I never run a post-build patch script again.

**Why this priority**: 5 patch scripts are fragile, unreproducible, and silently skipped.

**Independent Test**: the V22 build path produces a Zarr dataset with all required arrays and zero missing signals, and downstream consumers load tiles from that Zarr dataset without touching the game client.

**Acceptance Scenarios**:

1. **Given** a staged 3.3.5 client,
   **When** the V22 build pipeline completes,

   **Then** `signal_validation.json` reports zero missing signals and no `patch-*` commands were run.

2. **Given** that same store,
   **When** a V22 consumer reads any tile from Zarr,
   **Then** every documented batch key is present with correct shape/dtype/range.
### User Story 2 — Clean V22 Dataset Contract (Priority: P1)

As a downstream consumer, I want one fixed V22 dataset contract that produces every signal I need without fallback chains or optional `has_*` guards.

**Why this priority**: `V161Dataset` has a complex fallback chain for `object_precise_mask > object_filtered_mask > object_mask` and conditional keys. Consumers handle 3+ code paths per signal.

**Independent Test**: the V22 reader returns the same keys for every tile. Signal arrays are never absent.

**Acceptance Scenarios**:

1. **Given** a V22 Zarr store with 100+ tiles,
    **When** the V22 reader is initialized with no special flags,
   **Then** all signal arrays are present and populated — no missing keys.

2. **Given** a tile with zero placements,
   **When** `__getitem__` reads that tile,
   **Then** all mask arrays are all-zero, placement arrays are zero-length, model IDs are empty. No None returns.

### User Story 3 — Pre-Decoded Object Placements In The Dataset (Priority: P1)

As a model builder, I want MDDF and MODF placement data stored directly as dataset arrays so I can build models that predict 3D object positions, orientations, and identities from minimap data without touching sidecar files or re-parsing ADT chunks.

**Why this priority**: Placement data is the ground truth for any object-identification or pose-estimation model. Storing it in sidecars means every consumer needs a separate IO path. Storing it as native arrays means one code path.

**Independent Test**: the V22 reader returns `mddf_placement_data` and `modf_placement_data` arrays for every tile with correct shapes and dtypes.

**Acceptance Scenarios**:
1. **Given** a V22 Zarr store,

   **When** a tile with 3 MDDF and 1 MODF placement is queried,
   **Then** `mddf_placement_data` has shape (3, 9) with NameId, UniqueId, Position, Rotation, Scale,
    And `modf_placement_data` has shape (1, 17) with NameId, UniqueId, Position, Rotation, Flags, DoodadSet, NameSet, BoundsMin, and BoundsMax.

2. **Given** a tile with zero placements,
   **When** queried,
   **Then** both placement arrays have shape (0, N) — not None.

### User Story 4 — Pre-Decoded 3D Model Geometry In The Dataset (Priority: P1)

As a model builder, I want every M2 and WMO model referenced by a build's placements to have its raw geometry stored directly in the dataset — vertex positions, triangle indices, normals, bounds — so models can learn 3D object identity from actual 3D data instead of regenerating or synthesizing data further downstream.

**Why this priority**: Building a model that identifies objects from minimap data currently requires (a) knowing the model path, (b) opening the MPQ, (c) parsing M2/WMO binary at runtime. Pre-decoded 3D assets in the dataset mean a model can tensor-lookup `model_vertices[id]` in one call.

**Independent Test**: `V22Store` has a `models/` group. Each entry has `vertices` (N, 3), `triangles` (M, 3), `normals` (N, 3), `bounds` (2, 3). Placements reference models by integer ID. Every unique model path in the build has an entry.

**Acceptance Scenarios**:

1. **Given** a V22 store for 3_3_5_12340 Azeroth,
   **When** `models/model_paths` is enumerated,
   **Then** every unique M2 and WMO model referenced by placements has a corresponding entry.

2. **Given** a known model path,
   **When** its model entry is loaded,
   **Then** `vertices` shape is (N, 3), `triangles` is (M, 3), `normals` is (N, 3), all float32, finite, M >= 1.

3. **Given** a tile with 10 MDDF placements,
   **When** `mddf_model_ids[tile]` is read,
   **Then** length is 10, every ID is a valid index into `models/`, and every referenced model has valid geometry.

### User Story 5 — Forward-Compatible Signal Layout (Priority: P2)

As a maintainer, I want the V22 signal layout to include optional slots for future signals (normals model, albedo, object class, etc.) so that adding a new signal does not break existing consumers.

**Independent Test**: the V22 reader accepts an available-signal list and returns zero-filled arrays for any requested signal not yet built.

## Requirements

### Foundational — V18 consolidation (zero patches)

- **FR-001**: V22 MUST contain all V18 base arrays (20): `height_257`, `normal_xyz`, `normal_mask`, `alpha_256`, `holes_16`, `liquid_mask`, `liquid_height`, `object_mask`, `object_precise_mask`, `object_instance_mask`, `mcnk_flags_16`, `mddf_mask`, `modf_mask`, `object_filtered_mask`, `object_roof_mask`, `object_roof_confidence`, `minimap_rgb`, `shadow_mask`, `mcly_texture_ids`, `mcly_layer_mask`.
- **FR-002**: V22 MUST integrate 3 patched signals into the base build: `mcnr_mask_257` (bool checkerboard), `liquid_type_256` (uint8 liquid class), `ground_intent_height_257` (float32 inpainted height).
- **FR-003**: V22 MUST use the fixed M2 precise mask pipeline (spec 001) — no fallback rectangles in `object_precise_mask`.
- **FR-004**: V22 MUST produce correct `holes_16` using WDT bitmasks (no post-build patch).
- **FR-005**: V22 build script MUST produce all arrays in a single `build` pass — no `patch-*` subcommands, no separate promotion step.
- **FR-006**: The V22 reader MUST return a fixed set of batch keys with no conditional `has_*` guards. Missing per-tile signals produce zero-filled arrays; missing per-build model or tileset entries produce empty arrays with `load_error` flag.

### Placement data as native arrays

- **FR-007**: V22 MUST store `mddf_placement_data` as a flat float32 array (total_MDDF, 9) with per-tile offset indexing. Columns: `nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, scale`.
- **FR-008**: V22 MUST store `modf_placement_data` as a flat float32 array (total_MODF, 17) with per-tile offset indexing. Columns: `nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, flags, doodadSet, nameSet, boundsMinX, boundsMinY, boundsMinZ, boundsMaxX, boundsMaxY, boundsMaxZ`.
- **FR-009**: V22 MUST store `mddf_count` and `modf_count` per tile (int32, N). Empty tiles = 0.
- **FR-010**: V22 MUST store `mddf_unique_ids` and `modf_unique_ids` (int32, flat) for cross-referencing with `object_instance_mask`.
- **FR-011**: V22 MUST store `mddf_model_ids` and `modf_model_ids` (int32, flat, one per placement) indexing into the `models/` group — so tile placements can be matched to their 3D geometry.
- **FR-011A**: V22 MUST preserve the canonical asset path for every MDDF and MODF placement, not only position/rotation/scale fields. Per-placement asset paths may be emitted as metadata or a string table plus placement indices, but a downstream consumer must be able to resolve `placement row -> canonical asset path` without rereading ADT chunks or MPQs.

### Pre-decoded tilesets (terrain BLP textures)

- **FR-012**: V22 MUST contain a `tilesets/` group with one entry per unique terrain texture path referenced by `mcly_texture_ids` across the build. Entries keyed by integer ID, with a string array `tileset_paths` mapping ID to normalized path.
- **FR-012A**: V22 MUST preserve each tile's MTEX texture path table as `mtex_texture_paths` so every `mcly_texture_ids` value can resolve to the exact source texture path before build-wide tileset remapping.
- **FR-013**: Each tileset entry MUST contain: `texture_rgb` (uint8, H×W×3) containing the decoded BLP pixel data, and `texture_shape` (int32, 2) recording the original width/height.
- **FR-014**: Tileset textures MUST be decoded by the C# harvester from BLP files in the MPQ archives — one decode per unique path. The Python writer then caches and re-emits the decoded RGB by path inside the Zarr dataset.
- **FR-015**: Per-tile `mcly_texture_ids` values (which index into the per-tile texture name list) MUST be remapped to index into the per-build `tileset_paths` array, stored as `mcly_tileset_ids` (int32, N×16×16×4).
- **FR-016**: Tilesets with unloadable BLP data (corrupt or missing file) MUST produce an entry with zero-filled `texture_rgb` and `load_error` flag set to 1.

### Pre-decoded 3D model geometry (full parsed asset)

V22 stores the **complete** parsed model document for every unique M2 and WMO, not just extracted vertices — so masking models have everything the C# harvester produces: render flags, blend modes, textures, bone lookups, and all 3D geometry. The Python Zarr writer stores that decoded payload, but does not reparse the game client.

- **FR-017**: V22 MUST contain a `models/` group with one entry per unique model path referenced by placements. Entries keyed by integer ID, with a string array `model_paths` mapping ID to normalized path.
- **FR-018**: Each M2 model entry MUST contain the full `M2GeometryDocument` structure as flat arrays:
  - `vertices` (float32, N×3) — vertex positions in model space
  - `normals` (float32, N×3)
  - `texcoords_0` (float32, N×2) — UV set 0
  - `texcoords_1` (float32, N×2) — UV set 1
  - `bone_indices` (uint8, N×4) — bone index per vertex
  - `bone_weights` (float32, N×4) — bone weight per vertex
  - `triangles` (int32, M×3) — triangle indices from companion `.skin` (spec 001 path)
  - `render_flags` (uint32, R) — per-batch render flags
  - `blend_modes` (uint8, R) — per-batch blend mode (Opaque=0, AlphaKey=1, AlphaBlend=2, etc.)
  - `texture_lookup` (uint16, T) — texture ID per batch
  - `texture_paths` (string, P) — texture filenames from `M2GeometryTexture`
  - `texture_replaceable_ids` (uint32, P) — replaceable texture IDs
  - `texture_flags` (uint32, P) — texture flags
  - `transparency_lookup` (uint16, R) — transparency index per render flag
  - `bone_lookup` (uint16, B)
  - `bounds` (float32, 2×3: min, max)
- **FR-019**: Each WMO model entry MUST contain:
  - `vertices` (float32, N×3) — all groups merged into one vertex buffer
  - `triangles` (int32, M×3) — all groups merged into one index buffer
  - `normals` (float32, N×3) — derived or from MOVT
  - `group_counts` (int32, G) — vertex/triangle count per group for decomposition
  - `group_indices` (int32, G) — group index offsets into the merged buffers
  - `materials` (int32, K×8) — per-material: flags, shader, blendMode, 3×texture name indices
  - `material_texture_paths` (string, P) — deduplicated texture names from all materials
  - `bounds` (float32, 2×3: min, max)
  - `portal_vertices` (float32, PV×3) — portal geometry for occlusion
  - `portal_indices` (int32, PI×3) — portal triangle indices
  - `doodad_set_paths` (string, DS) — referenced doodad model paths per set
  - `flags` (uint32) — WMO root flags
  - `version` (uint32)
- **FR-020**: Model extraction MUST happen in the C# harvester during the V22 stream. Models are read from MPQ archives and parsed via `M2GeometryReader` + `M2SkinReader` (M2) or `WmoRenderDocumentReader` (WMO), then handed off to the Python Zarr writer as decoded data. The Python writer must not reparse the game client.
- **FR-021**: The C# harvester MUST de-duplicate models by canonical path (`M2ModelIdentity.FromPath`) before handing decoded data to the Python Zarr writer — same model path referenced by 100 placements produces one entry.
- **FR-022**: Models with unloadable data (corrupt M2, missing `.skin`, unreadable WMO) MUST produce an entry with zero-length arrays and `load_error` flag set to 1. Build must not crash.
- **FR-023**: Per-tile `mddf_model_ids` and `modf_model_ids` reference model entries by position in `model_paths`, enabling direct tensor lookup of the full parsed structure.
- **FR-024**: The `models/` group MUST preserve per-batch render flags and blend modes because they determine how model triangles contribute to the rendered mask — opaque batches fill solid, alpha-blend batches contribute partially, and the masking model needs this distinction to learn correct footprints.

### Dataset contract

- **FR-025**: The V22 tile record MUST include: all 20 V18 signal arrays, `mcnr_mask_257`, `liquid_type_256`, `ground_intent_height_257`, `model_focus_mask`, `model_above_terrain_mask`, `mddf_placement_data`, `modf_placement_data`, `mddf_count`, `modf_count`, `mddf_unique_ids`, `modf_unique_ids`, `mddf_model_ids`, `modf_model_ids`, `mcly_tileset_ids`.
- **FR-026**: The V22 dataset contract MUST expose model entries addressable as `{model_id: {vertices, triangles, normals, bounds}}`.
- **FR-027**: The V22 dataset contract MUST expose tileset entries addressable as `{tileset_id: texture_rgb}`.
- **FR-028**: Renderer-truth promotion (`object_visibility_mask`, `no_object_minimap`) MUST be integrated into the base build, not a separate step.
- **FR-029**: V22 MUST derive `model_above_terrain_mask` by comparing each placement's world Z against the heightmap at the projected tile pixel. Placements whose Z is below the terrain height by more than 1.0 world unit are underground and must not appear in the mask, since they are invisible on the minimap.
- **FR-030**: V22 MUST emit `model_focus_mask` as the renamed successor to `object_filtered_mask`. The old name is kept for backward compatibility; `model_focus_mask` is the canonical V22 signal.

### Key Entities

- **C# harvester** (`WowViewer.Tool.Harvest`): Already reads M2+skin for geometry + mask rasterization (spec 001), WMO for mask rasterization (`TryPaintWmoFootprint`), and BLP textures for synthetic minimap. V22 adds: (1) final V22 tile signal emission, (2) full `M2GeometryDocument` + `M2SkinDocument` emission as structured arrays in the binary V22 stream, (3) full `WmoRenderDocument` emission, (4) decoded BLP RGB for all unique terrain textures. The C# harvester only writes the binary stream; Zarr persistence is Python's job.
- **Python V22 writer** (`wow-viewer/data-harvester/scripts/build_v22_dataset.py`): Consumes the decoded C# V22 stream, accumulates per-build model/tileset libraries, and writes the canonical Zarr dataset. No `patch-*` subcommands, no client reparse, no Python-side patch derivation.

## Success Criteria

- **SC-001**: V22 store for 3_3_5_12340 Azeroth builds in one pass with zero patches.
- **SC-002**: The V22 reader returns identical batch keys for every tile — no conditional None.
- **SC-003**: `object_precise_mask` shows triangle-fill M2 footprints (spec 001 match).
- **SC-004**: Every unique M2/WMO referenced by azeroth_32_32 (764 MDDF + 7 MODF) has a valid entry in `models/`.
- **SC-005**: Every unique terrain texture referenced by azeroth_32_32's MCLY layers has a valid entry in `tilesets/` with decoded RGB pixels.
- **SC-006**: A tile's `mddf_model_ids` length matches its `mddf_count`, and every ID indexes a valid model entry. Same for `mcly_tileset_ids` vs MCLY layers.
- **SC-007**: Placement arrays and tileset arrays round-trip correctly against source placement data (10 random tiles).
- **SC-008**: WMO `modf_mask` pixel-identical to V18 (no regression).
- **SC-009**: A downstream consumer reads V22 precomputed arrays from Zarr without MPQ reparse or post-build patch derivation.

## Assumptions

- C# harvester already reads M2/MDX/WMO from MPQ during the stream and already parses full `M2GeometryDocument` + `M2SkinDocument` (for spec 001 masks) and `WmoRenderDocument` (for WMO masks). Adding serialization of the full parsed structures into the binary stream is a new encoding step on already-loaded data — no new file reads, no new parsers. The Python writer only reads that binary stream.
- `M2GeometryReader` + `M2SkinReader` produce correct geometry and skin data (verified by spec 001).
- `WmoRenderDocumentReader` produces correct group-level geometry, materials, portals, and doodad sets (verified by existing WMO mask path).
- Model deduplication by canonical path is sufficient — same path always means same geometry.
- Per-build unique model count is manageable (~10k for Azeroth at 3_3_5_12340).
- The canonical store format is a Zarr dataset written/read with the Python package. The C# harvester only emits the binary V22 stream consumed by the Python writer; there is no C# Zarr implementation and no Python reparse of the game client.
