# Feature Specification: Terrain MCAL Rendering Parity

**Feature Branch**: `014-terrain-mcal-rendering-parity`

**Created**: 2026-05-23

**Status**: Active

**Input**: wow-viewer GPU preview renderer produces broken MCAL terrain output compared to MdxViewer reference. Coordinate and alpha blending bugs prevent correct object mask generation for V16.2.

## Problem Statement

The wow-viewer `WorldGpuPreviewRenderer` has multiple terrain rendering bugs that cause MCAL alpha blending and chunk placement to diverge from the MdxViewer reference. This breaks terrain rendering and corrupts object mask artifacts needed by V16.1.1 and V16.2 training pipelines.

Three categories of bugs were identified by comparing against MdxViewer:
1. **Coordinate bugs** — chunk placement and camera positioning use swapped IndexX/IndexY
2. **MCAL alpha bugs** — implicit alpha layers left at 0 instead of 255, shadow maps missing
3. **Shader UV bugs** — texture coordinate mapping may be inverted for the corrected coordinate system

## User Scenarios & Testing

### User Story 1 - Chunk Positions Match MdxViewer (Priority: P1)

A terrain researcher renders a single tile in wow-viewer and the chunk layout matches MdxViewer exactly — no transposed or scrambled terrain geometry.

**Why this priority**: If chunks are placed wrong, everything downstream (MCAL, normals, textures) is wrong.

**Independent Test**: Render a tile with known asymmetric chunk features (e.g., a road or river) and compare chunk positions side-by-side with MdxViewer.

**Acceptance Scenarios**:

1. **Given** a tile with non-symmetric chunk layout, **When** wow-viewer renders the tile, **Then** chunk positions match MdxViewer's output for the same tile.
2. **Given** engine tile (48,30) for Azeroth, **When** `BuildChunkPositions` is called, **Then** `chunkWorldX = MapOrigin - (tileX * TileSize) - (chunk.IndexY * ChunkSize)` and `chunkWorldY = MapOrigin - (tileY * TileSize) - (chunk.IndexX * ChunkSize)`.

---

### User Story 2 - MCAL Alpha Layers Blend Correctly (Priority: P1)

A terrain researcher renders a tile with multiple texture layers and sees correct alpha blending — overlay textures appear where they should, not missing or fully opaque.

**Why this priority**: Alpha blending is the core MCAL function. Without it, terrain textures are wrong.

**Independent Test**: Render a tile with known multi-layer terrain (e.g., grass over dirt) and verify the blend pattern matches MdxViewer.

**Acceptance Scenarios**:

1. **Given** a chunk with a texture layer that has MCLY flag `0x100` set, **When** the alpha shadow slice is filled, **Then** the decoded alpha map is written into the correct RGBA channel.
2. **Given** a chunk with a texture layer that does NOT have MCLY flag `0x100` set, **When** the alpha shadow slice is filled, **Then** the corresponding RGBA channel is filled with 255 (fully opaque), not left at 0.
3. **Given** a chunk with no texture layer at a given index, **When** the alpha shadow slice is filled, **Then** the corresponding RGBA channel remains at 0.

---

### User Story 3 - Shadow Maps Appear on Terrain (Priority: P2)

A terrain researcher sees terrain shadow maps applied to terrain rendering, matching MdxViewer's shadow output.

**Why this priority**: Shadow maps affect visual quality but not object mask correctness.

**Independent Test**: Render a tile with known shadow features and verify the alpha channel of the shadow texture contains shadow data.

**Acceptance Scenarios**:

1. **Given** a chunk with a non-null shadow map, **When** the alpha shadow slice is filled, **Then** channel 3 (alpha) contains the shadow map values.
2. **Given** the terrain shader reads `alphaShadow.a`, **When** rendering, **Then** shadow darkening is applied to the terrain.

---

### User Story 4 - Camera Position Matches Tile Center (Priority: P1)

The validation capture camera is centered over the correct tile, so terrain and objects are visible in the capture frame.

**Why this priority**: If the camera points at the wrong location, captures are empty.

**Independent Test**: Run `--real-scene-dry-run` and verify WMO/MDX visible counts are non-zero for a tile known to contain placements.

**Acceptance Scenarios**:

1. **Given** engine tile (48,30) for Azeroth, **When** `ComputeTileCenter(48, 30)` is called, **Then** the result is `rendererX = MapOrigin - 48.5 * TileSize`, `rendererY = MapOrigin - 30.5 * TileSize`.
2. **Given** a tile with known placements, **When** the validation capture camera is solved, **Then** the camera eye position is within the tile's coordinate range.

---

### User Story 5 - UniqueID Deduplication for Multi-Tile Rendering (Priority: P2)

When rendering multiple adjacent tiles, objects straddling tile boundaries appear only once.

**Why this priority**: Duplicate rendering causes double-blending artifacts in object masks.

**Independent Test**: Render two adjacent tiles that share a WMO placement and verify the WMO appears once, not twice.

**Acceptance Scenarios**:

1. **Given** two adjacent tiles that both contain the same WMO uniqueID, **When** the bridge builds instances from both tiles, **Then** only one `WorldObjectInstance` for that uniqueID exists in the merged instance list.

### Edge Cases

- What happens when a chunk has zero texture layers? The fallback color path should render.
- What happens when `DecodedAlpha` is null but the layer has the `0x100` flag? Treat as 0 alpha (transparent), not implicit 255.
- What happens when a chunk has more than 4 texture layers? Only the first 4 are rendered (existing behavior).
- What happens when `IndexX` or `IndexY` is out of range? Existing bounds checks should handle this.

## Requirements

### Functional Requirements

- **FR-001**: `BuildChunkPositions` MUST use `chunk.IndexY` for the worldX offset and `chunk.IndexX` for the worldY offset, matching the MdxViewer convention.
- **FR-002**: `FillAlphaShadowSlice` MUST write 255 into the RGBA channel for texture layers that exist but lack the `0x100` MCLY flag (implicit full alpha).
- **FR-003**: `FillAlphaShadowSlice` MUST write the chunk shadow map into channel 3 (alpha) of the RGBA shadow texture when shadow data is available.
- **FR-004**: `ComputeTilePlanarMin/Max/Center` MUST compute `rendererX = MapOrigin - tileX * TileSize` and `rendererY = MapOrigin - tileY * TileSize`, matching the MdxViewer convention.
- **FR-005**: The validation capture camera solver MUST receive engine tile coordinates without additional swapping that was compensating for the old buggy coordinate convention.
- **FR-006**: The bridge instance builder MUST deduplicate placement instances by uniqueID when building from multiple adjacent tiles.
- **FR-007**: The terrain shader diffuse UV computation MUST produce correct tiling for the corrected coordinate system.
- **FR-008**: All fixes MUST work on both staged alpha (`0_5_3_3368`) and LK (`3_3_5_12340`) clients.

### Key Entities

- **WorldGpuPreviewRenderer**: the GPU preview renderer that builds terrain meshes, uploads textures, and draws frames.
- **FillAlphaShadowSlice**: the method that writes MCAL alpha data and shadow data into the RGBA texture array.
- **BuildChunkPositions**: the method that computes world-space positions for terrain chunk vertices.
- **ComputeTilePlanarMin/Max/Center**: the bridge methods that compute tile coordinate ranges for camera placement.
- **ValidationCaptureCameraSolver**: the camera solver used by the headless capture runner.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `no_objects.png` from wow-viewer shows terrain with correct MCAL blending matching MdxViewer reference for `Azeroth_30_48` on `3_3_5_12340`.
- **SC-002**: `object_visibility_mask.png` contains non-zero pixel coverage for `Azeroth_30_48` on `3_3_5_12340`.
- **SC-003**: Bounded proof exists on both `0_5_3_3368` and `3_3_5_12340` staged clients.
- **SC-004**: `dotnet build` and `dotnet test` pass with no new errors.

## Assumptions

- MdxViewer's terrain rendering is the ground truth reference.
- The wow-viewer terrain shader fragment code at lines 1549-1598 is structurally correct but depends on correct alpha data being uploaded.
- The `AdtTextureChunkLayer.Flags` field correctly reflects the MCLY flags from the ADT file.
- The `WorldTerrainChunkData.ShadowMap` field exists and contains decoded shadow data (needs verification).
- The `WorldTerrainChunkData.IndexX` and `IndexY` follow the WoW convention: IndexX = column (east-west), IndexY = row (north-south).

## Relationship to Other Specs

- **Enables**: `013-object-mask-rendering-fix` (terrain must render correctly for object mask diffs)
- **Enables**: `007-v16-1-1-curated-normal-acceleration` (object masks for no-object guidance)
- **Enables**: `011-v16-2-patched-signal-expansion` (precise masks for sidecar signals)
- **Constitution**: Section "Terrain Alpha Risk Area" — any MCAL change must be checked against both Alpha and LK terrain
