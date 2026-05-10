# Alpha MCNK Flags and Metadata Tracking Plan

**Created**: 2026-05-09
**Status**: In Progress (Phase 0 complete — height bug fixed)
**Blocked by**: None (height base bug resolved 2026-05-09)

## Problem

Alpha terrain parsing in both `AlphaWdtReader` and `AlphaTerrainAdapter` reads a fraction of the MCNK header fields. Most flags, offsets, and metadata are silently ignored. This causes:

1. **Incorrect height handling** (FIXED 2026-05-09): MCVT heights in alpha are absolute world-space Z values. The writer (`AlphaWdtWriter`) was incorrectly subtracting `chunkBaseHeight` from MCVT and MCLQ heights; the readers (`AlphaWdtReader`, `AlphaEmbeddedAdtReader`) were incorrectly adding a base height. Both have been corrected — heights are now absolute end-to-end. MCNK offsets 0x68/0x6C store the chunk's Position.Z (used by the client for vertex relativization, not as a height delta). The round-trip tests (`LkToAlphaRoundTripTests`) now pass.
2. **Missing flags**: MCNK flags control liquid type, shadow presence, vertex coloring, holes, and more — but most are not tracked through the pipeline.
3. **Lost metadata**: Fields like Position (X, Y, Z), AreaId, M2/WMO reference counts, and effect maps are parsed in isolation or not at all, making it impossible to faithfully reconstruct alpha data or use it as an interchange format.
4. **Shard metadata gaps**: When building tensor-pack shards, we discard MCNK metadata that could be essential for downstream model training and format roundtripping.

## Goals

1. **Parse every MCNK header field** in both `AlphaWdtReader` and `AlphaTerrainAdapter`, storing them in a structured record.
2. **Track all flags bits** explicitly with named constants and documentation of their alpha-specific semantics.
3. **Propagate MCNK metadata through the shard pipeline** so that tensor-pack outputs include per-chunk metadata alongside height/alpha/normal data.
4. **Use the MCNK Position (X, Z, Y) fields correctly** — they store the chunk's world position for bounding box math, NOT an additive base height for MCVT.
5. **Serve as authoritative reference** for alpha MCNK field layout, validated against Ghidra decompilation of `CMapChunk::Create` (0x5.3.3368).

## Alpha MCNK Header Layout (Ghidra-Verified)

Offsets are from the start of MCNK header data (i.e., after the 8-byte FourCC+size chunk header). The absolute offset in the file adds 8 to each value.

Based on Ghidra decompilation of `CMapChunk::Create` (0x5.3.3.3368) and cross-referenced with `CMapChunk::CreateVertices`, `CMapChunk::AsyncCallback`, and `CMapChunk::UnpackAlphaBits`.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | Flags | MCNK flags (see Flags section below) |
| 0x04 | 4 | IndexX | Chunk X index within tile (0-15) |
| 0x08 | 4 | IndexY | Chunk Y index within tile (0-15) |
| 0x0C | 4 | NLayers | Number of texture layers (1-4) |
| 0x10 | 4 | Unknown1 | Purpose unknown (often 0) |
| 0x14 | 4 | M2Number | Count of M2/doodad references in MCRF |
| 0x18 | 4 | McvtOffset | Offset from MCNK data start to MCVT sub-chunk |
| 0x1C | 4 | McnrOffset | Offset from MCNK data start to MCNR sub-chunk |
| 0x20 | 4 | MclyOffset | Offset from MCNK data start to MCLY sub-chunk |
| 0x24 | 4 | McrfOffset | Offset from MCNK data start to MCRF sub-chunk |
| 0x28 | 4 | McalOffset | Offset from MCNK data start to MCAL sub-chunk |
| 0x2C | 4 | McalSize | Size of MCAL data in bytes |
| 0x30 | 4 | McshOffset | Offset from MCNK data start to MCSH shadow sub-chunk |
| 0x34 | 4 | McshSize | Size of MCSH shadow data in bytes |
| 0x38 | 4 | AreaId | Ghidra: `zoneId = *(uint*)(param_1 + 0x40)` where `param_1` includes the 8-byte MCNK chunk header |
| 0x3C | 4 | WMOGroup | Count of WMO/map object references in MCRF; passed to `CreateRefs` from `*(uint*)(param_1 + 0x44)` |
| 0x40 | 4 | Holes | 16-bit hole mask (low 16 bits); Ghidra: `holes = *(ushort*)(param_1 + 0x48)` |
| 0x44 | 4 | Unknown5 | Purpose unknown |
| 0x48 | 4 | Unknown6 | Purpose unknown |
| 0x4C | 4 | PredTex1 | Predetermined texture data (4 × uint) |
| 0x50 | 4 | PredTex2 | |
| 0x54 | 4 | PredTex3 | |
| 0x58 | 4 | PredTex4 | |
| 0x5C | 4 | NoEffectDoodad1 | No-effect doodad refs (2 × uint) |
| 0x60 | 4 | NoEffectDoodad2 | |
| 0x64 | 4 | MclqOffset | Offset from MCNK data start to MCLQ liquid sub-chunk |
| 0x68 | 4 | Unknown8 | Purpose unknown (NOT Position.X — see Position below) |
| 0x6C | 4 | Unknown9 | Purpose unknown |
| 0x70 | 4 | Unknown10 | Purpose unknown |
| 0x74 | 4 | Unknown11 | Purpose unknown |
| 0x78 | 4 | Unknown12 | Purpose unknown |
| 0x7C | 4 | Unknown13 | Purpose unknown |
| 0x80 | 4 | **Position.X** | X component of chunk world position (float) |
| 0x84 | 4 | **Position.Z** | Z component = height (float) — used as chunk Z in `field_0x6C` |
| 0x88 | 4 | **Position.Y** | Y component of chunk world position (float) |

**CRITICAL NOTE**: The Position fields at offsets 0x80-0x88 (relative to MCNK data start after chunk header) were confirmed via Ghidra. `CMapChunk::Create` reads `*(float*)(param_1 + 0x88)` as `field_0x6C` (the Z/height position) and `CreateVertices` uses it for bounding box computation. The client stores the chunk's world position here and then subtracts it from vertex positions to make them relative to the chunk center for rendering. The `field_0x64` and `field_0x68` are then overwritten with `MapOrigin - cOffset.Y * ChunkSmall` and `MapOrigin - cOffset.X * ChunkSmall` respectively.

This means offsets 0x68-0x7C (our old `Unused1/2/3/4/5/6`) are NOT the Position fields. They are genuinely unused/reserved fields in alpha 0.5.3.

## MCNK Flags (Alpha 0.5.3)

Ghidra-verified from `CMapChunk::Create` using `*puVar10 = *(uint*)(param_1 + 0x08)` where `puVar10` points to `this->field_0x74 | flags`.

| Bit | Mask | Name | Description |
|-----|------|------|-------------|
| 0 | 0x01 | HasMCSH | Chunk has shadow map (MCSH sub-chunk) |
| 1 | 0x02 | Impassable | Chunk is impassable (old "has all liquid" flag) |
| 2 | 0x04 | HasLiquid1 | Liquid flag bit 0 (inside bitfield 2-5) |
| 3 | 0x08 | HasLiquid2 | Liquid flag bit 1 / ocean override |
| 4 | 0x10 | LiquidType1 | Liquid type bit 0 (0=water,1=ocean,2=magma,3=slime) |
| 5 | 0x20 | LiquidType2 | Liquid type bit 1 |
| 6 | 0x40 | HasMCCV | Chunk has vertex colors (MCCV sub-chunk) |
| 7-15 | 0xFF80 | Unknown | Reserved / unused in alpha |

Type classification: `(flags >> 4) & 3` gives: 0=water, 1=ocean, 2=magma, 3=slime. If bit 3 (0x08) is set, force ocean.

## Execution Phases

### Phase 1: Structured MCNK Header Record

**Deliverable**: A new `AlphaMcnkHeader` record type in `WowViewer.Core` that captures every field listed above, with named constants for flags.

- Create `WowViewer.Core.Maps.AlphaMcnkHeader` with all 128 bytes parsed into named fields
- Use `System.Numerics.Vector3` for Position
- Add `[Flags] enum AlphaMcnkFlags` with the flags above
- Read the header from offset 0 of the MCNK data region (after chunk header)
- Validate IndexX/IndexY against the MCIN slot index

### Phase 2: Wire AlphaMcnkHeader Through AlphaWdtReader

**Deliverable**: `AlphaTileData` carries `AlphaMcnkHeader[]` array (256 entries) alongside existing data.

- Parse full 128-byte header per MCNK in `TryParseMcnk`
- Store in `AlphaTileData.McnkHeaders`
- Use `Position.X/Z/Y` from the header for chunk world-position validation
- Remove `ReadAlphaBaseHeight` entirely (already done)
- Use `Flags` to determine shadow/liquid/vertex-color presence

### Phase 3: Propagate Metadata to Shard Pipeline

**Deliverable**: Tensor-pack shards include per-chunk MCNK metadata.

- Add MCNK flags, holes, area ID, position, layer count, and effect maps to `AlphaTileData`
- Expose via `TerrainChunkData` so the shard builder can write them
- Include position and flags in NPZ metadata

### Phase 4: Legacy AlphaTerrainAdapter Alignment

**Deliverable**: `gillijimproject_refactor` reads the same full header and uses Position correctly.

- Update `McnkAlphaHeader` struct to match Phase 1 layout
- Replace `Unused1/2/3` with named `PositionX/Z/Y`
- Use the position fields for bounding-box computation (not height base)
- Verify MdxViewer renders alpha terrain correctly with absolute heights

## Validation

- For each phase: run `WowViewer.Tool.Inspect` on alpha WDT test data
- Compare rendered terrain against known-good screenshots
- Verify chunks form a continuous mesh (no vertical seams)
- Verify MdxViewer still renders alpha terrain correctly
- Verify MCLQ liquid heights are at the correct absolute elevation

## Known Issues to Address Separately

1. **WDT MAIN column-major indexing**: `ReadExistingTiles` and `TryReadTile` may use row-major indexing (needs verification against the spec's `tileIndex = tileX * 64 + tileY`)
2. **FillHeightmapGaps assumes 0.0f means unfilled**: With absolute heights, 0.0f could be a legitimate sea-level value. The fix would require tracking which heightmap positions were written by a chunk (e.g., using a bool array or NaN sentinel). Low priority because sea-level vertices are rarely exactly 0.0f in alpha data. `AlphaEmbeddedAdtReader` already uses NaN-based gap filling; the shared path in `AlphaWdtReader` and `LkToAlphaConverter` still uses 0.0f.

## Height Convention Contract (Verified 2026-05-09)

The following contract is Ghidra-verified and MUST be maintained across all dataset tooling:

| Path | MCVT Storage | Reading Convention | Tensor-Pack (height_257) |
|------|-------------|-------------------|--------------------------|
| Alpha WDT → AlphaWdtReader | **Absolute** world-space Z | Read directly, no addition | Absolute Z |
| Alpha WDT → AlphaEmbeddedAdtReader | **Absolute** world-space Z | Read directly, no addition | Absolute Z |
| Alpha WDT → AlphaWdtWriter | **Absolute** world-space Z | Write directly, no subtraction | N/A (write path) |
| LK ADT → AdtTensorPackBuilder | **Relative** to BaseHeight | Read BaseHeight at 0x70, add to MCVT | Absolute Z |
| LK ADT → WorldTerrainTileBuilder | **Relative** to BaseHeight | Read BaseHeight at 0x70, add to MCVT | Absolute Z |
| LK → Alpha (LkToAlphaConverter) | **Relative** → **Absolute** | `Heights[idx] + BaseHeight` | Absolute Z |
| Alpha → LK (AlphaToLkConverter) | **Absolute** → **Relative** | `ComputeBaseHeight`, subtract from MCVT | N/A (write path) |
| NPZ → Python training | N/A | All height_257 arrays are absolute Z | Absolute Z |

**Key rule**: `AlphaWdtWriter.BuildAlphaMcvt` writes absolute heights (no base-height subtraction). `AlphaWdtWriter.BuildAlphaMclq` writes absolute liquid heights (no base-height subtraction). MCNK offsets 0x68/0x6C store Position.Z (heights[0]) for client vertex relativization, NOT as a height delta to add to MCVT.

## MDDF/MODF Coordinate Transform Contract (Ghidra-Verified 2026-05-09)

The following coordinate transforms are Ghidra-verified from `CMapChunk::CreateRefs` (0x69a0c0), `CMap::CreateDoodadDef(SMDoodadDef&, C3Vector&)` (0x6805e0), and `CMap::CreateMapObjDef(SMMapObjDef&, C3Vector&)` (0x681250). The area offset `(17066.666, 17066.666, 0.0)` is hardcoded in `CreateRefs` and applied uniformly to ALL tile references.

### Position Transform (identical for MDDF and MODF)

```
renderer_x = -file_pos.z + MapOrigin
renderer_y = -file_pos.x + MapOrigin
renderer_z = file_pos.y
```

Where MapOrigin = 17066.666... (32/3 * TILE_SIZE).

### Rotation Transform (identical for MDDF and MODF)

```
renderer_rot.x = file_rot.z * π/180      (Roll, applied around X axis)
renderer_rot.y = file_rot.x * π/180      (Pitch, applied around Y axis)
renderer_rot.z = file_rot.y * π/180 + π  (Yaw, applied around Z axis, with 180° offset)
```

Applied in order: Translate → RotateZ(yaw) → RotateY(pitch) → RotateX(roll) → Scale.

### Bounds Transform (MODF only)

```
renderer_min.x = -file_extents_max.z + MapOrigin
renderer_min.y = -file_extents_max.x + MapOrigin
renderer_min.z = file_extents_min.y
renderer_max.x = -file_extents_min.z + MapOrigin
renderer_max.y = -file_extents_min.x + MapOrigin
renderer_max.z = file_extents_max.y
```

Negation of Z/X axes swaps min/max for those components.

### Round-Trip Convention

All read/write paths store the position in renderer coordinates:
- `Position = (MapOrigin - file_z, MapOrigin - file_x, file_y)`

All read/write paths store the rotation with axis-swap convention:
- `Rotation = (file_rot.x, file_rot.z, file_rot.y)`

This IS NOT the true renderer rotation (which would be `(file_rot.z, file_rot.x, file_rot.y)`), but the round-trip is correct because the writer reverses the same mapping. The `BuildLegacyMdxPlacementTransform` function in the viewer compensates for this by applying negated X/Y rotations and a separate Rz(π) prefix.

**Important**: The +π yaw offset (`renderer_rot.z = file_rot.y * π/180 + π`) is not captured in the `Rotation` vector. The viewer's `BuildLegacyMdxPlacementTransform` applies `Matrix4x4.CreateRotationZ(MathF.PI)` separately to account for this.

### Scale Transform (MDDF only)

```
scale = uint16_scale / 1024.0
```

Ghidra-verified at `0x6805e0`: scale field at file offset 0x20 is read as `uint16`, multiplied by `1.0/1024.0`.
