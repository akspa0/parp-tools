# Alpha 0.5.3 Placement Coordinate Transforms (Ghidra-Verified)

**Created**: 2026-05-09
**Client**: WoWClient.exe 0.5.3.3368
**Status**: Verified against Ghidra decompilation

This document records the coordinate transform conventions for MDDF (model/doodad placements) and MODF (world model/WMO placements) in alpha 0.5.3, as verified by Ghidra decompilation of the game client. It exists so that future sessions do not need to reverse-engineer these transforms again.

## Source Functions

All transforms were verified against these Ghidra-decompiled functions:

| Function | Address | Purpose |
|----------|---------|---------|
| `CMapChunk::CreateRefs` | `0x69a0c0` | Iterates MDDF/MODF entries, calls CreateDoodadDef/CreateMapObjDef |
| `CMap::CreateDoodadDef(SMDoodadDef&, C3Vector&)` | `0x6805e0` | Converts raw MDDF entry to renderer doodad |
| `CMap::CreateMapObjDef(SMMapObjDef&, C3Vector&)` | `0x681250` | Converts raw MODF entry to renderer map object |
| `CMap::CreateDoodadDef(uint, SMODoodadDef&, char*, uint, C44Matrix&)` | `0x6808f0` | Runtime-only doodad creation (area-local matrix transform) |
| `CMap::CreateMapObjDef(char*, C3Vector&, float, int)` | `0x680f50` | Runtime-only WMO creation (name + position + rotation) |
| `CMapArea::Create` | `0x6aad30` | Reads MHDR, MCIN, MTEX, MDDF, MODF from ADT blob |

## Area Offset (MapOrigin)

`CreateRefs` passes a hardcoded area offset `(17066.666, 17066.666, 0.0)` to both `CreateDoodadDef` and `CreateMapObjDef`. This value is 32/3 * TILE_SIZE, which is the global MapOrigin constant used throughout the client.

**Key implication**: MDDF and MODF positions are stored in a global coordinate space relative to MapOrigin. They are NOT tile-relative. The `pos` parameter in `CreateDoodadDef` and `CreateMapObjDef` is always `(MapOrigin, MapOrigin, 0)` regardless of which tile is being processed.

## Placement Instantiation: MCRF Is Required

MDDF and MODF are only global placement definition tables. The client does **not** instantiate every entry in these tables automatically. Objects are created from each chunk's `MCRF` reference list in `CMapChunk::CreateRefs`.

Ghidra evidence from `CMapChunk::Create` (`0x699000`):

```
CreateRefs(this, area, mcrfPtr, *(uint *)(mcnk + 0x1c), *(uint *)(mcnk + 0x44));
```

Because `mcnk` points at the MCNK chunk header including the 8-byte FourCC+size prefix, these are data-relative MCNK header offsets:

| Data Offset | Field | Meaning |
|-------------|-------|---------|
| `0x14` | `nDoodadRefs` | Count of MDDF indices at the start of MCRF |
| `0x24` | `mcrfOffset` | Offset from MCNK data start to wrapped `MCRF` subchunk |
| `0x3C` | `nMapObjRefs` | Count of MODF indices after doodad refs in MCRF |

`CreateRefs` consumes one contiguous `uint32[]` list:

```
MCRF[0 .. nDoodadRefs-1]                         -> indices into MDDF
MCRF[nDoodadRefs .. nDoodadRefs+nMapObjRefs-1]    -> indices into MODF
```

If `MDDF`/`MODF` are populated but each MCNK has `MCRF` empty and both counts set to zero, terrain loads correctly but no doodads or WMOs appear. This was the root cause of the "terrain works, objects missing" alpha writer bug fixed on 2026-05-09.

Current writer policy:

- MDDF references are assigned to the chunk containing the doodad `Position`.
- MODF references are assigned to every chunk whose planar bounds overlap the WMO `BoundsMin`/`BoundsMax`; if bounds do not overlap any chunk, the writer falls back to the chunk containing `Position`.
- Within each MCRF payload, all MDDF indices are written first, then all MODF indices, matching `CreateRefs`.

## Raw File Layout

### MDDF Entry (0x24 bytes = 36 bytes)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0x00 | 4 | nameId | uint32 |
| 0x04 | 4 | uniqueId | uint32 |
| 0x08 | 4 | pos.x | float |
| 0x0C | 4 | pos.y | float |
| 0x10 | 4 | pos.z | float |
| 0x14 | 4 | rot.x | float |
| 0x18 | 4 | rot.y | float |
| 0x1C | 4 | rot.z | float |
| 0x20 | 2 | scale | uint16 (scale/1024) |
| 0x22 | 2 | flags | uint16 |

### MODF Entry (0x40 bytes = 64 bytes)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0x00 | 4 | nameId | uint32 |
| 0x04 | 4 | uniqueId | uint32 |
| 0x08 | 4 | pos.x | float |
| 0x0C | 4 | pos.y | float |
| 0x10 | 4 | pos.z | float |
| 0x14 | 4 | rot.x | float |
| 0x18 | 4 | rot.y | float |
| 0x1C | 4 | rot.z | float |
| 0x20 | 4 | extents.b.x | float (lower corner X in file axis) |
| 0x24 | 4 | extents.b.y | float (lower corner Y in file axis) |
| 0x28 | 4 | extents.b.z | float (lower corner Z in file axis) |
| 0x2C | 4 | extents.t.x | float (upper corner X in file axis) |
| 0x30 | 4 | extents.t.y | float (upper corner Y in file axis) |
| 0x34 | 4 | extents.t.z | float (upper corner Z in file axis) |
| 0x38 | 2 | flags | uint16 |
| 0x3A | 2 | doodadSet | uint16 |
| 0x3C | 2 | nameSet | uint16 |
| 0x3E | 2 | scale | uint16 (scale/1024, alpha only?) |

## Position Transform (identical for MDDF and MODF)

The Ghidra decompilation for both `CreateDoodadDef` and `CreateMapObjDef` shows:

```
renderer_pos.x = -file_pos.z + MapOrigin     // field at offset 0x10 -> renderer X
renderer_pos.y = -file_pos.x + MapOrigin      // field at offset 0x08 -> renderer Y
renderer_pos.z = file_pos.y                    // field at offset 0x0C -> renderer Z
```

Where `file_pos.x/y/z` refer to the struct fields at offsets 0x08/0x0C/0x10, and `MapOrigin = 17066.666...`.

**Note on naming confusion**: Our reader code names the raw bytes `rawX` (offset+8), `rawZ` (offset+12), `rawY` (offset+16). This looks like it swaps Y and Z, but it matches the Ghidra naming where the struct field `pos.y` is at offset 0x0C and `pos.z` is at offset 0x10. The transform itself is correct regardless of names.

### Inverse (writer)

```
file_pos_at_offset_0x08 = MapOrigin - renderer.Y    (rawX in our code)
file_pos_at_offset_0x0C = renderer.Z                 (rawZ in our code, "height")
file_pos_at_offset_0x10 = MapOrigin - renderer.X    (rawY in our code)
```

### Our Code Convention

Our internal `Position` vector uses renderer coordinates:
- `Position = (MapOrigin - rawY, MapOrigin - rawX, rawZ)`

Reader (AlphaWdtReader.cs:755,799):
```csharp
Position = new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ)
```

Writer (AlphaWdtWriter.cs:547-549,571-573):
```csharp
Write(MapOrigin - Position.Y)   // offset+8
Write(Position.Z)                // offset+12
Write(MapOrigin - Position.X)    // offset+16
```

## Rotation Transform (identical for MDDF and MODF)

The Ghidra decompilation for both functions shows:

```
renderer_rot.x = file_rot.z * (π/180)        // Roll around renderer X axis
renderer_rot.y = file_rot.x * (π/180)        // Pitch around renderer Y axis
renderer_rot.z = file_rot.y * (π/180) + π     // Yaw around renderer Z axis, with +180° offset
```

Constants verified:
- `___real_3c8efa35` = 0x3C8EFA35 = `0.01745329238...` = π/180 (degrees to radians)
- `_DAT_00810e04` = `0x40490FDB` = `3.14159274101...` = π (the +180° yaw offset)

The matrix is built by: **Translate** → **RotateZ(yaw)** → **RotateY(pitch)** → **RotateX(roll)** → **Scale**.

### Inverse (writer)

The raw file rotation fields store **degrees**, and the axis mapping is:
```
file_rot_at_offset_0x14 = renderer_rot_in_degrees.X    (maps to file_rot.x -> renderer Y rotation)
file_rot_at_offset_0x18 = renderer_rot_in_degrees.Z    (maps to file_rot.y -> renderer Z rotation)
file_rot_at_offset_0x1C = renderer_rot_in_degrees.Y    (maps to file_rot.z -> renderer X rotation)
```

### Our Code Convention

Our `Rotation` vector stores the raw file values with the axis mapping **partially** applied:
```csharp
Rotation = (rotX, rotY, rotZ)  // where rotX=offset+20, rotZ=offset+24, rotY=offset+28
```

This is a **round-trip-safe** convention: reader writes `(rotX, rotY, rotZ)` and writer reads `(Rotation.X, Rotation.Z, Rotation.Y)` back to the same offsets. But it is NOT the true renderer rotation order; the viewer's `BuildLegacyMdxPlacementTransform` compensates:

```csharp
float rx = -DegreesToRadians(rotationDegrees.Y);  // -Y becomes roll
float ry = -DegreesToRadians(rotationDegrees.X);  // -X becomes pitch
float rz = DegreesToRadians(rotationDegrees.Z);   // Z becomes yaw
// Matrix: Rz(π) * Scale * Rx(rx) * Ry(ry) * Rz(rz) * Translate(position)
```

The separate `Matrix4x4.CreateRotationZ(MathF.PI)` compensates for the `+π` yaw offset in the client, and the negation of rx/ry compensates for the axis-negation in the position transform (since positions are negated in X/Y, the rotation senses are also negated).

### ⚠ IMPORTANT NOTE

The `+π` yaw offset (`renderer_rot.z = file_rot.y * π/180 + π`) means that a doodad with `file_rot.y = 0` will be rotated 180° (π radians) around the Z axis in the renderer. This is NOT captured in our `Rotation` vector — it's handled by the viewer's explicit `CreateRotationZ(π)` prefix. When writing rotation values to file, we do NOT subtract π from the yaw; we write `Rotation.Y` directly to offset+0x1C, and the `+π` is applied by the client at load time.

## Bounds Transform (MODF only)

The Ghidra decompilation for `CreateMapObjDef` shows:

```
renderer_bounds_min.x = -file_extents_max.z + MapOrigin
renderer_bounds_min.y = -file_extents_max.x + MapOrigin
renderer_bounds_min.z = file_extents_min.y

renderer_bounds_max.x = -file_extents_min.z + MapOrigin
renderer_bounds_max.y = -file_extents_min.x + MapOrigin
renderer_bounds_max.z = file_extents_max.y
```

Note: The negation of Z and X axes swaps min/max for those components. In the file:
- `extents.b` (at offset 0x20-0x28) is the **lower** corner in file axes
- `extents.t` (at offset 0x2C-0x34) is the **upper** corner in file axes

But after axis transform, `extents.b.z` maps to `-min_renderer_X` and `extents.t.z` maps to `-max_renderer_X`, so the negation flips the sense.

### Our Code Convention

```csharp
// Reader (AlphaWdtReader.cs:805-806)
BoundsMin = new Vector3(MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ)
BoundsMax = new Vector3(MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ)

// Writer (AlphaWdtWriter.cs:577-582)
Write(MapOrigin - BoundsMax.Y)   // offset+32  (file extents.b.z -> -max_renderer.X + MapOrigin)
Write(BoundsMin.Z)               // offset+36  (file extents.b.x -> renderer min.Z)
Write(MapOrigin - BoundsMax.X)   // offset+40  (file extents.b.y -> -max_renderer.Y + MapOrigin)
Write(MapOrigin - BoundsMin.Y)   // offset+44  (file extents.t.z -> -min_renderer.X + MapOrigin)
Write(BoundsMax.Z)               // offset+48  (file extents.t.x -> renderer max.Z)
Write(MapOrigin - BoundsMin.X)   // offset+52  (file extents.t.y -> -min_renderer.Y + MapOrigin)
```

The variable naming `bbMin/Max` in the reader matches the raw byte layout: `bbMin` reads from offset+32 (the file extents.b field), `bbMax` reads from offset+44 (the file extents.t field).

## Scale Transform (MDDF only)

```
scale_as_float = uint16_scale / 1024.0
```

Ghidra verified at `0x6805e0`:
```
uVar4 = param_1->scale;                    // uint16 at offset 0x20
fVar2 = (float)uVar4 * ___real_3a800000;   // ___real_3a800000 = 1/1024 = 0.0009765625
*(float *)&pCVar9->field_0x28 = fVar2;
```

## LK ADT Writer Alignment

The LK ADT writer (`LkAdtWriter.cs`) uses the **same** coordinate conventions as the alpha writer for both MDDF and MODF, because the client format did not change between alpha and LK for placement entries. The key difference is that the LK format adds `doodadSet`, `nameSet`, and `scale` fields to the MODF entry (expanding from 0x3C to 0x40 bytes).

## Round-Trip Test Coverage

`LkToAlphaRoundTripTests.cs` includes:

1. **`WriteAlphaWdt_UsesClientMainOrderAndMcnkSubchunkContract`** — structural validation
2. **`ConvertTile_AndWriteAlphaWdt_RoundTripsChunkHeightsAlphaAndLiquid`** — terrain/liquid
3. **`ConvertTile_ThroughAlphaWdt_BackToLkAdt_RoundTripsLiquidIntoMh2o`** — liquid → MH2O
4. **`LkAdtWriter_RoundTripsModfBoundsWithReaderOrientation`** — MODF through LK writer
5. **`AlphaWdt_RoundTripsMddfAndModfPlacements`** — MDDF/MODF through alpha WDT reader/writer

All tests pass as of 2026-05-09.

## Cross-Reference

- Height convention: `alpha-mcnk-flags-and-metadata-plan.md` Section "Height Convention Contract"
- MCNK field layout: `alpha-mcnk-flags-and-metadata-plan.md` Section "Alpha MCNK Header Layout"
- V14 model plan: `v14-model-and-refactor-plan-2026-05-06.md` Section 10.2
- Memory bank: `gillijimproject_refactor/memory-bank/activeContext.md`, `progress.md`
