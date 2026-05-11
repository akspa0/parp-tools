# Alpha 0.5.3 WDT/ADT Ghidra Research (Client 0.5.3.3368)

**Created**: 2026-05-10
**Client**: WoWClient.exe 0.5.3.3368
**Status**: Research-only. No code changes in this session.

## Key Finding: Alpha MCNK Subchunk Offsets Are Fixed-Position, Not Offset-Field-Based

The 0.5.3 client (`CMapChunk::Create` at 0x699000) does **NOT** use MCNK header offset fields (0x18 MCVTOffset, 0x1C MCNROffset, etc.) to locate subchunks. Instead, it uses **hardcoded positions** within the MCNK block:

| Subchunk | Offset from MCNK FourCC | Calculation |
|----------|--------------------------|-------------|
| MCNK header data | +8 (after FourCC+size) | Fixed 8-byte chunk header |
| Position.Z | +0x88 | `param_1 + 0x88` — chunks header(8) + header data offset 0x80 |
| MCVT (heights) | +0x88 | `CreateVertices(this, (float*)(param_1 + 0x88))` — same as Position.Z! Heights start immediately after the 128-byte MCNK header |
| MCNR (normals) | +0x2CC | `CreateNormals(this, (char*)(param_1 + 0x2CC))` |
| MCLY | +0x48C | Verified by FourCC check: `*(int*)(param_1 + 0x48C) == 0x4d434c59` |
| MCRF | Computed | `mLayer->pad + MCLY_size - 6` then walk through MCLY layers |
| MCRF nDoodadRefs | +0x1C | `*(uint*)(param_1 + 0x1C)` — read from MCNK header |
| MCRF nMapObjRefs | +0x44 | `*(uint*)(param_1 + 0x44)` — read from MCNK header |

All offsets above include the 8-byte FourCC+size chunk header. Subtract 8 for offsets from MCNK data start.

**Implication**: The alpha client ignores the offset fields at MCNK header offsets 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30. It hardcodes where each subchunk begins. Our writer must still produce the correct layout but the alpha client won't read these offset fields to find subchunks — it navigates the MCLY chunk chain and knows MCVT/MCNR at fixed positions.

However, the offset fields ARE read by our `AlphaWdtReader` and by `McnkAlpha` in the legacy codebase. So they must still be correct for round-trip fidelity.

## Key Finding: Heights Are Absolute (Confirmed)

`CMapChunk::Create` passes `(float*)(param_1 + 0x88)` to `CreateVertices`. In `CreateVertices`:
```c
v->z = *he;  // Read MCVT value directly
// ...
NTempest::C3Vector::operator-=(v, (C3Vector*)&this->field_0x64);
```

Where `field_0x64` is computed as:
```c
fVar3 = (float)(int)uVar21 * ___real_42055555;  // cOffset.x * UNITS_PER_TILE / 8
*(float*)&this->field_0x64 = fVar3;
fVar4 = (float)iVar7 * ___real_42055555;          // cOffset.y * UNITS_PER_TILE / 8
*(float*)&this->field_0x68 = fVar4;
*(float*)&this->field_0x6c = *(float*)(param_1 + 0x88);  // Position.Z from MCNK header
fVar3 = -fVar3 + ___real_46855555;  // MapOrigin - cOffsetX * chunkSize
*(float*)&this->field_0x64 = -fVar4 + ___real_46855555;  // MapOrigin - cOffsetY * chunkSize
*(float*)&this->field_0x68 = fVar3;
```

Then `operator-=` subtracts the entire vector `(MapOrigin - cOffsetY*chunkSize, MapOrigin - cOffsetX*chunkSize, Position.Z)` from each vertex position.

Result: `vertex.z = MCVT_height - Position.Z`. Heights are absolute world-space Z. Position.Z = first vertex height. The subtraction is for rendering-precision only (moving vertices close to origin).

## Key Finding: MCRF Consumption in CreateRefs

`CMapChunk::CreateRefs` (0x69a0c0) is called from `CMapChunk::Create`:

```c
CreateRefs(this, pCVar2, (uint*)puVar11, *(uint*)(param_1 + 0x1c), *(uint*)(param_1 + 0x44));
```

Parameters:
- `this`: CMapChunk
- `pCVar2`: CMapArea (parent tile)
- `puVar11`: pointer to MCRF data (uint32 array)
- `param_1 + 0x1C`: nDoodadRefs count from MCNK header
- `param_1 + 0x44`: nMapObjRefs count from MCNK header

Inside CreateRefs:
```c
pos.x = 17066.666;  // MapOrigin
pos.y = 17066.666;
pos.z = 0.0;

// Process doodad references (MDDF)
if (param_3 != 0) {  // nDoodadRefs
    do {
        uVar2 = *param_2;  // index into MDDF array
        // bounds check against doodadDefList size
        pCVar6 = CMap::CreateDoodadDef(
            (SMDoodadDef*)(doodadDefList.array_data + uVar2 * 0x24),
            &pos);
        // ... link to chunk
        param_2 = param_2 + 1;
        param_3 = param_3 - 1;
    } while (param_3 != 0);
}

// Process map object references (MODF)
if (param_4 != 0) {  // nMapObjRefs
    do {
        uVar2 = *param_2;  // index into MODF array
        // bounds check against mapObjDefList size
        pCVar9 = CMap::CreateMapObjDef(
            (SMMapObjDef*)(mapObjDefList.array_data + uVar2 * 0x40),
            &pos);
        // ... link to chunk
        param_2 = param_2 + 1;
        param_4 = param_4 - 1;
    } while (param_4 != 0);
}
```

**Critical observations**:
1. MCRF is a single contiguous `uint32` array. The first `nDoodadRefs` entries are MDDF indices, then the next `nMapObjRefs` entries are MODF indices.
2. `CMapArea::Create` loads the MDDF/MODF arrays into `doodadDefList` and `mapObjDefList` from the ADT-level MDDF and MODF chunks.
3. `CreateRefs` indexes into those arrays using `*param_2` as the index.
4. The area offset `(17066.666, 17066.666, 0.0)` is passed as the position offset to both CreateDoodadDef and CreateMapObjDef — this is the MapOrigin constant.
5. MDDF entry size = 0x24 (36 bytes), MODF entry size = 0x40 (64 bytes).

## Key Finding: Position Transform (Ghidra-Verified)

In `CMap::CreateDoodadDef(SMDoodadDef&, C3Vector&)` at 0x6805e0:

```c
// Position transform
*(float*)puVar1 = -(param_1->pos).z;           // renderer.x = -file_z
*(float*)&pCVar9->field_0x20 = -(param_1->pos).x; // renderer.y = -file_x
*(float*)&pCVar9->field_0x24 = (param_1->pos).y;  // renderer.z = file_y

// Then add area offset (MapOrigin)
*(float*)puVar1 = *(float*)puVar1 + param_2->x;   // + MapOrigin
*(float*)&pCVar9->field_0x20 = param_2->y + *(float*)&pCVar9->field_0x20; // + MapOrigin
*(float*)&pCVar9->field_0x24 = *(float*)&pCVar9->field_0x24 + param_2->z;   // + 0
```

So: `renderer = (MapOrigin - file_z, MapOrigin - file_x, file_y)`

In `CMap::CreateMapObjDef(SMMapObjDef&, C3Vector&)` at 0x681250 — identical position transform:
```c
*(float*)puVar1 = -(param_1->pos).z;
*(float*)&pCVar9->field_0x20 = -(param_1->pos).x;
*(float*)&pCVar9->field_0x24 = (param_1->pos).y;
// + MapOrigin
```

## Key Finding: Rotation Transform (Ghidra-Verified)

In `CreateDoodadDef(SMDoodadDef&)`:
```c
rot.x = (param_1->rot).z * ___real_3c8efa35;  // file_rot.z * π/180 (Roll)
rot.y = (param_1->rot).x * ___real_3c8efa35;  // file_rot.x * π/180 (Pitch)
rot.z = (param_1->rot).y * ___real_3c8efa35 + _DAT_00810e04;  // file_rot.y * π/180 + π

// Build matrix: Translate -> Rz(yaw) -> Ry(pitch) -> Rx(roll) -> Scale
```

Constants: `___real_3c8efa35 = 0.01745329238...` (π/180), `_DAT_00810e04 = 3.14159274...` (π).

In `CreateMapObjDef(SMMapObjDef&)` — identical rotation transform.

## Key Finding: Bounds Transform (MODF, Ghidra-Verified)

In `CreateMapObjDef(SMMapObjDef&)`:
```c
// Bounds transform (after matrix transform)
*(float*)&pCVar9->field_0x3c = -(param_1->extents).t.z + param_2->x;  // min.renderer.x = -extents_max.z + MapOrigin
*(float*)&pCVar9->field_0x40 = -(param_1->extents).t.x + param_2->y;  // min.renderer.y = -extents_max.x + MapOrigin
*(float*)&pCVar9->field_0x44 = (param_1->extents).b.y + param_2->z;    // min.renderer.z = extents_min.y + 0

*(float*)&pCVar9->field_0x48 = -(param_1->extents).b.z + param_2->x;  // max.renderer.x = -extents_min.z + MapOrigin
*(float*)&pCVar9->field_0x4c = -(param_1->extents).b.x + param_2->y;  // max.renderer.y = -extents_min.x + MapOrigin
*(float*)&pCVar9->field_0x50 = (param_1->extents).t.y + param_2->z;    // max.renderer.z = extents_max.y + 0
```

Note: `extents.t` = file upper corner, `extents.b` = file lower corner. The Z and X negation swaps min/max.

## Key Finding: Scale Transform (MDDF)

In `CreateDoodadDef(SMDoodadDef&)`:
```c
uVar4 = param_1->scale;
// ...
fVar2 = (float)uVar4 * ___real_3a800000;  // scale * (1/1024)
*(float*)&pCVar9->field_0x28 = fVar2;
```

`___real_3a800000 = 1/1024 = 0.0009765625`.

## Key Finding: MDDF Entry Size = 0x24, MODF Entry Size = 0x40

From `CMapArea::Create` (0x6aad30):
- MDDF: `uVar6 = *(uint*)(param_1 + iVar2 + 0xc) / 0x24` — total MDDF size divided by 36 bytes per entry
- MODF: `uVar6 = *(uint*)(param_1 + iVar2 + 0xc) >> 6` — total MODF size divided by 64 bytes per entry (shift right 6 = divide by 64)

## Key Finding: CMapArea::Create Reads MHDR, MCIN, MTEX, MDDF, MODF

The `CMapArea::Create` function reads:
1. MHDR FourCC at `param_1` 
2. MCIN at `param_1 + *(int*)(param_1 + 8) + 8` — offset from MHDR
3. MTEX FourCC check
4. MDDF — size divided by 0x24 to get count, bulk-copied into `doodadDefList`
5. MODF — size right-shifted by 6 (÷64) to get count, bulk-copied into `mapObjDefList`

The MDDF/MODF data is stored as raw `SMDoodadDef`/`SMMapObjDef` structs — the area reads the entire chunk and lays it out in a growable array. Then `CMapChunk::CreateRefs` indexes into that array using MCRF indices.

## Key Finding: Alpha WDT Embeds Tile Data Differently from LK ADTs

In the alpha WDT format, `CMapArea::Create` is called with the entire embedded tile data blob after MHDR/MCIN/MTEX/MDDF/MODF are parsed. The chunk data is located via MCIN entries which point to the start of each MCNK chunk. Each MCNK chunk contains MCVT/MCNR/MCLY/MCRF/MCSH/MCAL/MCLQ as subchunks, all referred to via their fixed positions within the MCNK block.

The key structural difference: alpha WDT embeds MDDF/MODF at the **tile level** (inside the WDT's per-tile data), while LK ADTs have them at the root ADT level. The `SMDoodadDef`/`SMMapObjDef` structs are the same binary format in both cases.

## Key Finding: ZoneId and Holes

From `CMapChunk::Create`:
```c
this->zoneId = *(uint*)(param_1 + 0x40);   // AreaId at offset 0x40 from MCNK data start
this->holes = *(ushort*)(param_1 + 0x48);  // Hole mask at offset 0x48
```

These are MCNK header fields at data offsets 0x38 and 0x40 (param_1 includes the 8-byte chunk header, so 0x40 - 8 = 0x38 for AreaId data offset, and 0x48 - 8 = 0x40 for Holes data offset). Note the discrepancy with the earlier plan doc which listed AreaId at "offset 0x3C" — I need to double-check whether param_1 includes the chunk header or not.

**Re-checking**: The function starts with `if (*(int*)param_1 != 0x4d434e4b)` which checks the MCNK FourCC. So `param_1` points to the MCNK FourCC byte. Therefore:
- `param_1 + 0x40` = FourCC byte + 0x40 = data offset 0x38 for AreaId
- `param_1 + 0x48` = data offset 0x40 for Holes

But our plan doc says AreaId is at "0x38" in the header. That's `param_1 + 8 + 0x38 = param_1 + 0x40`. This matches!

For holes: our doc says "offset 0x40" in the header data. `param_1 + 0x48 = param_1 + 8 + 0x40`. Also matches.

## Key Finding: PredTex and NoEffectDoodad Fields

```c
*(undefined4*)this->predTex = *(undefined4*)(param_1 + 0x4c);         // PredTex1
*(undefined4*)(this->predTex + 2) = *(undefined4*)(param_1 + 0x50); // PredTex2
*(undefined4*)(this->predTex + 4) = *(undefined4*)(param_1 + 0x54); // PredTex3
*(undefined4*)(this->predTex + 6) = *(undefined4*)(param_1 + 0x58); // PredTex4
*(undefined4*)this->noEffectDoodad = *(undefined4*)(param_1 + 0x5c); // NoEffectDoodad1
*(undefined4*)(this->noEffectDoodad + 4) = *(undefined4*)(param_1 + 0x60); // NoEffectDoodad2
```

Data offsets (subtract 8 for chunk header):
- PredTex1-4: 0x44, 0x48, 0x4C, 0x50 (our doc says 0x4C-0x58 — need to verify alignment)

Wait: `param_1 + 0x4c` = data offset 0x44, but our plan doc says PredTex1 is at offset 0x4C. Let me reconcile. Our doc lists offsets from MCNK data region (after 8-byte chunk header). So:
- param_1 points to FourCC byte
- data offset = (param_1 offset) - 8

| Ghidra param_1 offset | Data offset | Field | Our doc offset |
|------------------------|-------------|-------|----------------|
| 0x40 | 0x38 | AreaId | 0x38 ✓ |
| 0x48 | 0x40 | Holes | 0x40 ✓ (low 16 bits) |
| 0x4C | 0x44 | PredTex1 | 0x4C ✗ — Ghidra says 0x44 |
| 0x50 | 0x48 | PredTex2 | 0x50 ✗ — Ghidra says 0x48 |

There's an 8-byte offset discrepancy for PredTex. Our doc has PredTex starting at 0x4C, but the Ghidra decompilation shows it at data offset 0x44. Looking more carefully at our doc:

```
| 0x44 | 4 | Unknown5 | Purpose unknown |
| 0x48 | 4 | Unknown6 | Purpose unknown |
| 0x4C | 4 | PredTex1 | Predetermined texture data (4 × uint) |
```

Ghidra reads PredTex1 at `param_1 + 0x4c` = data offset 0x44. Our doc says offset 0x44 is "Unknown5" and PredTex1 starts at 0x4C. This means our doc had the PredTex offsets wrong — they start 8 bytes earlier than documented. The fields we labeled Unknown5 and Unknown6 are actually PredTex1 and PredTex2, and our PredTex1-4 are actually PredTex3-4 and NoEffectDoodad1-2.

Actually wait — looking again. `*(undefined4*)this->predTex` is an array indexed by uint4 values. The code does:
```c
*(undefined4*)this->predTex = *(undefined4*)(param_1 + 0x4c);     // predTex[0]
*(undefined4*)(this->predTex + 2) = *(undefined4*)(param_1 + 0x50); // predTex[2] (8 bytes into array)
*(undefined4*)(this->predTex + 4) = *(undefined4*)(param_1 + 0x54); // predTex[4] (16 bytes into array)
*(undefined4*)(this->predTex + 6) = *(undefined4*)(param_1 + 0x58); // predTex[6] (24 bytes into array)
```

Since `predTex` stores uint4 entries (4 bits each), `predTex[0]` = one uint32, `predTex[2]` = 8 bytes later. So `param_1 + 0x4c` through `param_1 + 0x58` = MCNK data offsets 0x44 through 0x50 are contiguous (16 bytes for PredTex), which matches 4 × uint(4 bytes each) = 16 bytes at data offset 0x44.

So our doc's PredTex1 should be at data offset 0x44, not 0x4C. The Unknown5 and Unknown6 fields at 0x44 and 0x48 in our plan doc ARE PredTex1 and PredTex2.

## Key Finding: Alpha MCNR Normal Encoding (Ghidra-Verified)

`CMapChunk::CreateNormals` at `0x699b60`:

```c
pCVar1->y = (float)(int)*pcVar3 * -1/127;      // byte[0] → Y component, negated
pCVar1->z = (float)(int)pcVar3[1] * 1/127;       // byte[1] → Z component, positive
pCVar1->x = (float)(int)pcVar3[2] * -1/127;      // byte[2] → X component, negated
```

Each normal is 3 bytes in the order `(-Y_signed, Z_signed, -X_signed)` where signed means `clamp(round(value * 127), -128, 127)`.

**This is different from LK format** which typically uses `(X, Z, Y)` or `(X, Y, Z)`. The alpha format negates X and Y, and the component order is `Y, Z, X`.

## Key Finding: Alpha MCNK Liquid Flags (Ghidra-Verified)

`CMapChunk::Create` at the liquid loop (disassembly around `0x698f70-0x699007`):

The loop starts with mask=4 and shifts left each iteration, testing MCNK flag bits 2, 3, 4, 5 against the chunk's flags field:
- **Bit 2 (0x04)**: Has water surface — liquid type 1 (river/lake)
- **Bit 3 (0x08)**: Has ocean/coast water — liquid type 2
- **Bit 4 (0x10)**: Has lava — liquid type 3 sub-type 0
- **Bit 5 (0x20)**: Has slime — liquid type 3 sub-type 1

For each set bit, the client allocates a `CChunkLiquid` and copies the corresponding MCLQ data entry.

Additionally, at `0x69919a`:
```c
TEST byte ptr [EDI], 0x2    ; test MCNK flags bit 1
JZ skip
MOV word ptr [ESI + 0x74], 0x100   ; set field_0x74 |= 0x100 if bit 1 set
```

And at `0x6992bc-0x6992da`:
```c
TEST byte ptr [EBX], 0x1   ; test MCNK flags bit 0
JZ skip_shadow
; if bit 0 set AND cvar check, create shadow texture from MCSH data
```

**Summary of MCNK flag bits for alpha client:**
- Bit 0 (0x01): Has shadow map (MCSH data present)
- Bit 1 (0x02): Unknown flag (sets chunk field_0x74 |= 0x100)
- Bits 2-5 (0x04, 0x08, 0x10, 0x20): Liquid type flags (each bit = one liquid entry)
- No other bits are used by the alpha client

## Key Finding: SMDoodadDef Struct Layout (0x24 = 36 bytes)

From `CreateDoodadDef(SMDoodadDef*, C3Vector*)` at 0x6805e0 and `CMapArea::Create` at 0x6aae00:

| Offset | Type | Field | Notes |
|--------|------|-------|-------|
| 0x00 | uint32 | nameId | Index into WDT-level doodadNamesIndex → byte offset into MDNM |
| 0x04 | uint32 | uniqueId | Unique doodad ID |
| 0x08 | float | pos.x | File X = MapOrigin - renderer.Y |
| 0x0C | float | pos.y | File Y = renderer.Z |
| 0x10 | float | pos.z | File Z = MapOrigin - renderer.X |
| 0x14 | float | rot.x | Degrees, maps to renderer pitch (Y rotation) |
| 0x18 | float | rot.y | Degrees, maps to renderer yaw (Z rotation) + 180° |
| 0x1C | float | rot.z | Degrees, maps to renderer roll (X rotation) |
| 0x20 | uint16 | scale | Scale value = rendering_scale * 1024 |
| 0x22 | uint16 | padding | Always 0 |

Struct layout verified by initialization in `CMapArea::Create`:
```c
NTempest::C3Vector::C3Vector((C3Vector*)(iVar8 + 8), 0.0);    // pos at offset 0x08
NTempest::C3Vector::C3Vector((C3Vector*)(iVar8 + 0x14), 0.0);  // rot at offset 0x14
```

And size confirmed: `uVar6 = *(uint*)(param_1 + iVar2 + 0xc) / 0x24` (total MDDF size ÷ 36).

## Key Finding: SMMapObjDef Struct Layout (0x40 = 64 bytes)

From `CreateMapObjDef(SMMapObjDef*, C3Vector*)` at 0x681250 and `CMapArea::Create`:

| Offset | Type | Field | Notes |
|--------|------|-------|-------|
| 0x00 | uint32 | nameId | Index into WDT-level mapObjNamesIndex → byte offset into MONM |
| 0x04 | uint32 | uniqueId | Unique map object ID |
| 0x08 | float | pos.x | File X = MapOrigin - renderer.Y |
| 0x0C | float | pos.y | File Y = renderer.Z |
| 0x10 | float | pos.z | File Z = MapOrigin - renderer.X |
| 0x14 | float | rot.x | Degrees, maps to renderer pitch |
| 0x18 | float | rot.y | Degrees, maps to renderer yaw + 180° |
| 0x1C | float | rot.z | Degrees, maps to renderer roll |
| 0x20 | float | extents.t.x | File bounds = MapOrigin - renderer_min_Y (LARGE file.x value) |
| 0x24 | float | extents.t.y | File bounds = renderer_max_Z |
| 0x28 | float | extents.t.z | File bounds = MapOrigin - renderer_min_X (LARGE file.z value) |
| 0x2C | float | extents.b.x | File bounds = MapOrigin - renderer_max_Y (SMALL file.x value) |
| 0x30 | float | extents.b.y | File bounds = renderer_min_Z |
| 0x34 | float | extents.b.z | File bounds = MapOrigin - renderer_max_X (SMALL file.z value) |
| 0x38 | uint16 | doodadSet | Doodad set index |
| 0x3A | uint16 | nameSet | Name set index |
| 0x3C | 4 bytes | padding | Always 0 |

Struct layout verified by initialization in `CMapArea::Create`:
```c
NTempest::C3Vector::C3Vector((C3Vector*)(iVar8 + 8), 0.0);    // pos at offset 0x08
NTempest::C3Vector::C3Vector((C3Vector*)(iVar8 + 0x14), 0.0);  // rot at offset 0x14
NTempest::CAaBox::CAaBox((CAaBox*)(iVar8 + 0x20), 0.0);        // extents at offset 0x20
```

And size confirmed: `uVar6 = *(uint*)(param_1 + iVar2 + 0xc) >> 6` (total MODF size ÷ 64).

## Key Finding: Full Coordinate Transform (Writing Direction)

**Position transform (renderer → file):**
```
file.pos.x = MapOrigin - renderer.Y
file.pos.y = renderer.Z
file.pos.z = MapOrigin - renderer.X
```

Where MapOrigin = (17066.666, 17066.666, 0.0).

**Rotation transform (renderer → file, input in degrees):**
```
file.rot.x = renderer_rot.Y    (pitch — renderer Y rotation)
file.rot.y = renderer_rot.Z - 180.0   (yaw minus π, renderer Z rotation)
file.rot.z = renderer_rot.X    (roll — renderer X rotation)
```

Applied by client as: `RotateZ(file_rot_y * π/180 + π) → RotateY(file_rot_x * π/180) → RotateX(file_rot_z * π/180)`.

**Bounds transform (renderer → file):**
```
extents.t.x = MapOrigin - renderer_bounds_min.Y    (large file.x = "top")
extents.t.y = renderer_bounds_max.Z                  (large file.y = "top")
extents.t.z = MapOrigin - renderer_bounds_min.X    (large file.z = "top")
extents.b.x = MapOrigin - renderer_bounds_max.Y    (small file.x = "bottom")
extents.b.y = renderer_bounds_min.Z                  (small file.y = "bottom")
extents.b.z = MapOrigin - renderer_bounds_max.X    (small file.z = "bottom")
```

Note: "t" and "b" in the struct refer to top/bottom in **file coordinates**, where X and Z are inverted relative to renderer coordinates. This means `extents.t` has the larger values for inverted axes (X, Z) and `extents.b` has the smaller ones.

**Scale (MDDF only):**
```
file.scale = (uint16)Math.Round(renderer_scale * 1024)
```

## Key Finding: WDT-Level Name Tables (MDNM/MONM)

`LoadDoodadNames` (0x680040) and `LoadMapObjNames` (0x6801a0):

1. Read MDNM/MONM chunk from WDT file
2. Build an offset index (doodadNamesIndex/mapObjNamesIndex) by scanning null-terminated strings
3. `nDoodadNames` from MPHD pre-allocates the index array size
4. `doodadNamesIndex[i]` stores the byte offset of the i-th string in the blob
5. `nameId` in MDDF/MODF is an INDEX into `doodadNamesIndex`/`mapObjNamesIndex`, NOT a direct byte offset

Name resolution: `modelName = names_blob[names_index[nameId]]`

The per-tile MDDF/MODF entries use WDT-level name IDs. Each tile's MDDF nameId indexes into the global doodadNamesIndex built from the WDT-level MDNM chunk.

## Key Finding: CMapArea::Create Per-Tile Data Layout

`CMapArea::Create` (0x6aae00) reads per-tile embedded data with:

1. **MHDR** at blob offset 0: 8-byte chunk header (MHDR + size) then 64 bytes of data
2. MHDR data contains offsets (relative to MHDR data start) to sub-chunks:
   - Data byte 0x00: MCIN offset → MCIN chunk at `blob + MHDR_data_start + offset`
   - Data byte 0x04: MTEX offset → MTEX chunk at `blob + MHDR_data_start + offset`
   - Data byte 0x0C: MDDF offset → MDDF chunk at `blob + MHDR_data_start + offset`
   - Data byte 0x14: MODF offset → MODF chunk at `blob + MHDR_data_start + offset`
3. Each sub-chunk token validated at `blob + offset + 8`, size at `blob + offset + 0xC`, data at `blob + offset + 0x10`
4. MDDF data bulk-copied into `doodadDefList` array
5. MODF data bulk-copied into `mapObjDefList` array
6. MCIN entries contain offsets to individual MCNK chunks

The current writer's MHDR layout (WriteMhdrData) is consistent with this:
```csharp
data[0x00] = 64;           // MHDR data size or MCIN offset
data[0x04] = mtexRelative; // MTEX offset relative to MHDR data start
data[0x08] = 0;            // unused
data[0x0C] = mddfRelative; // MDDF offset relative to MHDR data start
data[0x10] = 0;            // unused
data[0x14] = modfRelative; // MODF offset relative to MHDR data start
```