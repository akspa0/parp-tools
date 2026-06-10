# Wow.exe 3.3.5.12340 - MH2O Liquids Deep Dive

Scope: MH2O chunk discovery, liquid geometry build, rendering objects, and player/liquid queries.

## MH2O chunk discovery

### Function addresses
- FUN_007d6ef0: ADT parse and MHDR offset mapping

### Decompiled evidence

```c
// FUN_007d6ef0
if (*(int *)(pbVar1 + 0x28) != 0) {
  // MH2O chunk offset (relative to MHDR data start)
  FUN_007d4f10(pbVar1 + *(int *)(*(int *)(param_1 + 0x68) + 0x28) + 8);
}
```

### Definitive answer
- MH2O is found via **MHDR offset** at `+0x28`.
- The offset is **relative to MHDR data start** and the code adds `+8` to point at chunk data (skipping FourCC+size).
- This mirrors how MDDF/MODF/MH2O are all resolved: **offsets, not linear scan**.

---

## Liquid chunk objects and buffers

### Function addresses
- FUN_007cf140: allocate or reuse `CChunkLiquid` buffers
- FUN_007cefd0: allocate vertex/index buffers
- FUN_007cf790: allocate chunk-liquid records
- FUN_0079e7c0: pool registration (`WCHUNKLIQUID`)

### Decompiled evidence

```c
// FUN_0079e7c0
uVar7 = FUN_004d3650(0x444,0x40,"WCHUNKLIQUID",1);
*puVar5 = uVar7; // pool for liquid chunks
```

```c
// FUN_007cf140
uVar5 = FUN_007cefd0(param_1,param_2,param_3,iVar2); // allocate buffers
```

```c
// FUN_007cefd0
uVar2 = FUN_006876d0(0,1,param_1 * param_4,3,"CChunkBuf_Vertex");
uVar3 = FUN_006876d0(1,1,param_2 * 2,3,"CChunkBuf_Index");
...
puVar1[2] = param_1 * param_4; // vertex bytes
puVar1[3] = param_2 * 2;       // index bytes
```

### Definitive answer
- Liquid rendering uses a **chunk-based pipeline** (`WCHUNKLIQUID` pool).
- Per-chunk geometry is built into **vertex and index buffers** sized by the liquid sub-rects in MH2O.

---

## Liquid geometry build from MH2O

### Function addresses
- FUN_007d4ab0: builds liquid geometry and uploads buffers

### Decompiled evidence

```c
// FUN_007d4ab0
if (*(int *)(param_1 + 0x1c) == 0) {
  uVar5 = FUN_007cf140(local_8 & 0xffff,local_c & 0xffff,0,param_2);
  *(undefined4 *)(param_1 + 0x1c) = uVar5;
}
*param_3 = *(int *)(*(int *)(param_1 + 0x1c) + 0x10); // vertex buffer
*param_4 = *(int *)(*(int *)(param_1 + 0x1c) + 0x14); // index buffer
...
FUN_007ce390(local_78,local_28,&local_24,&local_1c,&local_18,&local_10,&local_14);
...
FUN_007ce270(iVar4 + param_5[2] * 2,sVar2,param_5); // index emit
```

### Definitive answer
- The engine **builds liquid geometry per MH2O chunk** using the sub-rect extents.
- Geometry is streamed into **vertex and index buffers**, with per-chunk caching and reuse.
- The build is skipped if cached buffers exist and the render state allows reuse.

---

## MH2O per-liquid struct layout (inferred)

### Function addresses
- FUN_007d4ab0: top-level builder
- FUN_007ce390: vertex emission (positions, UVs, optional attributes)
- FUN_007ce270: index emission (skips masked cells)
- FUN_007ce1f0: per-cell liquid mask check

### Decompiled evidence

```c
// FUN_007ce390
local_c = FUN_0079b870(*(undefined4 *)(param_1 + 4));
local_18 = (uint)(*(int *)(param_1 + 8) == 1);
local_14 = (*(int *)(param_1 + 0x3c) - *(int *)(param_1 + 0x34)) + 1;
local_2c = (*(int *)(param_1 + 0x40) - *(int *)(param_1 + 0x38)) + 1;
local_8 = param_1 + 0x78; // vertex position list (float3)
...
uVar5 = (**(code **)(**(int **)(param_1 + 0x44) + 8))(iVar8); // per-vertex data
...
puVar6 = (ushort *)(**(code **)(**(int **)(param_1 + 0x44) + 0xc))(iVar8); // UV data
```

```c
// FUN_007ce1f0
if (*(int *)(param_1 + 0x54) != 0) {
  return FUN_007ce180(...); // mask map lookup
}
if ((*(uint *)(param_1 + 0x38) <= param_2) && (*(uint *)(param_1 + 0x34) <= param_3) &&
    (param_2 < *(uint *)(param_1 + 0x40)) && (param_3 < *(uint *)(param_1 + 0x3c))) {
  return FUN_0095da20(...); // mask bit
}
```

```c
// FUN_007ce270
cVar3 = FUN_007ce1f0(iVar5,iVar6); // skip masked cells when emitting indices
```

### Inferred field map

- `+0x04`: liquid type id (used by `FUN_0079b870` for per-liquid color/param tables).
- `+0x08`: texture coordinate mode (when `== 1`, UVs are read from a separate table; otherwise derived from XY).
- `+0x10`: count of liquid sub-rects (used in `FUN_007d4ab0`).
- `+0x14`: pointer to array of sub-rect/layer records (used in `FUN_007d4ab0`).
- `+0x1c`: pointer to `CChunkLiquid` cache (vertex/index buffers).
- `+0x20..+0x2c`: cached buffer metadata written after build.
- `+0x34..+0x40`: liquid grid bounds: `minY, minX, maxY, maxX` (used for index/vertex loops and mask checks).
- `+0x44`: pointer to an auxiliary table with virtual methods for per-vertex extra data and UVs.
- `+0x54`: pointer to a per-cell mask map; if null, a simpler rectangular mask is used.
- `+0x78`: base pointer to float3 vertex positions (advanced by 0x0c per vertex).

### Definitive answer
- The engine builds MH2O liquids from a **per-liquid struct** with bounds, per-cell mask, and vertex list.
- Masking is handled by `FUN_007ce1f0`, which chooses between an explicit **mask map** (`+0x54`) or a default rect-based mask.
- Geometry emission loops are based on `(max - min + 1)` extents stored at `+0x34..+0x40`.

### Confidence
Medium (layout inferred from field access; names are derived from behavior).

---

## MH2O mask byte layout and meaning

### Function addresses
- FUN_007ce180: decode mask byte
- FUN_007ce1f0: mask lookup used by geometry and queries

### Decompiled evidence

```c
// FUN_007ce180
iVar2 = *(int *)(param_1 + 0x54); // mask base
iVar1 = *param_2 + param_2[1] * 8; // local cell index (8x8)
pbVar3 = (byte *)(iVar1 + iVar2);
*param_3 = *pbVar3 & 0xf;          // 4-bit cell flags
*param_4 = *pbVar3 >> 6 & 1;       // flag bit 6
*param_5 = (uint)(*pbVar3 >> 7);   // flag bit 7
```

### Definitive answer
- Each MH2O cell mask byte encodes:
  - **bits 0..3**: 4-bit value (0x0..0xf). `0xf` is treated as **no-liquid**.
  - **bit 6**: additional per-cell flag (used by callers via output param_4).
  - **bit 7**: additional per-cell flag (used by callers via output param_5).
- Mask addressing is **8x8** per liquid sub-rect row: `index = x + y * 8`.

### Practical decoding rule
- If the low nibble is `0xf`, the cell is **masked out** and geometry/query paths skip it.

### Return logic in `FUN_007ce1f0`

```c
// FUN_007ce1f0 (after FUN_007ce180)
if (lowNibble != 0xf) {
  return 1 - ((((liquidType - 1) ^ lowNibble) & 3) != 0);
}
return 0;
```

### Interpretation
- The **return value** is not a simple mask test; it also depends on `liquidType` (`param_1 + 4`).
- The low nibble appears to encode a **liquid kind variant** gated by `(liquidType - 1) & 3`.

### Variant decoding (most likely behavior)
- Variant id is **`lowNibble & 3`** (2-bit variant).
- A cell is **accepted** when `(lowNibble & 3) == ((liquidType - 1) & 3)`.
- `lowNibble == 0xf` means **no-liquid**, regardless of variant.

### Practical guidance
- Treat the low nibble as **four variants** per liquid type.
- If you ignore variants, you will mistakenly enable cells belonging to a different variant bucket.

### Bit 6 / Bit 7 usage (3.3.5)
- The mask decoder exposes bit 6 and bit 7, but in the traced 3.3.5 paths, those outputs are **not consumed** by rendering or basic query functions.
- If you need their semantics, search other versions for direct calls to `FUN_007ce180` or for usages of the mask byte with `>> 6` or `>> 7`.

---

## Liquid query / player interaction path

### Function addresses
- FUN_007a0820: liquid check at a world position
- FUN_007a3570: ray/segment intersection with liquids

### Decompiled evidence

```c
// FUN_007a0820
iVar2 = tile->chunk[(uVar1 >> 3 & 0xf, local_8 >> 3 & 0xf)];
...
uVar4 = *(uint *)(iVar2 + 0x108); // liquid list
...
cVar3 = FUN_007ce1f0(local_1c,local_18); // mask
if (cVar3 != '\0') {
  cVar3 = FUN_007ce0b0(&local_24,&local_1c,param_3); // liquid height
  if (cVar3 != '\0' && param_1[2] < *param_3 + _DAT_009f1968) { ... }
}
```

```c
// FUN_007a3570
uVar4 = *(uint *)(local_c + 0x108); // liquid list
...
cVar2 = FUN_007ce1f0(uVar7,uVar6); // mask
if (cVar2 != '\0') {
  // test against liquid triangles for hit
  cVar2 = FUN_009836b0(&local_90,uVar4 + 0x78,&local_50,&local_14,0,_DAT_009f1968);
}
```

### Definitive answer
- Player interaction and physics queries use **the same MH2O mask** as rendering.
- If your mask decode is wrong, both **rendering** and **liquid collision** will fail in the same places.
- The query path relies on `FUN_007ce0b0` (liquid height solve) and triangle tests using the liquid vertex list.

---

## Liquid height solver (grid-based)

### Function addresses
- FUN_007ce0b0: bilinear height interpolation

### Decompiled evidence

```c
// FUN_007ce0b0
iVar2 = (maxX - minX) + 1;
iVar3 = (y - minY) * iVar2 + (x - minX);
iVar2 = iVar3 + iVar2; // next row
f00 = v(iVar3);   f10 = v(iVar3 + 1);
f01 = v(iVar2);   f11 = v(iVar2 + 1);
f0 = (f10 - f00) * fracX + f00;
f1 = (f11 - f01) * fracX + f01;
height = (f1 - f0) * fracY + f0;
```

### Definitive answer
- Liquid height is computed by **bilinear interpolation** over a grid of height samples accessed via a virtual table (`param_1 + 0x44`).
- The cell-local fractional coordinates are `fracX = *param_2`, `fracY = param_2[1]`.
- The grid is indexed using MH2O bounds at `+0x34..+0x40`.

### Implementation notes
- If your heights are wrong, confirm:
  - Bounds are correct.
  - `fracX/fracY` are in the `[0,1]` range for the target cell.
  - You are reading the **same per-liquid height source** that the client uses (vtable call at `+0x44`).

---

## Per-liquid vtable interface (height/UV/extra)

### Function addresses
- FUN_007ce0b0: uses vtable +4 for height samples
- FUN_007ce390: uses vtable +8 and +0xc for per-vertex data and UVs

### Decompiled evidence

```c
// FUN_007ce0b0
f00 = (**(code **)(**(int **)(param_1 + 0x44) + 4))(iVar3);
f10 = (**(code **)(**(int **)(param_1 + 0x44) + 4))(iVar3 + 1);
f01 = (**(code **)(**(int **)(param_1 + 0x44) + 4))(iVar2);
f11 = (**(code **)(**(int **)(param_1 + 0x44) + 4))(iVar2 + 1);
```

```c
// FUN_007ce390
uVar5 = (**(code **)(**(int **)(param_1 + 0x44) + 8))(iVar8); // per-vertex extra
puVar6 = (ushort *)(**(code **)(**(int **)(param_1 + 0x44) + 0xc))(iVar8); // UV
```

### Definitive answer
- The MH2O per-liquid struct exposes a **vtable at +0x44** used to fetch:
  - Height samples (method at vtable+4).
  - Per-vertex extra data (method at vtable+8).
  - UV coordinates or packed UV data (method at vtable+0xc).
- If your decode ignores this interface and reads raw arrays directly, you will miss format-dependent behavior.

---

## Additional liquid intersection utilities

### Function addresses
- FUN_007c8dd0: liquid intersection helper (uses mask + triangle tests)
- FUN_007d8730: triangle test for terrain/liquid grid

### Evidence summary
- `FUN_007c8dd0` iterates liquid cells, applies mask checks, and uses `FUN_009836b0` for triangle intersection.
- This path is used for **advanced queries** (likely spell/LOS or specialized liquid tests).

### Definitive answer
- There are **multiple liquid query paths** beyond the basic height test; they all rely on the same MH2O mask and vertex grid.
- If one path fails, expect **secondary systems** (spells, footstep, waterline effects) to fail too.

---

## Liquid triangle test (ray/segment)

### Function addresses
- FUN_009836b0: triangle intersection test

### Decompiled evidence

```c
// FUN_009836b0
// triangle = v0,v1,v2 (float3 each)
// ray = origin+dir (param_1), with dir in param_1[3..5]
// returns barycentric + distance when hit
```

### Definitive answer
- The engine uses a **triangle test** for liquid collision/intersections.
- Inputs:
  - `param_1`: ray origin (0..2) and direction (3..5).
  - `param_2 + param_3[i] * 0xc`: vertex array (float3).
  - `param_6`: epsilon for near-plane checks.
- Outputs:
  - `param_4`: hit distance along ray.
  - `param_5`: barycentric coords (u,v) on hit.

---

## Why MH2O decoding often fails

1. Not using **bilinear height interpolation** for grid-based liquids.
2. Misinterpreting the per-liquid **vtable-based height source** at `+0x44`.
3. Mixing up bounds (min/max) or **swapping x/y** indices.

---

## Error modes caused by incorrect MH2O decode

1. Treating the mask byte as a boolean instead of a **4-bit value**.
2. Using **wrong local cell indexing** (must be `x + y * 8`).
3. Ignoring the **bounds** at `+0x34..+0x40` and iterating 8x8 globally.
4. Skipping the **aux table at +0x44**, which provides UVs and per-vertex extra data.

---

## Liquid instance creation (material + settings)

### Function addresses
- FUN_007cf200: builds liquid instance for a set of liquid chunks
- FUN_008a1fa0: material by LiquidType.dbc
- FUN_008a28f0: settings by LiquidMaterial.dbc

### Decompiled evidence

```c
// FUN_007cf200
puVar11 = (undefined4 *)FUN_008a1b00();
*puVar11 = FUN_008a1fa0(*(undefined4 *)(param_1 + 4));   // material
puVar11[1] = FUN_008a28f0(*(undefined4 *)(param_1 + 4)); // settings
puVar11[4] = FUN_007d5120(0);                            // client environment
...
*(undefined4 **)(DAT_00d2dd3c[uVar6] + 0x58) = puVar11;  // bind to chunks
```

### Definitive answer
- MH2O liquid types are **bound to a shared liquid instance** (material + settings + environment).
- The instance is **shared across chunks** that use the same liquid type id.

---

## Player interaction and liquid queries

### Function addresses
- FUN_0079c360: builds `LiquidQueryResult` from nearby liquid chunks
- FUN_0079ba30: resizes `LiquidQueryResult` buffer
- FUN_0079b370: resizes `LiquidChunkAndDistance` buffer

### Decompiled evidence

```c
// FUN_0079c360
// Convert player/world position into chunk grid indices
iVar11 = (int)ROUND(-(*param_2 - *(float *)(param_1 + 0x3c)) * _DAT_00a3fa38 - _DAT_00adf7fc);
iVar15 = (int)ROUND(-(param_2[1] - *(float *)(param_1 + 0x40)) * _DAT_00a3fa38 - _DAT_00adf7fc);
...
// iterate chunk liquids in radius, push results
puVar12[0] = *(undefined4 *)(iVar11 + 4); // liquid id
puVar12[1] = local_48;                    // delta x
puVar12[2] = fVar4;                       // delta y
puVar12[3] = fVar5;                       // delta z
```

```c
// FUN_0079c360
if ((*(byte *)(*(int *)(iVar11 + 0x54) + uVar9 + local_38 * 8) & 0xf) == 0xf) {
  ... // skip masked cells (holes or no-liquid bits)
}
```

### Definitive answer
- Player interaction uses **liquid query results** built from nearby liquid chunks.
- The engine converts the player position into **liquid grid coordinates** and iterates nearby liquid chunks.
- Results include **liquid type id and relative offsets** to compute immersion, splash, and effects.
- Per-cell masks are applied to skip **holes / no-liquid regions** (bitmask test on the MH2O cell map).

---

## Rendering update loop

### Function addresses
- FUN_007935a0: per-frame update path for liquid chunks
- FUN_007cf9a0: ensure instance exists and trigger render work

### Decompiled evidence

```c
// FUN_007935a0
if ((DAT_00cd774c & 0x1000000) != 0) {
  ...
  if ((FUN_007cf9a0(), DAT_00cd8610 != 0 && (*(int *)(uVar7 + 0x58) != 0))) {
    FUN_008a20c0(*(int *)(uVar7 + 0x58));
  }
}
```

### Definitive answer
- Liquid chunks participate in the **main map update/render loop**.
- Each chunk ensures a liquid instance exists, then triggers rendering through the liquid system.

---

## Practical takeaways for correct MH2O decode

1. **Use MHDR offsets** to locate MH2O (do not rely on linear chunk scan).
2. MH2O defines **per-chunk sub-rects** and **cell masks** that drive geometry and interaction.
3. Geometry buffers are sized from **sub-rect extents** and cached per chunk.
4. Player-liquid interaction depends on **query results** built from nearby chunk liquids and cell masks.
5. The MH2O per-liquid data includes **bounds + mask + vertex list**; decode these before building geometry.

### Confidence
Medium-High. The offsets, buffer build, and query logic are explicit; full MH2O binary struct layout is still inferred from usage, not a named struct in the binary.
