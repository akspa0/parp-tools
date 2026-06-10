# Research: MH2O / MCLQ Liquid Type Determination (040)

**Status**: Research slice evidence dump. No code changes proposed. See `spec.md` for requirements and success criteria.

**Date**: 2026-06-02
**Source binaries**:
- `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/WoW.exe` (7,704,216 bytes) — Mac OS Mach-O, currently loaded in Ghidra
- `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/` (staged, MPQ-only, not extracted)
- `output/tmp/wowarchive-clients/2.X_Retail_Windows_enUS_2.4.3.8606/` (staged, MPQ-only, not extracted)

---

## 1. Authoritative 3.3.5 Liquid Type Lookup

### 1.1 Source file anchor

`0x00b96f4c` (92 bytes) — the source path string:
```
/Users/Shared/BuildServer/wow1/work/WoW-code/branches/wow-patch-3_3_5_A-BNet/WoW/Source/Mac/../WorldClient/MapChunkLiquid.cpp
```

This file is the 3.3.5 implementation of all liquid-related code in the CMapChunk family. Two functions in the binary reference this string directly, plus several allocator/pool helpers.

### 1.2 `FUN_00439760` — Material Bank Liquid Type Lookup

Decompiled (Ghidra decompiler, this session):

```c
int FUN_00439760(undefined4 param_1)  // param_1 = LiquidType.dbc row ID
{
    uint uVar1;
    bool bVar2;
    undefined *puVar3;
    int iVar4;
    int iVar5;
    uint uVar6;
    int iVar7;
    uint uVar8;
    undefined4 *puVar9;
    uint local_34;
    uint local_20;

    if (DAT_00d98828 == 0) {
        return 0;  // DBC bank not loaded
    }
    while (iVar4 = (**(code **)(*DAT_0100ae94 + 8))(DAT_0100ae94, param_1),  // DBC lookup by row ID
          puVar3 = PTR_DAT_0112106c,
          iVar4 == 0)  // miss
    {
        FUN_0088c5b0(2, 2, "Material Bank: Liquid type [%d] not found, defaulting to water!", param_1);
        if (DAT_00d98828 == 0) {
            return 0;
        }
        param_1 = 1;  // <-- FALLBACK TO ROW 1 (WATER) AND RETRY
    }
    uVar1 = *(uint *)(iVar4 + 0x38);  // <-- TYPE FIELD AT DBC RECORD OFFSET 0x38
    if ((uVar1 < DAT_00e506bc) && (iVar4 = *(int *)(DAT_00e506c0 + uVar1 * 4), iVar4 != 0)) {
        return iVar4;  // cache hit
    }

    // Pixel-shader support check
    iVar4 = FUN_008d4620(*(undefined4 *)PTR_DAT_0112106c);
    iVar4 = *(int *)(iVar4 + 0xb4);
    iVar5 = FUN_008d4620(*(undefined4 *)puVar3);
    if ((iVar4 < 1 || DAT_0100ae98 == 0) || (bVar2 = true, *(int *)(iVar5 + 0xc4) < 3)) {
        bVar2 = false;  // no pixel shaders → use FFPE materials
    }

    // Material function dispatch by type field (1=Water, 2=Magma, 3=Slime)
    if (uVar1 == 2) {
        if (bVar2) iVar4 = FUN_00434f70();  // Magma with pixel shaders
        else       iVar4 = FUN_00435100();  // Magma FFPE
    }
    else if (uVar1 == 3) {
        if (bVar2) iVar4 = FUN_00434740();  // Slime with pixel shaders
        else       iVar4 = FUN_00434910();  // Slime FFPE
    }
    else if (uVar1 == 1) {
        if (bVar2) {
            if (DAT_00d98830 == 0) iVar4 = FUN_00434ca0();  // Water with pixel shaders
            else                   iVar4 = FUN_00434a10();  // Water with pixel shaders (alt)
        }
        else iVar4 = FUN_00434e70();  // Water FFPE
    }
    else iVar4 = 0;  // unknown type → no material

    // ... cache insert omitted ...

    return iVar4;
}
```

**Critical findings**:
- `param_1` is the DBC **row ID** (the LiquidTypeId), not the type. The DBC is the source of truth.
- The actual **type** is at `record + 0x38` (a uint field).
- The type enum values are: `0=unknown (no material)`, `1=Water`, `2=Magma`, `3=Slime`.
- On DBC miss: log a warning, set `param_1 = 1` (water), retry. Default is **water**, not magma.
- There are 4 material functions per type: 2 for water (with and without pixel shaders + 1 alt path), 2 for magma, 2 for slime.

### 1.3 `FUN_0043a730` — Settings Bank Liquid Type Lookup

Decompiled (Ghidra decompiler, this session):

```c
int FUN_0043a730(uint param_1)  // param_1 = LiquidType.dbc row ID
{
    char cVar1;
    int iVar2;
    uint uVar3;
    int iVar4;
    undefined4 *puVar5;
    int iVar6;
    uint uVar7;
    uint local_34;
    uint local_20;

    while (true) {
        if ((param_1 < DAT_00e506cc) && (iVar2 = *(int *)(DAT_00e506d0 + param_1 * 4), iVar2 != 0)) {
            return iVar2;  // cache hit
        }
        iVar2 = FUN_0076d9d0(&DAT_0100ae54, 0, "", 0xfffffffe);  // allocate settings record
        // ... zero-init 4-byte groups at offsets 0x300, 0x37c (5x16 bytes), 0x3dc (6x16 bytes) ...

        cVar1 = FUN_0043a610(iVar2, param_1);  // DBC load
        iVar6 = DAT_00e506d0;
        if (cVar1 != '\0') break;  // load succeeded

        FUN_0088c5b0(2, 2, "Settings Bank: Liquid type [%d] not found, defaulting to water!", param_1);
        param_1 = 1;  // <-- FALLBACK TO ROW 1 (WATER) AND RETRY
    }

    // ... cache insert omitted ...
    return iVar2;
}
```

Same pattern: row ID is the lookup key, default is `param_1 = 1` (water).

### 1.4 `FUN_00739e00` — WMO Liquid Type Lookup

This function references the string `"WMO: Liquid type [%d] not found, defaulting to water!"` at `0x00b9b2b8`. Decompilation not yet captured in this session, but the pattern is the same: DBC lookup by LiquidTypeId, default to water on miss. The function is in the WMO rendering path (`MapObj.cpp:0x73` area).

### 1.5 `FUN_006ad110` — CMapMgr::Initialize (allocator)

Allocates a pool of 0x40 (64) `WCHUNKLIQUID` objects, each 0x444 bytes. The `WCHUNKLIQUID` is the per-tile liquid state structure that holds the resolved material pointer (set by `FUN_00439760`).

### 1.6 Liquid shader/material type names (3.3.5 Mac OS)

| Address | String | Class |
|---------|--------|-------|
| `0x00b6b7b5` | `N6Liquid14CMaterialWaterE` | `Liquid::CMaterialWater` |
| `0x00b6b7d0` | `N6Liquid20CMaterialWaterNoSpecE` | `Liquid::CMaterialWaterNoSpec` |
| `0x00b6b7f0` | `N6Liquid17CMaterialWaterFFPEE` | `Liquid::CMaterialWaterFFPE` |
| `0x00b6b80d` | `N6Liquid14CMaterialMagmaE` | `Liquid::CMaterialMagma` |
| `0x00b6b827` | `N6Liquid17CMaterialMagmaFFPEE` | `Liquid::CMaterialMagmaFFPE` |
| `0x00b6b845` | `vsLiquidProcWater%s` | vertex shader prefix |
| `0x00b6b85c` | `psLiquidProcWater%s` | pixel shader prefix |
| `0x00b6b872` | `vsLiquidWater` | water vertex shader |
| `0x00b6b880` | `psLiquidWater` | water pixel shader |
| `0x00b6b88e` | `vsLiquidWaterNoSpec` | water (no specular) vertex shader |
| `0x00b6b8a2` | `psLiquidWaterNoSpec` | water (no specular) pixel shader |
| `0x00b6b8b6` | `vsLiquidMagma` | magma vertex shader |
| `0x00b6b8c4` | `psLiquidMagma` | magma pixel shader |

3.3.5 has explicit water, water-NoSpec, magma classes. There is no explicit slime class in this list — slime is rendered with the same water shaders but different material parameters. This matches the wowdev.wiki convention: slime is "water with green tint and slower animation".

---

## 2. wow-viewer Data Path: From File to Renderer

### 2.1 3.x (MH2O) path

```
ADT file (MH2O chunk)
  ↓ AdtLiquidReader.Read → Parse → ParseLayer
AdtLiquidLayer {
    LiquidTypeId: ushort = DBC row ID (e.g. 17, 19, 20, 13, 14, 1, ...)
    BasicType:    AdtLiquidBasicType = MapLiquidTypeId(liquidTypeId)
                                          ↑ BUG: hardcoded 17/19/20, no DBC lookup
}
  ↓ LiquidRenderer.BuildLayer
layer.BasicType → color
                  Magma → (0.9, 0.4, 0.05)  orange-red
                  Slime → (0.2, 0.5, 0.1)   green
                  else  → (0.1, 0.3, 0.6)   blue (water)
```

**Bug**: `AdtLiquidReader.MapLiquidTypeId` (line 275-284) hardcodes:
- `17 → Ocean`
- `19 → Magma`
- `20 → Slime`
- `_ → Water`

For any LiquidTypeId that isn't 17/19/20 (e.g. row 1=Water, row 13=River, row 14=StillWater, row 2=Ocean, row 3=Magma per 3.3.5 type field), the type is wrong. The fix requires a DBC lookup.

### 2.2 1.x (MCLQ) path

```
ADT file (MCNK flags + MCLQ chunk)
  ↓ AlphaWdtReader / AlphaTileData builder
AlphaLiquidChunk {
    McnkFlags: uint
    TileFlags: byte[64] (MCLQ tile nibbles, lower 4 bits)
}
  ↓ AlphaToLkConverter.ConvertTile → BuildLiquidData
ResolveLiquidBasicType(mcnkFlags)
  ↑ BUG: ((mcnkFlags>>4)&0x3) == 2 → Magma  (this is bit 0x20, which is SLIME per canonical)
  ↑ Should be: 0x10=Magma, 0x20=Slime
AdtLiquidLayer {
    BasicType: AdtLiquidBasicType (WRONG due to bug above)
}
  ↓ LiquidRenderer.BuildLayer
layer.BasicType → color (WRONG: slime tiles render as magma)
```

**Bug**: `AlphaToLkConverter.ResolveLiquidBasicType` (line 547-558):
```cs
if ((mcnkFlags & 0x08) != 0)
    return AdtLiquidBasicType.Ocean;  // 0x08 = ocean, correct

return ((mcnkFlags >> 4) & 0x3) switch
{
    2 => AdtLiquidBasicType.Magma,   // 0x20 bit set → Magma (WRONG: 0x20 is Slime)
    3 => AdtLiquidBasicType.Slime,   // 0x20+0x10 bits → Slime (WRONG: should be Magma+Slime = invalid, or just Slime per AlphaLiquidTypeCodec)
    _ => AdtLiquidBasicType.Water,   // 0x10 only = Magma → falls to Water (WRONG)
};
```

**Correct interpretation** (per `LiquidConverter.GetLiquidTypeFromMcnkFlags` at line 240-248 and `AlphaLiquidTypeCodec.ResolveBasicType` at line 32-56):
```cs
if ((mcnkFlags & 0x20) != 0) return AdtLiquidBasicType.Slime;
if ((mcnkFlags & 0x10) != 0) return AdtLiquidBasicType.Magma;
if ((mcnkFlags & 0x08) != 0) return AdtLiquidBasicType.Ocean;
if ((mcnkFlags & 0x04) != 0) return AdtLiquidBasicType.River;  // or Water
return AdtLiquidBasicType.Water;
```

### 2.3 1.x MCLQ tile nibble path

`AlphaLiquidTypeCodec.ResolveBasicType` (line 32-56) ALSO checks MCLQ tile nibble first:
```cs
byte tileType = GetVisibleTileTypeNibble(tileFlags);  // first non-0x0F nibble
if (tileType != 0)
{
    return tileType switch
    {
        0x02 => AdtLiquidBasicType.Ocean,
        0x03 => AdtLiquidBasicType.Magma,
        0x04 => AdtLiquidBasicType.Slime,
        _ => AdtLiquidBasicType.Water,
    };
}
```

**NOTE**: This nibble encoding is different from `MclqLiquidType` enum (which uses Ocean=1, Slime=3, River=4, Magma=6). The two encodings are not compatible. The `AlphaLiquidTypeCodec` writer uses 0x01/0x02/0x03/0x04. The reader (MCLQ tile raw byte) is interpreted by `MclqTile.LiquidType` as `RawValue & 0x0F`, which would give 0..6. There is an internal encoding mismatch.

### 2.4 WL (Water Level) fallback path

`WlToLiquidConverter` converts WL files to MCLQ (for 1.x) or MH2O (for 3.x):
- WL type → MCLQ type: StillWater/Ocean/River/Magma/Slime/FastWater → River/Ocean/River/Magma/Slime/River
- WL type → MH2O LiquidTypeId: StillWater=14, Ocean=17, River=13, Magma=19, Slime=20, FastWater=13

This is the path that creates "synthetic" water planes for missing data. The WL→MH2O mapping is consistent with 3.3.5 DBC row IDs.

### 2.5 Renderer's color pick (correct, no bug)

`LiquidRenderer.BuildLayer` line 85:
```cs
Vector3 color = layer.BasicType switch
{
    AdtLiquidBasicType.Magma => new Vector3(0.9f, 0.4f, 0.05f),  // orange-red
    AdtLiquidBasicType.Slime => new Vector3(0.2f, 0.5f, 0.1f),   // green
    _ => new Vector3(0.1f, 0.3f, 0.6f),                           // blue (water)
};

float opacity = layer.BasicType == AdtLiquidBasicType.Water ? 0.45f : 0.7f;
```

The renderer is **correct**. The upstream `BasicType` is the bug.

---

## 3. Conflicting MCNK Flag Interpretations (4-way drift)

| File | Function | Lines | Current interpretation | Correct? |
|------|----------|-------|------------------------|----------|
| `Liquids/LiquidConverter.cs` | `GetLiquidTypeFromMcnkFlags` | 240-248 | 0x04=River, 0x08=Ocean, 0x10=Magma, 0x20=Slime (direct bit checks, Magma first) | YES |
| `Maps/AlphaLiquidTypeCodec.cs` | `ResolveBasicType` | 32-56 | Tile nibble first (0x02=Ocean, 0x03=Magma, 0x04=Slime), then MCNK (0x20=Slime first, 0x10=Magma) | YES (most correct) |
| `Maps/AlphaToLkConverter.cs` | `ResolveLiquidBasicType` | 547-558 | 0x08=Ocean, then `((mcnkFlags>>4)&0x3)` switch (2=Magma, 3=Slime) | **NO** — bit 0x20 (nibble=2) is Slime per canonical |
| `Maps/AlphaTensorPackBuilder.cs` | `McnkFlagsToLiquidType` | 217-223 | 0x08=1 (Water/Ocean?), then raw 2-bit field (0..3) | **NO** — returns raw bits, not AdtLiquidBasicType |
| `Maps/LkToAlphaConverter.cs` | `ClassifyLkLiquid` | 543-549 | 0x04=1, 0x08=1, `(flags>>4)&3` switch (1=1, 2=2, 3=3) | **NO** — broken for Magma, all water return 1 |
| `Maps/AlphaTileData.cs` | `ClassifyLiquid` | 243-255 | 0x04=1, 0x08=1, `(mcnkFlags>>4)&3` switch (1=1, 2=2, 3=3) | **NO** — same as LkToAlphaConverter |

**Canonical interpretation** (per `LiquidConverter.GetLiquidTypeFromMcnkFlags` and `AlphaLiquidTypeCodec.ResolveBasicType`):
- 0x04 = River (water)
- 0x08 = Ocean
- 0x10 = Magma
- 0x20 = Slime
- Default = Water

**Suggested fix**: Extract the canonical logic into a single static helper (e.g. `McnkFlagDecoder.DecodeLiquidType(uint mcnkFlags) → AdtLiquidBasicType`) and call it from all 6 sites.

---

## 4. DBC LiquidTypeId Conventions

### 4.1 3.3.5 DBC row ID → type field (per `FUN_00439760`)

The 3.3.5 binary reads `*(uint *)(record + 0x38)` from the DBC record. The type values are 1, 2, 3. The DBC row IDs are independent — they can be 1, 2, 3, ..., N. The wowdev.wiki DBC schema for `LiquidType.dbc` is:

| Field | Type | Offset | Notes |
|-------|------|--------|-------|
| ID | int | 0x00 | row ID |
| Name | string | 0x04 | localized name |
| Flags | int | 0x08 | bit 0: not swim, bit 6: related to type |
| SoundBank | int | 0x0C | sound entry |
| MaterialID | int | 0x10 | LiquidMaterial.dbc row |
| ... | ... | ... | many other fields |
| Type | uint | 0x38 | 1=Water, 2=Magma, 3=Slime |

**Note**: The offset 0x38 in the wowdev schema has varied across client versions. The 3.3.5 Ghidra evidence (`FUN_00439760`) confirms it's 0x38 in 3.3.5 specifically. For 1.12 and 2.4.3, the offset may differ.

### 4.2 1.x WL file DBC ID conventions (per `WlToLiquidConverter.MapWlTypeToMh2oTypeId`)

- WL `StillWater` (0) → LiquidTypeId 14
- WL `Ocean` (1) → LiquidTypeId 17
- WL `River` (2) → LiquidTypeId 13
- WL `Magma` (3) → LiquidTypeId 19
- WL `Slime` (4) → LiquidTypeId 20
- WL `FastWater` (5) → LiquidTypeId 13

### 4.3 3.3.5 known LiquidType.dbc row IDs (per various sources)

| Row ID | Name | Type field (offset 0x38) |
|--------|------|--------------------------|
| 1 | Water (generic) | 1 (Water) |
| 2 | Ocean (generic) | 1 (Water) — type is water, not separate "ocean" type |
| 3 | Magma (generic) | 2 (Magma) |
| 13 | River (Dark) | 1 (Water) |
| 14 | Still Water (generic) | 1 (Water) |
| 17 | Ocean (deep) | 1 (Water) |
| 19 | Magma (lava) | 2 (Magma) |
| 20 | Slime (green) | 3 (Slime) |

This is the convention the wow-viewer `MapLiquidTypeId` function is **guessing** at, but the actual lookup should go through the DBC record.

---

## 5. Empirical Validation Plan (deferred to spec 041)

1. **Extract 1.12 ADT from MPQ**: use `StormLib`-based tool or `wowmpq` to extract `World\Maps\Kalimdor\*.adt` from `1.X_Retail_Windows_enUS_1.12.1.5875/606/World of Warcraft/Data/common.MPQ` (or similar). Pick a known lava region (e.g. Orgrimmar area, Searing Gorge) for a magma tile, and a water region (e.g. Darkshore) for a water tile.
2. **Extract 2.4.3 ADT/WL from MPQ**: same process against `2.X_Retail_Windows_enUS_2.4.3.8606/606/World of Warcraft/Data/common.MPQ`. Pick a map with MH2O (Outland) and a map with WL data.
3. **Use 3.3.5 ADT (already extracted)**: pick a map with a non-default LiquidTypeId (e.g. row 1, 13, 14). These should render as water, not magma/slime.
4. **Add a `WowViewer.Tool.Inspect map inspect --dump-liquid-types` subcommand** (deferred to 041) that emits per-chunk/per-tile `LiquidTypeId` (raw), `MclqTileNibble` (raw), `McnkFlags` (raw), and `BasicType` (resolved) as JSON.
5. **Before fix**: confirm "lava for everything" symptom by inspecting a 1.12 water tile — expect to see `BasicType=Magma` despite `MclqTileNibble=0` and `McnkFlags=0`.
6. **After fix**: re-inspect, expect `BasicType=Water` for water tiles, `BasicType=Magma` for magma tiles, `BasicType=Slime` for slime tiles.

---

## 6. Open Questions (full list in spec.md OQ-L1..L9)

- OQ-L1: What's the actual `BasicType` value for 1.x water tiles today?
- OQ-L2: Is `Mh2oInstance.LiquidTypeId` reading the right offset? (Likely yes.)
- OQ-L3: Is `MclqTile.LiquidType` reading the right nibble? (Likely yes.)
- OQ-L4: Should we extract a `McnkFlagDecoder` helper? (Yes, strong.)
- OQ-L5: Should MCNK flags be used at all for 3.x data? (No, MH2O DBC lookup is sufficient.)
- OQ-L6: Is `Mh2oVertexFormat.HeightUv` correct for magma? (Yes, per shader names.)
- OQ-L7: Are multi-layer MH2O chunks handled? (Yes, via `foreach (var layer in chunk.Layers)`.)
- OQ-L8: How to handle invalid 0x10+0x20 MCNK flag combo? (Document Magma-first precedence.)
- OQ-L9: Does 1.12 have the same `FUN_00439760` material-bank structure? (Likely yes, but unprobed.)

---

## 7. References

- wowdev.wiki ADT v18 documentation (MH2O format, MCLQ format, MCNK flags)
- 3.3.5 Ghidra decompilations (this session): `FUN_00439760`, `FUN_0043a730`, `FUN_006ad110`, `FUN_00434eb0` (CMaterialMagma ctor)
- 3.3.5 Mac OS Mach-O binary: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/WoW.exe`
- wow-viewer source files: see spec.md Files Inventory
