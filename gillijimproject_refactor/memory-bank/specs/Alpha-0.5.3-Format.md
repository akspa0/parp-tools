# Alpha 0.5.3 World Format Specification

**Version**: 1.0 (Verified)  
**Target Client**: WoW Alpha 0.5.3.3368  
**Date**: 2025-12-05  

This document defines the exact specification for the Alpha 0.5.3 Monolithic WDT format, derived from verified working C# implementation code (`AlphaWdtMonolithicWriter.cs`, `AlphaMcnkBuilder.cs`) and in-game verification.

## 1. High-Level Differences vs. LK (3.3.5)

| Feature | Alpha 0.5.3 | Lich King 3.3.5 |
| :--- | :--- | :--- |
| **File Structure** | **Monolithic** (`.wdt` embeds all ADT data) | **Split** (`.wdt` + `.adt` + `_obj` + `_tex`) |
| **FourCC** | **Reversed** on disk (e.g., `KNCM`) | **Reversed** on disk (e.g., `KNCM`) |
| **Chunk Headers** | **Mixed** (Some chunks raw, some with headers) | **All** chunks have headers (FourCC+Size) |
| **Liquids** | **MCLQ** (legacy format) | **MH2O** (modern format) |
| **Holes** | Low-res (per-chunk bitmask) | High-res (per-pixel bitmap) |
| **Vertex Lighting**| Not supported (MCNR only contains normals) | Supported (MCLV) |

---

## 2. Monolithic WDT Structure

The Alpha `map.wdt` file contains the entire map definition, including global metadata and all tile terrain data.

### Global Header Sequence (Start of File)

All offsets are absolute file positions unless otherwise noted.

1.  **MVER** (4 bytes + header)
    *   Version: `18` (uint32)
2.  **MPHD** (128 bytes + header)
    *   `0x00` `nTextures` (uint32): Count of M2 filenames.
    *   `0x04` `offsDoodadNames` (uint32): Absolute offset to `MDNM`.
    *   `0x08` `nMapObjNames` (uint32): Count of WMO filenames. **Must include trailing empty string in count** (Count = real_count + 1).
    *   `0x0C` `offsMapObjNames` (uint32): Absolute offset to `MONM`.
    *   `0x10..0x7F`: Padding (zeros).
3.  **MAIN** (4096 * 16 bytes + header)
    *   64x64 grid of `SMAreaInfo` entries.
    *   **SMAreaInfo** (16 bytes):
        *   `0x00` `offset` (uint32): Absolute offset to tile's `MHDR` **letters** (FourCC). 0 if tile missing.
        *   `0x04` `size` (uint32): Distance from `MHDR` letters to the **first MCNK** letters. **NOT** total tile size.
        *   `0x08` `flags` (uint32): Load flag (0 or 1).
        *   `0x0C` `pad` (uint32): 0.
4.  **MDNM** (Variable + header)
    *   Block of null-terminated M2 filename strings.
5.  **MONM** (Variable + header)
    *   Block of null-terminated WMO filename strings.
    *   **Must** end with an extra null terminator (empty string) if count > 0.
6.  **[Tile Data Regions]**
    *   Concatenated data for each present tile, pointed to by `MAIN`.

---

## 3. Per-Tile Structure (Embedded ADT)

Each tile referenced by `MAIN` has the following layout.

### 3.1. Tile Header & Global Chunks

1.  **MHDR** (64 bytes + header)
    *   **CRITICAL**: All offsets are relative to **MHDR.data** (8 bytes after letters).
    *   `0x00` `offsInfo` (uint32): Offset to `MCIN`. Fixed to `64` (immediately after MHDR).
    *   `0x04` `offsTex` (uint32): Offset to `MTEX`.
    *   `0x08` `sizeTex` (uint32): Size of `MTEX`.
    *   `0x0C` `offsDoodad` (uint32): Offset to `MDDF`.
    *   `0x10` `sizeDoodad` (uint32): Size of `MDDF`.
    *   `0x14` `offsMapObj` (uint32): Offset to `MODF`.
    *   `0x18` `sizeMapObj` (uint32): Size of `MODF`.
    *   `0x1C..0x3F`: Padding (zeros).
2.  **MCIN** (4096 bytes + header)
    *   256 entries of `SMChunkInfo` (16 bytes each).
    *   `offset` (uint32): **Absolute** file offset to MCNK chunk.
    *   `size` (uint32): Total size of MCNK chunk.
    *   `flags` (uint32): 0.
    *   `asyncId` (uint32): 0.
3.  **MTEX** (Variable + header)
    *   List of texture filenames (null-terminated).
4.  **MDDF** (Variable + header)
    *   M2 placements (`SMDoodadDef`).
    *   `uid` field is preserved from LK.
5.  **MODF** (Variable + header)
    *   WMO placements (`SMMapObjDef`).
    *   `uid` field is preserved from LK.

### 3.2. MCNK Chunk Structure (Terrain)

256 MCNK chunks follow the global chunks. Each MCNK is a self-contained terrain patch (33.33 yards).

**MCNK Header** (128 bytes + header `KNCM` + size)
*   Offsets are relative to the **start of MCNK header** (after `KNCM`+Size, i.e., offset 0 is the first byte of flags).
*   `0x00` `flags` (uint32): `0x1`=HasShadow, `0x2`=River, `0x4`=Ocean, `0x8`=Magma, `0x10`=Slime.
*   `0x04` `indexX` (uint32): 0-15.
*   `0x08` `indexY` (uint32): 0-15.
*   `0x0C` `radius` (float): **Must** be calculated from MCVT heights (approx `sqrt(23.57^2 + (h_range/2)^2)`). Cannot be 0.
*   `0x18` `offsHeight` (uint32): Offset to `MCVT`.
*   `0x1C` `offsNormal` (uint32): Offset to `MCNR`.
*   `0x20` `offsLayer` (uint32): Offset to `MCLY`.
*   `0x24` `offsRefs` (uint32): Offset to `MCRF`.
*   `0x28` `offsAlpha` (uint32): Offset to `MCAL`.
*   `0x2C` `sizeAlpha` (uint32).
*   `0x30` `offsShadow` (uint32): Offset to `MCSH`.
*   `0x34` `sizeShadow` (uint32).
*   `0x38` `areaId` (uint32): Zone ID.
*   `0x40` `holes` (uint16): Low-res hole map.
*   `0x5C` `offsSnd` (uint32): Offset to `MCSE`.
*   `0x60` `nSnd` (uint32): Count of sound emitters.
*   `0x64` `offsLiquid` (uint32): Offset to `MCLQ`. **Must** point to end of chunk if no liquid.

### 3.3. Sub-Chunk Definitions (Strict)

Order **MUST** be: `Header` -> `MCVT` -> `MCNR` -> `MCLY` -> `MCRF` -> `MCSH` -> `MCAL` -> `MCSE` -> `MCLQ`.

| Sub-Chunk | Description | Header? | Format |
| :--- | :--- | :--- | :--- |
| **MCVT** | Height Map | **NO** | 145 floats (9x9 + 8x8). Raw data only. |
| **MCNR** | Normals | **NO** | 145 entries * 3 bytes (int8 X,Y,Z) + 13 bytes pad = 448 bytes. Raw. |
| **MCLY** | Texture Layers | **YES** | `YLCM` + Size + Array of 16-byte layer defs. |
| **MCRF** | Doodad/WMO Refs | **YES** | `FRCM` + Size + Array of uint32 indices. |
| **MCSH** | Shadow Map | **NO** | Raw bitmask/shadow map data. **0 bytes if empty**. |
| **MCAL** | Alpha Map | **NO** | Raw alpha data. 4-bit packed (2048 bytes) or uncompressed. **NO Header**. |
| **MCSE** | Sound Emitters | **NO** | Raw `SndEmitter` structs. **0 bytes if empty**. |
| **MCLQ** | Liquid (Legacy) | **YES** | `QLCM` + Size + Liquid Data. |

**NOTE**: The "Header?" column refers to the standard 8-byte Chunk Header (FourCC + Size).
*   **YES**: You must write `FourCC` and `Size` before the data.
*   **NO**: You must write the data **directly**. The size/offset is tracked solely in the MCNK header.

## 4. Liquid Format (MCLQ)

Alpha uses `MCLQ`, distinct from `MH2O`.
*   Contains 9x9 height map for liquid surface.
*   Contains 8x8 flags/type map.
*   Generated from `MH2O` by sampling/synthesizing heights and mapping types (Ocean/River/Magma/Slime).

## 5. Common Pitfalls & "Gotchas"

1.  **MONM Counting**: You **must** add 1 to the WMO count for the trailing empty string. Failure to do so causes client crashes.
2.  **Relative Offsets**: MHDR offsets are relative to `MHDR.data`. MCNK offsets are relative to `MCNK.header`.
3.  **Absolute Offsets**: MAIN and MCIN use absolute file offsets.
4.  **MCAL Header**: Do **NOT** write a header for MCAL. It is a raw blob pointed to by `offsAlpha`.
5.  **MAIN.size**: This is **NOT** the tile size. It is the distance from `MHDR` start to `MCNK` start.
6.  **Radius**: Must be calculated. 0 radius causes culling bugs.
7.  **MCLQ**: Must have a header (`QLCM`).
