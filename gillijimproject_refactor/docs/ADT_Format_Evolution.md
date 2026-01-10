# ADT Format Evolution (Alpha 0.6.0)

**Based on Reverse Engineering of WoW Alpha 0.6.0 Client**

This document details the first appearance of the `.adt` file format in World of Warcraft Alpha 0.6.0. Prior versions (0.5.3/0.5.5) likely relied on WDT-based terrain or a different structure.

## Overview

Alpha 0.6.0 ADT files are surprisingly modern, establishing the standard structure used through Vanilla, TBC, and WotLK.

*   **Header**: `MVER` + `MHDR` (with valid offsets).
*   **Structure**: 16x16 Grid (`MCNK` chunks) defined by `MCIN`.
*   **Assets**: Standard lists (`MMDX`, `MWMO`) and placements (`MDDF`, `MODF`).

---

## File Structure

**Function:** `FUN_006b4180` (Root Parser)

| Chunk | Token | Notes |
| :--- | :--- | :--- |
| **MVER** | `0x4D564552` | Version chunk. |
| **MHDR** | `0x4D484452` | Header. Contains offsets to other chunks (Data+0x04 to Data+0x20). |
| **MCIN** | `0x4D43494E` | Chunk Index (16x16 lookup table). |
| **MTEX** | `0x4D544558` | Texture Names. |
| **MMDX** | `0x4D4D4458` | M2 Model Names. |
| **MMID** | `0x4D4D4944` | M2 Model Offsets/Indices. |
| **MWMO** | `0x4D574D4F` | WMO Model Names. |
| **MWID** | `0x4D574944` | WMO Model Offsets/Indices. |
| **MDDF** | `0x4D444446` | **M2 Placements**. Entry Size: **0x24** (36 bytes). |
| **MODF** | `0x4D4F4446` | **WMO Placements**. Entry Size: **0x40** (64 bytes). |

### MHDR Offsets Data
The `MHDR` chunk data (after 8-byte header) contains 32-bit offsets to the chunks listed above.
- 0x00: Flags?
- 0x04: MCIN
- 0x08: MTEX
- 0x0C: MMDX
- 0x10: MMID
- 0x14: MWMO
- 0x18: MWID
- 0x1C: MDDF
- 0x20: MODF

---

### MCNK (Map Chunk) Structure

**Function:** `FUN_006a6d00` (Chunk Parser)

The 16x16 grid chunks (`MCNK`) contain the actual terrain data. The header contains offsets to sub-chunks.

| Sub-Chunk | Token | Description |
| :--- | :--- | :--- |
| **MCVT** | `0x4D435654` | Height Map (Vertices). |
| **MCNR** | `0x4D434E52` | Normals. |
| **MCLY** | `0x4D434C59` | Texture Layers. |
| **MCRF** | `0x4D435246` | Object References (Doodads/WMOs). |
| **MCSH** | `0x4D435348` | **Shadow Map** (Static shadows). |
| **MCAL** | `0x4D43414C` | Alpha Maps. |
| **MCLQ** | `0x4D434C51` | Liquid Data (Old format). |
| **MCSE** | `0x4D435345` | Sound Emitters. |

**Notable**: `MCCV` (Vertex Colors) is **missing** from the verified check list in 0.6.0. It might be absent or implicit? (Verification: `FUN_006a6d00` did not check for `MCCV`). `MCSH` (Shadows) is present.

---

## Placement Data (MDDF / MODF)

**Function:** `FUN_00691640` (Object Initialization)

Analysis confirms that Alpha 0.6.0 uses the **Modern 64-byte MODF** structure (identical to 3.3.5) and the **Standard 36-byte MDDF**.

### Coordinate System (CRITICAL)
Like Alpha 0.5.3, the 0.6.0 client expects **XZY** coordinate order (where Z is Height).
- **Position**: X, Height(Z), Y.
- **Extents**: MinX, MinHeight(Z), MinY, MaxX, MaxHeight(Z), MaxY.
- **Rotation**: Euler angles in degrees (X, Y, Z).

### MDDF (M2 Placement) - 36 Bytes
| Offset | Type | Description |
| :--- | :--- | :--- |
| 0x00 | uint32 | Name ID (in MMDX) |
| 0x04 | uint32 | Unique ID |
| 0x08 | float | Position X |
| 0x0C | float | **Position Z (Height)** |
| 0x10 | float | Position Y |
| 0x14 | float | Rotation X |
| 0x18 | float | Rotation Y |
| 0x1C | float | Rotation Z |
| 0x20 | uint16 | Scale (1024 = 1.0) |
| 0x22 | uint16 | Flags |

### MODF (WMO Placement) - 64 Bytes
| Offset | Type | Description |
| :--- | :--- | :--- |
| 0x00 | uint32 | Name ID (in MWMO) |
| 0x04 | uint32 | Unique ID |
| 0x08 | float | Position X |
| 0x0C | float | **Position Z (Height)** |
| 0x10 | float | Position Y |
| 0x14 | float | Rotation X |
| 0x18 | float | Rotation Y |
| 0x1C | float | Rotation Z |
| 0x20 | float[6]| **Extents** (MinX, **MinZ**, MinY, MaxX, **MaxZ**, MaxY) |
| 0x38 | uint16 | Flags |
| 0x3A | uint16 | Doodad Set |
| 0x3C | uint16 | Name Set |
| 0x3E | uint16 | Scale? / Padding |

## Converter Implications

To generate valid 0.6.0 ADTs:
1.  Use the **Standard ADT structure** (like 3.3.5).
2.  **MDDF**: Write 36-byte entries.
3.  **MODF**: Write 64-byte entries.
4.  **Coordinates**: Ensure Z (Height) is in the 2nd position (0xC) for vectors.
5.  **MCNK**: Ensure `MCSH` is generated (or zeroed) if required.
