# Alpha 0.5.5 ADT Format Specification

**Version**: Alpha 0.5.5 (Prototype / Internal)
**Status**: Experimental. Gated by `0x008ab3e4`.
**Validation**: Decompiled `FUN_006b6390` (ADT), `FUN_006a7230` (MCNK).

This format is a unique "Headerless + Padded" variant of the Alpha ADT.

## 1. Global File Structure
The file has **NO** `MVER` chunk. It consists of `MHDR` followed by padded chunks.

**The "8-Byte Pad" Rule**: All top-level chunks (except MHDR) must be preceded by 8 bytes of padding (likely zeros), and the MHDR offsets point to this padding.

| Offset | Data | Description | Notes |
| :--- | :--- | :--- | :--- |
| 0x00 | `MHDR` Chunk | Standard MHDR Token + Size + Data | **No Padding** before this. |
| ... | **8 Bytes** | **Padding** | Required. |
| ... | `MCIN` Chunk | Standard MCIN Token + Size + Data | |
| ... | **8 Bytes** | **Padding** | Required. |
| ... | `MTEX` Chunk | Standard MTEX Token + Size + Data | |
| ... | **8 Bytes** | **Padding** | Required. |
| ... | `MDDF` Chunk | M2 Placements | Entry Size: 36 bytes. |
| ... | **8 Bytes** | **Padding** | Required. |
| ... | `MODF` Chunk | WMO Placements | Entry Size: 64 bytes. |
| ... | `MCNK`... | Map Chunks | Variable padding? likely standard. |

### 1.1 MHDR Structure (Modified)
The MHDR chunk offsets point to the **Padding** before the target chunk.

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `uint` | flags | ? |
| 0x04 | `uint` | **mcin_ofs** | Offset to MCIN Padding. |
| 0x08 | `uint` | **mtex_ofs** | Offset to MTEX Padding. |
| 0x0C | `uint` | *unused* | Gap. |
| 0x10 | `uint` | **mddf_ofs** | Offset to MDDF Padding. |
| 0x14 | `uint` | *unused* | Gap. |
| 0x18 | `uint` | **modf_ofs** | Offset to MODF Padding. |
| 0x1C | `uint` | *unused* | Gap. |

## 2. MCNK (Map Chunk) Structure
Fixed layout with an extra padding gap.

**Base Offset**: 0 (Start of MCNK)

| Relative | Size | Description | Token | Notes |
| :--- | :--- | :--- | :--- | :--- |
| +0 | 128 | **Header** | `MCNK` | Standard 128-byte header. |
| +128 | 8 | **Padding** | | Gap before heights. |
| +136 | 580 | **MCVT** | | Implicit Heights (9x9 + 8x8). |
| +716 | 435 | **MCNR** | | Implicit Normals (Packed). |
| +1151 | 13 | **Padding** | | Alignment gap to 1164. |
| +1164 | Var | **MCLY** | `MCLY` | Explicit Layer Chunk. |
| *seq* | Var | **MCRF** | `MCRF` | Explicit Refs Chunk. |
| *seq* | Var | **MCAL** | | Alpha Maps (Calculated pos). |

## 3. Implementation Checklist
1.  **Format Version**: Do **NOT** write `MVER`.
2.  **MHDR**: Use the modified struct (gaps at 0xC, 0x14, 0x1C).
3.  **Chunk Padding**: Write 8 bytes of zeros before `MCIN`, `MTEX`, `MDDF`, `MODF`.
4.  **MHDR Offsets**: Compute offsets pointing to the *start of the padding*.
5.  **MCNK**:
    *   Write 8 bytes padding after Header (before MCVT).
    *   Ensure `MCLY` starts exactly at relative offset **1164**.
    *   Write 13 bytes padding after MCNR.
6.  **Object Data**: Use 36-byte MDDF and 64-byte MODF entries.
