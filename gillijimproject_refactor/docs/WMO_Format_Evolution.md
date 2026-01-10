# WMO Format Evolution (Alpha 0.5.3 - 0.6.0)

**Based on Reverse Engineering of WoW Alpha Client Binaries**

This document details the critical format changes in World Map Objects (WMO) across early World of Warcraft Alpha builds. This research is vital for maintaining the `WoWMapConverter` tool and ensuring correct asset conversion.

## Version Timeline

| Client Version | WMO Version | Key Characteristic | Status |
| :--- | :--- | :--- | :--- |
| **Alpha 0.5.3** | **v14** | Embedded Lightmaps, Small Headers | **Supported** |
| **Alpha 0.5.5** | **v14** | (Identical to 0.5.3) | **Supported** |
| **Alpha 0.6.0** | **v16** (Hybrid) | No Embedded LMs, Hybrid Headers | **Documented** |
| **Alpha 0.7.0+** | **v17** | Modern Format (Standard) | Standard |

---

## 1. Alpha 0.5.3 / 0.5.5 (Format v14)

**File Spec**: `v14`
**Status**: Fully analyzed and supported.

### Root File Structure
The Root WMO file uses smaller, older header variants compared to modern (v17) files.

| Chunk | Token | Size of Entry | Notes |
| :--- | :--- | :--- | :--- |
| **MOMT** | `0x4D4F4D54` | **0x2C** (44 bytes) | Materials. Smaller than v17 (0x40). |
| **MOGI** | `0x4D4F4749` | **0x28** (40 bytes) | Group Info. Larger than v17 (0x20). |
| **MOLT** | `0x4D4F4C54` | **0x20** (32 bytes) | Lights. Smaller than v17 (0x30). |

### Group File Structure (MOGP)
The Group file supports embedded lightmaps, a feature removed in later versions.

*   **MOPY (Polys)**: Present. Explicit list of triangle faces.
*   **MOIN (Indices)**: Standard index list (uint16).
*   **Embedded Lightmaps**:
    *   Flag `0x0002` in MOGP header enables this.
    *   Chunks: `MOLM` (Info) and `MOLD` (Data).
    *   **Data Size**: `0x8004` (32KB + 4 bytes header) per lightmap.
    *   **MOLV**: Explicit Lightmap UV coordinates chunk.
*   **Vertex Colors**:
    *   Flag `0x0004` enables `MOCV` chunk.
    *   Standard RGBA format.

---

## 2. Alpha 0.6.0 (Format v16 - "The Hybrid")

**File Spec**: `v16` (Transitional)
**Status**: Analyzed. Represents the transition point between v14 architecture and v17 data structures.

### Key Changes
1.  **Root File Modernized**: The Root file structure jumps immediately to v17 standards.
2.  **Embedded Lightmaps Removed**: The engine no longer supports `MOLM`/`MOLD`. Lightmaps are likely externalized (BLP) or disabled.
3.  **Geometry Hybrid**: The Group file keeps the old `MOPY` (Polys) chunk but adopts the modern `MOVI` (Indices) token.

### Root File Structure (v17-style)

| Chunk | Token | Size of Entry | Change vs v14 |
| :--- | :--- | :--- | :--- |
| **MOMT** | `0x4D4F4D54` | **0x40** (64 bytes) | **Expanded** to modern size. |
| **MOGI** | `0x4D4F4749` | **0x20** (32 bytes) | **Shrunk** to modern size. |
| **MOLT** | `0x4D4F4C54` | **0x30** (48 bytes) | **Expanded** to modern size. |

### Group File Structure (Hybrid)

| Chunk | Token | Entry Size | Notes |
| :--- | :--- | :--- | :--- |
| **MOPY** | `0x4D4F5059` | **4 bytes** | Polygon Materials/Flags. |
| **MOVI** | `0x4D4F5649` | **2 bytes** | **Vertex Indices** (uint16). Replaces `MOIN`. |
| **MOLV** | `MOLV` | - | **Removed**. No lightmap UVs in main sequence. |
| **MOBA** | `0x4D4F4241` | **32 bytes** | **Expanded**. Material ID is at **offset 0x17** (1 byte). |

### Lighting & Flags
*   **Flag 0x2 (Embedded LMs)**: **No longer checked**. The code blocks for `MOLM`/`MOLD` are removed.
*   **Flag 0x4 (Vertex Colors)**: **Still supported**. `MOCV` chunk is read if flag is set.

---

## 3. Summary of Converter Requirements

To support these versions, the converter must handle:

1.  **v14 (0.5.3/0.5.5)**:
    *   Write `MOMT` as 0x2C bytes.
    *   Write `MOGI` as 0x28 bytes.
    *   Support embedding lightmaps via `MOLD` if preserving lighting (or strip them).

2.  **v16 (0.6.0)**:
    *   Write `MOMT` as 0x40 bytes (v17 style).
    *   Write `MOGI` as 0x20 bytes (v17 style).
    *   **Do not** write embedded lightmaps (`MOLD`/`MOLM`).
    *   Use `MOVI` token instead of `MOIN`.
    *   Keep writing `MOPY` chunk.

3.  **WDT Compatibility**:
    *   WDT format (MDDF/MODF) is **identical** across 0.5.3, 0.5.5, and 0.6.0.
