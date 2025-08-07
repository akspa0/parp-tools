# MSUR Surface Chunk Reference

_Last updated: 2025-07-12_

This document captures all confirmed and hypothesised knowledge about the **MSUR** ("Surface") chunk gained from recent CSV dumps and OBJ-export analysis.  Keep this file in sync with further findings – it is our single source of truth for MSUR fields and relationships.

---

## 1. MSUR Entry Structure (64 bytes)
| Offset | Field                           | Size | Meaning (current understanding) |
|-------:|---------------------------------|-----:|---------------------------------|
| 0x00   | `FlagsOrUnknown_0x00`           | 1    | Bit-flags controlling render state; **bit 0 == 1** when surface participates in M2 bucket export. |
| 0x01   | `SurfaceGroupKey`               | 1    | Surface grouping discriminator; equals low-byte of `Unknown_0x1C` (see below). |
| 0x02   | `IsM2Bucket`                    | 1    | Boolean derived from `FlagsOrUnknown_0x00` (see exporter logic). |
| 0x03   | `IndexCount`                    | 1    | Number of indices contributed to **MSVI/MSPI** (multiple of 3). |
| 0x04   | `Unknown_0x02`                  | 2    | Always ‑ currently seen **0**. Suspected padding. |
| 0x06   | `SurfaceAttributeMask`          | 2    | Bit-mask of surface attributes (material, collision flags).  Values `0x0000`, `0x0004`, `0x000C` observed. |
| 0x08   | `IsLiquidCandidate`             | 1    | `true` when `SurfaceAttributeMask & 0x000C != 0`. Appears to mark water surfaces. |
| 0x09   | `Padding_0x03`                  | 1    | Alignment padding. Always **0**. |
| 0x0A   | `UnknownFloat_0x04`             | 4    | Surface normal X component (pre-normalisation). |
| 0x0E   | `UnknownFloat_0x08`             | 4    | Surface normal Y component. |
| 0x12   | `UnknownFloat_0x0C`             | 4    | Surface normal Z component. Together with the two values above these form the **raw surface normal** prior to normalisation. |
| 0x16   | `UnknownFloat_0x10`             | 4    | Plane distance (height) from origin; corresponds to **SurfaceHeight** in exporter. |
| 0x1A   | `SurfaceNormalX/Y/Z`            | 12   | **Normalised** surface normal vector (XYZ order). Computed at load-time from raw values above. |
| 0x26   | `SurfaceHeight`                 | 4    | Signed float plane D value. |
| 0x2A   | —                               | —    | (Struct packing gap) |
| 0x2C   | `MsviFirstIndex`                | 4    | First index into **MSVI** for this surface. |
| 0x30   | `FirstIndex`                    | 4    | Alias of `MsviFirstIndex` (legacy). |
| 0x34   | `MdosIndex`                     | 4    | Index into **MDOS** (surface materials); `-1` when no MDOS. |
| 0x38   | `Unknown_0x1C` (“SurfaceKey”)   | 4    | **Primary grouping key** for render objects.  Surfaces sharing the same 32-bit value form one visual object.  Low word matches `MSLK.LinkSubHigh/Low`; high word clusters by tile/material. |
| 0x3C   | `PackedParams`                  | 4    | Two 16-bit fields packed:<br>• **Low word** frequently `0x6870` / `0x6960` on candidate liquid surfaces.<br>• **High word** correlates with collision flag clusters. |

> The exporter writes `LowWord_0x1C` and `HighWord_0x1C` as aliases that split `Unknown_0x1C` for easier CSV filtering.

---

## 2. Relationships

### 2.1 MSUR ⇄ Geometry Chunks
```
MSUR.MsviFirstIndex → MSVI[range]
MSVI indices → (vertex buffer)
               ├─ MSPV – navigation vertices
               └─ MSVT – render vertices (axis-flipped)
```
* `IndexCount` gives the length of this index range (`IndexCount / 3` triangles).

### 2.2 MSUR ⇄ MSLK (Logical Objects)
* `Unknown_0x1C` **low word** equals `(LinkSubHighByte << 8) | LinkSubLowByte` of corresponding MSLK entries.
* Sum of `IndexCount` over all MSUR surfaces with the same key equals the total triangle budget of that object.

### 2.3 MSUR ⇄ MDOS (Material Definition)
`MdosIndex` maps into the MDOS chunk providing shader / texture material data.  Zero indicates default material; ‑1 means **no MDOS entry**.

---

## 3. Observations from 2025-07-12 CSV Dump
* Two main value bands detected in `PackedParams`:
  * `0x6870 / 0x6960` → Surfaces flagged as **liquid**; normals typically horizontal.
  * `0x40AA / 0x42??` → Solid geometry; normals varied.
* `SurfaceAttributeMask` aligns with **WMO render flags** (needs more samples).
* `IsM2Bucket` derived from `FlagsOrUnknown_0x00 & 0x10` – surfaces in bucket **0** belong to the overlaid M2 model geometry.
* Very small `IndexCount` (≤6) surfaces are often stray triangles at tile borders.

---

## 4. Outstanding Unknowns
1. Full bit-layout of `FlagsOrUnknown_0x00` – 8 distinct bits observed but not yet mapped.
2. Semantic split of `PackedParams`; names `MaterialId`, `CollisionFlags` are hypotheses.
3. Relation between `SurfaceAttributeMask` and MDOS material flags.
4. Whether `UnknownFloat_0x04..0x10` store something other than raw normals (e.g. tangent-space components).

---

## 5. References
* Source: `MSURChunk.cs`, `MslkObjectMeshExporter.cs` exporter logic.
* Diagnostic CSV: `msur.csv` in `project_output/<timestamp>/dump_all/<tile>/`.
* Cross-reference: `mslk_linkage_reference.md` for MSLK linkage rules.

---

## 6. Statistical Findings – 2025-07-12

Analysis of `msur_stats.csv` generated from `development_22_18.pm4` produced these key insights:

| Category | Notes |
|----------|-------|
| **Ground bucket** | `(High1C, Low1C) = 0x0000,0x0000` dominates (≈ 2 900 surfaces). Likely base terrain triangles. |
| **AttributeMask progression** | For ground bucket masks increment from `0x0000` to `0x000A`; average normal-Z declines gradually. Appears to encode material layers (grass → rock → cliff / water). |
| **Object buckets** | Each non-zero `Unknown_0x1C` key (e.g. `0x42084CF4`) groups ~150-200 surfaces with internally consistent normals → discrete doodad / collision objects. |
| **PackedParams formula** | Confirmed `PackedParams == (HighWord1C << 16) | LowWord1C`. |
| **Normals by mask** | AttributeMask 0 is flattest; masks 4-6 trend steeper; sparse masks 9-A only in ground bucket (suspected water or invisible collision). |

### Next Correlation Targets
1. Map `SurfaceAttributeMask` values to **MSVT** material/texture rows.
2. Compute an AABB per `(High,Low)` MSUR group via OBJ export and compare with ADT/WMO extents.
3. Examine **MSCN** chunk fields for indices/pointers that might reference MSUR/MSPV data.

_Last updated 2025-07-12 by Cascade_
