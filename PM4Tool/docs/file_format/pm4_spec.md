# PM4 File Format – Verified Specification (Pre-WotLK Build)

This document captures **only the structures and behaviours we have decoded and confirmed in code/tests**.  Any field whose purpose is not yet confirmed is marked `Unknown`.  No conjecture or historical speculation is included.

---

## Global Characteristics
* Little-endian.
* Four-character chunk signatures, **reversed** in the file header (e.g. `NSVM` on disk → `MSVN` in memory).
* All chunk headers use the common IFF header (4-byte signature, 4-byte size).

---

## Chunk List (confirmed)

| ID  | Description | Notes |
|-----|-------------|-------|
| MVER | Version header | 32-bit `Version` (always `4` in samples) |
| MSHD | Scene header  | 8 × `uint32` fields – all values stable across tiles. Exact semantics unknown but used in exporters for provenance. |
| **Vertex / Index Data** | | |
| MSPV | Vertex positions | `C3Vectorf` list. Stride auto-detected: *12 bytes* (`XYZ`) or *24 bytes* (`XYZ` + 3 extra floats). Extra floats correlate with per-vertex normals but are **ignored** by current tooling. |
| MSVT | Texture coordinates | `C2Vectorf` list. Count always matches `MSPV`. |
| MSVI | Triangle indices  | `uint16[]` list. Length divisible by 3; guard present to skip trailing bytes when not. |
| MSPI | Vertex island index ranges | Array of `{ FirstIndex : uint32, Count : uint32 }`. Used to group subsets of `MSPV` for path-finding. |
| **Surface Metadata** | | |
| MSUR | Surface headers | Array of `{ StartIndex : uint32, Count : uint32, UnknownAABB[4] : float }`. `StartIndex/Count` reference **triangles** in `MSVI`. |
| MSRN | Vertex normals | *PD4 only* – `C3Vectori` list; scale by **1/8192** to get unit normals. Not observed in PM4 tiles to date. |
| MSCN | Exterior boundary vertices | List of `C3Vectorf`; used by terrain-stamp utilities. |
| **Scene-Graph / Links** | | |
| MSLK | Link table | Array of `{ SurfaceIndex : uint32, SecondaryReference : uint32, AdjacentMspiStart : uint32, Flags : uint32 }`. Confirmed fields drive object/region extraction. |
| **Path Region** | | |
| MPRL | Tile-coord → RegionID lookup | Dense array `[64×64]` of `uint16` covering ADT tile. |
| MPRR | Region bounding ranges | List of `uint16[]`; each sequence describes extents used by server ROI queries. |
| **Destructible Geometry** | Seen only in tile `00_00` | |
| MDOS | Object set table | 24-byte records – confirmed fields: `SurfaceID : uint32`, `Flags : uint32`, remainder unknown. |
| MDSF | Object state frames | 36-byte records – contains AABB + state flags for phased destruction. |
| MDBH | Destructible header | Single struct describing MD* entry counts. |

---

## Signature Order in Sample Tiles
The following sequence is typical (development_22_18.pm4):
```
MVER → MSHD → MSPI → MSPV → MSVI → MSVT → MSLK → MSUR → MSCN → MPRL → MPRR
```

Destructible chunks (`MD*`) appear only in `development_00_00.pm4` **after** the standard chunks.

---

## Validation Status
* All loaders implemented in `WoWToolbox.Core.v2` have unit tests against **real sample tiles** from `test_data/original_development`.
* `batch-dump` command serialises every confirmed chunk to CSV for inspection.

---

_Last updated: 2025-07-12_
