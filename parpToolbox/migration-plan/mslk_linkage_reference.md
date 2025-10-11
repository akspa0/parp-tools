# MSLK Cross-Chunk Linkage Reference

_Last updated: 2025-07-10_

This document consolidates everything confirmed about how the **MSLK** chunk connects to other PM4 chunks.  It is the single source of truth; keep it updated as new discoveries are validated.

---

## 1. MSLK Entry Structure (20 bytes)
| Offset | Field                   | Size | Meaning (confirmed) |
|-------:|-------------------------|-----:|---------------------|
| 0x00   | `Unknown_0x00`          | 1    | Object-type flag ("flag") – high-level category used in grouping. |
| 0x01   | `Unknown_0x01`          | 1    | Sub-type (global to file, **not** object-local). |
| 0x02   | `Unknown_0x02`          | 2    | Always 0 – padding. |
| 0x04   | `GroupObjectId`         | 4    | Legacy name _Unknown_0x04_. Appears to group geometry by logical sub-object. |
| 0x08   | `MspiFirstIndex` (24-bit) | 3+1* | First index into **MSPI** for geometry triangles. -1 if entry is a doodad/reference only. |
| 0x0B   | `MspiIndexCount`        | 1    | Number of indices in **MSPI** (multiple of 3 for triangles). |
| 0x0C   | `LinkIdRaw`             | 4    | Links across tiles. Split into:<br>• `LinkPadWord` = high 16-bit (always `0xFFFF`).<br>• `LinkSubHighByte`, `LinkSubLowByte` – low 16-bit halves. |
| 0x10   | `ReferenceIndex`        | 2    | Split into `RefHighByte`, `RefLowByte`. `RefHighByte` is the container ID used by current grouping; `RefLowByte` sequences child nodes. |
| 0x12   | `Unknown_0x12`          | 2    | Constant `0x8000`. |

\* `MspiFirstIndex` is stored as a _signed_ 24-bit little-endian integer followed by the `MspiIndexCount` byte.

---

## 2. Relationships

### 2.1 MSLK ⇄ MSPI / MSVI / MSPV / MSVT (Geometry)
```
MSLK.MspiFirstIndex → MSPI[range]
MSPI indices → MSVI indices
MSVI indices → (vertex buffer choice)
                ├─ Navigation vertices  – MSPV ( X Y Z )
                └─ Render   vertices    – MSVT ( Y X Z )  ← flipped axes
```
Use **MSLK.IsGeometryNode** (`MspiIndexCount > 0`) to decide whether geometry exists.

### 2.2 MSLK ⇄ MSUR (Surface / Render object)
`MSUR` surfaces reference the same **MSVI** indices but provide per-surface normals and material hints.

Empirical findings:
* `MSUR.Unknown_0x1C` **uniquely identifies a render-object**.  All surfaces that share the same 32-bit key form one logical object.
* The low word of that key aligns to **MSLK.LinkSubHigh/Low**; high word groups by tile/material (0x40AA etc.).

Table:
| Observation | Correlation |
|-------------|-------------|
| `MSUR.Unknown_1C LowWord` | == concatenation of `LinkSubHighByte` + `LinkSubLowByte` |
| `MSUR.IndexCount`         | == sum of `MspiIndexCount` for matching MSLK nodes |

### 2.3 MSLK Containers & Flags
* **Container ID** = `(Unknown_0x00 << 8) | RefHighByte`  (used by exporter `--by-container`).
* Containers close in space and sequential IDs cluster into full props (exporter `--by-cluster`).

### 2.4 Inter-Tile Links
`LinkPadWord == 0xFFFF` signals the object crosses ADT tile boundaries.  The low 16-bit of `LinkIdRaw` matches the neighbouring tile’s matching object.

---

## 3. Export Strategies Implemented
| CLI Switch / Mode        | Description |
|--------------------------|-------------|
| `mslk-export --by-container` | One OBJ per container ID. |
| `mslk-export --by-cluster`   | Spatially merges nearby containers. |
| `mslk-export --by-objectcluster` | Merges overlapping clusters. |
| `msur-export` (new)      | One OBJ per `MSUR.Unknown_1C` key – **best isolation of individual render objects discovered so far**. |
| `bulk-extract`           | Runs all of the above and produces diagnostic CSVs. |

---

## 4. Outstanding Unknowns
1. Exact meaning of `ObjectSubtype` (Unknown_0x01)
2. Whether `GroupObjectId` (_Unknown_0x04_) is ever used in game logic.
3. Full semantics of `LinkIdRaw` when high word ≠ `0xFFFF` (not yet observed).
4. Relationship between `MSLK.Unknown_0x12 (0x8000)` and navigation flags.

---

## 5. Practical Examples
*Tile `development_00_00.pm4`*
| MSUR Key | Containers | Notes |
|----------|------------|-------|
| `0x40AA0A7E` | 3 | Sentinal tower cap (partial) |
| `0x421B098A` | 5 | Tree stump cluster |

Refer to `project_output/<timestamp>/bulk_extract/<tile>/` for the OBJ files corresponding to these IDs.

---

## 6. References
* `PM4Documentation/pm4_format_reference.md` – base format spec.
* Source code: `MSLKChunk.cs`, `MSURChunk.cs`, exporters in `Services/PM4`.
* Diagnostic CSV: `mslk_linkscan.csv` (produced by `MslkLinkScanner`).

---

> **Always update this file** whenever a new linkage rule is confirmed.
