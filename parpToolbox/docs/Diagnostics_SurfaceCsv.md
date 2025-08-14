# PM4 Next Exporter – Surface CSV Diagnostics

This document describes the **`surfaces.csv`** diagnostic output produced by the PM4 Next Exporter when the `--csv-diagnostics` flag is enabled.

## File Location
The file is written to the selected diagnostics output directory ( `--csv-out <dir>` if specified, otherwise the main exporter output directory) with the fixed name `surfaces.csv`.

## Column Reference
| Column | Description |
|--------|-------------|
| `index` | Zero-based position of the surface entry inside the loaded scene collection. |
| `compositeKey` | Full 24-bit CompositeKey of the MSUR surface, rendered as hexadecimal `0xABCDEF`. |
| `msviFirstIndex` | Starting index inside the global triangle index buffer for this surface. |
| `indexCount` | Number of indices (triangles × 3) belonging to the surface. |
| `groupKey` | Primary grouping byte (formerly `FlagsOrUnknown_0x00`) indicating surface/object category. |

## Usage Hints
* Import into Excel, Google Sheets, or Python/Pandas for analysis.
* Correlate `indexCount` with your assembly strategy to detect outlier surfaces (e.g., tiny or unusually large meshes).
* Join with other chunk dumps (future `mslk.csv`, `mprl.csv` etc.) using `compositeKey` once those diagnostics are added.

---
## MSCN Vertices CSV (`mscn_vertices.csv`)

This file lists **all** MSCN vertex anchors extracted from the loaded tiles.  It is written alongside `surfaces.csv` whenever the scene contains at least one MSCN chunk.

| Column | Description |
|--------|-------------|
| `index` | Zero-based vertex position within the aggregated MSCN list. |
| `rawX` / `rawY` / `rawZ` | Raw vertex coordinates as stored in the MSCN chunk. |
| `worldX` / `worldY` / `worldZ` | Canonical world-space coordinates after tile transform (same space as OBJ/glTF exports). |
| `tileId` | Numerical tile identifier (e.g., `33_27`) where the vertex originated. |

This CSV enables offline correlation between collision anchors (MSCN) and surface geometry (MSUR/MSLK/MPRL). Join on `tileId` to relate anchors back to surfaces loaded from the same grid cell.

---
_Last updated: 2025-08-14_
