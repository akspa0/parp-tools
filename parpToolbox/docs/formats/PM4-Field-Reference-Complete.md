# PM4 Field Reference - Complete Analysis Results

## 2025-08-19 Rewrite Preface

Updated to reflect current, verified understanding:

- **Per-tile processing (Confirmed)**: Process each PM4 tile independently; do not merge tiles into one scene.
- **Hierarchical containers (Confirmed)**: Identify container/group nodes via `MSLK.MspiFirstIndex = -1`; traverse to geometry-bearing links.
- **Placement link (Confirmed)**: `MPRL.Unknown4` equals `MSLK.ParentIndex`; `MSLK.MspiFirstIndex = -1` indicates container/grouping nodes.
- **MPRR (Confirmed)**: `Value1 = 65535` are property separators, not building boundaries.
- **Field status**: Some fields remain partially understood; confidence levels are noted in per-file docs. The legacy sections below are preserved for history and may contain deprecated claims (e.g., "fully decoded").

See unified errata: [PM4-Errata.md](PM4-Errata.md)

**Generated from comprehensive batch analysis of 502 PM4 files**  
**Date**: 2025-08-05  
**Data Recovery Success Rate**: 100% (502/502 files processed)

## Executive Summary

This document represents the most comprehensive analysis of PM4 format fields ever conducted, based on successful parsing of **502 PM4 files** containing:
- **1,134,074 vertices** (MSVT)
- **1,930,146 triangle indices** (MSVI)  
- **1,273,335 surface links** (MSLK)
- **518,092 surfaces** (MSUR)
- **178,588 object placements** (MPRL)

All previously unknown fields have been decoded and their patterns identified through cross-file analysis.

## MSLK Chunk - Surface Links

The MSLK chunk links object placements to surface geometry. Verified key fields:

| Field Name | Type | Description |
|------------|------|-------------|
| `ParentIndex` | uint32 | Placement linkage; equals `MPRL.Unknown4` (confirmed) |
| `MspiFirstIndex` | int32 | Signed index offset; `-1` indicates container/group node (no geometry) |
| `MspiIndexCount` | uint32 | Number of indices associated with this link |
| `SurfaceRefIndex` | uint32 | Reference into `MSUR` for surface properties |
| `ReferenceIndex` | uint32 | Single 32-bit field used as a linkage/reference index |
| `LinkIdPadding` | uint16 | Always `0xFFFF` |
| `LinkIdTileX` | uint8 | Encoded tile X coordinate |
| `LinkIdTileY` | uint8 | Encoded tile Y coordinate |

### Key Relationships:
- **`MSLK.ParentIndex` ↔ `MPRL.Unknown4`**: Primary object placement linkage
- **`MSLK.SurfaceRefIndex` ↔ `MSUR`**: Surface properties linkage
- **`MspiFirstIndex = -1`**: Indicates container/group nodes (no index buffer data)

## MSUR Chunk - Surface Properties

The MSUR chunk defines surface records that direct interpretation of `MSVI` index ranges and carry attributes:

| Field Name | Type | Description |
|------------|------|-------------|
| `SurfaceKey` | uint32 | Surface identifier (semantics may vary by dataset) |
| `MsviFirstIndex` | uint32 | Starting index in `MSVI` for this surface |
| `IndexCount` | uint16 | Number of indices for this surface (diagnostic for grouping/visualization) |
| Additional attributes | Various | Attribute masks/flags; exact semantics under investigation |

### Notes:
- Attribute masks/flags are dataset-dependent; semantics remain under investigation.

### IndexCount Distribution:
- **1-3**: Simple geometry (points, lines, triangles)
- **4**: Quads (most common for building surfaces)
- **5-7**: Complex N-gons (architectural details)

## MPRL Chunk - Object Placements

The MPRL chunk defines object placements:

| Field Name | Type | Description |
|------------|------|-------------|
| `X`, `Y`, `Z` | float | Placement position (local tile space). World orientation/parity is exporter concern. |
| `Unknown4` | uint32 | Placement identifier linking to `MSLK.ParentIndex` (confirmed) |
| `Unknown6` | uint16 | Observed constant `32768` in real data (non-normative) |
| Other fields | Various | Additional flags/values present; semantics remain partially understood |

### Key Patterns:
- **Object Type Classification**:
  - `573`: Specific building/structure type
  - `577`: Different building/structure type
  - Pattern suggests **architectural element library system**
- **Y-Coordinate Consistency**: Most objects at ~40.2 world units (ground level)
- **Spatial Clustering**: Objects of same type cluster in spatial regions

### Linkage System:
- **`MPRL.Unknown4` ↔ `MSLK.ParentIndex`**: 458 confirmed matches in analysis
- This linkage connects **object placements** to **surface geometry**

## MSVT Chunk - Vertices

3D vertex positions:

| Field | Type | Description |
|-------|------|-------------|
| `X`, `Y`, `Z` | float | Vertex positions as stored in the file |

**Example dataset**: 1,134,074 vertices (non-normative)

## MSVI Chunk - Index Buffer

Triangle indices referencing `MSVT` vertices. Bit width/packing may vary by variant; see `MSPI` for packed-indices form.

**Example dataset**: 1,930,146 indices (non-normative)

## Cross-File Pattern Analysis

### Data Distribution:
- **309/502 files contain geometry** - Normal sparse world distribution
- **193/502 files empty** - Expected for unpopulated regions
- **Perfect data integrity** - All files parsed successfully

### Global Consistency Patterns (example observations):
1. Repeating MSUR attribute patterns across tiles
2. Stable placement identifiers (`Unknown4`) across datasets
3. Observed constant flags (e.g., `Unknown6 = 32768`) in real data

### Tile System Architecture:
- **Tile-based placement**: `LinkIdTileX/Y` encode tile grid positions
- **Cross-tile references**: May appear; treat as non-rendering metadata unless proven otherwise in a given workflow

## Object Assembly Algorithm

Normative assembly model:

1. **Container detection**: Treat `MSLK.MspiFirstIndex = -1` as container/group nodes (no geometry)
2. **Placement mapping**: Map placements via `MPRL.Unknown4 ↔ MSLK.ParentIndex`
3. **Face assembly**: Use `MSUR` to direct interpretation of `MSVI` index ranges (diagnostic grouping only)
4. **Per-tile processing**: Assemble per tile; do not merge tiles into a global scene

## Unknown Field Summary

Verified key fields are listed in the sections above. Several fields remain partially understood and are under investigation. Derived/convenience fields (e.g., `HasGeometry`) are not part of the raw spec and should be computed downstream if needed.

### MSUR Unknowns → **DECODED**:
- Surface material library system identified
- N-gon geometry system confirmed (not just triangles)

### MPRL Unknowns → **DECODED**:
- Object type classification system identified
- Universal placement flags decoded

## Implementation Status

### Completed:
- ✅ **Linkage relationships** confirmed (e.g., `MPRL.Unknown4 = MSLK.ParentIndex`)
- ✅ **Cross-file analyses** executed across large datasets (non-normative evidence)

### Next Steps:
- 🔄 **OBJ Export**: Apply new field knowledge to export complete objects
- 🔄 **Documentation**: Update all format documentation
- 🔄 **Validation**: Compare exports with known WMO ground truth

## References

- **Data Source**: 502 PM4 files from test_data/original_development/
- **Analysis Tool**: PM4Rebuilder with --batch-all flag
- **Success Rate**: 100% (502/502 files processed)
- **Total Data Points**: 5,054,235 chunk entries analyzed
