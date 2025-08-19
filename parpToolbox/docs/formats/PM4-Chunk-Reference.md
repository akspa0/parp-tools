# PM4 Chunk Reference

## 2025-08-19 Rewrite Preface

This chunk reference has been updated to reflect current understanding:

- **Per-tile processing (Confirmed)**: Process each PM4 tile independently. Do not build a unified global scene across tiles.
- **Hierarchical containers (Confirmed)**: Identify containers via `MSLK.MspiFirstIndex = -1` and traverse to geometry-bearing links.
- **Placement link (Confirmed)**: `MPRL.Unknown4` equals `MSLK.ParentIndex`.
- **MPRR sentinels (Confirmed)**: `MPRR.Value1 = 65535` are property separators, not building boundaries.

See unified errata: [PM4-Errata.md](PM4-Errata.md)

Legacy content below is preserved for historical context and may contain deprecated claims.

## [Deprecated] CRITICAL ARCHITECTURE BREAKTHROUGH (2025-07-27)

**GLOBAL MESH SYSTEM DISCOVERED:** PM4 chunks implement a **cross-tile linkage system** requiring **multi-tile processing** for complete geometry assembly.

### **Mathematical Validation:**
- **58.4% of triangles** reference vertices from adjacent tiles (30,677 out of 52,506)
- **63,297 cross-tile vertex indices** in perfect sequential range: 63,298-126,594
- **Zero gap** between local (0-63,297) and cross-tile vertex ranges
- **Complete architectural assembly requires directory-wide PM4 processing**

### **Surface Encoding System in MSUR:**
- **GroupKey determines field interpretation**: spatial vs encoded vs mixed
- **GroupKey 3**: Spatial coordinates, local tile geometry
- **GroupKey 18**: Mixed data, boundary objects spanning tiles  
- **GroupKey 19**: Encoded linkage data, cross-tile references (74% of surfaces)
- **BoundsMaxZ in encoded groups**: Hex-encoded tile/object references, NOT coordinates

**IMPACT:** All single-tile chunk processing methods are fundamentally incomplete. **Cross-tile vertex resolution mandatory.**

---

## Overview

PM4 files use an IFF-style chunk format to store building interior geometry. This document provides a comprehensive reference for all PM4 chunks and their relationships.

## Chunk Structure

Each chunk follows the IFF format:
- **4-byte signature** (FourCC)
- **4-byte size** (payload length)
- **Variable payload** (chunk-specific data)

## Core Geometry Chunks

### MSVT (Vertex Table)
- **Purpose**: Primary vertex storage
- **Structure**: Array of Vector3 positions
- **Coordinate System**: Y-up, X-axis requires inversion for proper orientation
- **Usage**: Referenced by MSVI indices

### MSPV (Packed Vertices)
- **Purpose**: Alternative vertex storage format
- **Structure**: Packed vertex data
- **Usage**: Used when MSVT is absent

### MSVI (Index Buffer)
- **Purpose**: Triangle indices referencing MSVT vertices
- **Structure**: Array of 16-bit indices
- **Triangulation**: Automatic conversion to triangle list
- **Notes**: Indices reference vertices within the current tile's data scope for per-tile processing workflows

### MSPI (Packed Indices)
- **Purpose**: Alternative index storage
- **Structure**: Packed triangle data
- **Usage**: Used when MSVI is absent

## Surface and Grouping Chunks

### MSUR (Surface Records)
- **Purpose**: Geometry subdivision and material assignment
- **Structure**: 32-byte entries
- **Key Fields**:
  - `SurfaceKey`: Subdivision level identifier (1,301 unique values)
  - `MsviFirstIndex`: Starting index in MSVI
  - `IndexCount`: Number of indices for this surface
  - `IsM2Bucket`: Flag for overlay models (should be ignored)
- **Important**: SurfaceKey represents subdivision levels, NOT object groups

### MSLK (Link Table)
- **Purpose**: Hierarchical linking between placements and geometry
- **Structure**: 20-byte entries
- **Key Fields**:
  - `ParentIndex`: Links to `MPRL.Unknown4` (confirmed mapping)
  - `MspiFirstIndex`: Signed index offset; `-1` indicates a container node (no geometry)
  - `MspiIndexCount`: Geometry index count
  - `SurfaceRefIndex`: Reference into `MSUR` for surface properties
  - `ReferenceIndex`: Single 32-bit field; any High/Low split is software-derived
  - `LinkIdPadding`: Always `0xFFFF`
  - `LinkIdTileX`, `LinkIdTileY`: Encode PM4 tile grid coordinates
 - **Coverage**: 125.3% over-indexing indicates hierarchical relationships

## Object Definition Chunks

- ### MPRR (Properties Record)
- **Purpose**: Records segmented properties. Sentinel markers separate property blocks, not building/object boundaries.
- **Structure**: 4-byte entries (2 ushort values)
- **Key Pattern**: 
  - `Value1 = 65535 (0xFFFF)`: Sentinel marker separating property sections
  - `Value2`: Component/property type identifier
  - **Data Scale**: 81,936 properties with 15,427 sentinels (example dataset)

### MPRL (Placement List) ⭐ **OBJECT INSTANCE SYSTEM**
- **Purpose**: Object instance positions with LOD control and state management
- **Structure**: 24-byte entries with sophisticated field encoding
- **Decoded Fields** (from pattern analysis):
  - `Unknown0`: **Object Category ID** (4630 = building type)
  - `Unknown2`: **State Flag** (-1 = active/enabled)
  - `Unknown4`: **Object Instance ID** (227 unique objects, links to MSLK.ParentIndex)
  - `Unknown6`: **Property Flag** (32768 = 0x8000, bit 15 set)
  - `Position`: **Vector3 local tile coordinates** (validated ranges)
  - `Unknown14/Unknown16`: **LOD Control System**
    - (-1, 16383) = Full detail rendering (906 instances)
    - (0-5, 0) = LOD levels 0-5 (667 instances)
- **Scale**: 1,573 object instances with advanced rendering control
- **Coordinate System**: Local tile space, requires XX*533.33 + YY*533.33 offset for world coordinates
- **Important**: Sophisticated object management system with LOD, not simple placements

## [Deprecated] Cross-Tile Reference System / Global Mesh Architecture

The following sections previously described a multi-tile/global processing model with cross-tile vertex resolution and GroupKey-driven encoding. These claims are deprecated for object assembly workflows. Current guidance is to process PM4 strictly per tile and assemble objects via hierarchical container traversal. See rewrite preface and unified errata for details.

## Data Analysis Results

### Scale Analysis (development_00_00.pm4)
- **Vertices**: 812,648 (after cross-tile resolution)
- **Indices**: 1,930,146
- **Triangles**: 643,382
- **Surfaces**: 518,092 (1,301 unique SurfaceKeys)
- **Links**: 1,273,335 (598,882 with valid geometry)
- **Placements**: 178,588 (6,470 unique Unknown4 values)
- **Properties**: 81,936 (15,427 sentinels, 15,428 object groups)

### Coverage Analysis
- **MSUR Coverage**: 100.0% (complete index coverage)
- **MSLK Coverage**: 125.3% (over-indexing from hierarchical links)
- **Vertex Usage**: 98.8% (minimal unused vertices)
- **MPRL-MSLK Overlap**: 98.8% (confirmed relationship)

## [Deprecated] Implementation Notes (Global Mesh Architecture)

Legacy guidance on multi-tile/global pipelines and MPRR-based grouping is preserved for history only and should not be used for current implementations.

### Performance Considerations
- **Large Objects**: Building objects can contain 600K+ triangles
- **Memory Usage**: Complete region loading requires significant RAM
- **Export Time**: Large object assembly and export may take considerable time

## Validation and Testing

### CLI Commands
- `pm4-analyze-data`: Comprehensive chunk relationship analysis
- `pm4-mprr-objects`: Export using MPRR-based object grouping
- `pm4-export-scene`: Export complete scene as unified OBJ

### Expected Results
- **Object Scale**: 38K-654K triangles per building object
- **Object Count**: ~15,400 complete building objects
- **Geometry Quality**: Connected, realistic building-scale structures

## Common Pitfalls

### ❌ Incorrect Approaches
1. **Treating MPRR as object boundaries**: MPRR are property separators only
2. **Surface group as object ID**: MSUR.SurfaceKey are subdivisions, not objects
3. **Global/unified scene building**: Merging tiles into one scene causes fragmentation

### ✅ Correct Approach
1. **Per-tile processing**: Build an isolated scene per PM4 tile
2. **Hierarchical container traversal**: Identify containers via `MSLK.MspiFirstIndex = -1`; map placements via `MPRL.Unknown4 ↔ MSLK.ParentIndex`
3. **Use MSUR/MSVI for faces**: Treat `MSUR.IndexCount` as diagnostic, not an object identifier

## Conclusion

PM4 chunk relationships support a hierarchical, per-tile assembly workflow:
- **MPRR separates property sections**, not building/object boundaries
- **Container traversal** via `MSLK` and placement mapping via `MPRL.Unknown4 ↔ MSLK.ParentIndex` drives assembly
- **Faces** come from `MSUR → MSVI`; treat `IndexCount` as diagnostic only

See the rewrite preface and unified errata for the authoritative guidance.
