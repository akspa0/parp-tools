# PM4 Chunk Reference

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
- **Cross-Tile References**: May reference vertices in adjacent tiles (requires region loading)

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
  - `ParentIndex`: Links to MPRL.Unknown4 (98.8% overlap confirmed)
  - `MspiFirstIndex`: Geometry start index
  - `MspiIndexCount`: Geometry index count
  - `LinkIdRaw`: 32-bit coordinate field
- **Coverage**: 125.3% over-indexing indicates hierarchical relationships

## Object Definition Chunks

### MPRR (Properties Record) ⭐ **CRITICAL FOR OBJECT GROUPING**
- **Purpose**: Defines true object boundaries using sentinel markers
- **Structure**: 4-byte entries (2 ushort values)
- **Key Pattern**: 
  - `Value1 = 65535 (0xFFFF)`: Sentinel marker separating objects
  - `Value2`: Component type identifier
- **Object Count**: ~15,400 building objects per PM4 file
- **Data Scale**: 81,936 properties with 15,427 sentinels

### MPRL (Placement List)
- **Purpose**: Instance positions and transformations
- **Structure**: 24-byte entries
- **Key Fields**:
  - `Unknown4`: Component type (links to MSLK.ParentIndex)
  - `Position`: Vector3 world position
  - Additional unknown fields (likely rotation/scale)
- **Scale**: 178,588 placements (many instances of same components)
- **Important**: These are instances/copies, NOT object definitions

## Cross-Tile Reference Chunks

### MSCN (Scene Connectivity)
- **Purpose**: External vertex references for cross-tile geometry
- **Structure**: Array of Vector3 positions
- **Usage**: Provides vertices referenced by out-of-bounds MSVI indices
- **Processing**: Appended to main vertex pool during region loading

## Chunk Relationships and Data Flow

### Object Assembly Pipeline
```
MPRR (sentinels) → Object Groups
    ↓
MPRL (placements) → Component Instances
    ↓
MSLK (links) → Geometry References
    ↓
MSUR (surfaces) → Triangle Data
    ↓
MSVI (indices) + MSVT (vertices) → Final Geometry
```

### Key Relationships
1. **MPRR.Value1=65535** separates object groups
2. **MPRR.Value2** identifies component types within objects
3. **MPRL.Unknown4 ↔ MSLK.ParentIndex** (98.8% overlap)
4. **MSLK** links component instances to geometry fragments
5. **MSUR** provides subdivision-level geometry organization
6. **MSVI** references vertices (may cross tile boundaries)

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

## Implementation Notes

### Cross-Tile Loading
PM4 files reference vertices from adjacent tiles. Complete geometry requires:
1. **Region Loading**: Load up to 502 tiles in 64x64 grid
2. **MSCN Processing**: Append exterior vertices to main pool
3. **Index Remapping**: Resolve out-of-bounds references

### Object Grouping Strategy
1. **Parse MPRR sentinels** to identify object boundaries
2. **Group component types** between sentinel markers
3. **Map components to geometry** via MPRL→MSLK→MSUR chain
4. **Assemble complete objects** from all component geometry

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
1. **Surface Group Grouping**: MSUR.SurfaceKey represents subdivisions, not objects
2. **Placement Instance Grouping**: MPRL entries are instances, not definitions
3. **Single-Tile Loading**: Results in 64% data loss from missing cross-tile references

### ✅ Correct Approach
1. **MPRR-Based Grouping**: Use Value1=65535 sentinels for object boundaries
2. **Hierarchical Assembly**: Combine all component types within object groups
3. **Region Loading**: Load complete tile regions for cross-tile vertex resolution

## Conclusion

PM4 chunk relationships form a complex instanced geometry system where:
- **MPRR defines object boundaries** (not surface groups or placements)
- **Cross-tile loading is essential** for complete geometry
- **Hierarchical assembly produces realistic building objects** (not fragments)

Understanding these relationships is crucial for successful PM4 object extraction and export.
