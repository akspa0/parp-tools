# PM4 Spatial Clustering Approach Documentation

## 🚀 REVOLUTIONARY BREAKTHROUGH (January 29, 2025)

### Cross-Tile Object Architecture Discovered
**FUNDAMENTAL PARADIGM SHIFT**: PM4 objects are not tile-local but span multiple tiles through ParentId linkage.

#### Data Analysis Results
- **27,087 ParentIds span multiple tiles** - Objects scattered across hundreds of tiles
- **364 tiles detected** in single PM4 file
- **0 SurfaceKeys span tiles** - SurfaceKeys are tile-local geometry fragments
- **ParentIds = Master cross-tile object identifiers**
- **SurfaceKeys = Tile-local surface geometry fragments**

#### Revolutionary Understanding
**Previous Approach (WRONG)**: Group by SurfaceKey → Get tile-local fragments
**Correct Approach (NEW)**: Group by ParentId → Assemble cross-tile complete objects

#### PM4 Object Assembly Model
```
Complete Object = ParentId + Σ(Geometry Fragments from ALL tiles)
```

### Cross-Tile Object Assembler
New `Pm4CrossTileObjectAssembler` implements the correct approach:
- Groups MSLK entries by ParentId (cross-tile identifier)
- Collects geometry fragments from all tiles for each ParentId
- Assembles complete objects with vertex deduplication
- Tracks source tiles and surface keys for diagnostics

### 🧠 PM4 COORDINATE SYSTEM ARCHITECTURE (USER INSIGHTS)

#### Confirmed PM4 Data Organization
- **SurfaceKey = Level of detail** arranged as **Y axis**
- **Banding slices** arranged as **Z axis**
- **MSCN data = Collision normals** per polygon (1/64th density of other chunks)
- **Cross-tile objects** still subdivided instead of single objects per PM4 tile
- **Complex nested coordinate systems** requiring careful preservation

#### 4D Architecture Model (REFINED)
```
Complete Object = ParentId + Σ(All Tiles) + Σ(SurfaceKey LOD layers)
                = Cross-tile + Level-of-detail + Z-axis banding
```

#### Critical Requirements
- **Z-axis alignment** needed for functional banding
- **Coordinate system preservation** within original chunk data
- **Per-tile object consolidation** instead of cross-tile subdivision
- **LOD layer integration** respecting Y-axis arrangement

## Overview
This document describes the implemented spatial clustering approach for PM4 object assembly, which is based on the working implementation extracted from the POC `poc_exporter.cs` file.

## Implementation Details

### Core Components
1. **Pm4SpatialClusteringAssembler.cs** - Main assembler class implementing the spatial clustering logic
2. **Pm4ExportSpatialClusteringCommand.cs** - CLI command for invoking the assembler
3. **Program.cs** - Command registration and argument parsing

### Key Algorithm Steps

1. **Root Node Identification**
   - Find MSLK entries where `ParentIndex == entry_index` (self-referencing)
   - These represent the root nodes of building objects

2. **Structural Element Grouping**
   - For each root node, collect all MSLK entries with matching `ParentIndex`
   - Filter to only entries with valid structural geometry (`MspiFirstIndex >= 0` and `MspiIndexCount > 0`)

3. **Bounds Calculation**
   - Calculate bounding box from MSPV vertices via MSLK → MSPI → MSPV chain
   - This defines the spatial extent of the structural elements

4. **Spatial Clustering**
   - Expand the structural bounds by 50.0f tolerance in all directions
   - Find MSUR surfaces whose vertices fall within the expanded bounds

5. **Hybrid Assembly**
   - Combine MSPV structural elements with nearby MSUR render surfaces
   - Export as complete building-scale OBJ files

### Data Flow

1. **Input:** PM4 scene loaded via `Pm4Adapter.LoadRegion()`
2. **Processing:** 
   - Root node detection from MSLK entries
   - Structural element collection and bounds calculation
   - Spatial clustering to find nearby render surfaces
   - Vertex and triangle assembly
3. **Output:** Building-scale OBJ files with proper geometry

### Coordinate Transformations

- **MSPV:** Direct coordinates `(vertex.X, vertex.Y, vertex.Z)`
- **MSVT:** Swapped coordinates `(vertex.Position.Y, vertex.Position.X, vertex.Position.Z)`

### CLI Usage

```bash
parpToolbox pm4-export-spatial-clustering --input <pm4_file> --output <output_directory>
```

Supports both `--output=value` and `--output value` formats.

## Architecture Insights

### Why Spatial Clustering?

Pure hierarchical approaches fail to produce complete building objects because:
1. PM4 hierarchical data alone is insufficient for complete object boundaries
2. Building geometry is fragmented across multiple chunks
3. Spatial proximity is needed to group related surfaces

### Cross-Tile Reference Resolution

The implementation uses region loading (`Pm4Adapter.LoadRegion()`) to:
1. Resolve cross-tile vertex references
2. Access ~63,000 missing vertices that would otherwise be out-of-bounds
3. Produce complete building-scale objects instead of fragments

## Validation

### Working Approach Verification

- Extracted from proven POC implementation (`poc_exporter.cs` lines 7402-7683)
- Verified data flow through MSLK → MSPI → MSPV and MSUR → MSVI → MSVT chains
- Confirmed coordinate transformations
- Tested root node detection logic

### Expected Results

- 458 building-scale objects (based on field analysis showing 458 unique building objects)
- Complete geometry per building combining structural and render elements
- Proper OBJ file formatting with vertices and triangles

## Next Steps

1. Test with real PM4 data to validate building count
2. Analyze output geometry quality
3. Compare with reference implementation outputs
4. Document any issues or improvements needed
