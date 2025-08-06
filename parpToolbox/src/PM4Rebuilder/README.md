# PM4Rebuilder

A comprehensive toolkit for analyzing, reconstructing, and exporting World of Warcraft PM4 terrain/building data to various formats.

## Overview

PM4Rebuilder provides multiple approaches for extracting and exporting geometry from WoW PM4 files, with a focus on producing accurate building-level exports for 3D analysis and reconstruction.

## Features

### 🏗️ **Direct PM4 to OBJ Exporter** ⭐ *Latest Feature*
- **Purpose**: Export PM4 scenes as multiple building-level OBJ files
- **Approach**: Direct PM4 processing without database intermediary
- **Output**: Individual OBJ files per building/object
- **Status**: ⚠️ *Active development - grouping refinements ongoing*

### 🗄️ **Database-Driven Building Exporter**
- **Purpose**: Export buildings from SQLite database using MPRR sentinel boundaries
- **Approach**: PM4 → SQLite → Grouped OBJ exports
- **Output**: Complete buildings based on MPRR boundary detection
- **Status**: ✅ *Functional with MPRR-based grouping*

### 🔍 **Analysis and Diagnostic Tools**
- **Batch PM4 Analysis**: Multi-file processing and statistics
- **MSCN Linkage Analysis**: Vertex relationship mapping
- **Surface Completeness Analysis**: Data integrity validation
- **Building Aggregation Analysis**: Spatial grouping validation

## Commands

### `direct-export` - Direct PM4 to Multiple OBJ Files
```bash
dotnet run -- direct-export <pm4_file_path> <output_directory>
```

**Example:**
```bash
dotnet run -- direct-export "data/development_00_00.pm4" "exported_buildings"
```

**Current Behavior:**
- Loads PM4 scene directly (bypasses database)
- Groups geometry using MSLK linkage data and MSUR surface fragments
- Outputs multiple OBJ files (one per building/object)
- Applies coordinate system corrections (X-axis flip)
- Includes diagnostic logging for troubleshooting

**Known Issues:**
- Over-aggregation: Some buildings contain mixed geometry
- Cross-contamination: Objects may include unrelated surfaces
- Grouping refinement needed for optimal building boundaries

### `export-buildings` - Database-Driven Building Export
```bash
dotnet run -- export-buildings <scene.db> [output_directory]
```

**Example:**
```bash
dotnet run -- export-buildings "scene_analysis.db" "buildings_output"
```

**Features:**
- Uses MPRR sentinel boundaries for building detection
- Exports complete buildings with proper component aggregation
- Includes building classification and metadata

### Analysis Commands

#### `analyze-batch` - Batch PM4 Analysis
```bash
dotnet run -- analyze-batch <pm4_directory> <output_directory>
```

#### `analyze-mscn` - MSCN Linkage Analysis
```bash
dotnet run -- analyze-mscn <pm4_file> <output_directory>
```

#### `validate-surfaces` - Surface Data Validation
```bash
dotnet run -- validate-surfaces <pm4_file> <output_directory>
```

## Current Development Status

### ✅ **Working Components**
- PM4 scene loading and data extraction
- Vertex pool resolution (regular + MSCN vertices)
- OBJ file generation with proper vertex remapping
- Coordinate system correction
- MSLK linkage data processing
- MSUR surface fragment aggregation

### ⚠️ **Active Development**
- **Building grouping optimization**: Refining MSLK ParentId-based grouping
- **Cross-contamination resolution**: Preventing mixed geometry in buildings
- **Spatial validation**: Ensuring building groups are spatially coherent
- **Performance optimization**: Handling large PM4 files efficiently

### 📋 **Planned Features**
- Alternative grouping strategies (spatial clustering + MSLK)
- MPRR sentinel integration with direct exporter
- Multi-level building hierarchy support
- Automated validation against reference data

## Technical Architecture

### Data Flow - Direct Exporter
```
PM4 File → Pm4Scene → MSLK Links → ParentId Groups → MSUR Fragments → Vertex Pools → OBJ Files
```

### Key Components
- **PM4 Scene Loading**: `SceneLoaderHelper.LoadSceneAsync`
- **Building Assembly**: `AssembleBuildingsFromMslkLinkage`
- **Vertex Resolution**: `ResolveVertexFromIndex` (regular + MSCN pools)
- **OBJ Export**: Standard OBJ format with 1-based indexing

### PM4 Data Relationships
- **MSLK**: Linkage table connecting fragments to buildings
- **MSUR**: Surface groups containing geometry fragments  
- **MSCN**: Additional vertex data for complete geometry
- **MPRR**: Property records with building boundary markers

## Output Examples

### Direct Export Output Structure
```
output_directory/
├── building_000_triangles_1097.obj  # Large building (walls, roof, floors)
├── building_001_triangles_243.obj   # Medium structure
├── building_002_triangles_89.obj    # Small object/detail
└── ...
```

### OBJ File Format
```obj
# PM4 Building Export - Building ID 0
# Triangle Count: 1097
# Triangle Range: ParentId 12345

v -32.826920 317.932159 12.033226
v -33.800953 316.848389 12.934059
# ... vertices ...

f 1 2 3
f 4 5 6
# ... faces ...
```

## Dependencies

- **.NET 9**: Target framework
- **parpToolbox**: PM4 parsing and data structures
- **wow.tools.local**: Additional WoW format support
- **System.Numerics**: Vector3 operations

## Build Instructions

```bash
# Navigate to project directory
cd src/PM4Rebuilder

# Build project
dotnet build

# Run with commands
dotnet run -- <command> [arguments]
```

## Known Limitations

1. **Building Grouping**: MSLK ParentId grouping may create incorrect building boundaries
2. **Performance**: Large PM4 files may require extended processing time
3. **Coordinate Systems**: Some coordinate transformations may need refinement
4. **Validation**: Limited ground truth comparison for building accuracy

## Troubleshooting

### Common Issues

**"Too many objects output"**
- MSLK ParentId grouping is over-aggregating
- Check diagnostic logs for cross-contamination warnings
- Consider using surface group fallback mode

**"Objects contain mixed geometry"**  
- Cross-contamination in MSLK assembly
- Review ParentId distribution in diagnostic output
- May require spatial clustering refinement

**"Coordinate system appears flipped"**
- X-axis flip correction is applied automatically
- If still incorrect, check original PM4 coordinate conventions

### Debug Logging

Set environment or enable verbose logging to see:
```
[DIAGNOSTIC] Building 0: MSLK entry references surface 5
[DIAGNOSTIC] Building 0: Surface 5 ('G40AA0A7E') has 943 faces  
[DIAGNOSTIC] Building 0 data sources:
  - Surface groups used: [5, 12, 8, 15]
  - Vertex pool usage: regular=1200, mscn=400
[CROSS-CONTAMINATION WARNING] Building 0 uses 12 surface groups - may be over-aggregating!
```

## Contributing

This project is under active development. Key areas for contribution:
- Building grouping algorithm refinement
- Performance optimization for large datasets
- Validation framework development
- Alternative export format support

## Documentation

- **Development Log**: `docs/DirectPM4Exporter-Development-Log.md`
- **Technical Issues**: `docs/PM4-Building-Assembly-Issues.md`  
- **PM4 Format Docs**: `docs/formats/` directory
- **Building Export Plans**: `docs/BuildingLevelExporter_Plan.md`

---

**Last Updated**: 2025-08-06  
**Version**: Active Development  
**Status**: Direct PM4 exporter functional with ongoing grouping refinements
