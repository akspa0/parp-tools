# parpToolbox

A comprehensive toolkit for analyzing and extracting World of Warcraft PM4 and WMO files with advanced spatial correlation capabilities.

## Core Capabilities

### PM4 Processing
- **Scene Graph Export**: Extract complete building objects using hierarchical scene traversal
- **Cross-Tile Resolution**: Handle global mesh system requiring multi-tile processing
- **Object Assembly**: Assemble building-scale objects from fragmented PM4 data
- **Spatial Clustering**: Group geometry using verified spatial proximity algorithms
- **Database Export**: Export PM4 scenes to SQLite for detailed analysis
- **Field Analysis**: Analyze chunk field distributions and correlations

### WMO Processing
- **Group-Based Export**: Export individual WMO groups or complete models
- **Collision Geometry**: Extract collision meshes with optional inclusion flags
- **Texture Handling**: Process WMO materials and textures
- **Coordinate Transformation**: Apply proper coordinate system fixes for visualization

### MSCN-WMO Correlation
- **Anchor Point Extraction**: Extract MSCN collision anchor points from PM4 files
- **Spatial Correlation**: Correlate MSCN data with WMO geometry using 3D spatial algorithms
- **Batch Processing**: Process large datasets of PM4/WMO pairs
- **Validation Reports**: Generate detailed correlation analysis reports

## Installation

Requires .NET 9.0 or later.

```bash
git clone <repository-url>
cd parpToolbox
dotnet build
```

## PM4 Processing

### Basic Export

```bash
# Export PM4 to OBJ using default scene graph approach
dotnet run -- export "development_00_00.pm4"

# Export with single-tile mode (for performance testing)
dotnet run -- export "development_00_00.pm4" --single-tile

# Export each surface group as separate OBJ files
dotnet run -- export "development_00_00.pm4" --per-object
```

### Advanced PM4 Analysis

```bash
# Analyze PM4 file structure and generate detailed reports
dotnet run -- analyze "development_00_00.pm4" --report

# Analyze with single-tile mode
dotnet run -- analyze "development_00_00.pm4" --single-tile --report

# Export PM4 data to SQLite database for detailed analysis
dotnet run -- pm4-export-json "development_00_00.pm4" --output "analysis.db"
```

### PM4 Field Analysis

```bash
# Analyze MPRL placement fields
dotnet run -- mprl-pattern-analysis "analysis.db"

# Analyze MSLK link fields
dotnet run -- mslk-pattern-analysis "analysis.db"

# Analyze surface reference index patterns
dotnet run -- analyze-surface-ref-index "analysis.db"

# Diagnose linkage patterns between chunks
dotnet run -- diagnose-linkage "analysis.db"
```

### PM4 Export Methods

The parpToolbox implements multiple PM4 export approaches, each optimized for different use cases:

1. **Scene Graph Export** (`pm4-export-scene-graph`): Uses hierarchical traversal following PM4's built-in scene graph structure. This is the recommended approach for most use cases.

2. **Spatial Clustering** (`pm4-export-spatial-clustering`): Combines hierarchical grouping with spatial proximity for robust object assembly.

3. **WMO-Inspired Export** (`pm4-export-wmo-inspired`): Applies WMO organizational logic to PM4 data for familiar export structure.

4. **Per-Object Export** (`export --per-object`): Exports each surface group as a separate OBJ file.

## WMO Processing

### Basic WMO Export

```bash
# Export complete WMO model
dotnet run -- wmo "building.wmo" --output "./exports"

# Export each WMO group as separate OBJ files
dotnet run -- wmo "building.wmo" --split-groups --output "./exports"

# Include collision geometry in export
dotnet run -- wmo "building.wmo" --collision --output "./exports"

# Include facade groups in export
dotnet run -- wmo "building.wmo" --facades --output "./exports"
```

### WMO Analysis

```bash
# Analyze WMO structure
dotnet run -- analyze "building.wmo" --report
```

## MSCN-WMO Correlation

### Single File Correlation

```bash
# Correlate MSCN anchors with WMO geometry
dotnet run -- mscn-wmo-compare "development_00_00.pm4" "building.wmo" --tolerance 5.0

# Generate detailed correlation report
dotnet run -- mscn-wmo-compare "development_00_00.pm4" "building.wmo" --report
```

### Batch Processing

```bash
# Batch correlate all PM4 files with WMO files
dotnet run -- batch-mscn-wmo-correlation --pm4-dir "./pm4_files" --wmo-dir "./wmo_files" --output-dir "./results"
```

## Key Concepts

### PM4 Scene Graph Architecture

PM4 files are structured as hierarchical scene graphs with nested coordinate systems:

```
PM4 Scene Graph Hierarchy:
MPRL (Building Root Nodes) - 458 buildings
  ├─ Coordinate Transform Matrix
  ├─ MSLK (Child Sub-objects) - ~13 per building
  │   ├─ Local Transform
  │   ├─ MSUR (Fine Geometry) → n-gon faces at full ADT resolution
  │   └─ MSCN (Spatial Bounds) → 1/4096 scale spatial anchors
  └─ Export as Single Unified Object
```

### Global Mesh System

PM4 implements a cross-tile linkage system requiring multi-tile processing:
- 58.4% of triangles reference vertices from adjacent tiles
- Cross-tile vertex resolution is mandatory for complete geometry
- Single-tile processing produces fragmented geometry

### Object Assembly Methods

The toolbox implements multiple verified object assembly methods:
1. **Scene Graph Traversal**: Follows hierarchical PM4 structure (recommended)
2. **Spatial Clustering**: Combines hierarchy with spatial proximity
3. **Surface Field Grouping**: Groups by MSUR field patterns

## Output Organization

All output files are written to timestamped directories in `project_output/` to prevent contamination of input data:

```
project_output/
├── session_20250803_143022/
│   ├── exported_models/
│   ├── analysis_reports/
│   ├── databases/
│   └── logs/
└── session_20250803_154501/
    ├── ...
```

### File Types

- `*.obj` - 3D geometry models
- `*.mtl` - Material files accompanying OBJ exports
- `*.db` - SQLite databases with extracted PM4 data
- `*_analysis.json` - Detailed analysis reports
- `*_correlation.json` - MSCN-WMO correlation results
- `*.log` - Processing logs with detailed diagnostics

## Documentation

- [PM4 Format](docs/formats/PM4.md) - Comprehensive PM4 format specification
- [PM4 Chunk Reference](docs/formats/PM4-Chunk-Reference.md) - Detailed chunk structure documentation
- [Object Grouping](docs/formats/PM4-Object-Grouping.md) - Verified object assembly methods
- [Surface Fields](docs/MSUR_FIELDS.md) - MSUR chunk field analysis

## Troubleshooting

### Common Issues

1. **Incomplete Geometry**: Use region loading mode (default) instead of single-tile mode
2. **Fragmented Objects**: Ensure cross-tile references are resolved
3. **Coordinate System Issues**: Apply X-axis inversion (-vertex.X) for proper orientation
4. **Memory Issues**: Process smaller regions or use single-tile mode for analysis

### Diagnostic Commands

```bash
# Diagnose tile contamination issues
dotnet run -- diagnose-tile-contamination "development_00_00.pm4"

# Analyze chunk field distributions
dotnet run -- pm4-analyze-fields "development_00_00.pm4"

# Test chunk combination patterns
dotnet run -- chunk-test "development_00_00.pm4"
```

## License

MIT License
