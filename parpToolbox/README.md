# parpToolbox

A comprehensive toolkit for analyzing and extracting World of Warcraft PM4 and WMO files with advanced spatial correlation capabilities.

## Tooling Overview

- **parpToolbox**: Core library + analysis/diagnostics CLI for PM4/PD4/WMO. Provides research exporters, analysis tools, correlation utilities, and diagnostics.
- **PM4NextExporter**: Focused PM4 export CLI (`pm4next-export`) with multiple assembly strategies and robust per-object/per-tile exporting. Prefer this for PM4 exports.

All tools write outputs exclusively to timestamped directories under `project_output/`, with logs captured in `run.log`.

### Which CLI should I use?

- **Export PM4 to OBJ (current path)**: Use `PM4NextExporter`. See `src/PM4NextExporter/README.md`.
- **Analyze PM4/PD4, WMO export, correlation, diagnostics**: Use parpToolbox commands (this README).

### Build & Run Quickstart

```bash
# Build repository
dotnet build

# Run PM4NextExporter (preferred for PM4 export)
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" --format obj

# Run parpToolbox CLI (analysis/diagnostics)
dotnet run --project src/parpToolbox -- analyze "development_00_00.pm4" --report
```

For PM4NextExporter usage and options, see: `src/PM4NextExporter/README.md`.

## parpToolbox CLI Command Index

- **Primary**
  - `analyze` — Analyze PM4/PD4 files; generate reports/DB
  - `export` — Export PM4/PD4/WMO to OBJ/MTL (research/legacy paths)
  - `wmo` — Export WMO; supports `--split-groups`, `--collision`, `--facades`
  - `test` — Run regression/validation tests

- **PM4 analysis and diagnostics**
  - `pm4-analyze-fields` — Chunk field distribution analysis
  - `analyze-pm4-keys` — Key-like fields (Data Web model) analysis
  - `diagnose-linkage` — Cross-chunk linkage diagnostics
  - `analyze-surfacerefindex` — MSUR surface-ref index patterns
  - `analyze-mprl-fields` — MPRL field statistics
  - `analyze-mslk-fields` — MSLK field statistics
  - `chunk-validation` (`cv`) — Validate MSUR chunk integrity
  - `pm4-analyze-data-banding` — Data banding analysis
  - `surface-encoding-analysis` (`sea`) — Surface encoding patterns
  - `bounds-decoder` (`bd`) — Decode surface bounds
  - `hierarchical-decoder` (`hd`) — Container hierarchy analysis
  - `global-mesh-analysis` / `gma`, `mprl-analysis` / `mfa` — Global mesh and MPRL analyses
  - `export-pm4-dataweb` — Export structured PM4 Data Web dataset

- **PM4 exporters (research/experimental)**
  - `pm4-export-scene-graph` — Scene graph traversal export
  - `pm4-export-spatial-clustering` — Spatial clustering export
  - `refined-hierarchical-export` — Refined hierarchical grouping
  - `spatial-clustering`, `diagnostic-spatial-clustering`, `enhanced-diagnostic-spatial`, `fixed-spatial-clustering` — Clustering diagnostics
  - `pm4-export-4d-objects` — Experimental 4D object exporter
  - `pm4-export-json` — Export PM4 to JSON/DB/aux files

- **Correlation**
  - `pm4-wmo-match` — Spatial match PM4 to WMO geometry
  - `mscn-wmo-compare` — Compare MSCN anchors to WMO
  - `batch-mscn-wmo-correlation` — Batch correlation across directories

- **Utilities**
  - `chunk-test` — Explore chunk combination patterns
  - `mprl-pattern-analysis` (`mpa`) — Analyze MPRL patterns (DB)
  - `mslk-pattern-analysis` (`mla`) — Analyze MSLK patterns (DB)
  - `quality-analysis` (`qa`) — Data quality metrics

Notes:
- Legacy command aliases are supported with deprecation warnings; prefer unified commands above.
- All commands write to timestamped `project_output/` directories and log to `run.log` inside each session.

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

Prefer the dedicated PM4 export CLI: `PM4NextExporter` (`src/PM4NextExporter/README.md`).
The commands below refer to parpToolbox's analysis/experimental exporters.

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

## GLB-RAW Export (parpDataHarvester)

Use the GLB-RAW exporter to pack raw PM4 geometry into GLB for quick inspection.

```bash
# Per-region (recommended: resolves cross-tile MSCN remaps)
dotnet run --project src/parpDataHarvester/parpDataHarvester.csproj -- \
  export-glb-raw --in ".\\test_data\\original_development" --out ".\\project_output\\glb_raw" \
  --per-region --mode objects

# Optional: flip X for visualization parity
dotnet run --project src/parpDataHarvester/parpDataHarvester.csproj -- \
  export-glb-raw --in ".\\test_data\\original_development" --out ".\\project_output\\glb_raw" \
  --per-region --mode objects --flip-x

# Per-tile (no cross-tile remap; may miss geometry)
dotnet run --project src/parpDataHarvester/parpDataHarvester.csproj -- \
  export-glb-raw --in ".\\test_data\\original_development" --out ".\\project_output\\glb_raw" \
  --mode surfaces
```

Notes:
- Positions are written as-is by default (no X-axis flip). Use `--flip-x` only if needed.
- Per-region uses `Pm4GlobalTileLoader` to aggregate vertices and apply MSCN remapping.
- `RawGeometryAssembler` clamps index slices, drops invalid triangles, and prints diagnostics
  (clamped slices, dropped triangles, emitted triangles).
- GLB uses a default double-sided material to avoid backface culling issues when mirroring.

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
- [PM4 Field Reference (Complete)](docs/formats/PM4-Field-Reference-Complete.md)
- [PM4 Assembly Relationships](docs/formats/PM4_Assembly_Relationships.md)
- [PM4 Object Grouping](docs/formats/PM4-Object-Grouping.md) - Verified object assembly methods
- Archived (historical): [Surface Fields analysis](docs/_archive/MSUR_FIELDS.md)

## Troubleshooting

### Common Issues

1. **Incomplete Geometry**: Use region loading mode (default) instead of single-tile mode
2. **Fragmented Objects**: Ensure cross-tile references are resolved
3. **Coordinate System Issues**: For GLB-RAW exports, use `--flip-x` if visualization parity requires X inversion (default is no flip)
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
