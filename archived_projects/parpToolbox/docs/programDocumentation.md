# Program Documentation - parpToolbox

## Implementation Status

### MSCN-WMO Spatial Correlation
**Status**: Production Ready
- Single file and batch correlation analysis
- Spatial hash grid optimization for large datasets
- 3D visualization export (OBJ format)
- Fixed critical correlation bug (August 2025)
- Visual validation confirms spatial alignment

### PM4 Analysis
**Status**: Stable
- SQLite database export with complete relational structure
- Cross-tile vertex resolution (58.4% triangles span tiles)
- Object assembly from hierarchical MPRL/MSLK relationships
- Field analysis and pattern detection
- Decoded MPRL.Unknown4 → MSLK.ParentIndex relationships (458 confirmed)

### WMO Processing
**Status**: Stable
- Native WMO file loading via WowToolsLocalWmoLoader
- Group-based filtering and vertex extraction
- OBJ export for visualization
- Integration with spatial correlation analysis

## Key Discoveries

### PM4 Format
- Cross-tile architecture: ~58.4% triangles reference vertices in adjacent tiles
- Object assembly: MPRL.Unknown4 links to MSLK.ParentIndex (458 matches)
- Surface encoding: GroupKey-based spatial vs linkage data classification
- Hierarchical structure: Buildings contain ~13 MSLK sub-objects on average

### Spatial Correlation
- MSCN anchors are in world space, WMO vertices in local space
- Coordinate normalization required for meaningful correlation
- Multi-tile MSCN aggregation improves coverage
- Spatial hash grids enable O(1) nearest neighbor lookup

## Current Commands

### MSCN-WMO Correlation
- `mscn-wmo-compare` - Single file spatial correlation analysis
- `batch-mscn-wmo-correlation` - Large-scale batch processing

### PM4 Analysis
- `export` - Export PM4 to SQLite database
- `sea` - Surface encoding analysis
- `mpa` - MPRL placement analysis
- `gma` - Global mesh analysis
- `qa` - Quality analysis

## PM4NextExporter (2025-08-10)

- **Default assembler**: composite-hierarchy.
- **Per-tile export**: `--export-tiles` writes tile OBJs. Current behavior merges geometry; refactor pending to output distinct assembled objects per tile.
- **Projection**: `--project-local` applies an export-time centroid translation (optional).
- **MSCN controls**: `--export-mscn-obj` (point OBJ), `--no-remap` to skip remapping.
- **Legacy parity**: `--legacy-obj-parity` for winding/naming parity.

### WMO Processing
- `wmo` - Export WMO to OBJ format

## Output Files

- `*.obj` - 3D geometry for visualization
- `*.db` - SQLite databases with extracted data
- `*_results.json` - Analysis reports
- `*_comparison.txt` - Correlation reports
