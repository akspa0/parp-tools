# parpToolbox

A tool for analyzing World of Warcraft PM4 and WMO files, with spatial correlation capabilities between MSCN collision data and WMO geometry.

## Features

### MSCN-WMO Spatial Correlation
- Extract MSCN anchor points from PM4 files
- Load WMO geometry with group filtering
- Spatial correlation analysis between MSCN and WMO data
- 3D visualization export (OBJ format)
- Batch processing for large datasets

### PM4 Analysis
- Export PM4 scenes to SQLite database
- Cross-tile geometry resolution
- Object assembly from hierarchical data
- Field analysis and pattern detection
### WMO Processing
- WMO file loading and vertex extraction
- Group-based filtering
- OBJ export for visualization

## Installation

Requires .NET 9.0 or later.

```bash
git clone <repository-url>
cd parpToolbox
dotnet build
```

## Usage

### MSCN-WMO Correlation

```bash
# Single file analysis
dotnet run -- mscn-wmo-compare input.pm4 building.wmo --tolerance 5.0

# Batch processing
dotnet run -- batch-mscn-wmo-correlation --pm4-dir ./pm4_files --wmo-dir ./wmo_files --output-dir ./results
```

### PM4 Export

```bash
# Export to SQLite database
dotnet run export "input.pm4"

# Analysis commands
dotnet run sea "database.db"  # Surface encoding analysis
dotnet run mpa "database.db"  # MPRL placement analysis
dotnet run gma "database.db"  # Global mesh analysis
```

### WMO Export

```bash
# Export WMO to OBJ
dotnet run wmo "building.wmo" --output "./exports"
```

## Output Files

- `*.obj` - 3D geometry for visualization
- `*.db` - SQLite databases with extracted data
- `*_results.json` - Analysis reports
- `*_comparison.txt` - Detailed correlation reports

## Documentation

- [PM4 Format](docs/formats/PM4.md)
- [PM4 Chunk Reference](docs/formats/PM4-Chunk-Reference.md)
- [Object Grouping](docs/formats/PM4-Object-Grouping.md)
- [Surface Fields](docs/MSUR_FIELDS.md)

## License

MIT License
