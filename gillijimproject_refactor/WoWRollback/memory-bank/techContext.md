# Technical Context - WoWRollback Implementation

## Runtime Environment
- **Target Framework**: .NET 9.0
- **Platform**: Windows x64
- **Build System**: dotnet CLI with MSBuild
- **Dependencies**: 
  - `GillijimProject.WowFiles` (Alpha/LK ADT parsing)
  - `System.Text.Json` (Configuration)

## Project Structure
```
WoWRollback/
├── WoWRollback.Core/           # Shared libraries
│   ├── Models/                 # Data models (PlacementRange, etc.)
│   └── Services/              # Core services
│       ├── AlphaWdtAnalyzer   # Alpha WDT archaeological analysis
│       ├── AdtPlacementAnalyzer # LK ADT preservation analysis  
│       ├── RangeScanner       # Map-level aggregation
│       └── Config/            # Configuration management
├── WoWRollback.Cli/           # Console application
├── memory-bank/               # Project documentation
└── docs/                      # Usage and specifications
```

## Key Dependencies
- **GillijimProject.WowFiles**: Existing C# port of WoW file format parsers
- **Alpha Classes**: `WdtAlpha`, `AdtAlpha`, `McnkAlpha`
- **LK Classes**: `Mhdr`, `Mddf`, `Modf` (for LK ADT analysis)

## Data Flow Architecture
```
Alpha WDT Files → AlphaWdtAnalyzer → Archaeological Ranges → CSV Output
LK ADT Files → AdtPlacementAnalyzer → Preservation Ranges → CSV Output  
Configuration → RangeSelector → Rollback Filtering → Modified ADTs
```

## File Format Considerations

### Alpha WDT Structure
- **WDT File**: Contains tile map and references to individual ADT files
- **ADT Files**: Individual tiles with MCNK chunks containing placement data
- **UniqueID Storage**: Located within chunk structures (format needs verification)

### LK ADT Structure  
- **MHDR**: Header with chunk offsets
- **MDDF**: M2 (doodad) placement data with UniqueIDs at offset +4
- **MODF**: WMO (building) placement data with UniqueIDs at offset +4
- **Entry Sizes**: MDDF=36 bytes, MODF=64 bytes per entry

## Output Formats

### Archaeological CSV Schema
```csv
map,tile_row,tile_col,kind,count,min_unique_id,max_unique_id,file
```

### Session Management
- **Output Root**: `rollback_outputs/` (default)
- **Session Directories**: `session_YYYYMMDD_HHMMSS/`
- **Per-Map Files**: `<session>/<map>/alpha_ranges_by_map_<map>.csv`

## Implementation Gaps

### Alpha Parsing (Critical)
The `AlphaWdtAnalyzer` currently has placeholder implementations:
- Alpha MCNK chunk structure analysis
- UniqueID field locations within Alpha format
- WDT tile presence detection

### Required Research
- Study existing `AlphaWDTAnalysisTool` parsing logic
- Verify Alpha format compatibility with current parsers
- Map Alpha chunk structures to UniqueID locations

## Error Handling Strategy
- **Resilient Analysis**: Continue processing if individual files fail
- **Logging**: Console output with archaeological terminology
- **Skip-if-Missing**: Don't abort on missing optional files

## Performance Considerations
- **Batch Processing**: Handle entire map directories
- **Memory Management**: Process files individually, not all in memory
- **Parallel Processing**: Future enhancement for large datasets

## Integration Points
- **AlphaWDTAnalysisTool**: Reference existing parsing logic
- **GillijimProject.WowFiles**: Leverage existing format parsers
- **Configuration System**: JSON/YAML for rollback rules

## Archaeological Terminology Integration
- **Console Output**: Uses archaeological language ("excavation", "preservation", "artifacts")
- **Variable Naming**: "sedimentary layers", "volumes of work", "ancient developers"
- **Documentation**: Consistently applies archaeological metaphor
