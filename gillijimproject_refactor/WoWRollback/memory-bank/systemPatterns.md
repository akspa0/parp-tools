# System Patterns - WoWRollback Architecture

## Archaeological Data Model
The WoWRollback tool treats game data as archaeological layers:

```
Modern WoW (LK ADT files)
    ↑ Evolution/Preservation
Alpha WoW (Alpha WDT files)
    ↑ Original Development
```

## UniqueID as Historical Artifacts
- **Volume of Work**: Each contiguous range represents a discrete work session
- **Singleton Artifacts**: Isolated IDs showing experiments, tests, or technological changes
- **Gap Analysis**: Spaces between ranges indicate development phases or technological shifts
- **Clustering Strategy**: Preserve outliers rather than filtering them (unlike typical clustering)

## Core Services Architecture

### Alpha Analysis Path
```
Alpha WDT Files → AlphaWdtAnalyzer → UniqueID Extraction → Archaeological Clustering → alpha_ranges_by_map_<map>.csv
```

### LK Analysis Path (Existing)
```
LK ADT Files → AdtPlacementAnalyzer → UniqueID Extraction → Range Analysis → lk_ranges_by_map_<map>.csv
```

### Comparison Path (Future)
```
Alpha CSV + LK CSV → EvolutionAnalyzer → Content Preservation Analysis → evolution_report_<map>.csv
```

## Data Flow Patterns

### Input Sources
- **Alpha WDT Files**: Raw game client files from ~2000-2004
- **Converted LK ADT Files**: Output from AlphaWDTAnalysisTool conversion process
- **Configuration Files**: User-defined range filters for rollback functionality

### Output Artifacts
- **Per-Map Range CSVs**: Detailed UniqueID range data by map
- **Archaeological Reports**: Analysis of content preservation/evolution
- **Rollback Configurations**: JSON/YAML configs for selective content removal

## Error Handling Patterns
- **Missing Files**: Skip-if-missing approach for optional analysis
- **Parse Failures**: Log and continue rather than abort entire analysis
- **Invalid Ranges**: Preserve all data, mark questionable ranges for manual review

## Processing Patterns
- **Batch Processing**: Handle entire map collections
- **Session Management**: Timestamped output directories
- **Incremental Analysis**: Support partial re-runs on updated data

## Integration Patterns
- **Reuse AlphaWDTAnalysisTool Logic**: Leverage existing WDT parsing capabilities
- **Standalone Operation**: Independent tool that doesn't modify source tools
- **Configurable Thresholds**: Archaeological significance parameters (gap sizes, minimum cluster sizes)
