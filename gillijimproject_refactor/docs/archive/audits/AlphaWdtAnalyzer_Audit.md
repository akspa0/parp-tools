# AlphaWdtAnalyzer Tool Audit

**Date**: 2025-10-04  
**Purpose**: Document AlphaWdtAnalyzer analysis pipeline before Phase 6 migration  
**Target**: Migrate high-level logic to WoWRollback plugins

---

## Executive Summary

**AlphaWdtAnalyzer** is a **comprehensive analysis and conversion pipeline** that processes Alpha WDT files and exports viewer data + LK-format ADTs. It works well but is **separate from WoWRollback**.

### Current State
- **Location**: `AlphaWDTAnalysisTool/`
- **Technology**: C# .NET 9.0, CLI tool + Core library
- **Purpose**: Alpha WDT analysis → CSV reports → Viewer overlays → LK ADT export
- **Status**: Production-ready, used by rebuild-and-regenerate.ps1
- **Architecture**: 38-file analysis pipeline

### Migration Goal
**Migrate high-level analysis logic** to WoWRollback plugins while **keeping format readers** in gillijimproject-csharp library.

---

## Project Structure

### Two-Project Solution
```
AlphaWDTAnalysisTool/
├── AlphaWdtAnalyzer.Cli/        # CLI entry point
│   └── Program.cs (10KB)        # Command-line parsing
└── AlphaWdtAnalyzer.Core/       # Analysis pipeline (38 files)
    ├── Terrain/ (8 files)       # MCNK terrain/shadow extraction
    ├── Export/ (10 files)       # LK ADT writing + AreaID patching
    ├── Dbc/ (7 files)           # DBC integration
    ├── Assets/ (1 file)         # Resource management
    ├── Models.cs                # Shared data structures
    ├── AnalysisPipeline.cs      # Main orchestrator
    ├── BatchAnalysis.cs         # Multi-map processing
    ├── AdtScanner.cs            # Per-tile analysis
    ├── WdtAlphaScanner.cs       # WDT-level analysis
    ├── CsvReportWriter.cs       # CSV generation
    ├── ListfileLoader.cs        # Asset name resolution
    ├── MultiListfileResolver.cs # Listfile merging
    ├── UniqueIdClusterer.cs     # Object grouping
    ├── WebAssetsWriter.cs       # Viewer overlay generation
    └── WowPath.cs               # Path utilities
```

---

## Core Modules (38 files)

### 1. **Terrain Extraction** (8 files)
Extracts MCNK data for viewer overlays.

**Files**:
- `McnkTerrainExtractor.cs` (9KB) - Extract heights, flags, liquids
- `McnkTerrainEntry.cs` (629 bytes) - Per-chunk terrain data
- `McnkTerrainCsvWriter.cs` (2.5KB) - Generate terrain CSV
- `McnkShadowExtractor.cs` (7KB) - Extract shadow maps
- `McnkShadowEntry.cs` (332 bytes) - Per-chunk shadow data
- `McnkShadowCsvWriter.cs` (1.7KB) - Generate shadow CSV
- `McshDecoder.cs` (3.3KB) - Decode MCSH shadow data
- `LkAdtAreaReader.cs` (5.5KB) - Read AreaIDs from LK ADTs

**Purpose**: Generate CSVs consumed by WoWRollback viewer report writer.

**Output**:
```
csv/
└── DeadminesInstance/
    ├── DeadminesInstance_mcnk_terrain.csv   # Heights, flags, liquids
    └── DeadminesInstance_mcnk_shadows.csv   # Shadow map data
```

---

### 2. **Export Pipeline** (10 files)
Converts Alpha ADTs to LK format with AreaID patching.

**Files**:
- `AdtExportPipeline.cs` (28KB) - Orchestrates export process
- `AdtWotlkWriter.cs` (61KB) - **Largest file** - Writes LK ADT chunks
- `AreaIdMapper.cs` (17KB) - Maps Alpha AreaIDs → LK AreaIDs
- `DbcPatchMapping.cs` (33KB) - Loads DBCTool CSV crosswalks
- `AssetFixupPolicy.cs` (10KB) - Fixes M2/WMO/BLP paths
- `FixupLogger.cs` (3.5KB) - Logs asset corrections
- `MissingAssetsLogger.cs` (1.6KB) - Tracks missing assets
- `CsvReportBus.cs` (3KB) - Event bus for CSV reporting
- `CsvEvents.cs` (184 bytes) - Event definitions
- `Log.cs` (663 bytes) - Simple logging

**Purpose**: Generate LK-format ADTs for WoWRollback comparison.

**Output**:
```
World/Maps/DeadminesInstance/
├── DeadminesInstance.wdt
├── DeadminesInstance_32_18.adt
└── ... (all tiles)
```

---

### 3. **DBC Integration** (7 files)
Loads and applies AreaID mappings from DBCTool.

**Files**:
- `DbcAreaTableLoader.cs`
- `DbcAreaTableEntry.cs`
- `DbcCrosswalk.cs`
- `DbcMappingCsv.cs`
- `DbcAreaIdSuggester.cs`
- `DbcZoneResolver.cs`
- `DbcSubzoneResolver.cs`

**Purpose**: Apply AreaID patches using DBCTool crosswalk CSVs.

**Data Flow**:
```
DBCTool.V2/dbctool_outputs/*/compare/v2/
├── alpha_to_335_explicit_map.csv  # Alpha → LK mappings
├── alpha_areaid_decode.csv        # Zone/subzone decode
└── alpha_to_335_suggestions.csv   # Name-based suggestions
                                   ↓
                            AreaIdMapper.cs
                                   ↓
                        Patched MCNK AreaIDs in LK ADTs
```

---

### 4. **Analysis Orchestration** (7 files)
Coordinates the analysis pipeline.

**Files**:
- `AnalysisPipeline.cs` (6KB) - Per-map pipeline
- `BatchAnalysis.cs` (6KB) - Multi-map batch processing
- `AdtScanner.cs` (6KB) - Scans ADT for assets
- `WdtAlphaScanner.cs` (1KB) - Reads WDT metadata
- `ListfileLoader.cs` (2.4KB) - Loads listfiles
- `MultiListfileResolver.cs` (14KB) - Merges listfiles
- `UniqueIdClusterer.cs` (1.2KB) - Groups objects by UniqueID

**Purpose**: High-level workflow coordination.

---

### 5. **Output Generation** (2 files)
Generates viewer-ready assets.

**Files**:
- `CsvReportWriter.cs` (7KB) - Generates CSV reports
- `WebAssetsWriter.cs` (3KB) - **Viewer overlay generation**

**Purpose**: Create viewer overlay JSONs from terrain CSVs.

**Output**:
```
viewer/overlays/0.5.3.3368/DeadminesInstance/
├── terrain_complete/
│   └── tile_32_18.json    # Terrain properties overlay
└── shadow_map/
    ├── tile_32_18.json    # Shadow metadata
    └── shadow_32_18.png   # Shadow image
```

---

## Data Flow Pipeline

```
┌──────────────────┐
│  Alpha WDT File  │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ WdtAlphaScanner  │ ← Read MAIN table, MDNM, MONM
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  AdtScanner      │ ← Scan each tile
└────────┬─────────┘
         │
         ├─────────────────────┐
         ↓                     ↓
┌──────────────────┐  ┌──────────────────┐
│ TerrainExtractor │  │  ExportPipeline  │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ↓                     ↓
┌──────────────────┐  ┌──────────────────┐
│   Terrain CSV    │  │   LK ADT Files   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ↓
         ┌──────────────────┐
         │  CsvReportWriter │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ WebAssetsWriter  │ ← Consumed by WoWRollback
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ Viewer Overlays  │
         └──────────────────┘
```

---

## Critical Dependencies

### External Libraries
- **gillijimproject-csharp** - Format readers (WdtAlpha, AdtAlpha)
- **wow.tools.local** - DBC parsing (DBCD)
- **WoWDBDefs** - DBC definitions

### Listfiles
- **Community listfile** (required) - Asset name resolution
- **LK 3.x listfile** (optional) - Asset fixups

### DBCTool Integration
- **Crosswalk CSVs** (`alpha_to_335_explicit_map.csv`)
- **Decode CSVs** (`alpha_areaid_decode.csv`)
- **Suggestion CSVs** (`alpha_to_335_suggestions.csv`)

---

## Key Features

### 1. **Terrain Extraction**
Extracts per-chunk MCNK data for viewer visualization:
- Height maps (MCVT)
- Normal maps (MCNR)
- Alpha maps (MCAL)
- Liquid data (MCLQ)
- Terrain flags
- Holes
- AreaIDs

### 2. **Shadow Map Extraction**
Decodes MCSH shadow data and generates:
- Per-chunk shadow PNG images
- Shadow metadata JSON
- Shadow overview images

### 3. **AreaID Patching**
Applies DBCTool mappings to fix AreaIDs:
- Strict CSV-only patching (no heuristics)
- Per-source-map numeric mappings
- Writes `0` if no explicit mapping

### 4. **Asset Fixups**
Corrects asset paths using listfiles:
- M2 model paths
- WMO object paths
- BLP texture paths
- Case-sensitivity fixes

### 5. **Batch Processing**
Processes multiple maps in one run:
- Auto-discovers maps in directory
- Parallel tile processing (future)
- Progress reporting

---

## Integration with WoWRollback

### Current Integration (rebuild-and-regenerate.ps1)

```powershell
# Line 307: Calls AlphaWdtAnalyzer
$toolArgs = @(
    'run','--project',$alphaToolProject,'--',
    '--input', $wdtPath,
    '--listfile', $communityListfile,
    '--lk-listfile', $lkListfile,
    '--out', $analysisDir,
    '--export-adt',
    '--export-dir', $tempExportDir,
    '--extract-mcnk-terrain',    # Generate terrain CSV
    '--extract-mcnk-shadows',    # Generate shadow CSV
    '--no-web',
    '--profile', 'raw',
    '--no-fallbacks'
)
& dotnet @toolArgs

# Results:
# - cached_maps/analysis/0.5.3.3368/DeadminesInstance/csv/
# - cached_maps/0.5.3.3368/World/Maps/DeadminesInstance/ (LK ADTs)
```

### Data Handoff to WoWRollback

```
AlphaWdtAnalyzer Output:
├── csv/MapName/
│   ├── MapName_mcnk_terrain.csv
│   └── MapName_mcnk_shadows.csv
                  ↓
        (Copied by rebuild-and-regenerate.ps1)
                  ↓
rollback_outputs/0.5.3.3368/csv/MapName/
                  ↓
        (Read by WoWRollback.Cli)
                  ↓
WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs
                  ↓
        (Generates viewer overlays)
                  ↓
rollback_outputs/comparisons/*/viewer/overlays/
```

---

## Migration Strategy: High-Level Logic Only

### What to Migrate

**DO Migrate**:
- ✅ Terrain extraction logic → WoWRollback plugin
- ✅ Shadow extraction logic → WoWRollback plugin
- ✅ CSV report generation → WoWRollback service
- ✅ Viewer overlay generation → WoWRollback service
- ✅ Batch processing → WoWRollback CLI

**DON'T Migrate**:
- ❌ Format readers (keep in gillijimproject-csharp)
- ❌ LK ADT writing (keep as standalone tool initially)
- ❌ DBC parsing (use DBCD dependency)

---

## Phase 6 Migration Plan

### Week 9: Audit Complete ✓
- This document
- Decision: Migrate high-level logic, keep format readers

### Week 10-11: Plugin Architecture

**Create WoWRollback Plugins**:
```
WoWRollback.Plugins/
├── AlphaTerrainPlugin.cs       # Terrain extraction (from McnkTerrainExtractor)
├── AlphaShadowPlugin.cs        # Shadow extraction (from McnkShadowExtractor)
└── AlphaConverterPlugin.cs     # LK ADT export (from AdtExportPipeline)
```

**Create Services**:
```
WoWRollback.Core/Services/
├── TerrainExtractionService.cs  # Orchestrates terrain plugins
├── ViewerOverlayService.cs      # Generates overlay JSONs
└── BatchProcessingService.cs    # Multi-map processing
```

### Week 12-13: CLI Integration

**New WoWRollback.Cli Commands**:
```bash
# Convert Alpha WDT (new)
dotnet run --project WoWRollback.Cli -- convert-alpha \
  --input Azeroth.wdt \
  --listfile community.csv \
  --out output/ \
  --extract-terrain \
  --extract-shadows \
  --export-adt \
  --threads 8    # NEW: Multi-threading

# Old (still works via feature flag)
dotnet run --project AlphaWdtAnalyzer.Cli -- --input Azeroth.wdt ...
```

### Week 14-15: Validation

**Side-by-Side Testing**:
```powershell
# Old pipeline
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance
Get-FileHash rollback_outputs/**/*.json > old.txt

# New pipeline
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -UseNewConverter
Get-FileHash rollback_outputs/**/*.json > new.txt

# Must be identical
Compare-Object (Get-Content old.txt) (Get-Content new.txt)
```

**Feature Flag in rebuild-and-regenerate.ps1**:
```powershell
if ($UseNewConverter) {
    # WoWRollback.Cli convert-alpha
} else {
    # AlphaWdtAnalyzer.Cli (current)
}
```

---

## Code to Migrate (Prioritized)

### High Priority (Core Analysis)
1. **McnkTerrainExtractor.cs** (9KB) → `AlphaTerrainPlugin.cs`
2. **McnkShadowExtractor.cs** (7KB) → `AlphaShadowPlugin.cs`
3. **WebAssetsWriter.cs** (3KB) → `ViewerOverlayService.cs`
4. **AnalysisPipeline.cs** (6KB) → `BatchProcessingService.cs`

### Medium Priority (Export)
5. **AdtExportPipeline.cs** (28KB) → `AlphaConverterPlugin.cs`
6. **AreaIdMapper.cs** (17KB) → `AreaIdService.cs`
7. **AssetFixupPolicy.cs** (10KB) → `AssetFixupService.cs`

### Low Priority (Can Stay Standalone)
8. **AdtWotlkWriter.cs** (61KB) - Keep as standalone library
9. **DBC modules** (7 files) - Use DBCD directly
10. **Listfile resolution** - Reuse existing

---

## Multi-Threading Opportunities

**Current**: Single-threaded tile processing  
**Target**: Parallel tile processing

### Parallelization Points

1. **Per-Tile Analysis** (embarrassingly parallel)
```csharp
Parallel.ForEach(tiles, tile => {
    var terrain = terrainExtractor.Extract(tile);
    var shadow = shadowExtractor.Extract(tile);
});
```

2. **Per-Map Batch** (map-level parallelism)
```csharp
Parallel.ForEach(maps, map => {
    ProcessMap(map);
});
```

**Expected Speedup**: 6.5x (8 cores)

---

## Risks & Mitigations

### Risk 1: Breaking CSV Output Format
**Risk**: WoWRollback expects specific CSV schema  
**Mitigation**: SHA256 validation of CSV outputs

### Risk 2: Performance Regression
**Risk**: New code slower than old  
**Mitigation**: Benchmark before/after

### Risk 3: AreaID Patch Differences
**Risk**: New AreaID mapping differs from old  
**Mitigation**: Bit-identical validation required

### Risk 4: Missing Edge Cases
**Risk**: Old code handles cases new doesn't  
**Mitigation**: Comprehensive test suite with real Alpha WDTs

---

## Success Criteria

### Phase 6 Complete When:
- [ ] `WoWRollback.Cli convert-alpha` command works
- [ ] Terrain extraction SHA256 matches old
- [ ] Shadow extraction SHA256 matches old
- [ ] Viewer overlays SHA256 match old
- [ ] LK ADT exports SHA256 match old
- [ ] Multi-threading achieves 6x speedup
- [ ] rebuild-and-regenerate.ps1 uses new converter by default
- [ ] AlphaWdtAnalyzer archived to `_archived/`

---

## Deprecation Timeline

### Week 13: New Converter Default
```powershell
# Default switches to new
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance
# Uses WoWRollback.Cli convert-alpha

# Old available via flag
.\rebuild-and-regenerate.ps1 -UseOldConverter
```

### Week 14: Monitor for Issues
- 1 week of production use
- Zero regressions required

### Week 15: Archive Old Tool
```powershell
# Move to archive
mv AlphaWDTAnalysisTool _archived/AlphaWDTAnalysisTool
# Update docs
# Remove from build scripts
```

---

## Recommendations

### Short Term (Phase 6)
1. **Migrate high-level logic** - Terrain/shadow extraction
2. **Wrap as plugins** - Clean plugin interface
3. **Add multi-threading** - 6x speedup goal
4. **Side-by-side validation** - SHA256 all outputs

### Long Term (Post-Phase 6)
1. **Archive AlphaWdtAnalyzer** - After 2 weeks stable
2. **Performance tuning** - Optimize hot paths
3. **Test suite** - Unit + integration tests
4. **Documentation** - API docs for plugins

---

## Next Steps

1. **User Approval**: Review and approve this audit
2. **Phase 6 Week 10**: Create plugin interfaces
3. **Phase 6 Week 11**: Migrate terrain/shadow extractors
4. **Phase 6 Week 12**: Add multi-threading
5. **Phase 6 Week 13**: Validate and promote to default
6. **Phase 6 Week 15**: Archive old tool

---

**Status**: ✅ Complete, awaiting approval  
**Confidence**: High  
**Decision**: Migrate high-level logic, keep format readers in gillijimproject-csharp  
**Estimated Migration**: 6 weeks (Phase 6, Weeks 10-15)
