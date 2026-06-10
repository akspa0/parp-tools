# Data Pipeline Consolidation Plan

## Problem Statement

Currently, WoW data processing is fragmented across three tools:
- **DBCTool.V2**: DBC reading, AreaTable extraction and mapping
- **AlphaWDTAnalysisTool**: WDT/ADT analysis, terrain extraction, coordinate conversion
- **WoWRollback**: Comparison, viewer generation, CSV transformation

This causes:
- ❌ Path inconsistencies (test_data vs rollback_outputs vs cached_maps)
- ❌ Duplicate functionality (DBC readers, CSV parsers)
- ❌ No single source of truth for processed data
- ❌ Difficult maintenance (changes require updates in 3 places)
- ❌ Version/map mismatches in viewer (wrong data loaded)

---

## Proposed Architecture: Single Data Library

### New Project Structure

```
WoWRollback.Data/          ← NEW shared library
├── Dbc/
│   ├── DbcReader.cs       (Unified DBC reader)
│   ├── AreaTableParser.cs (Replaces DBCTool.V2 logic)
│   ├── DbcModels.cs       (Area, Map, etc.)
│   └── AreaTableMapper.cs (Alpha ↔ LK mapping)
├── Terrain/
│   ├── McnkReader.cs      (From AlphaWDTAnalysisTool)
│   ├── McnkModels.cs      (Unified terrain models)
│   └── TerrainExtractor.cs
├── Csv/
│   ├── CsvWriter.cs       (Standardized CSV output)
│   ├── CsvReader.cs       (Standardized CSV input)
│   └── CsvFormats.cs      (Column definitions)
└── DataPaths.cs           (Centralized path management)

WoWRollback.Core/          ← Existing, refactored
├── Services/
│   ├── VersionComparisonService.cs (Uses WoWRollback.Data)
│   └── Viewer/
│       ├── ViewerReportWriter.cs   (Uses WoWRollback.Data)
│       └── (overlay builders)
└── Models/
    └── (comparison models only)

AlphaWDTAnalysisTool/      ← Refactored to use shared library
└── (WDT analysis specific code, uses WoWRollback.Data for DBC/CSV)

DBCTool.V2/                ← Can be deprecated or refactored
└── (DBC comparison/analysis, uses WoWRollback.Data)
```

---

## Centralized Path Management

### DataPaths.cs (NEW)

```csharp
public static class DataPaths
{
    // Input paths (read-only)
    public static string GetAlphaWdtPath(string alphaRoot, string version, string map);
    public static string GetAlphaDbcPath(string alphaRoot, string version);
    public static string GetLkDbcPath(string lkRoot);
    
    // Output paths (write)
    public static string GetVersionRoot(string outputRoot, string version);
    public static string GetMapRoot(string outputRoot, string version, string map);
    public static string GetCsvRoot(string outputRoot, string version, string map);
    public static string GetViewerRoot(string outputRoot, string comparisonKey);
    public static string GetOverlayRoot(string viewerRoot, string version, string map);
    
    // Standardized CSV paths
    public static string GetTerrainCsvPath(string outputRoot, string version, string map);
    public static string GetAreaTableCsvPath(string outputRoot, string version);
    public static string GetRangesCsvPath(string outputRoot, string version, string map);
}
```

### Standardized Output Structure

```
rollback_outputs/
├── {version}/                         ← Version-specific data
│   ├── AreaTable_Alpha.csv           ← From DBC extraction
│   ├── AreaTable_335.csv             ← From DBC extraction (if LK data available)
│   └── {map}/
│       ├── {map}_mcnk_terrain.csv    ← Terrain data
│       ├── {map}_mcnk_shadow.csv     ← Shadow maps
│       └── id_ranges_by_map_alpha_{map}.csv
└── comparisons/
    └── {comparison_key}/
        └── viewer/
            ├── overlays/{version}/{map}/
            │   ├── combined/
            │   ├── m2/
            │   ├── wmo/
            │   └── terrain_complete/     ← Generated from CSVs
            └── (HTML/JS assets)
```

---

## Phase 1: Create Shared Library (Immediate)

### 1.1: Create WoWRollback.Data Project

```bash
dotnet new classlib -n WoWRollback.Data -f net9.0
# Add to WoWRollback.sln
# Reference from WoWRollback.Core and AlphaWDTAnalysisTool
```

### 1.2: Move DBC Reading Logic

**Move from**: DBCTool.V2/AreaTableReader.cs  
**Move to**: WoWRollback.Data/Dbc/AreaTableParser.cs

- Unified CSV format (5 columns: row_key, id, parent, continentId, name)
- Extract both Alpha and LK AreaTables
- Store in standardized locations

### 1.3: Move Terrain Extraction Logic

**Move from**: AlphaWDTAnalysisTool/McnkTerrainExtractor.cs  
**Move to**: WoWRollback.Data/Terrain/McnkReader.cs

- Consolidate McnkTerrainEntry, McnkShadowEntry
- Standardize CSV output format

### 1.4: Centralize CSV I/O

**Create**: WoWRollback.Data/Csv/CsvWriter.cs, CsvReader.cs

- Single place for CSV serialization/deserialization
- Standardized error handling
- Type-safe column definitions

---

## Phase 2: Fix Viewer Data Loading (Next Step)

### 2.1: Ensure Correct Map/Version Matching

**Problem**: Viewer can load wrong overlay data if files exist but don't match

**Solution**: Add validation to overlay loaders:

```javascript
// overlayManager.js
loadVisibleOverlays(map, version) {
    const tileKeys = this.getVisibleTiles();
    for (const layer of this.enabledLayers) {
        for (const tileKey of tileKeys) {
            const url = `/overlays/${version}/${map}/terrain_complete/tile_r${row}_c${col}.json`;
            // VALIDATE response contains correct map/version before displaying
            this.loadAndValidate(url, map, version);
        }
    }
}

loadAndValidate(url, expectedMap, expectedVersion) {
    fetch(url).then(data => {
        if (data.map !== expectedMap || !data.layers.some(l => l.version === expectedVersion)) {
            console.warn(`Overlay data mismatch: expected ${expectedMap}/${expectedVersion}`);
            return null;
        }
        return data;
    });
}
```

### 2.2: Generate All Maps

**Update**: rebuild-and-regenerate.ps1

```powershell
# Default to all discovered maps
$Maps = @("auto")  # Auto-discover from AlphaRoot

# Extract terrain for all maps
foreach ($map in $discoveredMaps) {
    AlphaWdtAnalyzer --input "$alphaRoot\...\$map.wdt" `
        --extract-mcnk-terrain `
        --out "rollback_outputs\$version\$map\"
}
```

### 2.3: Copy AreaTable CSVs During Generation

**Update**: ViewerReportWriter.cs

```csharp
private static void GenerateTerrainOverlays(...)
{
    // Look for AreaTable in version root (not map-specific)
    var areaTablePath = DataPaths.GetAreaTableCsvPath(rootDirectory, version);
    
    if (!File.Exists(areaTablePath))
    {
        // Extract from DBC if not already done
        ExtractAreaTableFromDbc(rootDirectory, version);
    }
    
    var areaLookup = AreaTableReader.ReadAreaTableCsv(areaTablePath);
    // ... rest of overlay generation
}
```

---

## Phase 3: Deprecate Duplicate Code (Future)

### 3.1: Refactor DBCTool.V2

- Keep: Comparison/analysis UI
- Remove: DBC readers (use WoWRollback.Data)
- Remove: CSV writers (use WoWRollback.Data)

### 3.2: Refactor AlphaWDTAnalysisTool

- Keep: WDT/ADT format parsing
- Remove: DBC readers (use WoWRollback.Data)
- Remove: McnkExtractor (use WoWRollback.Data)
- Remove: CSV writers (use WoWRollback.Data)

### 3.3: Benefits

- ✅ Single source of truth for data formats
- ✅ Consistent paths everywhere
- ✅ Easier testing (mock DataPaths)
- ✅ No duplicate code
- ✅ Easier to add new DBC tables (CharacterRaces, CreatureDisplayInfo, etc.)

---

## Immediate Action Items (This Session)

### 1. Fix AreaTable Reading ✅ DONE
- Updated AreaTableReader to parse 5-column CSV format
- Handles commas in names properly

### 2. Update Script to Copy AreaTable CSVs
- Modify rebuild-and-regenerate.ps1 to copy AreaTable from DBCTool output

### 3. Test End-to-End
- Extract terrain data for Azeroth
- Generate viewer with correct area names
- Verify overlays load/unload correctly per map

### 4. Document Current State
- Create migration guide for other developers
- Document temporary paths until consolidation complete

---

## Timeline

### Week 1 (Current)
- ✅ Fix immediate AreaTable issue
- ⏳ Test multi-map viewer loading
- ⏳ Document temporary workflow

### Week 2
- Create WoWRollback.Data project
- Move AreaTableParser
- Move McnkReader
- Update references

### Week 3
- Create DataPaths utility
- Update all tools to use DataPaths
- Standardize CSV formats

### Week 4
- Refactor DBCTool.V2 to use shared library
- Refactor AlphaWDTAnalysisTool to use shared library
- Integration testing

---

## Success Criteria

✅ **Single source of truth**: One library for all DBC/CSV/terrain data
✅ **Consistent paths**: DataPaths utility used everywhere
✅ **Correct viewer data**: Map/version validation prevents mismatches
✅ **All maps supported**: Viewer can load any map with terrain data
✅ **Maintainable**: Changes in one place propagate everywhere
✅ **Testable**: Mock DataPaths for unit tests

---

## Notes

- This is a significant refactoring but will pay off long-term
- Can be done incrementally without breaking existing functionality
- Improves code quality and reduces bugs
- Makes adding new features (CreatureDisplayInfo overlays, etc.) much easier
