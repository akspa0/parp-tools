# Viewer Pipeline Improvements - Plan

## Status: 📋 Planning Phase

Major improvements to make the WoWRollback pipeline easier to use and maintain.

---

## Goals

1. ✅ Make `--prefer-raw` the default (no flag needed)
2. ✅ Auto-discover all Alpha WDT files from test_data structure
3. ✅ Match Map.dbc data for proper minimap generation
4. ✅ Update README with clear examples
5. ✅ Refactor into single unified codebase
6. ✅ Enforce 750 LOC limit per module

---

## Problem 1: `--prefer-raw` Should Be Default

### Current State:
```bash
# Users must remember to add --prefer-raw flag
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth") -Versions @("0.5.3.3368") --prefer-raw
```

### Proposed:
```bash
# --prefer-raw is the default, add --use-converted if you want transformed coordinates
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth") -Versions @("0.5.3.3368")
```

### Changes Needed:
1. **AlphaWdtAnalyzer.Cli/Program.cs**: Default `--profile raw` instead of `modified`
2. **rebuild-and-regenerate.ps1**: Remove `--profile modified`, add `--profile raw` by default
3. **WoWRollback.Cli/Program.cs**: Document that raw coordinates are standard

### Rationale:
- ✅ Raw coordinates match original game client
- ✅ No confusing transformations
- ✅ Better debugging (matches wow.tools)
- ✅ Simpler mental model

---

## Problem 2: Auto-Discover All Maps from test_data

### Current State:
```bash
# Must manually specify maps
-Maps @("Azeroth","Kalimdor","Kalidar","DeadminesInstance","Shadowfang")
```

### Proposed:
```bash
# Discovers all maps automatically
-Maps @("auto")  # or omit entirely for default

# Still allows manual override
-Maps @("Azeroth","Kalimdor")  # Only process these
```

### Discovery Logic:

```powershell
function Get-AllAlphaMapsFromTestData {
    param([string]$AlphaRoot, [string[]]$Versions)
    
    $discovered = @{}
    
    foreach ($version in $Versions) {
        # Pattern: test_data/0.5.3/tree/World/Maps/<MapName>/<MapName>.wdt
        $versionPaths = @(
            Join-Path $AlphaRoot "$version\tree\World\Maps",
            Join-Path $AlphaRoot "$version\World\Maps"
        )
        
        foreach ($path in $versionPaths) {
            if (Test-Path $path) {
                Get-ChildItem -Path $path -Directory | ForEach-Object {
                    $mapName = $_.Name
                    $wdtPath = Join-Path $_.FullName "$mapName.wdt"
                    
                    if (Test-Path $wdtPath) {
                        if (!$discovered.ContainsKey($mapName)) {
                            $discovered[$mapName] = @()
                        }
                        $discovered[$mapName] += @{
                            Version = $version
                            WdtPath = $wdtPath
                        }
                    }
                }
            }
        }
    }
    
    return $discovered
}
```

### Benefits:
- ✅ No manual map list maintenance
- ✅ Discovers all available data
- ✅ Scales to hundreds of maps
- ✅ Never miss a map

---

## Problem 3: Match Map.dbc for Proper Minimap Generation

### Current Issue:
Minimaps may not match map names/IDs correctly because we don't validate against Map.dbc

### Proposed Solution:

**Phase 1: Extract Map.dbc Data**
```bash
cd DBCTool.V2
# Extract Map.dbc alongside AreaTable.dbc
# Output: dbctool_outputs/session_X/compare/v2/Map_dump_0.5.5.csv
```

**Phase 2: Match During Processing**
```csharp
public class MapValidator
{
    private Dictionary<string, MapDbcEntry> _mapsByName;
    
    public void ValidateMap(string mapName, int tileRow, int tileCol)
    {
        if (!_mapsByName.TryGetValue(mapName, out var mapEntry))
        {
            Console.WriteLine($"[warn] Map '{mapName}' not found in Map.dbc");
            return;
        }
        
        // Validate tile is within map bounds
        if (tileRow < 0 || tileRow >= 64 || tileCol < 0 || tileCol >= 64)
        {
            Console.WriteLine($"[warn] Tile [{tileRow},{tileCol}] out of bounds for {mapName}");
        }
        
        // Check if this is an instance map (should have different handling)
        if (mapEntry.IsInstance && (tileRow != 0 || tileCol != 0))
        {
            Console.WriteLine($"[warn] Instance map {mapName} should only have tile [0,0]");
        }
    }
}
```

**Map.dbc CSV Format**:
```csv
row_key,id,directory,instanceType,mapType,name
1,0,Azeroth,0,0,Eastern Kingdoms
2,1,Kalimdor,0,0,Kalimdor
3,13,test,0,0,Testing
4,17,Kalidar,0,0,Kalidar
5,30,PVPZone01,0,0,Alterac Valley
6,33,Shadowfang,1,1,Shadowfang Keep
7,34,StormwindJail,1,1,The Stockade
```

**Integration Points**:
1. Copy Map.dbc CSV during rebuild-and-regenerate.ps1
2. Load Map.dbc in ViewerReportWriter
3. Validate map names before generating minimaps
4. Log warnings for mismatches

---

## Problem 4: Update README with Clear Examples

### Current README Issues:
- ❌ Too technical, assumes prior knowledge
- ❌ Missing common workflows
- ❌ No troubleshooting section
- ❌ Doesn't explain test_data structure

### Proposed README Structure:

```markdown
# WoWRollback Viewer

Visual comparison tool for World of Warcraft Alpha versions.

## Quick Start (5 Minutes)

### 1. Organize Your Data
```
test_data/
├── 0.5.3/
│   └── tree/
│       └── World/
│           ├── Maps/
│           │   ├── Azeroth/Azeroth.wdt
│           │   ├── Kalimdor/Kalimdor.wdt
│           │   └── ...
│           └── Minimaps/
│               ├── Azeroth/
│               └── Kalimdor/
└── 0.5.5/
    └── (same structure)
```

### 2. Generate DBCs (One-Time Setup)
```bash
cd DBCTool.V2
# Extract AreaTable and Map DBCs
# (specific command here)
```

### 3. Generate Viewer
```bash
cd WoWRollback
.\rebuild-and-regenerate.ps1 `
  -Versions @("0.5.3.3368","0.5.5.3494") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

Open http://localhost:8080 and explore!

## Common Workflows

### Generate Specific Maps Only
```bash
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth","Kalimdor") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\
```

### Refresh Cached Maps
```bash
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth") `
  -Versions @("0.5.3.3368") `
  -RefreshCache `
  -AlphaRoot ..\test_data\
```

### Compare Two Versions
```bash
.\rebuild-and-regenerate.ps1 `
  -Versions @("0.5.3.3368","0.5.5.3494") `
  -AlphaRoot ..\test_data\
```

## Troubleshooting

### "Map not found in Map.dbc"
- Make sure DBCTool.V2 has generated Map.dbc CSVs
- Check that map name matches directory name exactly

### "No terrain CSV found for X"
- Run AlphaWdtAnalyzer with --extract-mcnk-terrain flag
- Or let rebuild-and-regenerate.ps1 do it automatically

### Minimaps not appearing
- Verify minimaps exist in test_data/<version>/tree/World/Minimaps/<map>/
- Check file naming: <map>_<col>_<row>.blp

## Architecture

See [docs/architecture/](docs/architecture/) for detailed documentation.
```

---

## Problem 5: Consolidate Under One Roof

### Current Structure (Fragmented):
```
parp-tools/
├── AlphaWDTAnalysisTool/        (WDT/ADT analysis)
├── DBCTool.V2/                  (DBC reading)
└── WoWRollback/                 (Comparison + Viewer)
    ├── WoWRollback.Core/
    ├── WoWRollback.Cli/
    └── ViewerAssets/
```

### Proposed Structure (Unified):
```
WoWRollback/
├── WoWRollback.Data/            ← NEW shared library
│   ├── Dbc/
│   │   ├── DbcReader.cs
│   │   ├── AreaTableParser.cs
│   │   └── MapTableParser.cs
│   ├── Wdt/
│   │   ├── WdtReader.cs
│   │   └── AdtReader.cs
│   ├── Terrain/
│   │   ├── McnkExtractor.cs
│   │   └── MinimapReader.cs
│   └── Csv/
│       ├── CsvWriter.cs
│       └── CsvReader.cs
├── WoWRollback.Core/            (Comparison logic)
├── WoWRollback.Cli/             (CLI interface)
├── WoWRollback.Tools/           ← NEW (consolidates AlphaWDT + DBC tools)
│   ├── WdtAnalyzer.cs
│   ├── DbcComparer.cs
│   └── BatchProcessor.cs
└── ViewerAssets/                (Web UI)
```

### Migration Plan:

**Phase 1: Create WoWRollback.Data**
- Move DBC readers from DBCTool.V2
- Move WDT/ADT readers from AlphaWDTAnalysisTool
- Move CSV I/O logic
- Add DataPaths utility

**Phase 2: Refactor Existing Projects**
- AlphaWDTAnalysisTool → depends on WoWRollback.Data
- DBCTool.V2 → depends on WoWRollback.Data
- WoWRollback.Core → depends on WoWRollback.Data

**Phase 3: Merge Tool CLIs**
- Combine AlphaWdtAnalyzer.Cli and DBCTool.V2 into WoWRollback.Tools
- Single entry point for all tools
- Consistent CLI interface

**Phase 4: Deprecate Old Projects**
- Mark AlphaWDTAnalysisTool as legacy
- Mark DBCTool.V2 as legacy
- Update all documentation

---

## Problem 6: Enforce 750 LOC Limit Per Module

### Current Large Files (>750 LOC):

```bash
# Find files over 750 lines
Get-ChildItem -Path WoWRollback -Recurse -Filter *.cs | 
    Where-Object { (Get-Content $_.FullName).Count -gt 750 } |
    Select-Object FullName, @{Name='Lines';Expression={(Get-Content $_.FullName).Count}}
```

### Strategy for Splitting:

**Example: ViewerReportWriter.cs (~500 LOC, but could grow)**

**Before** (single file):
```csharp
public static class ViewerReportWriter
{
    public static string Generate(...) { }
    public static void WriteIndexJson(...) { }
    public static void WriteConfigJson(...) { }
    private static void GenerateTerrainOverlays(...) { }
    private static void CopyViewerAssets(...) { }
    // ... 500+ lines
}
```

**After** (split into modules):
```csharp
// ViewerReportWriter.cs (main orchestrator, <100 LOC)
public static class ViewerReportWriter
{
    public static string Generate(...)
    {
        var paths = ViewerPaths.Setup(...);
        var minimaps = MinimapGenerator.Generate(...);
        var overlays = OverlayGenerator.Generate(...);
        var index = IndexWriter.Write(...);
        var config = ConfigWriter.Write(...);
        AssetCopier.Copy(...);
        return paths.ViewerRoot;
    }
}

// ViewerPaths.cs (<150 LOC)
// MinimapGenerator.cs (<200 LOC)
// OverlayGenerator.cs (<250 LOC)
// IndexWriter.cs (<100 LOC)
// ConfigWriter.cs (<100 LOC)
// AssetCopier.cs (<100 LOC)
```

### Enforcement Strategy:

**Add to .editorconfig**:
```ini
[*.cs]
# Warn if file exceeds 750 lines
dotnet_diagnostic.CA1506.severity = warning
file_length_limit = 750
```

**Add Pre-Commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit

find . -name "*.cs" | while read file; do
    lines=$(wc -l < "$file")
    if [ $lines -gt 750 ]; then
        echo "ERROR: $file has $lines lines (limit: 750)"
        exit 1
    fi
done
```

**CI/CD Check**:
```yaml
# .github/workflows/ci.yml
- name: Check file sizes
  run: |
    find . -name "*.cs" -exec wc -l {} + | awk '$1 > 750 {print "File too large: "$2" ("$1" lines)"; exit 1}'
```

---

## Implementation Timeline

### Week 1: Defaults & Discovery
- [x] Make --prefer-raw default
- [x] Implement auto-discovery from test_data
- [x] Update rebuild-and-regenerate.ps1
- [x] Test with multiple versions/maps

### Week 2: Map.dbc Integration
- [ ] Extract Map.dbc with DBCTool.V2
- [ ] Create MapTableParser
- [ ] Add validation to ViewerReportWriter
- [ ] Update README with Map.dbc workflow

### Week 3: README & Examples
- [ ] Rewrite README with Quick Start
- [ ] Add common workflows section
- [ ] Add troubleshooting guide
- [ ] Add test_data structure diagram

### Week 4: Begin Consolidation
- [ ] Create WoWRollback.Data project
- [ ] Move DBC readers
- [ ] Move WDT/ADT readers
- [ ] Update references

### Week 5-6: Complete Consolidation
- [ ] Merge tool CLIs
- [ ] Split large files (>750 LOC)
- [ ] Add file size checks
- [ ] Update all documentation

---

## Success Criteria

✅ **Usability**:
- User runs one command: `.\rebuild-and-regenerate.ps1 -AlphaRoot ..\test_data\ -Serve`
- All maps discovered automatically
- Minimaps generated correctly
- Area names displayed properly

✅ **Maintainability**:
- Single codebase (no AlphaWDT/DBCTool duplication)
- All modules <750 LOC
- Clear separation of concerns
- Comprehensive README

✅ **Quality**:
- Map.dbc validation catches errors
- Raw coordinates by default
- No manual configuration needed
- Works out-of-the-box

---

## Open Questions

1. **Map.dbc Format**: What's the exact CSV structure from DBCTool.V2?
2. **Instance Maps**: How should we handle instance maps differently?
3. **Minimap Fallback**: What if minimap doesn't exist? Generate placeholder?
4. **Tool CLI Merge**: Keep separate executables or single entry point?
5. **Backward Compatibility**: Support old --profile modified flag?

---

## Next Steps

**Immediate** (this session):
1. Make --prefer-raw default
2. Implement auto-discovery
3. Update README Quick Start section

**Short-term** (next session):
1. Map.dbc integration
2. Validation logic
3. Complete README

**Long-term** (future sessions):
1. Create WoWRollback.Data
2. Consolidate codebases
3. Enforce file size limits

---

## Summary

This plan transforms WoWRollback from a collection of loosely-coupled tools into a **unified, easy-to-use pipeline** with:
- ✅ Sensible defaults (raw coordinates)
- ✅ Auto-discovery (no manual map lists)
- ✅ Validation (Map.dbc integration)
- ✅ Great docs (README with examples)
- ✅ Clean code (single codebase, <750 LOC modules)

Ready to start with **Phase 1: Defaults & Discovery**! 🚀
