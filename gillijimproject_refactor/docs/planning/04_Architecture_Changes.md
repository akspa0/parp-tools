# Architecture Changes - Before & After 🏗️

Visual comparison of current vs target architecture.

---

## 📐 Current Architecture (Before)

```
┌──────────────────────────────────────────────────────┐
│                  User Request                        │
│  "Convert Alpha maps and generate viewer"           │
└─────────────────┬────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│         rebuild-and-regenerate.ps1                   │
│         (573 lines of orchestration)                 │
└─────┬────────────┬────────────┬──────────────────────┘
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│  Alpha   │ │  DBC     │ │  WoWRollback │
│  WDT     │ │  Tool    │ │              │
│  Tool    │ │  .V2     │ │  (Viewer)    │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │               │
     │ 45 min     │ 2 min         │ 5 min
     │ (2-8% CPU) │               │
     │            │               │
     ▼            ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ LK ADTs  │ │AreaTable │ │ Viewer JSONs │
│ + CSVs   │ │ Mappings │ │              │
└──────────┘ └──────────┘ └──────────────┘

Total: 52 minutes
```

### Problems
- ❌ 3 separate processes
- ❌ Complex PowerShell orchestration
- ❌ Sequential, single-threaded
- ❌ Poor error visibility
- ❌ Hard to debug
- ❌ Code duplication (WDT reading, listfiles)

---

## 🎯 Target Architecture (After)

```
┌──────────────────────────────────────────────────────┐
│                  User Request                        │
│  "dotnet run -- compare-versions ..."               │
└─────────────────┬────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│              WoWRollback.Cli                         │
│           (Beautiful Spectre.Console UI)             │
└─────────────────┬────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│           WoWRollback.Core                           │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Formats/   │  │ Processing/  │  │ Services/  │ │
│  │             │  │              │  │            │ │
│  │ Alpha/      │  │ MapConverter │  │ Comparison │ │
│  │ Lk/         │  │ Terrain      │  │ Viewer     │ │
│  │ Dbc/        │  │ Shadow       │  │ Listfile   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│                                                      │
│           ▲ Multi-threaded (8 threads) ▲            │
└───────────┼────────────────────────────┼────────────┘
            │                            │
            └──── Parallel.ForEachAsync ─┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
       ┌────────┐  ┌────────┐  ┌────────┐
       │ ADT 0  │  │ ADT 1  │  │ ADT N  │
       └───┬────┘  └───┬────┘  └───┬────┘
           │           │           │
           └───────────┴───────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ LK ADTs + CSVs   │
              │ AreaTable Maps   │
              │ Viewer JSONs     │
              └──────────────────┘

Total: 7-8 minutes (6.5x faster!)
```

### Benefits
- ✅ Single unified tool
- ✅ One command, no orchestration
- ✅ Multi-threaded (60-80% CPU usage)
- ✅ Beautiful progress bars
- ✅ Easy to debug (one process)
- ✅ Clean code, no duplication

---

## 📦 Package Structure Changes

### Before
```
parp-tools/gillijimproject_refactor/
├── AlphaWDTAnalysisTool/
│   ├── AlphaWdtAnalyzer.Core/
│   │   ├── WdtAlphaScanner.cs
│   │   ├── AdtScanner.cs
│   │   ├── Terrain/McnkTerrainExtractor.cs
│   │   └── Terrain/McnkShadowExtractor.cs
│   └── AlphaWdtAnalyzer.Cli/
│       └── Program.cs (complex arg parsing)
│
├── DBCTool.V2/
│   ├── DBCTool.V2.Core/
│   │   ├── Dbc/DbcFile.cs
│   │   └── AreaTable/AreaTableReader.cs
│   └── DBCTool.V2.Cli/
│       └── Program.cs
│
└── WoWRollback/
    ├── WoWRollback.Core/
    │   ├── Formats/Lk/          (LK only)
    │   └── Services/Viewer/
    └── WoWRollback.Cli/
        └── Commands/CompareVersionsCommand.cs
```

### After
```
parp-tools/gillijimproject_refactor/
├── _archived/                   ← Old tools (for reference)
│   ├── AlphaWDTAnalysisTool/
│   └── DBCTool.V2/
│
└── WoWRollback/                 ← Everything here!
    ├── WoWRollback.Core/
    │   ├── Formats/
    │   │   ├── Alpha/           ← NEW: Alpha WDT/ADT reading
    │   │   │   ├── WdtAlphaReader.cs
    │   │   │   ├── AdtAlphaReader.cs
    │   │   │   └── AdtAlphaConverter.cs
    │   │   ├── Lk/              ← ENHANCED: Add writer
    │   │   │   ├── AdtLkReader.cs
    │   │   │   ├── AdtLkWriter.cs
    │   │   │   └── WdtLkReader.cs
    │   │   └── Dbc/             ← NEW: DBC support
    │   │       ├── DbcReader.cs
    │   │       └── AreaTableReader.cs
    │   ├── Processing/          ← NEW: Multi-threaded pipelines
    │   │   ├── MapConverter.cs
    │   │   ├── TerrainExtractor.cs
    │   │   ├── ShadowExtractor.cs
    │   │   └── AreaTableProcessor.cs
    │   └── Services/
    │       ├── ListfileService.cs    ← NEW: Shared listfile
    │       ├── Comparison/           ← EXISTING
    │       └── Viewer/               ← EXISTING
    ├── WoWRollback.Cli/
    │   └── Commands/
    │       ├── ConvertMapCommand.cs       ← NEW
    │       ├── ExtractTerrainCommand.cs   ← NEW
    │       ├── ProcessAreaTableCommand.cs ← NEW
    │       └── CompareVersionsCommand.cs  ← ENHANCED
    └── WoWRollback.Tests/       ← NEW: Comprehensive tests
        ├── Formats/
        ├── Processing/
        └── Integration/
```

---

## 🔄 Data Flow Changes

### Before: Sequential Pipeline
```
Alpha WDT
    │
    ▼ [Tool 1: AlphaWDTAnalysisTool - 45 min]
    ├─> LK ADT files (128 tiles × ~20 sec each = 42 min)
    ├─> terrain CSV (extract from Alpha)
    └─> shadow CSV (extract from Alpha)
    │
    ▼ [Tool 2: DBCTool.V2 - 2 min]
    └─> AreaTable mappings
    │
    ▼ [Tool 3: WoWRollback - 5 min]
    ├─> Read LK ADTs + CSVs
    ├─> Generate viewer JSONs
    └─> Serve web viewer

Total: 52 minutes (sequential)
```

### After: Parallel Pipeline
```
Alpha WDT
    │
    ▼ [WoWRollback.Core.Processing.MapConverter]
    │
    ├─> Parallel.ForEachAsync(tiles, threads: 8)
    │   │
    │   ├─> Thread 1: Tiles 0-15   (1 min)
    │   ├─> Thread 2: Tiles 16-31  (1 min)
    │   ├─> Thread 3: Tiles 32-47  (1 min)
    │   ├─> Thread 4: Tiles 48-63  (1 min)
    │   ├─> Thread 5: Tiles 64-79  (1 min)
    │   ├─> Thread 6: Tiles 80-95  (1 min)
    │   ├─> Thread 7: Tiles 96-111 (1 min)
    │   └─> Thread 8: Tiles 112-127(1 min)
    │
    │   └─> All complete in ~7 min (vs 45 min!)
    │
    ├─> TerrainExtractor (parallel, <1 min)
    ├─> ShadowExtractor (parallel, <1 min)
    ├─> AreaTableProcessor (<1 min)
    └─> ViewerReportGenerator (<1 min)

Total: 7-8 minutes (6.5x faster!)
```

---

## 💻 Code Pattern Changes

### Before: Sequential Loop
```csharp
// AlphaWDTAnalysisTool - AdtScanner.cs
foreach (var adtNum in wdt.AdtNumbers)
{
    var adt = new AdtAlpha(wdt.WdtPath, adtNum, offset);
    
    // Process ADT (takes ~20 seconds)
    ProcessAdt(adt);
    
    // Next ADT... (20 more seconds)
}
// Total: 128 tiles × 20 sec = 42 min
```

**Problems**:
- One ADT at a time
- CPU mostly idle (2-8% usage)
- No progress visibility
- Errors hidden until end

### After: Parallel Processing
```csharp
// WoWRollback.Core - MapConverter.cs
var options = new ParallelOptions 
{ 
    MaxDegreeOfParallelism = 8,
    CancellationToken = ct
};

var results = new ConcurrentBag<AdtResult>();
var progress = 0;

await Parallel.ForEachAsync(wdtInfo.AdtTiles, options, 
    async (adtNum, token) =>
{
    try
    {
        var adt = await ProcessAdtAsync(adtNum, token);
        results.Add(new AdtResult { Success = true, AdtNum = adtNum });
        
        // Progress reporting
        var current = Interlocked.Increment(ref progress);
        if (current % 10 == 0)
        {
            _logger.LogInformation("Progress: {Current}/{Total}", 
                current, wdtInfo.AdtTiles.Count);
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Failed ADT {AdtNum}", adtNum);
        results.Add(new AdtResult { Success = false, Error = ex.Message });
    }
});

// Total: 128 tiles / 8 threads × 20 sec = ~5 min
// (Plus ~2 min overhead = 7 min total)
```

**Benefits**:
- ✅ 8 ADTs processed simultaneously
- ✅ High CPU usage (60-80%)
- ✅ Real-time progress updates
- ✅ Per-tile error handling
- ✅ Graceful cancellation support

---

## 🎨 CLI Changes

### Before: PowerShell Wrapper
```powershell
# rebuild-and-regenerate.ps1 (573 lines!)
param(
    [string[]]$Maps = @("Azeroth"),
    [string[]]$Versions = @("0.5.3.3368"),
    [string]$AlphaRoot = "..\test_data\",
    [switch]$RefreshCache,
    [switch]$Serve
)

# Step 1: Build solution (30 sec)
& dotnet build WoWRollback.sln

# Step 2: Run AlphaWDTAnalysisTool (45 min)
foreach ($map in $Maps) {
    foreach ($version in $Versions) {
        & dotnet run --project AlphaWdtAnalyzer.Cli -- `
            --input "$AlphaRoot\$version\$map.wdt" `
            --listfile "community-listfile.csv" `
            --out "cached_maps\$version\$map" `
            --export-adt `
            --extract-mcnk-terrain
    }
}

# Step 3: Run DBCTool.V2 (2 min)
& dotnet run --project DBCTool.V2.Cli -- ...

# Step 4: Run WoWRollback (5 min)
& dotnet run --project WoWRollback.Cli -- compare-versions ...

# Step 5: Serve viewer
if ($Serve) {
    python -m http.server 8080 --directory viewer
}
```

**Problems**:
- ❌ Complex orchestration
- ❌ No progress bars
- ❌ Errors easy to miss
- ❌ Hard to customize

### After: Single Command
```powershell
# One command does everything!
dotnet run --project WoWRollback.Cli -- compare-versions `
  --alpha-root ..\test_data `
  --versions 0.5.3.3368,0.5.5.3494 `
  --maps Azeroth,Kalimdor `
  --threads 8 `
  --viewer-report `
  --serve
```

**Output** (with Spectre.Console):
```
╭─────────────────────────────────────────────╮
│  WoWRollback - Alpha Map Converter          │
╰─────────────────────────────────────────────╯

[1/5] Reading Alpha WDTs...
  ✓ Azeroth.wdt (128 tiles)
  ✓ Kalimdor.wdt (140 tiles)

[2/5] Converting to LK format (8 threads)...
  Azeroth   ████████████████████ 100% | 128/128 | 7m 23s
  Kalimdor  ████████████████████ 100% | 140/140 | 8m 15s

[3/5] Extracting terrain data...
  ✓ Azeroth_mcnk_terrain.csv (2,048 chunks)
  ✓ Kalimdor_mcnk_terrain.csv (2,240 chunks)

[4/5] Processing AreaTables...
  ✓ Matched 342/350 Alpha → LK areas (97.7%)
  ⚠ Unmatched: DuskwoodTest, ElwynnPlaceholder

[5/5] Generating viewer...
  ✓ Created 268 overlay JSONs

╭─────────────────────────────────────────────╮
│  Conversion Complete!                       │
│  Time: 15m 38s (was 104m → 6.7x faster!)   │
│  Success: 268/268 tiles                     │
╰─────────────────────────────────────────────╯

Server starting: http://localhost:8080
Press Ctrl+C to stop...
```

**Benefits**:
- ✅ Beautiful progress bars
- ✅ Clear success/warning/error messages
- ✅ Real-time stats
- ✅ Single command
- ✅ Easy to understand

---

## 🎯 Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Tools** | 3 separate | 1 unified | -67% complexity |
| **Time** | 52 min | 7-8 min | 6.5x faster |
| **CPU** | 2-8% | 60-80% | 10x utilization |
| **Commands** | 1 PS script | 1 dotnet command | Simpler |
| **Progress** | Hidden | Beautiful UI | Better UX |
| **Errors** | Hard to find | Clear messages | Easier debug |
| **Tests** | Minimal | 90%+ coverage | More reliable |
| **Docs** | Scattered | Comprehensive | Easier onboard |

**Bottom Line**: Faster, simpler, more maintainable! 🚀
