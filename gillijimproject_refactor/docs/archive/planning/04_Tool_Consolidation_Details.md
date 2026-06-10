# WoWRollback Consolidation Plan 🎯

**Goal**: Consolidate AlphaWDTAnalysisTool + DBCTool.V2 functionality into WoWRollback with multi-threading optimizations.

---

## Current Architecture Problems

### 1. Tool Fragmentation
```
AlphaWDTAnalysisTool     → Reads Alpha WDTs, converts to LK ADTs
       ↓
DBCTool.V2               → Processes AreaTable DBCs
       ↓
WoWRollback              → Compares versions, generates viewer
```

**Problems**:
- ❌ 3 separate tools, 3 separate processes
- ❌ Slow (sequential, single-threaded)
- ❌ Complex PowerShell orchestration
- ❌ Hard to debug/maintain
- ❌ Duplicate code (WDT reading, listfile handling)

### 2. Performance Issues
- **AlphaWDTAnalysisTool**: 2-8% CPU usage (sequential)
- **DBCTool.V2**: Single-threaded DBC processing
- **Total time**: 45-60 minutes for large maps

---

## Target Architecture

```
WoWRollback.Core
├── Formats/
│   ├── Alpha/
│   │   ├── WdtAlphaReader.cs        ← From AlphaWDTAnalysisTool
│   │   ├── AdtAlphaReader.cs        ←
│   │   └── AdtAlphaConverter.cs     ← New: Alpha → LK conversion
│   ├── Lk/
│   │   ├── AdtLkReader.cs           ← Existing
│   │   └── AdtLkWriter.cs           ← New: Write LK ADTs
│   └── Dbc/
│       ├── DbcReader.cs             ← From DBCTool.V2
│       ├── AreaTableReader.cs       ←
│       └── AreaTableProcessor.cs    ← Area ID matching logic
├── Processing/
│   ├── MapConverter.cs              ← Orchestrates Alpha → LK pipeline
│   ├── TerrainExtractor.cs          ← MCNK terrain to CSV (multi-threaded)
│   ├── ShadowExtractor.cs           ← MCSH shadows to CSV (multi-threaded)
│   └── AreaTableMatcher.cs          ← Match Alpha AreaIDs to LK
└── Services/
    ├── Comparison/
    │   └── VersionComparisonService.cs  ← Existing
    └── Viewer/
        └── ViewerReportGenerator.cs      ← Existing

WoWRollback.Cli
├── Commands/
│   ├── ConvertMapCommand.cs         ← New: Alpha WDT → LK ADTs
│   ├── ExtractTerrainCommand.cs     ← New: ADT → terrain CSV
│   ├── ProcessAreaTableCommand.cs   ← New: DBC → AreaTable CSV
│   └── CompareVersionsCommand.cs    ← Existing (enhanced)
└── Program.cs
```

**Benefits**:
- ✅ Single tool, single process
- ✅ Multi-threaded (6-7x faster!)
- ✅ Unified error handling
- ✅ Better code reuse
- ✅ Easier testing

---

## Migration Strategy

### Phase 1: Core Format Readers (Foundation)
**Goal**: Move file format reading into WoWRollback.Core

#### 1.1 Alpha WDT/ADT Reading
**Source**: `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/`

**Files to migrate**:
```
WdtAlphaScanner.cs         → WoWRollback.Core/Formats/Alpha/WdtAlphaReader.cs
AdtScanner.cs              → WoWRollback.Core/Formats/Alpha/AdtAlphaReader.cs
```

**Dependencies**:
- `GillijimProject.WowFiles.Alpha` (keep as-is, it's stable)
- Listfile loading → Move to `WoWRollback.Core/Services/ListfileService.cs`

**Refactoring**:
```csharp
// Old (AlphaWDTAnalysisTool)
public class WdtAlphaScanner
{
    public List<int> AdtNumbers { get; }
    // ... constructor loads WDT immediately
}

// New (WoWRollback.Core)
public class WdtAlphaReader
{
    public static WdtInfo ReadWdt(string wdtPath)
    {
        var wdt = new WdtAlpha(wdtPath);
        return new WdtInfo
        {
            MapName = Path.GetFileNameWithoutExtension(wdtPath),
            AdtTiles = wdt.GetExistingAdtsNumbers(),
            AdtOffsets = wdt.GetAdtOffsetsInMain(),
            DoodadFiles = wdt.GetMdnmFileNames(),
            ObjectFiles = wdt.GetMonmFileNames()
        };
    }
}

public record WdtInfo
{
    public string MapName { get; init; }
    public List<int> AdtTiles { get; init; }
    public List<int> AdtOffsets { get; init; }
    public List<string> DoodadFiles { get; init; }
    public List<string> ObjectFiles { get; init; }
}
```

#### 1.2 LK ADT Writing
**New functionality** - currently AlphaWDTAnalysisTool writes, but not cleanly.

```csharp
namespace WoWRollback.Core.Formats.Lk;

public class AdtLkWriter
{
    public static async Task WriteAdtAsync(string outputPath, AdtLkData data)
    {
        // Write MVER, MHDR, MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, MH2O, MCNK chunks
        // Properly formatted for 3.3.5a client
    }
}

public record AdtLkData
{
    public List<McnkChunk> Chunks { get; init; }
    public List<string> Textures { get; init; }
    public List<string> Doodads { get; init; }
    public List<string> WorldObjects { get; init; }
    // ... all chunk data
}
```

#### 1.3 DBC Reading
**Source**: `DBCTool.V2.Core/`

**Files to migrate**:
```
Dbc/DbcFile.cs             → WoWRollback.Core/Formats/Dbc/DbcReader.cs
AreaTable/AreaTable.cs     → WoWRollback.Core/Formats/Dbc/AreaTableReader.cs
```

**Refactoring**:
```csharp
namespace WoWRollback.Core.Formats.Dbc;

public class AreaTableReader
{
    public static List<AreaTableEntry> ReadAreaTable(string dbcPath, string version)
    {
        // Use DBCD.IO for proper reading
        // Return structured AreaTable entries
    }
}

public record AreaTableEntry
{
    public uint ID { get; init; }
    public uint ParentAreaID { get; init; }
    public uint AreaBit { get; init; }
    public string AreaName { get; init; }
    // ... other fields
}
```

---

### Phase 2: Processing Pipeline (Multi-Threaded!)
**Goal**: Implement conversion and extraction with parallelization

#### 2.1 MapConverter (Alpha → LK Pipeline)
**New file**: `WoWRollback.Core/Processing/MapConverter.cs`

```csharp
namespace WoWRollback.Core.Processing;

public class MapConverter
{
    private readonly ILogger<MapConverter> _logger;
    private readonly ListfileService _listfileService;
    
    public async Task<ConversionResult> ConvertMapAsync(
        string alphaWdtPath,
        string outputDirectory,
        ConversionOptions options,
        CancellationToken ct = default)
    {
        _logger.LogInformation("Reading Alpha WDT: {Path}", alphaWdtPath);
        var wdtInfo = WdtAlphaReader.ReadWdt(alphaWdtPath);
        
        _logger.LogInformation("Converting {Count} ADT tiles (using {Threads} threads)", 
            wdtInfo.AdtTiles.Count, options.MaxThreads);
        
        var results = new ConcurrentBag<AdtConversionResult>();
        var progress = 0;
        
        var parallelOptions = new ParallelOptions 
        { 
            MaxDegreeOfParallelism = options.MaxThreads,
            CancellationToken = ct
        };
        
        await Parallel.ForEachAsync(wdtInfo.AdtTiles, parallelOptions, async (adtNum, token) =>
        {
            try
            {
                // Read Alpha ADT
                var alphaAdt = AdtAlphaReader.ReadAdt(alphaWdtPath, adtNum, wdtInfo.AdtOffsets[adtNum]);
                
                // Convert to LK format
                var lkAdt = AdtAlphaConverter.ConvertToLk(alphaAdt, _listfileService);
                
                // Write LK ADT
                var outputPath = Path.Combine(outputDirectory, 
                    $"{wdtInfo.MapName}_{adtNum / 64}_{adtNum % 64}.adt");
                await AdtLkWriter.WriteAdtAsync(outputPath, lkAdt, token);
                
                results.Add(new AdtConversionResult { AdtNum = adtNum, Success = true });
                
                // Progress reporting
                var current = Interlocked.Increment(ref progress);
                if (current % 10 == 0 || current == wdtInfo.AdtTiles.Count)
                {
                    _logger.LogInformation("Progress: {Current}/{Total} ADTs converted", 
                        current, wdtInfo.AdtTiles.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to convert ADT {AdtNum}", adtNum);
                results.Add(new AdtConversionResult { AdtNum = adtNum, Success = false, Error = ex.Message });
            }
        });
        
        return new ConversionResult
        {
            MapName = wdtInfo.MapName,
            TotalTiles = wdtInfo.AdtTiles.Count,
            SuccessfulTiles = results.Count(r => r.Success),
            FailedTiles = results.Count(r => !r.Success),
            Errors = results.Where(r => !r.Success).Select(r => r.Error).ToList()
        };
    }
}

public record ConversionOptions
{
    public int MaxThreads { get; init; } = Environment.ProcessorCount;
    public bool ExtractTerrain { get; init; } = true;
    public bool ExtractShadows { get; init; } = true;
    public bool GenerateAreaTable { get; init; } = true;
}
```

#### 2.2 TerrainExtractor (Multi-Threaded)
**New file**: `WoWRollback.Core/Processing/TerrainExtractor.cs`

```csharp
public class TerrainExtractor
{
    private readonly SemaphoreSlim _csvLock = new(1, 1);
    
    public async Task ExtractTerrainAsync(
        string wdtPath,
        string outputCsvPath,
        int maxThreads = 0,
        CancellationToken ct = default)
    {
        var wdtInfo = WdtAlphaReader.ReadWdt(wdtPath);
        var csvRows = new ConcurrentBag<string>();
        
        // Write CSV header
        await File.WriteAllTextAsync(outputCsvPath, 
            "Map,TileX,TileY,ChunkX,ChunkY,AreaID,Flags,Holes,LiquidType,LiquidLevel,LiquidFlags\n", 
            ct);
        
        var parallelOptions = new ParallelOptions 
        { 
            MaxDegreeOfParallelism = maxThreads > 0 ? maxThreads : Environment.ProcessorCount,
            CancellationToken = ct
        };
        
        await Parallel.ForEachAsync(wdtInfo.AdtTiles, parallelOptions, async (adtNum, token) =>
        {
            var alphaAdt = AdtAlphaReader.ReadAdt(wdtPath, adtNum, wdtInfo.AdtOffsets[adtNum]);
            var rows = ExtractTerrainFromAdt(alphaAdt, adtNum, wdtInfo.MapName);
            
            foreach (var row in rows)
                csvRows.Add(row);
        });
        
        // Write all rows at once (faster than synchronous appending)
        await File.AppendAllLinesAsync(outputCsvPath, csvRows, ct);
    }
    
    private List<string> ExtractTerrainFromAdt(AdtAlphaData adt, int adtNum, string mapName)
    {
        var rows = new List<string>();
        var tileX = adtNum % 64;
        var tileY = adtNum / 64;
        
        foreach (var chunk in adt.Chunks)
        {
            rows.Add($"{mapName},{tileX},{tileY},{chunk.X},{chunk.Y}," +
                     $"{chunk.AreaId},{chunk.Flags},{chunk.Holes}," +
                     $"{chunk.LiquidType},{chunk.LiquidLevel},{chunk.LiquidFlags}");
        }
        
        return rows;
    }
}
```

#### 2.3 ShadowExtractor (Multi-Threaded)
Similar pattern to TerrainExtractor, but for MCSH data.

#### 2.4 AreaTableProcessor
**New file**: `WoWRollback.Core/Processing/AreaTableProcessor.cs`

```csharp
public class AreaTableProcessor
{
    public async Task<AreaTableMapping> ProcessAreaTablesAsync(
        string alphaDbcPath,
        string lkDbcPath,
        CancellationToken ct = default)
    {
        var alphaAreas = AreaTableReader.ReadAreaTable(alphaDbcPath, "0.5.3");
        var lkAreas = AreaTableReader.ReadAreaTable(lkDbcPath, "3.3.5");
        
        // Match Alpha → LK using name/hierarchy matching
        var mappings = MatchAreaTables(alphaAreas, lkAreas);
        
        return new AreaTableMapping
        {
            AlphaVersion = "0.5.3",
            LkVersion = "3.3.5",
            Mappings = mappings
        };
    }
    
    private List<AreaMapping> MatchAreaTables(
        List<AreaTableEntry> alpha, 
        List<AreaTableEntry> lk)
    {
        // Smart matching logic from DBCTool.V2
        // But cleaner and more maintainable!
    }
}
```

---

### Phase 3: CLI Integration
**Goal**: Expose all functionality through clean CLI commands

#### 3.1 ConvertMapCommand
**New file**: `WoWRollback.Cli/Commands/ConvertMapCommand.cs`

```csharp
[Command("convert-map", Description = "Convert Alpha WDT to LK ADTs")]
public class ConvertMapCommand : AsyncCommand<ConvertMapCommand.Settings>
{
    public class Settings : CommandSettings
    {
        [CommandArgument(0, "<input-wdt>")]
        public string InputWdt { get; set; }
        
        [CommandOption("--output|-o")]
        [DefaultValue("./output")]
        public string OutputDirectory { get; set; }
        
        [CommandOption("--threads|-t")]
        [DefaultValue(0)]
        public int MaxThreads { get; set; }
        
        [CommandOption("--extract-terrain")]
        public bool ExtractTerrain { get; set; }
        
        [CommandOption("--extract-shadows")]
        public bool ExtractShadows { get; set; }
    }
    
    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        var converter = new MapConverter(logger, listfileService);
        
        var options = new ConversionOptions
        {
            MaxThreads = settings.MaxThreads > 0 ? settings.MaxThreads : Environment.ProcessorCount,
            ExtractTerrain = settings.ExtractTerrain,
            ExtractShadows = settings.ExtractShadows
        };
        
        var result = await converter.ConvertMapAsync(
            settings.InputWdt, 
            settings.OutputDirectory, 
            options);
        
        AnsiConsole.MarkupLine($"[green]✓[/] Converted {result.SuccessfulTiles}/{result.TotalTiles} tiles");
        
        if (result.FailedTiles > 0)
        {
            AnsiConsole.MarkupLine($"[red]✗[/] Failed: {result.FailedTiles} tiles");
            foreach (var error in result.Errors)
                AnsiConsole.MarkupLine($"  [red]•[/] {error}");
            return 1;
        }
        
        return 0;
    }
}
```

**Usage**:
```powershell
# Convert single map (multi-threaded)
dotnet run --project WoWRollback.Cli -- convert-map \
  path/to/Azeroth.wdt \
  --output cached_maps/0.5.3.3368/World/Maps/Azeroth \
  --extract-terrain \
  --extract-shadows \
  --threads 8

# Output:
# [1/3] Reading WDT: Azeroth (128 tiles)
# [2/3] Converting ADTs (8 threads)...
# Progress: 10/128 ADTs converted
# Progress: 20/128 ADTs converted
# ...
# Progress: 128/128 ADTs converted
# [3/3] Extracting terrain data...
# ✓ Converted 128/128 tiles in 7m 23s
```

#### 3.2 Enhanced CompareVersionsCommand
Update existing command to use new conversion pipeline:

```csharp
[Command("compare-versions")]
public class CompareVersionsCommand : AsyncCommand<CompareVersionsCommand.Settings>
{
    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        // If alpha-root provided, convert maps first
        if (!string.IsNullOrEmpty(settings.AlphaRoot))
        {
            AnsiConsole.MarkupLine("[yellow]Converting Alpha maps to LK format...[/]");
            
            foreach (var map in settings.Maps)
            {
                var wdtPath = FindWdt(settings.AlphaRoot, map);
                if (wdtPath == null) continue;
                
                var outputDir = Path.Combine(settings.ConvertedAdtCache, 
                    settings.Versions[0], "World", "Maps", map);
                
                await _mapConverter.ConvertMapAsync(wdtPath, outputDir, new ConversionOptions
                {
                    MaxThreads = settings.MaxThreads,
                    ExtractTerrain = true,
                    ExtractShadows = true
                });
            }
        }
        
        // Continue with existing comparison logic...
        return await base.ExecuteAsync(context, settings);
    }
}
```

---

### Phase 4: Remove Dependencies
**Goal**: Clean up old tools and simplify build

#### 4.1 Solution Structure
```
parp-tools/gillijimproject_refactor/
├── WoWRollback/                      ← Keep (enhanced)
│   ├── WoWRollback.Core/
│   ├── WoWRollback.Cli/
│   └── WoWRollback.sln
├── AlphaWDTAnalysisTool/             ← Archive/deprecate
├── DBCTool.V2/                       ← Archive/deprecate
└── lib/
    ├── Warcraft.NET/                 ← Keep (dependency)
    └── gillijimproject-csharp/       ← Keep (Alpha file formats)
```

#### 4.2 Update rebuild-and-regenerate.ps1
**Simplified script** (no more external tool calls):

```powershell
# Before: 3 tools, 3 processes
& dotnet run --project AlphaWdtAnalyzer.Cli ...    # 45 min
& dotnet run --project DBCTool.V2.Cli ...          # 2 min
& dotnet run --project WoWRollback.Cli ...         # 5 min

# After: 1 tool, 1 process, multi-threaded
& dotnet run --project WoWRollback.Cli -- compare-versions \
  --alpha-root ..\test_data \
  --versions 0.5.3.3368,0.5.5.3494 \
  --maps Azeroth,Kalimdor \
  --threads 8 \
  --viewer-report
# Total: 8-10 min! (6-7x faster)
```

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Create `WoWRollback.Core/Formats/Alpha/` namespace
  - [ ] Migrate `WdtAlphaReader`
  - [ ] Migrate `AdtAlphaReader`
  - [ ] Create `AdtAlphaConverter`
- [ ] Create `WoWRollback.Core/Formats/Lk/` namespace
  - [ ] Create `AdtLkWriter`
  - [ ] Test round-trip (Alpha → LK → Game client)
- [ ] Create `WoWRollback.Core/Formats/Dbc/` namespace
  - [ ] Migrate `DbcReader`
  - [ ] Migrate `AreaTableReader`
- [ ] Create `ListfileService` (shared listfile loading)

### Week 2: Processing Pipeline
- [ ] Create `MapConverter` (Alpha → LK orchestration)
  - [ ] Implement multi-threaded conversion
  - [ ] Add progress reporting
  - [ ] Add error handling
- [ ] Create `TerrainExtractor` (multi-threaded)
- [ ] Create `ShadowExtractor` (multi-threaded)
- [ ] Create `AreaTableProcessor`

### Week 3: CLI Integration
- [ ] Add `Spectre.Console` for beautiful CLI
- [ ] Create `ConvertMapCommand`
- [ ] Enhance `CompareVersionsCommand`
- [ ] Add `--threads` option everywhere
- [ ] Update `rebuild-and-regenerate.ps1`

### Week 4: Testing & Optimization
- [ ] Test small map (DeadminesInstance)
- [ ] Test large map (Azeroth)
- [ ] Benchmark: Sequential vs Parallel
- [ ] Memory profiling (ensure no leaks)
- [ ] Document performance gains

### Week 5: Cleanup
- [ ] Archive AlphaWDTAnalysisTool
- [ ] Archive DBCTool.V2
- [ ] Update documentation
- [ ] Update README

---

## Testing Strategy

### 1. Unit Tests
```csharp
[Fact]
public async Task WdtAlphaReader_ReadValidWdt_ReturnsCorrectTileCount()
{
    var wdt = WdtAlphaReader.ReadWdt("test_data/Azeroth.wdt");
    Assert.Equal(128, wdt.AdtTiles.Count);
}

[Fact]
public async Task MapConverter_ConvertMap_ProducesValidLkAdts()
{
    var result = await _converter.ConvertMapAsync(
        "test_data/DeadminesInstance.wdt",
        "output/DeadminesInstance",
        new ConversionOptions { MaxThreads = 2 });
    
    Assert.Equal(10, result.SuccessfulTiles);
    Assert.Equal(0, result.FailedTiles);
}
```

### 2. Integration Tests
```powershell
# Convert map and verify client can load it
dotnet run -- convert-map test_data/Azeroth.wdt -o output/Azeroth
# Launch 3.3.5a client and fly around → Should render correctly
```

### 3. Performance Tests
```powershell
# Benchmark sequential vs parallel
Measure-Command { 
    dotnet run -- convert-map Azeroth.wdt --threads 1 
} # ~50 min

Measure-Command { 
    dotnet run -- convert-map Azeroth.wdt --threads 8 
} # ~7 min ✅ 7x faster!
```

---

## Expected Performance Gains

### Before (Current)
```
AlphaWDTAnalysisTool: 45 min (sequential, 2-8% CPU)
DBCTool.V2:            2 min (sequential)
WoWRollback:           5 min (sequential)
────────────────────────────────
Total:                52 min
```

### After (Consolidated + Multi-Threaded)
```
WoWRollback (all-in-one): 7-8 min (parallel, 60-80% CPU)
────────────────────────────────
Total:                    7-8 min ✅ 6.5x faster!
```

---

## Risk Mitigation

### Risk 1: Data Correctness
**Concern**: Refactored code produces different outputs  
**Mitigation**:
- Keep AlphaWDTAnalysisTool for now (archive, don't delete)
- Compare outputs byte-by-byte
- Test in-game (load maps in 3.3.5a client)

### Risk 2: Memory Issues
**Concern**: Parallel processing uses too much RAM  
**Mitigation**:
- Limit parallelism: `MaxDegreeOfParallelism = Math.Min(cores, 8)`
- Profile memory usage during large map conversions
- Add `--max-memory` CLI option if needed

### Risk 3: Thread Safety
**Concern**: Race conditions in CSV writing or shared state  
**Mitigation**:
- Use `ConcurrentBag<T>` for collection
- Use `SemaphoreSlim` for file I/O
- Thorough testing with high thread counts

---

## Success Metrics

✅ **Performance**: 6-7x faster than current workflow  
✅ **Simplicity**: Single tool instead of 3  
✅ **Correctness**: 100% match with existing outputs  
✅ **Maintainability**: Clean architecture, good tests  
✅ **User Experience**: Beautiful CLI with progress bars

---

## Next Steps

1. **Review this plan** - Get approval before starting
2. **Start with Phase 1** - Migrate format readers first
3. **Test incrementally** - Verify each phase before moving on
4. **Document as we go** - Update docs with new commands

**Estimated Timeline**: 3-4 weeks for full implementation + testing.

Ready to begin when you approve! 🚀
