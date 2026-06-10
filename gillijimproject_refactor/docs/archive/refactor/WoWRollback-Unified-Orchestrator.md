# WoWRollback Unified Orchestrator - Complete Refactor Plan

**Status**: Planning  
**Goal**: Replace PowerShell/Python orchestration with self-contained C# solution  
**Timeline**: 3-4 days  
**Start Date**: TBD

---

## Executive Summary

Replace the fragile PowerShell script and external Python web server with a unified C# orchestrator that provides:
- Single executable entry point
- Direct library integration (no shell exec)
- Predictable output structure
- Embedded web server
- Full debugging and testing capability

---

## Current Problems

### Architecture Issues
1. **DBCTool.V2**: Standalone CLI, ignores `--out-root` parameter, writes to `./dbctool_out`
2. **AlphaWDTAnalysisTool**: Standalone CLI, complex argument parsing, shell execution overhead
3. **PowerShell Script**: 900+ lines, fragile path handling, hard to debug, OS-dependent
4. **Python Web Server**: External dependency, manual startup, no integration
5. **Output Scatter**: Files in `dbctool_out/`, `parp_out/`, `rollback_outputs/`, `cached_maps/`

### Data Flow Issues
- No central coordination
- Unclear data dependencies
- Hard to trace errors
- Difficult to verify completeness

---

## Target Architecture

```
WoWRollback.Orchestrator (new C# console app)
├─ WoWRollback.DbcModule       ← DBCTool.V2.Core as library
├─ WoWRollback.AdtModule        ← AlphaWdtAnalyzer.Core as library  
├─ WoWRollback.ViewerModule     ← Viewer gen + embedded server
├─ WoWRollback.Core             ← Shared models, I/O utilities
└─ WoWRollback.Cli              ← Keep existing comparison logic
```

### Single Entry Point
```bash
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang,DeadminesInstance \
  --versions 0.5.3.3368,0.5.5.3494 \
  --alpha-root ../test_data \
  --output parp_out \
  --serve
```

### Unified Output Structure
```
parp_out/
└─ session_20251006_030000/
   ├─ 01_dbcs/              ← DBC dumps (AreaTable CSVs)
   ├─ 02_crosswalks/        ← DBCTool mapping CSVs
   ├─ 03_adts/              ← Converted LK ADTs
   ├─ 04_analysis/          ← Terrain/shadow CSVs
   ├─ 05_viewer/            ← HTML + overlay JSONs
   │  ├─ index.html
   │  ├─ overlays/
   │  └─ cached_maps/
   ├─ logs/                 ← All logs
   └─ manifest.json         ← Session metadata
```

---

## Implementation Plan

### Phase 1: Extract Core Libraries (Day 1-2)

#### 1.1 Create WoWRollback.DbcModule

**File**: `WoWRollback.DbcModule/DbcOrchestrator.cs`

```csharp
namespace WoWRollback.DbcModule;

public class DbcOrchestrator
{
    private readonly string _dbdDir;
    private readonly string _locale;
    
    public DbcOrchestrator(string dbdDir, string locale = "enUS")
    {
        _dbdDir = dbdDir;
        _locale = locale;
    }
    
    public DbcDumpResult DumpAreaTables(
        string srcAlias,        // e.g., "0.5.3"
        string srcDbcDir,       // path to source DBCs
        string tgtDbcDir,       // path to 3.3.5 DBCs
        string outDir)          // output directory
    {
        // Use existing DBCTool.V2.Core logic
        // Return: paths to dumped CSVs
    }
    
    public CrosswalkResult GenerateCrosswalks(
        string srcAlias,
        string srcDbcDir,
        string tgtDbcDir,
        string outDir)
    {
        // Use existing DBCTool.V2.Core compare logic
        // Return: paths to maps.json, crosswalk CSVs
    }
}

public record DbcDumpResult(string SrcCsvPath, string TgtCsvPath);
public record CrosswalkResult(string MapsJsonPath, string CrosswalkV2Dir, string CrosswalkV3Dir);
```

**Dependencies**:
- Reference `DBCTool.V2` project (Core, Domain, IO, Crosswalk)
- Extract reusable logic from `DBCTool.V2/Cli/Program.cs`
- Remove CLI arg parsing, expose direct method calls

#### 1.2 Create WoWRollback.AdtModule

**File**: `WoWRollback.AdtModule/AdtOrchestrator.cs`

```csharp
namespace WoWRollback.AdtModule;

public class AdtOrchestrator
{
    public AdtConversionResult ConvertAlphaToLk(
        string wdtPath,
        string exportDir,
        string crosswalkDir,
        string lkDbcDir,
        ConversionOptions opts)
    {
        // Use AlphaWdtAnalyzer.Core.Export.AdtExportPipeline
        // Return: paths to converted ADTs, CSVs
    }
    
    public TerrainCsvResult ExtractTerrainFromLkAdts(
        string lkAdtDir,
        string mapName,
        string csvOutDir)
    {
        // Extract terrain CSVs from already-converted LK ADTs
        // Return: paths to terrain/shadow CSVs
    }
}

public record AdtConversionResult(
    string AdtOutputDir,
    string TerrainCsvPath,
    string ShadowCsvPath,
    int TilesProcessed,
    int AreaIdsPat ched);

public record TerrainCsvResult(string TerrainCsvPath, string ShadowCsvPath);
```

**Dependencies**:
- Reference `AlphaWdtAnalyzer.Core` project
- Extract from `AlphaWdtAnalyzer.Cli/Program.cs`
- Remove CLI parsing, expose direct API

#### 1.3 Create WoWRollback.ViewerModule

**File**: `WoWRollback.ViewerModule/ViewerServer.cs`

```csharp
namespace WoWRollback.ViewerModule;

public class ViewerServer : IDisposable
{
    private HttpListener? _listener;
    
    public void Start(string viewerDir, int port = 8080)
    {
        // Serve static files from viewerDir
        // Use HttpListener or minimal ASP.NET Core
    }
    
    public void Stop() { ... }
}
```

**Implementation Options**:
- **Option A**: Simple `HttpListener` wrapper (no dependencies)
- **Option B**: Minimal ASP.NET Core static files (cleaner, more features)

**Recommendation**: Start with Option A for simplicity, migrate to B if needed.

---

### Phase 2: Build Unified Orchestrator (Day 2-3)

#### 2.1 Create Pipeline Orchestrator

**File**: `WoWRollback.Orchestrator/PipelineOrchestrator.cs`

```csharp
public class PipelineOrchestrator
{
    private readonly DbcOrchestrator _dbc;
    private readonly AdtOrchestrator _adt;
    private readonly ComparisonGenerator _viewer;  // existing WoWRollback.Cli logic
    
    public PipelineResult RunFullPipeline(PipelineOptions opts)
    {
        var session = CreateSessionDir(opts.OutputRoot);
        var manifest = new SessionManifest { SessionId = session.Timestamp };
        
        // Stage 1: DBCTool - Dump & Generate Crosswalks
        var dbcResult = RunDbcStage(opts, session);
        manifest.DbcStage = dbcResult;
        
        // Stage 2: ADT Conversion (per map)
        var adtResults = new List<AdtConversionResult>();
        foreach (var map in opts.Maps)
        {
            foreach (var version in opts.Versions)
            {
                var result = RunAdtStage(map, version, opts, session, dbcResult);
                adtResults.Add(result);
            }
        }
        manifest.AdtStage = adtResults;
        
        // Stage 3: Viewer Generation
        var viewerResult = RunViewerStage(adtResults, opts, session);
        manifest.ViewerStage = viewerResult;
        
        // Save manifest
        SaveManifest(session.ManifestPath, manifest);
        
        return new PipelineResult
        {
            SessionDir = session.Path,
            ViewerDir = viewerResult.ViewerPath,
            ManifestPath = session.ManifestPath,
            Success = true
        };
    }
}
```

#### 2.2 Session Management

```csharp
public record SessionDirectories(
    string Path,              // parp_out/session_YYYYMMDD_HHMMSS
    string Timestamp,
    string DbcDir,           // 01_dbcs
    string CrosswalkDir,     // 02_crosswalks
    string AdtDir,           // 03_adts
    string AnalysisDir,      // 04_analysis
    string ViewerDir,        // 05_viewer
    string LogsDir,          // logs
    string ManifestPath);    // manifest.json

public class SessionManager
{
    public static SessionDirectories CreateSession(string outputRoot)
    {
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var sessionPath = Path.Combine(outputRoot, $"session_{timestamp}");
        
        var dirs = new SessionDirectories(
            Path: sessionPath,
            Timestamp: timestamp,
            DbcDir: Path.Combine(sessionPath, "01_dbcs"),
            CrosswalkDir: Path.Combine(sessionPath, "02_crosswalks"),
            AdtDir: Path.Combine(sessionPath, "03_adts"),
            AnalysisDir: Path.Combine(sessionPath, "04_analysis"),
            ViewerDir: Path.Combine(sessionPath, "05_viewer"),
            LogsDir: Path.Combine(sessionPath, "logs"),
            ManifestPath: Path.Combine(sessionPath, "manifest.json"));
        
        // Create all directories
        Directory.CreateDirectory(dirs.DbcDir);
        Directory.CreateDirectory(dirs.CrosswalkDir);
        Directory.CreateDirectory(dirs.AdtDir);
        Directory.CreateDirectory(dirs.AnalysisDir);
        Directory.CreateDirectory(dirs.ViewerDir);
        Directory.CreateDirectory(dirs.LogsDir);
        
        return dirs;
    }
}
```

#### 2.3 CLI Interface

**File**: `WoWRollback.Orchestrator/Program.cs`

```csharp
public class Program
{
    public static async Task<int> Main(string[] args)
    {
        var opts = ParseArguments(args);
        if (opts == null) return 1;
        
        ConfigureLogging(opts.Verbose);
        
        var orchestrator = new PipelineOrchestrator(
            dbdDir: opts.DbdDir,
            lkDbcDir: opts.LkDbcDir);
        
        try
        {
            var result = orchestrator.RunFullPipeline(opts);
            
            Console.WriteLine($"✓ Pipeline complete: {result.SessionDir}");
            Console.WriteLine($"  Viewer: {result.ViewerDir}");
            Console.WriteLine($"  Manifest: {result.ManifestPath}");
            
            if (opts.Serve)
            {
                using var server = new ViewerServer();
                server.Start(result.ViewerDir, opts.Port);
                Console.WriteLine($"Viewer running at http://localhost:{opts.Port}");
                Console.WriteLine("Press Ctrl+C to stop...");
                await WaitForCancellation();
            }
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Pipeline failed: {ex.Message}");
            if (opts.Verbose) Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }
}
```

**Arguments**:
```
--maps <map1,map2,...>           Required: Map names
--versions <v1,v2,...>           Required: Version numbers
--alpha-root <path>              Required: Path to Alpha test data
--output <path>                  Output directory (default: parp_out)
--dbd-dir <path>                 WoWDBDefs path (default: ../lib/WoWDBDefs/definitions)
--lk-dbc-dir <path>              3.3.5 DBCs (default: auto-detect)
--serve                          Start web server after generation
--port <number>                  Web server port (default: 8080)
--verbose                        Detailed logging
--help                           Show usage
```

---

### Phase 3: Testing & Migration (Day 3-4)

#### 3.1 Unit Tests

Create test projects for each module:
- `WoWRollback.DbcModule.Tests`
- `WoWRollback.AdtModule.Tests`
- `WoWRollback.Orchestrator.Tests`

**Critical Tests**:
1. DBCTool integration produces valid crosswalks
2. ADT conversion writes files to correct paths
3. AreaID patching actually works
4. Viewer generation creates overlay JSONs
5. Web server serves files correctly

#### 3.2 Integration Test

**Test Case**: Shadowfang end-to-end
```bash
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3.3368 \
  --alpha-root ../test_data \
  --output test_output \
  --verbose
```

**Success Criteria**:
- Exit code 0
- manifest.json exists with all stages completed
- CSVs have non-zero AreaIDs (not all 0)
- Viewer loads in browser
- Overlays render on map

#### 3.3 Parallel Operation

**DO NOT delete `rebuild-and-regenerate.ps1` yet**

Both systems should work in parallel:
- Old: `.\rebuild-and-regenerate.ps1`
- New: `dotnet run --project WoWRollback.Orchestrator`

Users can choose which to use during transition period.

#### 3.4 Cutover Plan

Once new orchestrator is proven stable (1-2 weeks of use):
1. Update README with new CLI usage
2. Mark PowerShell script as deprecated
3. Move to `scripts/deprecated/`
4. Remove after confirmed no regression

---

## Project Structure

```
gillijimproject_refactor/
├─ WoWRollback/
│  ├─ WoWRollback.Orchestrator/       ← NEW: Main entry point
│  │  ├─ Program.cs
│  │  ├─ PipelineOrchestrator.cs
│  │  ├─ SessionManager.cs
│  │  └─ PipelineOptions.cs
│  ├─ WoWRollback.DbcModule/          ← NEW: DBCTool wrapper
│  │  ├─ DbcOrchestrator.cs
│  │  └─ Models.cs
│  ├─ WoWRollback.AdtModule/          ← NEW: AlphaWDT wrapper
│  │  ├─ AdtOrchestrator.cs
│  │  └─ Models.cs
│  ├─ WoWRollback.ViewerModule/       ← NEW: Web server
│  │  ├─ ViewerServer.cs
│  │  └─ StaticFileHandler.cs
│  ├─ WoWRollback.Core/               ← NEW: Shared utilities
│  │  ├─ IO/
│  │  ├─ Logging/
│  │  └─ Models/
│  ├─ WoWRollback.Cli/                ← EXISTING: Keep as-is
│  ├─ WoWRollback.Viewer/             ← EXISTING: Keep as-is
│  └─ rebuild-and-regenerate.ps1      ← DEPRECATED: Keep until cutover
├─ DBCTool.V2/                        ← EXISTING: Reference as library
├─ AlphaWDTAnalysisTool/              ← EXISTING: Reference as library
└─ docs/refactor/                     ← This document
```

---

## Success Criteria

### Functional Requirements
- [ ] Single `dotnet run` command executes full pipeline
- [ ] All outputs in predictable `parp_out/session_*/` structure
- [ ] AreaIDs correctly patched in CSVs (not all 0)
- [ ] Viewer loads in browser at `http://localhost:8080`
- [ ] Overlays render correctly on maps
- [ ] Works cross-platform (Windows, Linux, macOS)

### Technical Requirements
- [ ] No PowerShell dependency
- [ ] No Python dependency
- [ ] No shell exec (`Process.Start`) for main tools
- [ ] All file paths under control
- [ ] Comprehensive error handling and logging
- [ ] Unit test coverage ≥70%

### Performance Requirements
- [ ] Pipeline completes in ≤ current PowerShell time
- [ ] Memory usage reasonable (< 4GB for typical maps)
- [ ] Web server responsive (< 100ms for static files)

---

## Non-Functional Requirements

### Logging
- Structured logging to console and file
- Log levels: Debug, Info, Warn, Error
- Timestamp, stage, and context in every log entry
- Separate log file per stage in `logs/` directory

### Error Handling
- Graceful failures with clear error messages
- Partial success allowed (e.g., some maps fail, continue others)
- Manifest tracks success/failure per stage
- Stack traces only in verbose mode

### Extensibility
- Easy to add new stages to pipeline
- Pluggable module architecture
- Clean interfaces between components

---

## Migration Strategy

### Week 1: Development
- Day 1-2: Phase 1 (extract libraries)
- Day 3: Phase 2 (orchestrator)
- Day 4: Phase 3 (testing)

### Week 2: Testing & Refinement
- Run both systems in parallel
- Compare outputs (CSVs should match)
- Fix any discrepancies
- Performance tuning

### Week 3: Deprecation
- Update documentation
- Mark PowerShell as deprecated
- Add deprecation warnings

### Week 4+: Removal
- After 2 weeks of stable operation
- Remove PowerShell script
- Archive to `scripts/deprecated/`

---

## Risk Mitigation

### Risk: Breaking existing functionality
**Mitigation**: Parallel operation, extensive testing, gradual migration

### Risk: Web server complexity
**Mitigation**: Start simple (HttpListener), upgrade only if needed

### Risk: Performance regression
**Mitigation**: Benchmark before/after, optimize hot paths

### Risk: Cross-platform issues
**Mitigation**: Test on Windows and Linux early, use Path.Combine everywhere

### Risk: Scope creep
**Mitigation**: Strict adherence to plan, no new features during refactor

---

## Open Questions

1. **Web Server Implementation**: HttpListener vs ASP.NET Core?
   - **Recommendation**: Start with HttpListener, migrate if needed
   
2. **Logging Framework**: Console.WriteLine vs Serilog/NLog?
   - **Recommendation**: Start simple, upgrade if needed
   
3. **Configuration**: Command-line only vs config file support?
   - **Recommendation**: Command-line only for Phase 1

---

## Appendix A: Key File Locations

### Input Paths (User-provided)
- `--alpha-root`: `../test_data/` (contains Alpha WDTs, DBCs)
- `--dbd-dir`: `../lib/WoWDBDefs/definitions/`
- `--lk-dbc-dir`: `../test_data/3.3.5/tree/DBFilesClient/`

### Output Paths (Generated)
- Session root: `parp_out/session_YYYYMMDD_HHMMSS/`
- DBCs: `01_dbcs/{version}/AreaTable.csv`
- Crosswalks: `02_crosswalks/{version}/compare/v2/*.csv`
- ADTs: `03_adts/{version}/World/Maps/{map}/*.adt`
- Analysis: `04_analysis/{version}/{map}/csv/*.csv`
- Viewer: `05_viewer/index.html`
- Logs: `logs/stage_*.log`
- Manifest: `manifest.json`

---

## Appendix B: Manifest Schema

```json
{
  "session_id": "20251006_030000",
  "start_time": "2025-10-06T03:00:00Z",
  "end_time": "2025-10-06T03:05:23Z",
  "duration_seconds": 323,
  "options": {
    "maps": ["Shadowfang"],
    "versions": ["0.5.3.3368"],
    "alpha_root": "../test_data"
  },
  "stages": {
    "dbc": {
      "status": "completed",
      "duration_seconds": 45,
      "outputs": {
        "src_csv": "01_dbcs/0.5.3/AreaTable.csv",
        "tgt_csv": "01_dbcs/3.3.5/AreaTable.csv",
        "maps_json": "02_crosswalks/0.5.3/maps.json",
        "crosswalks": "02_crosswalks/0.5.3/compare/v2/"
      }
    },
    "adt": [
      {
        "map": "Shadowfang",
        "version": "0.5.3.3368",
        "status": "completed",
        "tiles_processed": 25,
        "area_ids_patched": 6400,
        "outputs": {
          "adt_dir": "03_adts/0.5.3.3368/World/Maps/Shadowfang/",
          "terrain_csv": "04_analysis/0.5.3.3368/Shadowfang/terrain.csv"
        }
      }
    ],
    "viewer": {
      "status": "completed",
      "overlay_count": 1250,
      "viewer_path": "05_viewer/"
    }
  }
}
```

---

## End of Document

**Next Steps**: 
1. Review and approve this plan
2. Start Phase 1.1: Extract DbcOrchestrator
3. Commit progress after each phase
