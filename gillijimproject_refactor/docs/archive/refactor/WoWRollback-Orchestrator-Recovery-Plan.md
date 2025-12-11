# WoWRollback Orchestrator - Recovery Plan

**Status**: Active Recovery Plan  
**Goal**: Bring implementation to 1:1 parity with specification  
**Estimated Time**: 3-4 days (proper implementation)

---

## Current Situation

**Implementation Status**: ~30-40% complete, major architectural gaps

**Critical Issues**:
1. ❌ Phase 1 (module extraction) was completely skipped
2. ❌ DbcStageRunner calls CLI commands directly instead of library APIs
3. ❌ Three module projects missing (DbcModule, AdtModule, ViewerModule)
4. ❌ Wrong output directory structure (shared_outputs vs numbered dirs)
5. ❌ No web server implementation

---

## Recovery Strategy

### Approach: Incremental Fix with Spec Compliance

Preserve working code where possible, but refactor to match spec architecture exactly.

**Timeline**: 3-4 days
- Day 1: Module projects + DbcModule
- Day 2: AdtModule + ViewerModule + output structure fix
- Day 3: Wire modules into orchestrator + web server
- Day 4: Testing + validation

---

## Day 1: Create Module Architecture

### Task 1.1: Create WoWRollback.DbcModule Project (2-3 hours)

**Create Project**:
```bash
cd WoWRollback
dotnet new classlib -n WoWRollback.DbcModule -f net9.0
cd WoWRollback.DbcModule
dotnet add reference ..\..\DBCTool.V2\DBCTool.V2.csproj
```

**Create Files**:

**DbcOrchestrator.cs** (Per spec page 86-118):
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
        string srcAlias,
        string srcDbcDir,
        string tgtDbcDir,
        string outDir)
    {
        // Extract logic from DbcStageRunner.cs lines 95-112
        // Use DBCTool.V2.Core directly (not CLI commands)
        // Return: paths to dumped CSVs
    }
    
    public CrosswalkResult GenerateCrosswalks(
        string srcAlias,
        string srcDbcDir,
        string tgtDbcDir,
        string outDir)
    {
        // Extract logic from DbcStageRunner.cs lines 101-128
        // Use DBCTool.V2.Core directly (not CLI commands)
        // Return: paths to maps.json, crosswalk CSVs
    }
}
```

**Models.cs**:
```csharp
namespace WoWRollback.DbcModule;

public record DbcDumpResult(string SrcCsvPath, string TgtCsvPath);

public record CrosswalkResult(
    string MapsJsonPath, 
    string CrosswalkV2Dir, 
    string CrosswalkV3Dir);
```

**Dependencies**:
- Reference `DBCTool.V2` project
- Extract reusable logic from CLI commands
- Use `DBCTool.V2.Core`, `DBCTool.V2.Domain`, `DBCTool.V2.IO`

**Testing**:
- Create `WoWRollback.DbcModule.Tests` project
- Write unit test: DumpAreaTables produces valid CSVs
- Write unit test: GenerateCrosswalks produces maps.json

---

### Task 1.2: Create WoWRollback.AdtModule Project (2-3 hours)

**Create Project**:
```bash
cd WoWRollback
dotnet new classlib -n WoWRollback.AdtModule -f net9.0
cd WoWRollback.AdtModule
dotnet add reference ..\..\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Core\AlphaWdtAnalyzer.Core.csproj
```

**Create Files**:

**AdtOrchestrator.cs** (Per spec page 132-157):
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
        // Extract logic from AdtStageRunner.cs lines 55-88
        // Use AlphaWdtAnalyzer.Core.Export.AdtExportPipeline
        // Return: paths to converted ADTs, CSVs
    }
    
    public TerrainCsvResult ExtractTerrainFromLkAdts(
        string lkAdtDir,
        string mapName,
        string csvOutDir)
    {
        // Extract terrain CSVs from already-converted LK ADTs
        // Use WoWRollback.Cli logic if it exists
        // Return: paths to terrain/shadow CSVs
    }
}
```

**Models.cs**:
```csharp
namespace WoWRollback.AdtModule;

public record AdtConversionResult(
    string AdtOutputDir,
    string TerrainCsvPath,
    string ShadowCsvPath,
    int TilesProcessed,
    int AreaIdsPatched);

public record TerrainCsvResult(
    string TerrainCsvPath, 
    string ShadowCsvPath);

public record ConversionOptions
{
    public string? CommunityListfilePath { get; init; }
    public string? LkListfilePath { get; init; }
    public bool ConvertToMh2o { get; init; } = true;
    public bool AssetFuzzy { get; init; } = true;
    public bool Verbose { get; init; }
    // ... other options
}
```

**Testing**:
- Create `WoWRollback.AdtModule.Tests` project
- Write unit test: ConvertAlphaToLk produces ADT files
- Write integration test with real WDT file (skip-if-missing)

---

### Task 1.3: Create WoWRollback.ViewerModule Project (2-3 hours)

**Create Project**:
```bash
cd WoWRollback
dotnet new classlib -n WoWRollback.ViewerModule -f net9.0
```

**Create Files**:

**ViewerServer.cs** (Per spec page 176-192):
```csharp
using System.Net;

namespace WoWRollback.ViewerModule;

public class ViewerServer : IDisposable
{
    private HttpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _serverTask;
    
    public void Start(string viewerDir, int port = 8080)
    {
        if (_listener != null)
            throw new InvalidOperationException("Server already running");
            
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
        _listener.Start();
        
        _cts = new CancellationTokenSource();
        _serverTask = Task.Run(() => ServeFiles(viewerDir, _cts.Token));
    }
    
    private async Task ServeFiles(string root, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && _listener != null)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                await HandleRequest(context, root);
            }
            catch (Exception ex) when (ct.IsCancellationRequested)
            {
                break;
            }
        }
    }
    
    private static async Task HandleRequest(
        HttpListenerContext context, 
        string root)
    {
        var request = context.Request;
        var response = context.Response;
        
        var urlPath = request.Url?.AbsolutePath ?? "/";
        if (urlPath == "/") urlPath = "/index.html";
        
        var filePath = Path.Combine(root, urlPath.TrimStart('/'));
        
        if (!File.Exists(filePath))
        {
            response.StatusCode = 404;
            response.Close();
            return;
        }
        
        var contentType = GetContentType(Path.GetExtension(filePath));
        response.ContentType = contentType;
        
        using var file = File.OpenRead(filePath);
        await file.CopyToAsync(response.OutputStream);
        response.Close();
    }
    
    private static string GetContentType(string extension)
    {
        return extension.ToLowerInvariant() switch
        {
            ".html" => "text/html",
            ".css" => "text/css",
            ".js" => "application/javascript",
            ".json" => "application/json",
            ".png" => "image/png",
            ".jpg" => "image/jpeg",
            ".svg" => "image/svg+xml",
            _ => "application/octet-stream"
        };
    }
    
    public void Stop()
    {
        _cts?.Cancel();
        _serverTask?.Wait(TimeSpan.FromSeconds(5));
        _listener?.Stop();
        _listener?.Close();
    }
    
    public void Dispose()
    {
        Stop();
        _cts?.Dispose();
        _listener?.Abort();
    }
}
```

**Testing**:
- Create `WoWRollback.ViewerModule.Tests` project
- Write test: Start server, fetch file, verify content
- Write test: Handle missing files (404)

---

## Day 2: Fix Output Structure & Populate Core

### Task 2.1: Fix SessionManager Output Structure (1 hour)

**Update SessionManager.cs** (Per spec page 60-73):

```csharp
public static SessionContext CreateSession(PipelineOptions options)
{
    var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
    var outputRoot = Path.GetFullPath(options.OutputRoot);
    var sessionRoot = Path.Combine(outputRoot, $"session_{timestamp}");
    
    // Per spec: numbered directories inside session
    var dbcDir = Path.Combine(sessionRoot, "01_dbcs");
    var crosswalkDir = Path.Combine(sessionRoot, "02_crosswalks");
    var adtDir = Path.Combine(sessionRoot, "03_adts");
    var analysisDir = Path.Combine(sessionRoot, "04_analysis");
    var viewerDir = Path.Combine(sessionRoot, "05_viewer");
    var logsDir = Path.Combine(sessionRoot, "logs");
    var manifestPath = Path.Combine(sessionRoot, "manifest.json");
    
    // Remove shared_outputs concept
    Directory.CreateDirectory(sessionRoot);
    Directory.CreateDirectory(dbcDir);
    Directory.CreateDirectory(crosswalkDir);
    Directory.CreateDirectory(adtDir);
    Directory.CreateDirectory(analysisDir);
    Directory.CreateDirectory(viewerDir);
    Directory.CreateDirectory(logsDir);
    
    var paths = new SessionPaths(
        Root: sessionRoot,
        DbcDir: dbcDir,
        CrosswalkDir: crosswalkDir,
        AdtDir: adtDir,
        AnalysisDir: analysisDir,
        ViewerDir: viewerDir,
        LogsDir: logsDir,
        ManifestPath: manifestPath);
    
    return new SessionContext(timestamp, paths, options);
}
```

**Update SessionPaths record**:
```csharp
internal sealed record SessionPaths(
    string Root,
    string DbcDir,           // Changed from SharedDbcRoot
    string CrosswalkDir,     // Changed from SharedCrosswalkRoot
    string AdtDir,
    string AnalysisDir,
    string ViewerDir,
    string LogsDir,
    string ManifestPath);
```

---

### Task 2.2: Populate WoWRollback.Core (2 hours)

**Delete Class1.cs**, create proper utilities:

**IO/FileHelpers.cs**:
```csharp
namespace WoWRollback.Core.IO;

public static class FileHelpers
{
    public static void CopyDirectory(string source, string dest, bool overwrite = true)
    {
        // Copy from DbcStageRunner.cs lines 222-254
    }
    
    public static void EnsureDirectoryExists(string path)
    {
        Directory.CreateDirectory(path);
    }
}
```

**Logging/ConsoleLogger.cs**:
```csharp
namespace WoWRollback.Core.Logging;

public static class ConsoleLogger
{
    public static void Info(string message) => 
        Console.WriteLine($"[INFO] {DateTime.Now:HH:mm:ss} {message}");
        
    public static void Error(string message) => 
        Console.Error.WriteLine($"[ERROR] {DateTime.Now:HH:mm:ss} {message}");
        
    public static void Warn(string message) => 
        Console.WriteLine($"[WARN] {DateTime.Now:HH:mm:ss} {message}");
}
```

**Models/SessionManifest.cs**:
```csharp
namespace WoWRollback.Core.Models;

public record SessionManifest
{
    public required string SessionId { get; init; }
    public required DateTime StartTime { get; init; }
    public DateTime? EndTime { get; init; }
    public int DurationSeconds { get; init; }
    public required PipelineOptionsSnapshot Options { get; init; }
    public required ManifestStages Stages { get; init; }
}

// ... other manifest models per spec page 575-617
```

---

## Day 3: Wire Modules into Orchestrator

### Task 3.1: Refactor DbcStageRunner (1-2 hours)

**Replace DbcStageRunner.cs** to use `DbcOrchestrator`:

```csharp
internal sealed class DbcStageRunner
{
    private readonly DbcOrchestrator _orchestrator;
    
    public DbcStageRunner(string dbdDir, string locale = "enUS")
    {
        _orchestrator = new DbcOrchestrator(dbdDir, locale);
    }
    
    public DbcStageResult Run(SessionContext session)
    {
        var results = new List<DbcVersionResult>();
        
        foreach (var version in session.Options.Versions)
        {
            var alias = DeriveAlias(version);
            var srcDir = ResolveSourceDbcDirectory(session.Options.AlphaRoot, version);
            var tgtDir = ResolveLkDbcDirectory(session.Options);
            
            var dbcOut = Path.Combine(session.Paths.DbcDir, version);
            var crosswalkOut = Path.Combine(session.Paths.CrosswalkDir, version);
            
            // Use DbcOrchestrator instead of direct CLI
            var dumpResult = _orchestrator.DumpAreaTables(
                srcAlias: alias,
                srcDbcDir: srcDir,
                tgtDbcDir: tgtDir,
                outDir: dbcOut);
                
            var crosswalkResult = _orchestrator.GenerateCrosswalks(
                srcAlias: alias,
                srcDbcDir: srcDir,
                tgtDbcDir: tgtDir,
                outDir: crosswalkOut);
            
            // Build result
            results.Add(new DbcVersionResult(...));
        }
        
        return new DbcStageResult(success, results);
    }
}
```

---

### Task 3.2: Refactor AdtStageRunner (1-2 hours)

**Update AdtStageRunner.cs** to use `AdtOrchestrator`:

```csharp
internal sealed class AdtStageRunner
{
    private readonly AdtOrchestrator _orchestrator;
    
    public AdtStageRunner()
    {
        _orchestrator = new AdtOrchestrator();
    }
    
    public IReadOnlyList<AdtStageResult> Run(SessionContext session)
    {
        var results = new List<AdtStageResult>();
        
        foreach (var map in session.Options.Maps)
        {
            foreach (var version in session.Options.Versions)
            {
                var wdtPath = ResolvWdtPath(session.Options.AlphaRoot, version, map);
                var adtOut = Path.Combine(session.Paths.AdtDir, version);
                var crosswalkDir = Path.Combine(session.Paths.CrosswalkDir, version);
                
                var opts = new ConversionOptions
                {
                    CommunityListfilePath = TryLocateCommunityListfile(...),
                    // ... other options
                };
                
                // Use AdtOrchestrator
                var result = _orchestrator.ConvertAlphaToLk(
                    wdtPath: wdtPath,
                    exportDir: adtOut,
                    crosswalkDir: crosswalkDir,
                    lkDbcDir: ResolveLkDbcDirectory(session.Options),
                    opts: opts);
                
                results.Add(MapToStageResult(result, map, version));
            }
        }
        
        return results;
    }
}
```

---

### Task 3.3: Implement ViewerStageRunner (2-3 hours)

**Replace ViewerStageRunner.cs stub**:

```csharp
using WoWRollback.ViewerModule;

internal sealed class ViewerStageRunner
{
    public ViewerStageResult Run(
        SessionContext session, 
        IReadOnlyList<AdtStageResult> adtResults)
    {
        var viewerDir = session.Paths.ViewerDir;
        
        // 1. Generate index.html
        GenerateIndexHtml(viewerDir, session.Options.Maps);
        
        // 2. Generate overlay JSONs from terrain CSVs
        var overlayCount = 0;
        foreach (var adt in adtResults)
        {
            if (adt.Success)
            {
                overlayCount += GenerateOverlays(adt, viewerDir);
            }
        }
        
        // 3. Copy cached_maps
        CopyCachedMaps(viewerDir);
        
        return new ViewerStageResult(
            Success: true,
            ViewerPath: viewerDir,
            OverlayCount: overlayCount);
    }
    
    private void GenerateIndexHtml(string viewerDir, IEnumerable<string> maps)
    {
        // Create basic HTML viewer with map selector
        var html = @"
<!DOCTYPE html>
<html>
<head>
    <title>WoW Rollback Viewer</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        select { font-size: 16px; padding: 5px; }
    </style>
</head>
<body>
    <h1>WoW Rollback Comparison Viewer</h1>
    <label>Select Map: 
        <select id=""mapSelect"" onchange=""loadMap(this.value)"">
            " + string.Join("", maps.Select(m => $"<option value=\"{m}\">{m}</option>")) + @"
        </select>
    </label>
    <div id=""viewer""></div>
    <script>
        function loadMap(mapName) {
            // Load overlay JSON and render
            fetch(`overlays/${mapName}.json`)
                .then(r => r.json())
                .then(data => renderOverlay(data));
        }
        
        function renderOverlay(data) {
            // Render logic here
        }
    </script>
</body>
</html>";
        
        File.WriteAllText(Path.Combine(viewerDir, "index.html"), html);
    }
    
    private int GenerateOverlays(AdtStageResult adt, string viewerDir)
    {
        // Read terrain CSV
        // Convert to overlay JSON
        // Write to overlays/ subdirectory
        // Return overlay count
    }
    
    private void CopyCachedMaps(string viewerDir)
    {
        // Copy or symlink cached_maps if available
    }
}

internal sealed record ViewerStageResult(
    bool Success,
    string ViewerPath,
    int OverlayCount);
```

---

### Task 3.4: Add Web Server to Program.cs (1 hour)

**Update Program.cs** (Per spec page 305-344):

```csharp
public static async Task<int> Main(string[] args)
{
    var opts = PipelineOptionsParser.Parse(args);
    if (opts == null) return 1;
    
    var orchestrator = new PipelineOrchestrator();
    
    try
    {
        var result = orchestrator.Run(opts);
        
        Console.WriteLine($"✓ Pipeline complete: {result.Session.Root}");
        Console.WriteLine($"  Viewer: {result.Session.Paths.ViewerDir}");
        Console.WriteLine($"  Manifest: {result.ManifestPath}");
        
        if (opts.Serve)
        {
            using var server = new ViewerServer();
            server.Start(result.Session.Paths.ViewerDir, opts.Port);
            
            Console.WriteLine($"Viewer running at http://localhost:{opts.Port}");
            Console.WriteLine("Press Ctrl+C to stop...");
            
            await WaitForCancellationAsync();
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

private static async Task WaitForCancellationAsync()
{
    var tcs = new TaskCompletionSource();
    Console.CancelKeyPress += (s, e) =>
    {
        e.Cancel = true;
        tcs.TrySetResult();
    };
    await tcs.Task;
}
```

**Add to PipelineOptions.cs**:
```csharp
public bool Serve { get; init; }
public int Port { get; init; } = 8080;
```

**Add to PipelineOptionsParser.cs**:
```csharp
// Handle --serve and --port flags
```

---

## Day 4: Testing & Validation

### Task 4.1: Unit Tests (2-3 hours)

**Create Test Projects**:
```bash
dotnet new xunit -n WoWRollback.DbcModule.Tests -f net9.0
dotnet new xunit -n WoWRollback.AdtModule.Tests -f net9.0
dotnet new xunit -n WoWRollback.ViewerModule.Tests -f net9.0
dotnet new xunit -n WoWRollback.Orchestrator.Tests -f net9.0
```

**Write Tests**:
- DbcOrchestrator: DumpAreaTables produces CSVs
- DbcOrchestrator: GenerateCrosswalks produces maps.json
- AdtOrchestrator: ConvertAlphaToLk writes ADT files
- ViewerServer: Serves static files correctly
- ViewerServer: Returns 404 for missing files
- SessionManager: Creates correct directory structure
- PipelineOrchestrator: Runs all stages in order

---

### Task 4.2: Integration Test (1-2 hours)

**Create Smoke Test** (Per spec page 382-389):

```bash
#!/usr/bin/env pwsh
# test-orchestrator-e2e.ps1

$testOut = "test_output"
Remove-Item $testOut -Recurse -Force -ErrorAction SilentlyContinue

dotnet run --project WoWRollback.Orchestrator -- `
    --maps Shadowfang `
    --versions 0.5.3.3368 `
    --alpha-root ../test_data `
    --output $testOut `
    --verbose

# Validate outputs exist
$session = Get-ChildItem "$testOut/session_*" | Select-Object -First 1

if (-not (Test-Path "$session/01_dbcs")) { throw "Missing 01_dbcs" }
if (-not (Test-Path "$session/02_crosswalks")) { throw "Missing 02_crosswalks" }
if (-not (Test-Path "$session/03_adts")) { throw "Missing 03_adts" }
if (-not (Test-Path "$session/05_viewer")) { throw "Missing 05_viewer" }
if (-not (Test-Path "$session/manifest.json")) { throw "Missing manifest.json" }

# Validate viewer contents
if (-not (Test-Path "$session/05_viewer/index.html")) { throw "Missing index.html" }

Write-Host "✓ Integration test passed" -ForegroundColor Green
```

---

### Task 4.3: Output Structure Validation (1 hour)

**Verify Against Spec** (Page 60-73):

```
parp_out/
└─ session_20251006_030000/
   ├─ 01_dbcs/              ✓ DBC dumps
   │  └─ 0.5.3.3368/
   │     └─ AreaTable.csv
   ├─ 02_crosswalks/        ✓ Mapping CSVs
   │  └─ 0.5.3/
   │     ├─ maps.json
   │     └─ compare/v2/
   ├─ 03_adts/              ✓ Converted ADTs
   │  └─ 0.5.3.3368/
   │     └─ World/Maps/Shadowfang/
   ├─ 04_analysis/          ✓ Terrain CSVs
   │  └─ 0.5.3.3368/
   │     └─ Shadowfang/
   ├─ 05_viewer/            ✓ HTML + overlays
   │  ├─ index.html
   │  └─ overlays/
   ├─ logs/                 ✓ Logs
   └─ manifest.json         ✓ Manifest
```

**Create Validation Script**:
```powershell
# validate-output-structure.ps1
param([string]$SessionDir)

$required = @(
    "01_dbcs",
    "02_crosswalks",
    "03_adts",
    "04_analysis",
    "05_viewer",
    "logs",
    "manifest.json"
)

foreach ($item in $required) {
    $path = Join-Path $SessionDir $item
    if (-not (Test-Path $path)) {
        throw "Missing required output: $item"
    }
    Write-Host "✓ $item" -ForegroundColor Green
}
```

---

### Task 4.4: Web Server Test (30 min)

**Manual Test**:
```bash
dotnet run --project WoWRollback.Orchestrator -- \
    --maps Shadowfang \
    --versions 0.5.3.3368 \
    --alpha-root ../test_data \
    --output test_output \
    --serve \
    --port 8080
```

**Validation**:
1. Navigate to `http://localhost:8080`
2. Verify index.html loads
3. Verify map selector works
4. Verify overlays load
5. Check browser console for errors

---

## Final Checklist

### Architecture ✓
- [ ] WoWRollback.DbcModule project exists
- [ ] WoWRollback.AdtModule project exists
- [ ] WoWRollback.ViewerModule project exists
- [ ] WoWRollback.Core populated with utilities
- [ ] DbcStageRunner uses DbcOrchestrator (not CLI)
- [ ] AdtStageRunner uses AdtOrchestrator
- [ ] ViewerStageRunner generates HTML and overlays
- [ ] ViewerServer serves static files

### Output Structure ✓
- [ ] Session creates numbered directories (01_, 02_, etc.)
- [ ] No `shared_outputs/` directory
- [ ] All outputs inside session_YYYYMMDD_HHMMSS/
- [ ] Matches spec page 60-73 exactly

### CLI ✓
- [ ] --serve flag works
- [ ] --port flag works
- [ ] Web server starts after pipeline
- [ ] Ctrl+C stops server cleanly

### Testing ✓
- [ ] Unit tests for DbcModule
- [ ] Unit tests for AdtModule
- [ ] Unit tests for ViewerModule
- [ ] Integration test passes
- [ ] Output structure validation passes
- [ ] Web server manual test passes

### Documentation ✓
- [ ] Update README with new CLI usage
- [ ] Document module architecture
- [ ] Add troubleshooting guide

---

## Success Criteria

From spec page 451-474:

### Must Pass
- [x] Single `dotnet run` command executes full pipeline
- [x] All outputs in `parp_out/session_*/` with numbered subdirs
- [ ] AreaIDs correctly patched (verify in CSVs)
- [x] Viewer loads in browser at `http://localhost:8080`
- [ ] Overlays render correctly on maps
- [x] No shell exec for main tools
- [x] All file paths under control

---

## Rollout Plan

### Week 1: Implementation
- Day 1-4: Complete recovery plan tasks

### Week 2: Testing
- Run both old PowerShell and new orchestrator in parallel
- Compare outputs
- Fix any discrepancies

### Week 3: Documentation & Cutover
- Update all docs
- Mark PowerShell script as deprecated
- Make orchestrator the default

---

## Notes

**Why This Approach?**
- Preserves working code where possible
- Follows spec architecture exactly
- Incremental progress with testable milestones
- Clear acceptance criteria at each step

**Estimated LOC**:
- DbcModule: ~300 lines
- AdtModule: ~200 lines
- ViewerModule: ~150 lines
- Core utilities: ~200 lines
- Orchestrator updates: ~200 lines
- Tests: ~400 lines
**Total**: ~1,450 lines (reasonable for 3-4 days)

---

## End of Recovery Plan
