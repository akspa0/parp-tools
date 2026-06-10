# WoWRollback Orchestrator - Gap Analysis

**Generated**: 2025-10-06  
**Status**: Implementation significantly incomplete  
**Severity**: High - Missing core architecture

---

## Executive Summary

The current implementation is **NOT a 1:1 match** to the specification. Critical Phase 1 (module extraction) was skipped entirely, resulting in:

- ❌ Missing 3 out of 4 required module projects
- ❌ CLI commands called directly instead of library APIs
- ❌ No web server implementation
- ❌ Wrong output directory structure
- ⚠️ WoWRollback.Core exists but is empty stub

**Estimated Completion**: ~30-40% of specification

---

## Architecture Gaps

### Required by Spec (Phase 1)

| Component | Status | Notes |
|-----------|--------|-------|
| WoWRollback.DbcModule | ❌ Missing | Should wrap DBCTool.V2 as library |
| WoWRollback.AdtModule | ❌ Missing | Should wrap AlphaWdtAnalyzer.Core |
| WoWRollback.ViewerModule | ❌ Missing | Should provide embedded HTTP server |
| WoWRollback.Core | ⚠️ Stub Only | Exists but contains only Class1.cs placeholder |
| WoWRollback.Orchestrator | ✓ Partial | Exists but wrong implementation approach |

### What Actually Exists

```
WoWRollback/
├─ WoWRollback.Orchestrator/        ✓ Created
│  ├─ DbcStageRunner.cs              ❌ Wrong: Direct CLI usage
│  ├─ AdtStageRunner.cs              ⚠️ Partial: Uses library but no module
│  ├─ ViewerStageRunner.cs           ❌ Stub only
│  ├─ PipelineOrchestrator.cs        ⚠️ Basic structure
│  ├─ SessionManager.cs              ⚠️ Wrong output structure
│  └─ ...
└─ WoWRollback.Core/                 ⚠️ Empty stub
   └─ Class1.cs
```

---

## Critical Implementation Errors

### 1. DbcStageRunner - Direct CLI Usage ❌

**Spec Says** (Page 86-118):
```csharp
// WoWRollback.DbcModule/DbcOrchestrator.cs
public class DbcOrchestrator
{
    public DbcDumpResult DumpAreaTables(
        string srcAlias, string srcDbcDir, 
        string tgtDbcDir, string outDir) { ... }
        
    public CrosswalkResult GenerateCrosswalks(...) { ... }
}
```

**Current Implementation** (DbcStageRunner.cs:31-32):
```csharp
var dumpCommand = new DumpAreaCommand();      // ❌ Direct CLI instantiation
var compareCommand = new CompareAreaV2Command();
```

**Problem**: 
- Calling CLI commands directly instead of library APIs
- No proper DbcModule project exists
- Cannot test without running full CLI
- Tight coupling to CLI implementation

---

### 2. Missing Module Projects ❌

**Spec Requires** (Page 48):
```
WoWRollback.DbcModule/       ← NEW: DBCTool wrapper
WoWRollback.AdtModule/        ← NEW: AlphaWDT wrapper  
WoWRollback.ViewerModule/     ← NEW: Web server
WoWRollback.Core/             ← NEW: Shared utilities
```

**Current State**:
- ❌ WoWRollback.DbcModule - Does not exist
- ❌ WoWRollback.AdtModule - Does not exist
- ❌ WoWRollback.ViewerModule - Does not exist
- ⚠️ WoWRollback.Core - Empty stub (Class1.cs only)

---

### 3. Wrong Output Directory Structure ❌

**Spec Says** (Page 60-73):
```
parp_out/
└─ session_20251006_030000/
   ├─ 01_dbcs/              ← DBC dumps
   ├─ 02_crosswalks/        ← Mapping CSVs
   ├─ 03_adts/              ← Converted ADTs
   ├─ 04_analysis/          ← Terrain CSVs
   ├─ 05_viewer/            ← HTML + overlays
   ├─ logs/
   └─ manifest.json
```

**Current Implementation** (SessionManager.cs:17-24):
```csharp
var sharedRoot = Path.Combine(outputRoot, "shared_outputs");  // ❌ Wrong
var sharedDbcRoot = Path.Combine(sharedRoot, "dbc");          // ❌ Wrong
var sharedCrosswalkRoot = Path.Combine(sharedRoot, "crosswalks");

var sessionAdtDir = Path.Combine(sessionRoot, "adt");         // ❌ Not "03_adts"
var sessionAnalysisDir = Path.Combine(sessionRoot, "analysis"); // ❌ Not "04_analysis"
var sessionViewerDir = Path.Combine(sessionRoot, "viewer");   // ❌ Not "05_viewer"
```

**Problem**:
- Uses `shared_outputs/` instead of numbered session subdirectories
- Missing `01_dbcs/`, `02_crosswalks/` prefix numbering
- Creates parallel "shared" structure not in spec

---

### 4. ViewerStageRunner - Stub Only ❌

**Spec Says** (Page 176-199):
```csharp
public class ViewerServer : IDisposable
{
    private HttpListener? _listener;
    
    public void Start(string viewerDir, int port = 8080) { ... }
    public void Stop() { ... }
}
```

**Current Implementation** (ViewerStageRunner.cs):
```csharp
internal sealed class ViewerStageRunner
{
    public ViewerStageResult Run(SessionContext session, 
                                 IReadOnlyList<AdtStageResult> adtResults)
    {
        // TODO: Implement viewer generation
        return new ViewerStageResult(Success: true, ViewerPath: session.Paths.ViewerDir);
    }
}
```

**Problem**:
- No web server implementation
- No viewer HTML generation
- No overlay JSON generation
- Just returns success without doing anything

---

### 5. Missing CLI Arguments ❌

**Spec Says** (Page 348-359):
```
--maps <map1,map2,...>           Required
--versions <v1,v2,...>           Required
--alpha-root <path>              Required
--output <path>                  Default: parp_out
--dbd-dir <path>
--lk-dbc-dir <path>
--serve                          Start web server    ← Missing
--port <number>                  Web server port     ← Missing
--verbose
--help
```

**Current Implementation**: No `--serve` or `--port` support

---

## Detailed Gap Matrix

| Feature | Spec (Page) | Status | Implementation | Gap Description |
|---------|-------------|--------|----------------|-----------------|
| **Module Architecture** |
| DbcModule project | 81-122 | ❌ Missing | N/A | Entire project missing |
| AdtModule project | 129-166 | ❌ Missing | N/A | Entire project missing |
| ViewerModule project | 174-199 | ❌ Missing | N/A | Entire project missing |
| Core shared utilities | 420-447 | ⚠️ Stub | Class1.cs only | No actual utilities |
| **DBC Stage** |
| DbcOrchestrator API | 86-118 | ❌ Missing | Uses CLI directly | Should be library API |
| DumpAreaTables method | 99-107 | ❌ Missing | Direct CLI call | Not wrapped |
| GenerateCrosswalks method | 109-118 | ❌ Missing | Direct CLI call | Not wrapped |
| DbcDumpResult model | 120 | ❌ Missing | Inline records | Not proper module |
| CrosswalkResult model | 121 | ❌ Missing | Inline records | Not proper module |
| **ADT Stage** |
| AdtOrchestrator API | 132-157 | ❌ Missing | Direct pipeline call | Should be wrapped |
| ConvertAlphaToLk method | 138-145 | ⚠️ Partial | AdtStageRunner.RunExport | No module wrapper |
| ExtractTerrainFromLkAdts | 149-156 | ❌ Missing | N/A | Not implemented |
| AdtConversionResult model | 159-164 | ⚠️ Partial | AdtStageResult | Wrong namespace |
| TerrainCsvResult model | 166 | ❌ Missing | N/A | Not implemented |
| **Viewer Stage** |
| ViewerServer class | 176-192 | ❌ Missing | N/A | Entire class missing |
| HttpListener implementation | 186-189 | ❌ Missing | N/A | No server code |
| Start/Stop methods | 187-191 | ❌ Missing | N/A | Not implemented |
| Static file serving | 188 | ❌ Missing | N/A | Not implemented |
| **Pipeline Orchestration** |
| PipelineOrchestrator | 209-252 | ⚠️ Partial | Basic structure | Wrong stage APIs |
| RunFullPipeline method | 216-251 | ⚠️ Partial | Run method | Different signature |
| Stage coordination | 221-239 | ⚠️ Partial | Present | Uses wrong APIs |
| Manifest writing | 242 | ✓ Present | ManifestWriter | Implemented |
| **Session Management** |
| SessionDirectories model | 258-267 | ⚠️ Different | SessionPaths | Wrong structure |
| CreateSession method | 271-297 | ⚠️ Different | SessionManager | Wrong dirs |
| Directory numbering (01_, 02_) | 278-285 | ❌ Missing | No prefixes | Uses wrong names |
| shared_outputs separation | N/A in spec | ❌ Wrong | Present | Not in spec |
| **CLI Interface** |
| Program.Main entry point | 305-344 | ⚠️ Partial | Present | Missing --serve |
| ParseArguments | 309 | ✓ Present | PipelineOptionsParser | Implemented |
| ConfigureLogging | 312 | ❌ Missing | N/A | Not implemented |
| --serve flag | 326 | ❌ Missing | N/A | Not supported |
| --port flag | 330 | ❌ Missing | N/A | Not supported |
| ViewerServer.Start call | 328-329 | ❌ Missing | N/A | No server |
| WaitForCancellation | 332 | ❌ Missing | N/A | Not needed (no server) |
| **Output Structure** |
| 01_dbcs/ directory | 278 | ❌ Wrong | shared_outputs/dbc | Wrong path |
| 02_crosswalks/ directory | 279 | ❌ Wrong | shared_outputs/crosswalks | Wrong path |
| 03_adts/ directory | 280 | ❌ Wrong | session_*/adt | No prefix |
| 04_analysis/ directory | 281 | ❌ Wrong | session_*/analysis | No prefix |
| 05_viewer/ directory | 282 | ❌ Wrong | session_*/viewer | No prefix |
| logs/ directory | 284 | ✓ Present | session_*/logs | Correct |
| manifest.json | 285 | ✓ Present | session_*/manifest.json | Correct |
| **Testing & Validation** |
| Unit test projects | 366-371 | ❌ Missing | N/A | No tests |
| Integration tests | 380-397 | ❌ Missing | N/A | No tests |
| Smoke test script | N/A | ⚠️ Basic | run-smoke-orchestrator.ps1 | Present but untested |

---

## Phase Completion Status

### Phase 1: Extract Core Libraries (Day 1-2)
**Status**: ❌ 0% Complete - Entirely Skipped

- [ ] 1.1 Create WoWRollback.DbcModule
- [ ] 1.2 Create WoWRollback.AdtModule
- [ ] 1.3 Create WoWRollback.ViewerModule

### Phase 2: Build Unified Orchestrator (Day 2-3)
**Status**: ⚠️ 40% Complete - Wrong Implementation

- [x] 2.1 Create PipelineOrchestrator (exists but wrong APIs)
- [x] 2.2 Session Management (exists but wrong structure)
- [~] 2.3 CLI Interface (exists but missing --serve/--port)

### Phase 3: Testing & Migration (Day 3-4)
**Status**: ❌ 0% Complete - Not Started

- [ ] 3.1 Unit Tests
- [ ] 3.2 Integration Test
- [ ] 3.3 Parallel Operation
- [ ] 3.4 Cutover Plan

---

## Critical Path to Specification Compliance

### Step 1: Create Missing Module Projects ⭐ CRITICAL
1. Create `WoWRollback.DbcModule/DbcOrchestrator.cs`
2. Create `WoWRollback.AdtModule/AdtOrchestrator.cs`
3. Create `WoWRollback.ViewerModule/ViewerServer.cs`
4. Populate `WoWRollback.Core` with shared utilities

### Step 2: Refactor DbcStageRunner ⭐ CRITICAL
- Remove direct CLI instantiation
- Call `DbcOrchestrator.DumpAreaTables()` and `GenerateCrosswalks()`
- Use library APIs, not CLI commands

### Step 3: Fix Output Directory Structure ⭐ HIGH
- Remove `shared_outputs/` concept
- Add numbered prefixes: `01_dbcs/`, `02_crosswalks/`, etc.
- Match spec exactly (Page 60-73)

### Step 4: Implement Viewer Server ⭐ HIGH
- Create `ViewerServer` class with `HttpListener`
- Implement static file serving
- Add `--serve` and `--port` CLI flags
- Wire into Program.cs

### Step 5: Implement Viewer Generation
- Generate overlay JSONs from terrain CSVs
- Create index.html
- Copy cached maps
- Proper ViewerStageRunner implementation

### Step 6: Add Proper Logging
- ConfigureLogging method
- Structured logging to console + file
- Separate log files per stage

### Step 7: Testing
- Unit tests for each module
- Integration test (Shadowfang end-to-end)
- Verify output structure matches spec

---

## Recommended Action

### Option A: Fix Current Implementation (3-4 days)
Follow the spec strictly:
1. Create all missing module projects (Day 1-2)
2. Refactor StageRunners to use modules (Day 2)
3. Fix output structure (Day 2)
4. Implement viewer server (Day 3)
5. Add tests (Day 3-4)

### Option B: Start Over (2-3 days)
Delete current implementation and follow spec from Phase 1:
1. Phase 1.1-1.3: Create modules (Day 1-2)
2. Phase 2: Wire into orchestrator (Day 2-3)
3. Phase 3: Test and validate (Day 3)

**Recommendation**: Option A - Fix incrementally while preserving working code

---

## Success Criteria (From Spec)

From page 451-474, the implementation must meet:

### Functional Requirements
- [ ] Single `dotnet run` command executes full pipeline
- [~] All outputs in predictable `parp_out/session_*/` structure (wrong structure)
- [ ] AreaIDs correctly patched in CSVs
- [ ] Viewer loads in browser at `http://localhost:8080`
- [ ] Overlays render correctly on maps
- [ ] Works cross-platform

### Technical Requirements
- [x] No PowerShell dependency (orchestrator is C#)
- [x] No Python dependency (no web server yet)
- [ ] No shell exec for main tools (DbcStageRunner uses CLI)
- [ ] All file paths under control (shared_outputs is wrong)
- [ ] Comprehensive error handling (basic)
- [ ] Unit test coverage ≥70% (0%)

**Current Score**: 2/12 (17%)

---

## End of Gap Analysis

**Bottom Line**: The implementation is approximately **30-40% complete** and deviates significantly from the specification. Phase 1 (module extraction) was entirely skipped, resulting in architectural debt that must be addressed before the orchestrator can be considered production-ready.
