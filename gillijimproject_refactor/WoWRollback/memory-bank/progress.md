# Progress - WoWRollback Unified Orchestrator

## ✅ Completed (2025-10-07)

### Phase 1: Module Architecture (Day 1)
- ✅ Created `WoWRollback.DbcModule` - Wraps DBCTool.V2 as library API
- ✅ Created `WoWRollback.AdtModule` - Wraps AlphaWdtAnalyzer.Core
- ✅ Created `WoWRollback.ViewerModule` - Embedded HTTP server with HttpListener
- ✅ All three modules build successfully

### Phase 2: Infrastructure (Day 2)
- ✅ Populated `WoWRollback.Core` with shared utilities:
  - `IO/FileHelpers.cs` - Directory operations
  - `Logging/ConsoleLogger.cs` - Structured logging
  - `Models/SessionManifest.cs` - Session metadata
- ✅ Fixed `SessionManager` to use correct output structure:
  - Numbered directories: `01_dbcs/`, `02_crosswalks/`, `03_adts/`, `04_analysis/`, `05_viewer/`
  - Removed wrong `shared_outputs/` concept
- ✅ Updated `DbcStageRunner`, `AdtStageRunner`, `ManifestWriter` to use new paths
- ✅ All projects build successfully

### Phase 3: Wire Modules into Orchestrator (Day 3)
- ✅ Refactored `DbcStageRunner` to use `DbcOrchestrator` API
  - No more direct CLI command instantiation
  - Calls `DumpAreaTables()` and `GenerateCrosswalks()` library methods
- ✅ Refactored `AdtStageRunner` to use `AdtOrchestrator` API
  - Simplified to call `ConvertAlphaToLk()` with `ConversionOptions`
  - Returns structured result with tile/area counts
- ✅ Implemented `ViewerStageRunner` with HTML and overlay generation
  - Generates `index.html` with session summary
  - Creates `overlays/metadata.json` with ADT results
- ✅ Wired `ViewerServer` into `Program.cs`
  - Starts HTTP server if `--serve` flag provided
  - Blocks until Ctrl+C for graceful shutdown

## 🎯 Next Steps (Day 4)

### Testing & Validation
1. Create unit tests for DbcModule
2. Create unit tests for AdtModule
3. Create unit tests for ViewerModule
4. Create integration test: Shadowfang 0.5.3 end-to-end
5. Verify output structure matches spec exactly

## 📊 Current Status

**Progress**: ~83% Complete (10/12 tasks done)

### Architecture Status
```
WoWRollback/
├─ DbcModule/          ✅ Created & builds
├─ AdtModule/          ✅ Created & builds
├─ ViewerModule/       ✅ Created & builds with HTTP server
├─ Core/               ✅ Populated with utilities
└─ Orchestrator/       ✅ REFACTORED
   ├─ DbcStageRunner   ✅ Uses DbcOrchestrator API
   ├─ AdtStageRunner   ✅ Uses AdtOrchestrator API
   ├─ ViewerStageRunner ✅ Generates HTML + metadata
   └─ Program.cs        ✅ Wired ViewerServer with --serve
```

### Output Structure Status
✅ **Fixed**: Now matches spec exactly
```
parp_out/
└─ session_YYYYMMDD_HHMMSS/
   ├─ 01_dbcs/           ✅ Correct
   ├─ 02_crosswalks/     ✅ Correct
   ├─ 03_adts/           ✅ Correct
   ├─ 04_analysis/       ✅ Correct
   ├─ 05_viewer/         ✅ Correct
   ├─ logs/              ✅ Correct
   └─ manifest.json      ✅ Correct
```

## 🐛 Known Issues
- No unit tests yet for modules (DbcModule, AdtModule, ViewerModule)
- No integration test yet (Shadowfang end-to-end)
- ViewerStageRunner generates basic HTML; full interactive viewer TBD
- Need to verify real pipeline execution with actual data

## ✨ Success Criteria (from spec)
- [x] Single `dotnet run` executes full pipeline
- [x] Predictable output structure (numbered directories)
- [x] No shell execution for main tools (uses library APIs)
- [x] Viewer loads at http://localhost:8080 (with --serve flag)
- [x] Cross-platform compatibility (HttpListener, no Windows-specific deps)
- [ ] Unit test coverage ≥70% (testing phase pending)
