# Active Context - WoWRollback Unified Orchestrator Refactor

## Current Focus (2025-10-07)
**Unified Orchestrator Implementation - Replacing PowerShell/Python Orchestration**

We are executing a **major architectural refactor** to replace fragmented PowerShell and Python orchestration with a single unified C# application (`WoWRollback.Orchestrator`). This is based on the detailed plan in `docs/refactor/WoWRollback-Unified-Orchestrator.md`.

## What We're Building
A single executable that orchestrates the entire Alpha→Lich King pipeline:
1. **DBC Analysis** (via DBCTool.V2) - Extract and compare AreaTables
2. **ADT Conversion** (via AlphaWdtAnalyzer.Core) - Convert Alpha WDT/ADT → LK format with AreaID patching
3. **Viewer Generation** - Generate static HTML viewer with overlays
4. **Embedded HTTP Server** - Serve viewer directly from orchestrator

## Recent Changes (Session: 2025-10-07)

### ✅ Phase 1 Complete: Module Architecture Created
Created three new module projects to wrap existing tools as library APIs:

1. **WoWRollback.DbcModule** (NEW)
   - `DbcOrchestrator.cs` - Wraps DBCTool.V2 CLI commands
   - `DumpAreaTables()` - Returns structured result instead of exit code
   - `GenerateCrosswalks()` - Returns paths to maps.json and crosswalk CSVs

2. **WoWRollback.AdtModule** (NEW)
   - `AdtOrchestrator.cs` - Wraps AlphaWdtAnalyzer.Core.Export
   - `ConvertAlphaToLk()` - Clean API for ADT conversion
   - `ConversionOptions` - Typed configuration instead of massive option struct

3. **WoWRollback.ViewerModule** (NEW)
   - `ViewerServer.cs` - Embedded HTTP server using HttpListener
   - `Start(viewerDir, port)` - Serves static files without external dependencies
   - `Stop()` / `Dispose()` - Clean shutdown

### ✅ Phase 2 Complete: Infrastructure Fixed

4. **WoWRollback.Core** - Populated with shared utilities
   - `IO/FileHelpers.cs` - Directory copy operations (extracted from DbcStageRunner)
   - `Logging/ConsoleLogger.cs` - Structured console output with timestamps
   - `Models/SessionManifest.cs` - Session metadata models for manifest.json

5. **SessionManager** - **CRITICAL FIX**: Correct output structure per spec
   - ✅ NOW: `session_YYYYMMDD_HHMMSS/01_dbcs/`, `02_crosswalks/`, `03_adts/`, `04_analysis/`, `05_viewer/`, `logs/`
   - ❌ BEFORE: `shared_outputs/dbc/`, `shared_outputs/crosswalks/` (WRONG)
   - All stage runners updated to use new paths

### 🚧 Current Work: Day 3 - Wire Modules into Orchestrator

**Next immediate tasks:**
1. Refactor `DbcStageRunner` to call `DbcOrchestrator` API (not CLI directly)
2. Refactor `AdtStageRunner` to call `AdtOrchestrator` API
3. Implement `ViewerStageRunner` with HTML and overlay generation
4. Add `--serve` and `--port` CLI flags and wire `ViewerServer`

## Key Architectural Decisions

### Module Separation
- DbcModule, AdtModule, ViewerModule are **thin wrappers** around existing tools
- They provide clean typed APIs instead of exit codes and shell execution
- Orchestrator consumes these modules, not CLI commands directly

### Output Structure
Per spec (page 60-73), all outputs go into numbered session subdirectories:
```
parp_out/
└─ session_20251007_001830/
   ├─ 01_dbcs/           ← DBC dumps per version
   ├─ 02_crosswalks/     ← Maps.json and area mapping CSVs
   ├─ 03_adts/           ← Converted ADTs per version/map
   ├─ 04_analysis/       ← Terrain CSVs (future)
   ├─ 05_viewer/         ← Static HTML viewer
   ├─ logs/              ← Per-stage logs
   └─ manifest.json      ← Session metadata
```

### Testing Strategy
- Unit tests for each module (DbcModule, AdtModule, ViewerModule)
- Integration test: Shadowfang 0.5.3 end-to-end
- Verify output structure matches spec exactly

## Challenges & Solutions

### Challenge: CLI Command Dependencies
**Problem**: DbcStageRunner was calling `new DumpAreaCommand()` and `new CompareAreaV2Command()` directly
**Solution**: Created DbcOrchestrator wrapper that provides library API, still uses CLI commands internally for now

### Challenge: Wrong Output Structure
**Problem**: Original implementation used `shared_outputs/` concept not in spec
**Solution**: Fixed SessionManager to use numbered directories per spec (01_, 02_, etc.)

### Challenge: No Web Server
**Problem**: ViewerStageRunner was a stub returning success without doing anything
**Solution**: Created ViewerModule with HttpListener-based server for zero dependencies

## Non-Negotiables
- Match spec exactly (docs/refactor/WoWRollback-Unified-Orchestrator.md)
- No shell execution for main pipeline tools (use library APIs)
- Predictable output structure (numbered directories)
- All file paths under orchestrator control
