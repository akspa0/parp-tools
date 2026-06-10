# Viewer Overlay Blocking Issue (2025-10-08)

## Summary
- **Problem**: The viewer overlays still depend on legacy placement CSVs and the `OverlayGenerator` (`WoWRollback/AnalysisModule/OverlayGenerator.cs`) can no longer be cleanly rewired to the new master index JSON after the recent `MapMasterIndexWriter` overhaul. Multiple patch attempts left the file in an inconsistent state with compilation errors (`} expected`) and partial logic.
- **Impact**: Current builds fail when trying to generate viewer outputs, and the viewer cannot render placements because JSON overlays are not produced from the normalized master index. Work on CSV reduction is blocked until overlays run off the new JSON.

## Current State
- `MapMasterIndexWriter` now emits normalized per-tile placement JSON (`analysis/master/<map>_master_index.json` and `<map>_id_ranges_by_tile.json`).
- `OverlayGenerator` still mixes CSV parsing, ADT reader stubs, and the new master-index approach. The class exceeds 600 lines, contains duplicate helpers, and the attempted refactor introduced syntax errors.
- The orchestrator (`AnalysisOrchestrator.RunAnalysis`) calls `OverlayGenerator.GenerateFromIndex(...)`, but that method currently fails to compile, so the stage cannot run.

## Blockers
- Need a clean rewrite of `OverlayGenerator` that:
  - Loads master index JSON from `analysis/master/` instead of regenerating overlays from CSV.
  - Writes per-tile overlay JSON under `viewer/overlays/<version>/<map>/objects_combined/`.
  - Keeps CSV fallback logic (optional) without duplicating the primary code path.
- Must restore class structure and remove dead legacy methods (`Generate`, `GenerateTerrainOverlay`, etc.) to eliminate syntax issues.
- Once overlays work, we can safely remove the redundant CSVs and consolidate fixup logs.

## Proposed Plan (next session)
1. **Reset `OverlayGenerator`**: Replace the file with a concise implementation that only consumes `MapMasterIndexWriter` outputs plus optional CSV fallback helpers.
2. **Update call sites**: Adjust `AnalysisOrchestrator` (and any tests) to pass the analysis output directory so the generator can locate `master_index.json`.
3. **Verify build/run**: Re-run `dotnet run --project WoWRollback.Orchestrator ... --serve` to ensure overlays are emitted and the viewer loads.
4. **Follow-up**: After overlays succeed, proceed with CSV cleanup and fixup consolidation.

## Notes
- Keep an eye on `WoWRollback.Core` viewer services once overlays change format.
- Consider adding small unit tests or a debug harness to validate overlay JSON structure before wiring to the UI.

---

## Assessment Documents (Created 2025-10-08)

**Detailed Assessment**: [overlay-generator-assessment.md](overlay-generator-assessment.md)
- Complete architecture analysis
- All 3 compilation errors identified with exact line numbers
- Type relationships and data flow
- Corrected helper method implementations
- Risk assessment and success criteria

**Fix Plan**: [overlay-generator-fix-plan.md](overlay-generator-fix-plan.md)
- 4-phase implementation strategy
- Specific code changes for OverlayGenerator.cs and AnalysisOrchestrator.cs
- Missing helper method implementations
- Dead code removal checklist
- Estimated 30 minutes to complete

**Status**: Ready for implementation in next session (type `ACT` to proceed)

**Architecture Plan**: [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md)
- Complete path/filename/format mismatches identified
- ObjectOverlayManager design (parallel to existing TerrainOverlayManager)
- main.js refactor plan (1033 → <400 lines)
- World → Pixel coordinate transform system
- 4-phase implementation (~14 hours / 2 days)
- Must fix OverlayGenerator first, then proceed with viewer improvements

**⚡ SIMPLIFIED PLAN**: [viewer-json-refactor-plan.md](viewer-json-refactor-plan.md)
- **Discovery**: You already have production-ready coordinate infrastructure!
- CoordinateTransformer.cs + OverlayBuilder.cs already do everything needed
- Just wire OverlayGenerator to delegate to existing OverlayBuilder
- **Total Time: 2 hours** (30 min fix + 1 hour integration + 30 min testing)
- Reuses proven, documented coordinate system from COORDINATES.md
