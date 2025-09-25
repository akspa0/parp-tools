# AreaID Mapping Regression Plan

## Objectives
- Document the legacy AreaID patching pipeline that successfully preserved zone/sub-zone fidelity on map 0.
- Contrast the legacy pipeline with the current implementation to expose the regression.
- Define concrete actions to regain functional parity while supporting multiple maps.
- Establish validation steps to prevent future regressions.

## Legacy Success Factors
- **Ordered Strategy Chain**: `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` in `older_working_version/AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Export/AdtWotlkWriter.cs` evaluated mappings in this sequence:
  - **Sub-zone lookup** via `TryMapSubZone(zoneBase, subLo, currentMapId, …)`
  - **Per-map numeric mapping** via `TryMapBySrcAreaSimple(mapName, alphaAreaNumber, …)`
  - **Global numeric fallback** via `TryMapBySrcAreaNumber(alphaAreaNumber, …)`
  - **Zone-level fallback** via `TryMapZone(zoneBase, currentMapId, …)`
  - **Target map/name checks** using `TryMapByTarget` and `TryMapByTargetName`
- **Child Preference**: `_byTgtMapX` and `_byTgtNameX` in `DbcPatchMapping` preferred LK children via `_tgtParentById`, ensuring sub-zones did not collapse to parent zones.
- **Implicit Map Bias**: `TryMap` fell back to scanning all `_bySrcMap` entries and preferred map IDs 0/1. This accidentally produced correct results for Eastern Kingdoms but failed when new maps were introduced.

## Regression Causes
- **Helper Removal**: Public helpers (`TryMapZone`, `TryMapSubZone`, `TryMapBySrcAreaNumber`, `TryGetMidInfo`) were removed in the refactor (`AlphaWdtAnalyzer.Core/Export/DbcPatchMapping.cs`), so the ordered strategy chain could no longer reach zone/sub or mid data.
- **Stricter Crosswalks**: DBCTool v3 split outputs per map and relied on `_midBySrcArea` / `_childIdsByTarget`. Without helper access, the runtime never consumed the new metadata.
- **Map-Locking Changes**: New map-aware routines avoided the old map-0 bias but exposed the missing helper logic, resulting in extensive unmapped tiles on map 1.

## Improvement Plan
- **Step 1 – Restore Helper Surface** *(done in current session)*
  - Re-introduce helper methods in `AlphaWdtAnalyzer.Core/Export/DbcPatchMapping.cs` (`TryMapZone`, `TryMapSubZone`, `TryMapBySrcAreaNumber`, `TryGetMidInfo`, `TryResolveSourceMapId`).
  - Ensure `_midBySrcArea` lookups respect map IDs via `TryResolveSourceMapId` before inserting fallback `-1` keys.

- **Step 2 – Verify Strategy Ordering**
  - Audit `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` to confirm the restored helpers are called before numeric fallbacks.
  - Add targeted logging (guarded by `Verbose`) to surface method hits per tile and validate mid-chain usage.

- **Step 3 – Crosswalk Consistency Checks**
  - Regenerate DBCTool outputs for maps 0 and 1.
  - Inspect `dbctool_out/<alias>/compare/v3/Area_hierarchy_src_*.yaml` to confirm sub-zone listings align with expectations.
  - Spot-check `Area_patch_crosswalk_*` CSVs for `mid060_*` columns referencing new pivots.

- **Step 4 – Map-Locked Validation Runs**
  - Execute AlphaWDTAnalysisTool on:
    - **Map 0** sample (e.g., `ElwynnForest_31_49`) and verify `method=patch_csv_sub` / `patch_csv_zone_via060` hits.
    - **Map 1** sample (e.g., `Darkshore_19_12`) to ensure no map-0 leakage occurs.
  - Compare results with older build outputs using `tools/agg-area-verify.ps1`.

- **Step 5 – Safeguards**
  - Add unit/integration tests that load precomputed CSV snippets and assert helper behaviors (sub-zone hits, mid lookups, map locking).
  - Document the intended mapping order inside `AdtWotlkWriter.cs` to future-proof against helper removal.

## Open Questions
- **DBCTool Coverage**: Do we need additional CSVs for zones lacking via060 rows (e.g., Kalimdor gaps)? Investigate 0.6.0 pivots before adding heuristics.
- **Map Name Resolution**: Should map names be resolved via `TryResolveSourceMapId` earlier to improve `_midBySrcArea` keys?

## Validation Checklist
- **CSV Regeneration**: `dotnet run --project DBCTool.V2 --inputs ...` producing per-map crosswalks with child listings.
- **Tile Patch Run**: `dotnet run --project AlphaWDTAnalysisTool --remap ... --verbose` for both map 0 and map 1 tiles.
- **Verification Script**: `tools/agg-area-verify.ps1` comparisons between newer and older outputs.
- **Spotlight Logs**: Confirm verbose output shows restored helper methods yielding `method=patch_csv_sub_*` entries.
