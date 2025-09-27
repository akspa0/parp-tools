# Progress

- **Works**
  - Core parsing/conversion pipeline (`Utilities.cs`, `WowChunkedFormat.cs`, `Chunk.cs`, `WdtAlpha`, `AdtAlpha`, `AdtLk`) remains stable; verified on representative LK tiles.
  - Forced-zero handling for prototype maps (currently Kalidar `mapId=17`) in `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` prevents inherited LK IDs, with logs showing `method=map_forced_zero` or `unmapped_zero`.
  - DBCTool V2 CSV exports include canonical child metadata and per-map splits required for strict matching.
  - Planning doc `docs/planning/003-areaid-regression-plan.md` updated to document forced-zero policy.

- **Pending**
  - Confirm DBCTool crosswalk regeneration after forced-zero adjustments; ensure `Area_crosswalk_v3_map17_*.csv` contains only zero/placeholder targets.
  - Add regression tests/validation scripts to detect any non-zero LK IDs on forced-zero maps during AlphaWDTAnalysisTool runs.
  - Resume GillijimProject.Core / GillijimProject.Cli refactor once mapping parity is locked down.
  - Integrate Warcraft.NET writer APIs and build test coverage after mapping stabilization.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- **Works**
  - `CompareAreaV2Command` continues to emit map-locked CSVs with optional 0.6.0 pivot data.
  - `Area_hierarchy_335.yaml`, `Area_hierarchy_mid_0.6.0.yaml`, `Area_hierarchy_src_<alias>.yaml` support manual verification of child hierarchies.
  - `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` now exits early for forced-zero maps and writes 0 for any unmatched candidate.

- **Known Issues**
  - Crosswalk CSVs still include `-1` mid entries; name-based inference pending.
  - Visualization tooling (`053-viz`) still relies on ADT metadata rather than new CSV outputs.
  - Need helper wiring (`TryResolveSourceMapId`, unified mid lookup) to guarantee mid data reaches mapping layer for non-prototype maps.

- **Next Steps**
  - Sweep all output directories for lingering non-zero LK IDs on forced-zero maps and add CI guardrails.
  - Extend forced-zero lists if additional prototype maps are identified during testing.
  - Proceed with mid inference and unified crosswalk emission to unblock stable mapping for standard maps.
