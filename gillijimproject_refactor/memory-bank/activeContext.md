# Active Context

- **Current Focus**: Stop cross-map leakage in Alpha (0.5.x) → LK (3.3.5) mapping, with emphasis on prototype maps such as Kalidar (`mapId=17`) defaulting to zero when no LK AreaTable entry exists. Stabilize DBCTool crosswalk CSVs so AlphaWDTAnalysisTool can rely exclusively on map-locked matches.
- **Recent Changes**:
  - Added `s_forceZeroMapIds` / `s_forceZeroMapNames` and an early exit in `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` so Kalidar tiles always write `method=map_forced_zero`.
  - Updated write logic to treat any unmatched candidate as `unmapped_zero`, preventing inherited LK IDs.
  - Captured the forced-zero requirement in `docs/planning/003-areaid-regression-plan.md` and the memory bank project brief.
  - Regenerated AlphaWDT outputs for Kalidar to validate that CSV `areaid_verify_*` rows now record `lk_areaid = -1` and on-disk `0`.
- **Next Steps**:
  - Sweep other prototype or developer maps; extend the forced-zero tables as similar gaps are discovered.
  - Re-run DBCTool crosswalk generation after zero-overrides to confirm LK IDs are absent for Kalidar in CSV outputs.
  - Wire stricter validation into AlphaWDTAnalysisTool logging/tests to fail when a forced-zero map produces non-zero LK IDs.
  - Once mapping is stable, resume the GillijimProject.Core refactor and public API design.
- **Decisions**:
  - Library-first architecture (CLI delegates to library) remains the end goal but its implementation is paused until mapping is deterministic.
  - FourCC handling, exception strategy, and immutable domain guidance remain unchanged.
