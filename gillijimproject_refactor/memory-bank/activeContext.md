# Active Context

- Current Focus: Restore sub-zone AreaID fidelity for Alpha (0.5.x) → LK (3.3.5) exports by enriching DBCTool crosswalks with child hierarchies, reintroducing the 0.6.0 pivot, normalizing 0.5.x map-scoped zone definitions, and keeping mid-pivot lookups map-locked.
- Recent Changes:
  - Achieved 1:1 C++ → C# parity for Alpha WDT → Wrath ADT conversion.
  - Fixed MMID/MWID off-by-one, MCVT FourCC orientation, MH2O omission, and MCNK `MCLQ` ordering; outputs validated with 010 templates.
  - Updated `CompareAreaV2Command` to emit `tgt_child_ids` / `tgt_child_names` so LK hierarchies are preserved in CSV outputs.
  - `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` now preserves existing alpha AreaIDs when no LK mapping exists (`method=unmapped_preserve`).
  - `DbcPatchMapping` persists mid entries keyed by `(src_mapId, src_areaNumber)` and `(mid_mapId, mid_areaId)` to avoid cross-continent collisions.
- Next Steps:
  - Finalize helper plumbing in `DbcPatchMapping` (`TryResolveSourceMapId`, shared `TryGetMidEntry`) and update `AdtWotlkWriter` to pass the correct source map ID into mid lookups.
  - Re-run AlphaWDTAnalysisTool on Azeroth after the helper wiring to verify Kalimdor pivots (e.g., Darkshore) no longer appear.
  - Regenerate crosswalks and verify instance maps (Deadmines, Wailing Caverns, etc.) patch to the correct LK child IDs once pivots are stable.
  - Continue long-term refactor toward `GillijimProject.Core` / `GillijimProject.Cli` once mapping stability is achieved.
- Decisions:
  - Library-first architecture; CLI delegates to library.
  - Forward FourCC in memory; reversed on disk by serializer.
  - Exceptions for error handling; immutable domain where practical; builders/writers handle mutation.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- Recent Changes
  - `CompareAreaV2Command` now writes LK child listings to the mapping CSVs, capturing full hierarchies per zone.
  - Added `Area_hierarchy_335.yaml`, `Area_hierarchy_mid_0.6.0.yaml`, and map-scoped `Area_hierarchy_src_<alias>.yaml` exports to inspect canonical zone/sub-zone relationships.
  - `Area_crosswalk_v3` and `Area_crosswalk_via060` CSVs now carry canonical source and pivot names, plus per-map splits.

- Current Focus
  - Implement mid-ID inference fallbacks and build unified crosswalk outputs linking 0.5.x → 0.6.0 → 3.3.5.

- Next Steps
  - Infer missing `mid060_*` IDs using canonical name lookups before writing unified outputs.
  - Emit unified per-map CSV/YAML linking source → mid → target IDs.
  - Teach `DbcPatchMapping`/`AdtWotlkWriter` to consume unified data and verify outdoor + instance tiles with new mappings.
