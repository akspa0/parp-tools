- Utilize YAML hierarchies to normalize map-scoped zone/area lookups prior to applying pivots.
# Plan 002 — Restore Sub-Zone AreaID Mapping (0.5.x → 0.6.0 → 3.3.5)

## Goal
Ensure Alpha (0.5.x) sub-zone AreaIDs map deterministically to their Wrath (3.3.5) counterparts by reintroducing the 0.6.0 pivot and enriching DBCTool / AWDT pipelines with full LK child hierarchy awareness.

## Context

- **Scope constraint**: Only update code under `DBCTool.V2/` and `AlphaWDTAnalysisTool/`; legacy `DBCTool/` remains untouched.
- Commit `ad8cccdc` correctly mapped Azeroth sub-zones but failed for Kalimdor because of 0.5.x ID collisions.
- Subsequent fixes introduced continent map gating, restoring Kalimdor but dropping Azeroth sub-zones.
- Current CSVs now emit child hierarchies (`tgt_child_ids`, `tgt_child_names`), yet Alpha ADTs often expose only zone-level IDs (e.g., Westfall), so exporters still write parent IDs.
- Previous workflow used a 0.6.0 pivot to disambiguate 0.5.x IDs before projecting to 3.3.5.

- Updated DBCTool compare output including 0.6.0 pivot columns across mapping/patch CSVs. **(Done)**
- YAML exports for 0.5.x and 3.3.5 hierarchies plus enriched crosswalk CSV with canonical names. **(Done)**
- Enhanced `DbcPatchMapping` that ingest pivot data, expose `TryMapViaMid060(...)`, and leverage child lists.
- `AdtWotlkWriter` logic that, when Alpha data is ambiguous, resolves through pivot and child hints before falling back to parent IDs.
- Per-map v3 crosswalk outputs aligned with YAML canonical naming to drive downstream tooling.
- Verification runs over Azeroth, Kalimdor, and instance maps (e.g., Deadmines, Wailing Caverns) showing `patch_csv_via060_sub` / `patch_csv_exact` with correct LK child IDs.

## Work Breakdown

### Part A — DBCTool Pivot Revival *(Status: Complete)*
- **Outputs**: `compare/v2/mapping.csv` and `compare/v2/patch*.csv` now emit `mid060_mapId`, `mid060_areaID`, `mid060_parentID`, and `mid060_chain` columns alongside `tgt_child_ids` / `tgt_child_names`.
- **Crosswalk artifacts**: `compare/v3/Area_crosswalk_v3.csv`, per-map `compare/v3/maps/<mapId>.csv`, and per-map unified CSVs incorporate canonical 0.5.x names, pivot chains, and LK targets.
- **YAML exports**: `Area_hierarchy_335.yaml`, `Area_hierarchy_mid_0.6.0.yaml`, and `Area_hierarchy_src_<alias>.yaml` establish the canonical hierarchies that downstream tooling will consume.
- **Follow-up**: Document these artifacts in `docs/` once downstream integration proves stable.

### Part B — DbcPatchMapping Enhancements *(Status: Pending)*
- **Parsing**: Extend `DBCTool.V2/Mapping/DbcPatchMapping.cs` to ingest the new pivot and child columns, populating `_midBySrcArea`, `_tgtByMidArea`, and `_childInfoByTgtZone` caches.
- **APIs**: Expose `TryMapViaMid(...)` and `TryGetChildCandidates(...)` so AWDT can request pivot-assisted and child-aware resolutions.
- **Diagnostics**: Thread method strings (`pivot_060`, `pivot_060_reparent`, etc.) through the mapping results for logging.
- **Validation**: Add focused unit/integration checks to ensure map-locking and child preference rules remain intact.

### Part C — AdtWotlkWriter Integration *(Status: Pending)*
- **Lookup order**: Update `AlphaWDTAnalysisTool`'s `AdtWotlkWriter` to attempt: packed hi/lo direct hit → per-map numeric match → `TryMapViaMid` → child inference via `TryGetChildCandidates` → zone fallback.
- **Logging**: Emit verbose diagnostics (guarded by `--verbose`) showing chosen method, candidate child IDs, and map context.
- **Remap behavior**: Preserve remap-only writes; ensure zero or unmapped areas remain 0 when all strategies fail.

### Part D — Verification & Tooling *(Status: Pending)*
- **DBCTool runs**: Regenerate compare outputs for 0.5.3, 0.6.0, and 3.3.5 with the enhanced mapping logic enabled.
- **Patch validation**: Re-run AlphaWDTAnalysisTool over representative tiles (Azeroth: Goldshire/Westfall; Kalimdor: Durotar/Ashenvale; Instances: Deadmines, Wailing Caverns, StormwindStockade) and inspect remap logs.
- **Evidence capture**: Archive `--verbose` traces and summarize before/after LK AreaID distributions for docs.
- **Regression harness**: Add spot tests or scripts that fail if a known sub-zone regresses to its parent ID.

## Risks & Mitigations
- **Pivot optional files missing**: fail gracefully with clear diagnostics; fallback to current behavior.
- **CSV schema changes**: update downstream consumers/tests simultaneously to avoid deserialization errors.
- **Performance**: cache pivot and child lookups to avoid per-tile recomputation.
- **Regression**: maintain existing map-ID gating to preserve Kalimdor fixes; add unit tests for both continents plus instances.

## Next Steps
1. Finish Part B by wiring pivot-aware caches and helper APIs inside `DBCTool.V2/Mapping/DbcPatchMapping.cs`.
2. Update `AlphaWDTAnalysisTool` exporters (Part C) to consume the new mapping APIs and log chosen strategies.
3. Run the Part D verification suite (DBCTool compare + AWDT remap smoke tests) and capture evidence for documentation.
