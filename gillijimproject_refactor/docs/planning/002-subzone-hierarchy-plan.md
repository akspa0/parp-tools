- Utilize YAML hierarchies to normalize map-scoped zone/area lookups prior to applying pivots.
# Plan 002 — Restore Sub-Zone AreaID Mapping (0.5.x → 0.6.0 → 3.3.5)

## Goal
Ensure Alpha (0.5.x) sub-zone AreaIDs map deterministically to their Wrath (3.3.5) counterparts by reintroducing the 0.6.0 pivot and enriching DBCTool / AWDT pipelines with full LK child hierarchy awareness.

## Context
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

### Part A — DBCTool Pivot Revival
- Re-enable loading of 0.6.0 AreaTable (opt-in flag or default when available).
- For each mapping row, emit:
  - `mid060_mapId`, `mid060_areaID`, `mid060_parentID`, `mid060_chain` (zone/sub path).
  - Preserve `tgt_child_ids`, `tgt_child_names` columns.
- Update per-map patch CSVs to include pivot IDs so downstream tools can correlate `src_areaNumber → mid → tgt`.
- Add canonical 0.5.x zone/area names and resolved map IDs to v3 crosswalk outputs. **(Done)**

### Part B — DbcPatchMapping Enhancements
- Parse new pivot columns, storing:
  - `_midBySrcArea` → (midAreaId, via060 flag).
  - `_tgtByMidArea` → target LK IDs with child preference logic.
  - `_childInfoByTgtZone` → parsed from `tgt_child_ids` / `tgt_child_names`.
- Provide APIs:
  - `TryMapViaMid(int srcAreaNumber, int? mapId, out int targetId, out string method)`.
  - `TryGetChildCandidates(int targetZoneId)` for diagnostics / overrides.

### Part C — ADT Exporter Integration
- Extend lookup order in `AdtWotlkWriter` to:
  1. Packed hi/lo match (sub-zone).
  2. Per-map direct numeric match.
  3. Pivot-assisted resolution (`patch_csv_via060_sub`).
  4. LK child inference when only parent zone is present (use child list + map context).
  5. Zone fallback.
- Emit detailed verbose logging including chosen strategy and candidate child IDs.

### Part D — Verification & Tooling
- Re-run DBCTool compare for 0.5.3, 0.6.0, 3.3.5.
- Patch ADTs for:
  - Azeroth outdoor tiles (Goldshire, Westfall farms).
  - Kalimdor outdoor tiles (Durotar, Ashenvale).
  - Instances (Deadmines, Wailing Caverns, StormwindStockade).
- Record `--verbose` logs demonstrating child mappings; capture before/after stats for documentation.
- Update docs and memory bank with findings; add regression tests where feasible.
- Generate per-map crosswalk artifacts and attach to docs for manual review.

## Risks & Mitigations
- **Pivot optional files missing**: fail gracefully with clear diagnostics; fallback to current behavior.
- **CSV schema changes**: update downstream consumers/tests simultaneously to avoid deserialization errors.
- **Performance**: cache pivot and child lookups to avoid per-tile recomputation.
- **Regression**: maintain existing map-ID gating to preserve Kalimdor fixes; add unit tests for both continents plus instances.

## Next Steps
1. Implement Part A with schema updates and regenerate CSVs.
2. Integrate Part B + Part C in AWDT with new lookup APIs.
3. Execute Part D verification suite and iterate on any remaining mismatches.
