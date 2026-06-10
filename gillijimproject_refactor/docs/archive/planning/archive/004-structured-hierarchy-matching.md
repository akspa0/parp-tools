# Plan 004 — Structured Hierarchy Matching (DBCTool.V2)

## Goal
Build the "v3" area crosswalk matcher on top of the generated YAML hierarchies so that 0.5.3 zones/subzones map deterministically onto their 3.3.5 counterparts (including `UNUSED` legacy nodes) without relying on fuzzy string heuristics.

## Context
- `compare/v3/Area_hierarchy_src_0.5.3.yaml` and `Area_hierarchy_335.yaml` already encode map → zone → subzone structure with canonical names.
- Current matching in `DBCTool.V2/Cli/CompareAreaV2Command.cs` still operates on flat CSV indices, leading to mis-maps (e.g., colon names, UNUSED children, map drift).
- We want a reusable pairing data set that future tooling (Alpha WDT patcher, audit reports) can consume.

## Deliverables
1. Hierarchy parser that loads YAML into strongly-typed graph models (map, zone, subzone, metadata).
2. Structured pairing generator that walks 0.5.3 graph and records candidate 3.3.5 matches with reasons/confidence.
3. Updated matcher that consumes the structured pairs instead of ad-hoc rename/fuzzy logic.
4. Diagnostics (CSV/JSON) highlighting ambiguous or unresolved mappings for manual review.

## Milestones & Steps

### Phase A — Hierarchy ingestion
- **A1** Read YAML files and deserialize into `HierarchyGraph` models (maps, zones, children, attributes like `UNUSED`).
- **A2** Unit-test deserialization against known samples (e.g., map 36 Deadmines, map 14 Durotar/Orgrimmar UNUSED).

### Phase B — Pair discovery
- **B1** For each source zone/subzone, build normalized keys (raw, colon → "Parent: Child", alias tokens).
- **B2** Locate LK candidates on the same map using hierarchy data (prefer exact name, then alias, then UNUSED child).
- **B3** Emit structured record: source node, candidate LK target(s), reason (`exact`, `alias`, `unused_child`, `ambiguous`), map IDs, parent chains.
- **B4** Surface ambiguities (multiple viable targets) in a review CSV.

### Phase C — Matcher integration
- **C1** Add new pairing loader to `CompareAreaV2Command` (or dedicated service) that consumes the structured records.
- **C2** Replace rename/fuzzy branches with lookups into the pairing table, keeping map/parent validation strict.
- **C3** Update verbose logging to reference pairing IDs and reasons instead of ad-hoc method labels.

### Phase D — Validation
- **D1** Re-run `--compare-area-v2` and confirm key regressions (Deadmines, Orgrimmar UNUSED, colon zones) now map correctly.
- **D2** Execute AlphaWDT → ADT pipeline and verify AreaID patch logs show `[AreaMap]` entries resolving to expected LK IDs.
- **D3** Document remaining ambiguous cases (if any) and note follow-up work.

## Open Questions
- Do we store structured pairs as CSV, JSON, or serialized YAML for downstream tools?
- How do we merge manual overrides (if needed) into the pairing data without rerunning discovery?

## Next Action
Proceed with Phase A: implement YAML hierarchy loader plus smoke tests, then integrate into the compare command.

## Progress Log
- **Phase A**: Loader implemented (`DBCTool.V2/Mapping/AreaHierarchyLoader.cs`, `AreaHierarchyGraph.cs`) and wired into `CompareAreaV2Command`. Optional YAML ingestion now occurs when hierarchy files exist. Tests deferred per user request.
- **Phase B**: Pending.
- **Phase C**: Pending.
- **Phase D**: Pending.
