# Progress

- Works:
  - CLI scaffolding and argument parsing.
  - Filesystem-only DBC export via DBCD and a CSV writer.
  - Area comparison pipeline (`--compare-area`) with map crosswalk and robust name matching.
  - Deterministic remap workflow: discover, export to JSON (`--export-remap`), and apply from JSON (`--apply-remap`).
  - Filtering of development placeholders and exclusion of "DO NOT USE" targets by default.
  - Support for multiple source builds (`0.5.3`, `0.5.5`, `0.6.0`) targeting `3.3.5`.
  - ADT documentation aligned: Alpha MCNK area packs zone/sub in uint16 halves (hi16=zone, lo16=sub).
  - Alpha decode and LK suggestions CSVs under `out/compare/`.
  - Strict zone/sub suggestion rules:
    - Zones are map-locked and top-level-only (ParentAreaID==ID). No global fallback for zones.
    - Sub-zones are matched within the chosen zone and same map; NameVariants exact first ("name"/"name_alias"), then fuzzy.
    - `alpha_raw == 0` rows are emitted with method `alpha_zero` and no LK IDs.

- Completed Parts A–E:
  - A: Remap export restricted to explicit-only.
  - B: Added `ChooseSubWithinZone(subName, zoneId)`.
  - C: `alpha_areaid_decode.csv` (strict hi16/lo16 + parent validation).
  - D: `alpha_to_335_suggestions.csv` (zone: map-locked, top-level; sub: within chosen zone; annotated methods).
  - E: Wrote CSVs and console logs.

- Next:
  - Author explicit remaps based on reviewed suggestions (export remains explicit-only).
  - Use writer verification CSVs (from AlphaWDTAnalysisTool in verbose mode) to confirm on-disk writes.
  - Continue tightening docs for external consumers.

- Known Issues / Follow-ups:
  - None blocking. Suggestions are advisory; only explicit remaps affect writes.

## 2025-09-15 — Collision remediation

- Issue
  - Map 0 rows were mapped under Azshara (map 1) due to name-only mapping allowing cross-map parents and sub-zones.

- Root cause
  - Mapping loop lacked hard map-locking for zones, did not anchor sub selection to the chosen zone, and had no cross-map guard.

- Fix plan
  - Zones: map-locked and top-level-only.
  - Subs: constrained to the chosen zone and same map; do not coerce to zone when sub is missing; mark `zone_only_no_sub`.
  - Cross-map guard demotes any cross-map selection to `unmatched` with `cross_map_violation`.
  - Emit per-map CSVs (map0/map1) in addition to global outputs to make QA explicit.
  - Keep remap export explicit-only; no heuristics.

- Status
  - Documented here; implementation next in `Program.CompareAreas`.
  - AlphaWDTAnalysisTool will validate `explicit_map` before writing and skip any cross-map or DBCD-missing entries, logging them for review.

- Acceptance criteria
  - No cross-map parents in mapping CSVs.
  - Per-map CSVs contain only rows for their respective map.
  - Methods reflect strict matching and rejections (`name+map`, `fuzzy+map`, `zone_only_no_sub`, `cross_map_violation`, `explicit*`).
