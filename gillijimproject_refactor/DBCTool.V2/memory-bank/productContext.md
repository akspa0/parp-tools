# Product Context — DBCTool.V2

Why this exists:
- Provide deterministic AreaTable remapping from Alpha/Classic to 3.3.5 for modding, analysis, and patch workflows.
- Remove cross-map noise by locking to the source row’s continent, yet still recover matches via safe, labeled rename fallbacks when needed.

How it should work:
- Input: Alpha/Classic AreaTable + Map DBCs and LK (3.3.5) DBCs.
- Output: Auditable CSVs (mapping/unmatched/patch), plus raw dumps and diagnostics.
- Contract: Never emit cross-map results during primary matching; any cross-map is explicit and labeled (`rename_*`).

User experience goals:
- CSVs are stable and predictable with headers consistent across files.
- Diagnostics make investigation straightforward (`*dump*.csv`, `zone_missing_in_target_map*.csv`).
- A small API surface (AreaIdMapperV2) enables reuse in other tools.
