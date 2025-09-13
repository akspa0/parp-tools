# Project Brief — AlphaWDTAnalysisTool Memory Bank

## Goal
Patch ADT MCNK.AreaId in exported LK ADTs by mapping Alpha (0.5.x) AreaTable records to LK (3.3.5.12340) AreaTable records. Maintain high-fidelity exports without modifying the core WowFiles library.

## Scope
- Decode Alpha and LK AreaTable.dbc into normalized CSVs using wow.tools.local DBCD and WoWDBDefs definitions.
- Build a robust Alpha→LK AreaID mapping using names, parents, continent ids, and area bits.
- Patch MCNK.AreaId on disk after ADT export using the mapping.
- Persist CSV artifacts under `export-dir/csv/dbc/` for auditability.
- [Current] Apply asset fixups for textures/models in ADTs using safe in-place string patching (no chunk size/offset changes).

## Inputs
- Alpha WDT/ADTs.
- Alpha `AreaTable.dbc` (0.5.x).
- LK `AreaTable.dbc` (hardcoded DBD build: 3.3.5.12340).
- WoWDBDefs definitions and DBCD library (as source deps under `lib/`).
- Community and LK listfiles for asset fixups (already used by pipeline).

## Outputs
- `World/Maps/<Map>/<Map>.wdt` and `<Map>_<x>_<y>.adt` files.
- `csv/maps/<Map>/areaid_mapping.csv` per tile.
- `csv/dbc/AreaTable_Alpha.csv` and `csv/dbc/AreaTable_335.csv` (decoded DBCs).
- `csv/dbc/AreaTable_Mapping_Alpha_to_335.csv` mapping with rationale. (Planned richer content)
- `csv/maps/<Map>/asset_fixups.csv` (fuzzy + capacity diagnostics).

## Non-goals
- No edits to `src/gillijimproject-csharp/WowFiles` for this feature.
- No speculative features beyond AreaTable mapping and patching.
- No new CLI flags; reuse existing options (`--area-alpha`, `--area-lk`, `--dbc-dir`).

## Current Implementation Snapshot
- In-place patchers for ADT name tables:
  - MTEX (BLP textures)
  - MMDX (MDX/M2 model names)
  - MWMO (WMO names)
- Capacity-aware replacements: write only if replacement fits in the original slot; otherwise attempt fallbacks (textures) or skip.
- Specular rule: never map non-`_s` → `_s`; allow `_s` → non-`_s` only when `_s` is missing.
- Extension parity: enforce MDX/M2 extension matching for fuzzy and fallbacks.
- Directory-aware fuzzy for textures, prioritizing same-folder candidates.

## Next Steps
- Log overflow_skip and capacity_fallback diagnostics alongside fuzzy rows.
- AreaTable CSVs: switch to DBCD-driven full exports for Alpha and LK; include richer columns.
- Improve per-tile mapping CSV: include decoded fields and mapping reasoning.

## Future Architecture (Do not implement now)
- Treat ADTs as hierarchical container objects (chunks as parent/child nodes with tracked offsets) to allow safe structural edits with dynamic resizing. This increases memory footprint but enables robust mutation without manual offset auditing.
