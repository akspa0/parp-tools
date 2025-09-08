# Project Brief — AlphaWDTAnalysisTool Memory Bank

## Goal
Patch ADT MCNK.AreaId in exported LK ADTs by mapping Alpha (0.5.x) AreaTable records to LK (3.3.5.12340) AreaTable records. Maintain high-fidelity exports without modifying the core WowFiles library.

## Scope
- Decode Alpha and LK AreaTable.dbc into normalized CSVs using wow.tools.local DBCD and WoWDBDefs definitions.
- Build a robust Alpha→LK AreaID mapping using names, parents, continent ids, and area bits.
- Patch MCNK.AreaId on disk after ADT export using the mapping.
- Persist CSV artifacts under `export-dir/csv/dbc/` for auditability.

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
- `csv/dbc/AreaTable_Mapping_Alpha_to_335.csv` mapping with rationale.

## Non-goals
- No edits to `src/gillijimproject-csharp/WowFiles` for this feature.
- No speculative features beyond AreaTable mapping and patching.
- No new CLI flags; reuse existing options (`--area-alpha`, `--area-lk`, `--dbc-dir`).
