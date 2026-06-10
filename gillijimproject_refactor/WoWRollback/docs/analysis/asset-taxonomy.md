# Asset Taxonomy (Heuristics)

Purpose: enrich CSVs and viewer filters with a coarse asset type derived from file paths and names.

## Proposed Types
- tree
- bush
- rock
- prop
- building
- cave
- dungeon
- wmo_misc
- m2_misc

## Heuristic Rules (initial)
- M2 (models):
  - world/tree/, world/vegetation/ → tree/bush
  - world/rocks/, world/stone/ → rock
  - world/props/, world/generic/ → prop
- WMO (world models):
  - wmo/buildings/ → building
  - wmo/dungeons/, wmo/instances/ → dungeon
  - wmo/caves/ → cave
  - otherwise → wmo_misc
- Fallbacks:
  - If extension is .m2 and no rule matched → m2_misc
  - If extension is .wmo and no rule matched → wmo_misc

## CSV Columns to Add (where applicable)
- asset_type (string): taxonomy type from heuristics
- source (enum): m2|wmo
- path (string): original resolved path

## Viewer Filters (planned)
- Filter overlays by `asset_type`
- Preset bundles can list `filters.assetTypes` to preselect types

## Notes
- Keep rules deterministic and easily testable.
- Start conservative (avoid over-classification).
- Revisit with data samples; add per-map exceptions if needed.
