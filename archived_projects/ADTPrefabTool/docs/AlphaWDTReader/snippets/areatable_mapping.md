# AreaTable.dbc Remapping (0.5.3 → 3.3.5)

Purpose: Rewrite Alpha AreaIDs to their 3.3.5 equivalents so editors/viewers show proper area names instead of "Unknown".

## Inputs
- test_data/0.5.3/.../AreaTable.dbc
- test_data/3.3.5/.../AreaTable.dbc
- DBCD loader (preferred) or CSV fallback

## Strategy
- Load both versions into maps:
  - alphaId → alphaName
  - lkNameNorm → list of lkRecords { id, name, mapId, parentName? }
- Normalize names to maximize matches:
  - Trim, collapse whitespace
  - Lower-case
  - Strip color codes and localization markers
- Disambiguation rules (in order):
  1) Prefer same continent/mapId when available from ADT tile context
  2) Prefer same parent area name (if present in both)
  3) Otherwise pick first and record a warning
- Fallback: if no match by name, keep original AreaID and log `status=unmapped`.

## Integration
- When writing each `MCNK` AreaID, call `AreaTableMapper.RemapAreaId(alphaId, mapContext)`.
- Export `areatable_remap.csv`: `tile_x,tile_y,alpha_id,alpha_name,lk_id,lk_name,status,note`.

## Notes
- Only AreaTable is remapped; doodad/WMO placements remain intact.
- Do not alter the ADT structure; just switch the ID values before writing.
- Use real data from `test_data/` to validate coverage and detect ambiguous cases.
