# DBCTool.V2 — CompareArea V2 Mapping

## Purpose
Document the CompareArea V2 logic and constraints used by `DBCTool.V2`, centered around `DBCTool.V2/Cli/CompareAreaV2Command.cs`.

## Data Sources
- Source storages: `AreaTable`, `Map` for src (0.5.3/0.5.5/0.6.0) and target (3.3.5).
- Crosswalks: `Build053To335` for 0.5.x → 3.3.5. Optional pivot via 0.6.0.

## Core Rules
- Compute `contResolved` from each source row and use it for `src_mapId`, crosswalk (`mapIdX`), and path prefix.
- Chain building:
  - If `lo16 == 0`: chain = `[zone]`.
  - Else: resolve zone name via `(contResolved, zoneBase)` then `chain = [zone, sub]`.
- Strict map locking: `TryMatchChainExact(mapIdX, chain, ...)` only; discard cross-map results.
- 0.6.0 pivot path:
  - Strict child resolution by name under pivot map (unique match).
  - Forced-parent overrides for declared oddities; minimal fuzzy (EditDistance ≤ 1) only for those oddities.
  - LK re-parenting where known (e.g., Darrowmere Lake → Western Plaguelands).
  - Fallback: treat sub name as top-level when chain match fails but target map is certain.
- Rename fallbacks (disabled when `chainVia060`): exact/fuzzy top-level and child with map preference.
- Special case: “On Map Dungeon” fallback forces target map 0.

## Outputs
- Stable CSVs:
  - `mapping.csv`, `unmatched.csv` share a common header: `src_row_id,src_areaNumber,src_parentNumber,src_name,src_mapId,src_mapName,src_mapId_xwalk,src_mapName_xwalk,src_path,tgt_id_335,tgt_name,tgt_parent_id,tgt_parent_name,tgt_mapId,tgt_mapName,tgt_path,match_method`.
  - `patch.csv`, `patch_via060.csv`, `trace.csv` as defined in `CompareAreaV2Command.cs`.

## Invariants
- No cross-map acceptance in V2; map locking enforced.
- Prefer exact chain; fallbacks are gated and conservative.

## Known Issues (Session)
- Instances like `DeadminesInstance` may appear as AreaID 0 in visualization if the renderer expects ADT-embedded area fields that are empty in Alpha.
- Confirm validation via DBCTool outputs rather than Alpha ADT metadata.

## Next Steps
- Expand tests for instances and oddities; add targeted logging around instance chains and pivot method selection.
- Rewire visualization to consume DBCTool CSV outputs for proofing AreaIDs.
