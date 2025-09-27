# Project Brief — AlphaWDTAnalysisTool Memory Bank

## Goal
Patch ADT MCNK.AreaId in exported LK ADTs by mapping Alpha (0.5.x) AreaTable records to LK (3.3.5.12340) AreaTable records. Maintain high-fidelity exports without modifying the core WowFiles library.

## Scope
- Consume DBCTool.V2 crosswalks (`compare/v2/`) to drive AreaID patching strictly by CSV, per source map.
- Do not perform internal DBCD-based mapping or name heuristics; this tool is a pure consumer of CSVs.
- Patch MCNK.AreaId on disk after ADT export using numeric `(src_mapName, src_areaNumber) -> tgt_areaID` rows.
- Apply asset fixups for textures/models in ADTs using safe in-place string patching (no chunk size/offset changes).

## Inputs
- Alpha WDT/ADTs.
- DBCTool.V2 `compare/v2/` directory (patch CSVs).
- Optional LK DBC dir for guarding/legend only.
- Community and LK listfiles for asset fixups.

## Outputs
- `World/Maps/<Map>/<Map>.wdt` and `<Map>_<x>_<y>.adt` files.
- `csv/maps/<Map>/areaid_verify_<x>_<y>.csv` per tile (verbose runs) to audit patch writes.
- `csv/maps/<Map>/asset_fixups.csv` (fuzzy + capacity diagnostics).

## Non-goals
- No edits to `src/gillijimproject-csharp/WowFiles` for this feature.
- No speculative features beyond AreaTable mapping and patching.
- No internal DBCD mapping flags; CSV-driven via `--dbctool-patch-dir` with optional `--dbctool-lk-dir` only.

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
- Add optional diagnostics to trace which CSV provided each `patch_csv_num` mapping.
- Document compound zone-name handling expectations (resolved in DBCTool.V2 via aliases/fuzzy).
- Expand verify CSV with optional target parent/name columns when present in CSV.

## Future Architecture (Do not implement now)
- Treat ADTs as hierarchical container objects (chunks as parent/child nodes with tracked offsets) to allow safe structural edits with dynamic resizing. This increases memory footprint but enables robust mutation without manual offset auditing.
