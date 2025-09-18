# Tech Context — AlphaWDTAnalysisTool

## Mapping inputs (DBCTool.V2)
- This tool consumes DBCTool.V2 per-source-map crosswalks from a `compare/v2/` directory via CLI `--dbctool-patch-dir`.
- Optionally, `--dbctool-lk-dir` supplies LK DBCs for target map guard and legend names only; mapping remains CSV-driven.
- Strict CSV-only: no heuristics, no zone-base fallback, no name matching inside this tool.

## AreaID write path
- For each present MCNK, read Alpha area number from `Unknown3` (uint32 at `mcnk+8+0x38`, packed `zone<<16|sub`).
- Look up `(src_mapName, src_areaNumber)` in per-map crosswalks. If found and non-zero, write LK `AreaId` (uint32 at `mcnk+8+0x34`).
- If no explicit per-map row exists, write `0`.
- Verbose runs emit `csv/maps/<MapName>/areaid_verify_<x>_<y>.csv` with: `tile_x,tile_y,chunk_index,alpha_raw,lk_areaid,on_disk,reason`.

## Asset Fixups (Strategy)
- In-place patchers for ADT string tables (no chunk growth, no offset changes):
  - `MTEX` (BLP textures): capacity-aware replacement, with tileset/non-tileset fallbacks if the resolved path is too long; else skip.
  - `MMDX` (MDX/M2 model names): capacity-aware replacement only (no growth); extension parity enforced.
  - `MWMO` (WMO names): capacity-aware replacement only (no growth).
- Fuzzy matching:
  - Directory-aware for textures (prioritize same-folder candidates).
  - Basename similarity threshold ≥ 0.70, with path segment Jaccard tiebreak.
- Specular rule:
  - Never map non-`_s` → `_s` textures.
  - Allow `_s` → non-`_s` downgrade only when the `_s` original is missing.
- Extension parity:
  - Do not flip MDX↔M2. Fuzzy and fallbacks restricted to original extension when known.

## Profiles and Toggles
- `--profile preserve|modified` (default: `modified`)
  - `modified`: fuzzy on, fallbacks on, fixups on.
  - `preserve`: fuzzy off, fallbacks off, fixups off (log only, preserve original paths).
- Independent toggles:
  - `--no-fallbacks` → disables fallbacks even in modified profile.
  - `--no-fixups` → disables tileset `_s` variant handling.

## Asset Fixup Logging
- `csv/maps/<MapName>/asset_fixups.csv` records actionable events:
  - `fuzzy:*` (suggested replacements)
  - `capacity_fallback:*` (fallback chosen because resolved path didn’t fit slot)
  - `overflow_skip:*` (replacement too long for slot, left original in file)

## Notes
- All changes implemented within the tool; core library (`src/gillijimproject-csharp/WowFiles`) remains untouched.
