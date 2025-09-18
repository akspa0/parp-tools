# Tech Context — AlphaWDTAnalysisTool

## DBCD + WoWDBDefs
- We use DBCD from `lib/wow.tools.local` with WoWDBDefs definitions from `lib/WoWDBDefs/definitions/`.
- LK DBD build is hardcoded to `3.3.5.12340` for all LK DBC loads.
- Alpha builds: we attempt `0.5.5.3494` then `0.5.3.3368` for 0.5.x data.
- We no longer use any RawDBC reader anywhere.

## AreaTable Decoding & Export
- When `--dbc-dir`, `--area-alpha`, and `--area-lk` are supplied, we export:
  - `csv/dbc/AreaTable_Alpha.csv`
  - `csv/dbc/AreaTable_335.csv`
- Mapping/patching is performed in the tool layer (no changes under `src/gillijimproject-csharp/WowFiles`).
- [Planned] Replace minimal CSVs with DBCD-driven rich CSVs (decoded fields, parent/continent/flags), and improve per-tile mapping CSV accordingly.

## AreaID Mapping & Writes Alignment
- Remap-only writes: MCNK AreaId is patched only when the mapper reports `reason == "remap_explicit"`.
- Map-aware stats: Placeholder summary counts a chunk as mapped only when a remap is explicit for the current map (passes `currentMapId`).
- Verification CSV (verbose mode): After patching, emit `csv/maps/<MapName>/areaid_verify_<x>_<y>.csv` with per-MCNK rows:
  - `tile_x,tile_y,chunk_index,alpha_raw,lk_areaid,on_disk,reason,lk_name`.
  - Purpose: confirm the numeric LK AreaID written on disk equals the resolved explicit remap.

No suggestion logic lives here; DBCTool is the single source of truth for zone/sub suggestions (now map-locked zones and within-zone subs). This tool consumes `remap.json` only.

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
