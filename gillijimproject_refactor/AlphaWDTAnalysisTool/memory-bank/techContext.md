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

## Asset Fixups (Defaults)
- Fuzzy matching: enabled by default (`--asset-fuzzy` defaults to on).
- Fallbacks: enabled by default.
- Tileset fixups: `_s` variant swap enabled by default.
- Fuzzy resolver behavior (`MultiListfileResolver.FindSimilar()`):
  - Exact basename checks (with and without extension), primary then secondary.
  - Heuristics for prefix variants: e.g., `AZ_<name>` (helps `SunkenTemple` → `AZ_SunkenTemple`).
  - Basename similarity with Levenshtein threshold ≥ 0.70, path segment Jaccard tie-break.

## Profiles and Toggles
- `--profile preserve|modified` (default: `modified`)
  - `modified`: fuzzy on, fallbacks on, fixups on.
  - `preserve`: fuzzy off, fallbacks off, fixups off (log only, preserve original paths).
- Independent toggles:
  - `--no-fallbacks` → disables fallbacks even in modified profile.
  - `--no-fixups` → disables tileset `_s` variant handling.

## Asset Fixup Logging
- For each map when exporting, we write `csv/maps/<MapName>/asset_fixups.csv` with columns:
  - `type, original, resolved, method`
  - Methods include: `exact`, `tileset_variant`, `fuzzy:primary`, `fuzzy:secondary`, `fallback`, `preserve`.

## Notes
- All changes implemented within the tool; core library (`src/gillijimproject-csharp/WowFiles`) remains untouched.
