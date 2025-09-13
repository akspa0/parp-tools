# Project Brief — AlphaWDTAnalysisTool Memory Bank

## Goal
Patch LK ADT `MCNK.areaid` using a remap-only workflow. Alpha (0.5.x) MCNK provides a 32‑bit area identifier (zone<<16 | subzone); we write mapped LK AreaIDs to ADT v18 at the correct header offset without touching other fields.

## Scope
- Read Alpha WDT/ADT to extract per‑MCNK alpha area ids from the Alpha header field.
- Apply `remap.json` (explicit map-aware mappings + ignore) to resolve LK AreaIDs.
- Patch LK ADTs on disk by writing `MCNK.areaid` at 0x34 (uint32 LE). Never modify `holes` at 0x3C.
- Keep existing asset fixups (MTEX/MMDX/MWMO) capacity-safe and optional.
- Emit concise per-map logs (and optional per-tile samples) for audit.

## Inputs
- Alpha `WDT`/`ADT` files.
- `remap.json` with:
  - `explicit_map`: entries may include `src_mapID` to enable map-aware keys (`"mapId:areaNumber"`).
  - `ignore_area_numbers`: explicit alpha `areaNumber` values to skip.
  - `options.disallow_do_not_use_targets`: boolean.
- Community and LK listfiles (for asset name fixups only).

## Outputs
- `World/Maps/<Map>/<Map>.wdt` and `<Map>_<x>_<y>.adt` files.
- `csv/maps/<Map>/asset_fixups.csv` (asset diagnostics when enabled).
- [Optional] Per‑tile sample CSV with a handful of MCNK rows to validate AreaID writes.

## Non-goals
- No DBCD/DWoWDBDefs fallback mapping in this tool (remap-only).
- No lazy auto‑discovery CLI.
- No structural chunk growth (all patching is in-place and size-preserving).

## Current Implementation Snapshot
- Alpha read:
  - Alpha MCNK `Unknown3` read at offset `mcnkOffset + 8 + 0x38` (uint32). We use the full 32‑bit value as `src_areaNumber` (zone<<16 | subzone).
- LK write:
  - LK MCNK `areaid` written at offset `mcnkOffset + 8 + 0x34` (uint32).
  - `holes` lives at 0x3C and is never modified.
- Remap application:
  - Map-aware explicit takes precedence when `src_mapID` is present (`"mapId:areaNumber"`).
  - Global explicit fallback by `areaNumber`.
  - `ignore_area_numbers` short-circuits mapping per MCNK.

## CLI (non‑lazy) examples
- Batch (0.5.3 example):
  - `dotnet run -- --input-dir <...>\0.5.3\tree\World\Maps --listfile "<community.csv>" --lk-listfile "<3x.txt>" --out <out> --export-adt --export-dir <out> --remap <path\to\053_to_335.remap.json> --verbose`
- Batch (0.5.5 example):
  - `dotnet run -- --input-dir <...>\0.5.5\tree\World\Maps --listfile "<community.csv>" --lk-listfile "<3x.txt>" --out <out> --export-adt --export-dir <out> --remap <path\to\055_to_335.remap.json> --verbose`

## Next Steps
- Add lightweight per‑tile AreaID sample CSV (first N entries) logging: `idx, alpha_raw_hex, zone, subzone, reason, lk_write, lk_readback_areaid, lk_readback_holes_hex`.
- Add an option to fail-fast if a tile has present alpha MCNKs but no explicit mapping.

## Future Architecture (Do not implement now)
- Expand mapping diagnostics and suggest candidates when unmapped, but keep writes remap-only.
- Optional integration tests that diff AreaIDs across tiles post‑patch.
