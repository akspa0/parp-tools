# System Patterns — AlphaWDTAnalysisTool

## Architecture boundaries
- Strict CSV-only AreaID mapping:
  - Inputs: DBCTool.V2 `compare/v2/Area_patch_*_to_335.csv` files.
  - Scope: Per-source-map numeric mapping only, keyed by `(src_mapName, src_areaNumber)`.
  - No name-based or zone-base fallbacks inside this tool.
- Binary editing contract:
  - Read Alpha `MCNK.Unknown3` at `mcnk+8+0x38` (uint32) → `zone<<16 | sub`.
  - Write LK `MCNK.AreaId` at `mcnk+8+0x34` (uint32).
  - Do not modify holes at `mcnk+8+0x3C`.

## Data flow
1) Discover tiles from WDT (MAIN offsets) and placements.
2) Export WDT/ADT for present tiles.
3) Patch `AreaId` per MCNK using CSV mapping.
4) (Verbose) Emit per-tile verify CSVs.

## Mapping precedence (numeric only)
- Use per-map numeric CSV row if present and `tgt_areaID > 0`.
- Prefer `via060`-sourced rows if both via and non‑via exist for the same `(map, area)`.
- Otherwise write `0` (including cases like removed maps or explicit `on map dungeon`).

## Logging
- Limit on-console per-tile diagnostics to first 8 MCNKs to avoid spam.
- Verify CSV provides full coverage when `--verbose`.

## Non‑negotiables
- No invented targets; only apply explicit CSVs.
- No cross-map leakage; mapping is gated by `src_mapName`.
- Keep CSV schema stable; any changes require README/docs update first.
