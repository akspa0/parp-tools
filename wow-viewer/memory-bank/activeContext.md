# Active Context — wow-viewer

**Last updated**: 2026-06-16 | **Focus**: Spec 064 — Blank map generation → PM4 match patching

## Direction
WoW viewer. Libraries bridge to Unreal Engine.

## Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## Current: 064 — Blank Map Generation + PM4 Match Patching

### What landed this session
- `BlankAdtFactory.CreateBlank()` — produces valid `LkAdtData` with 256 flat MCNKs, 1 L0 texture layer, zero placements
- `BlankAdtFactory.CreateBlankWdtOptions()` — WDT options for blank tiles
- `BlankAdtFactory.CreateBlankAlphaTile()` — produces `AlphaTileData` for inline alpha WDT
- `BlankAdtFactory.CreateBlankWdlTile()` — flat WDL height tile
- `map generate-blank` CLI — `--format lk|alpha --texture <path> --tile-x --tile-y --map-name --output-dir`
- Default map name is `testing` (exists in Map.dbc across all versions)
- Default texture is `tileset\ocean\westfallseafloor.blp` (configurable via `--texture`)
- Output goes to `World/Maps/<name>/` directory structure
- **Critical bug fix**: MHDR ofsMCIN pointed to MCIN data instead of MCIN header — fixed in `LkAdtWriter.cs`
- Alpha WDT generation works — `AlphaWdtWriter.Build()` with blank tile data produces valid inline MCNK WDT
- Viewer loads LK blank ADT+WDT without errors (MCNK signatures valid, MCIN offsets correct)

### Key insight from user
ADTs always require at least 1 L0 texture layer per MCNK. Blank tiles have no texture set in the asset, just the entry exists. `westfallseafloor.blp` is the teal-green ocean floor texture used for OOB areas.

### Next: PM4 match patching onto blank tiles
Generate fresh blank tiles, then inject PM4-matched M2/WMO placements from `pm4 match-report` markdown output. This proves objects can work from any tiles we write.

### Infrastructure
- `LkAdtWriter` — writes full LK ADT from `LkAdtData` (working, MHDR offset fix applied)
- `LkAdtReader` — reads LK ADT into `LkAdtData` (working)
- `LkWdtWriter`, `WdlWriter` — write WDT/WDL (working)
- `AlphaWdtWriter` — frozen (Rule 10), used via `BlankAdtFactory.CreateBlankAlphaTile()`
- `AdtPlacementWriter` — patches MDDF+MODF in existing ADT (working, may be reusable for match patching)
- `Pm4MatchSupport` — PM4 match report generation (markdown output)

### Previous work reverted
- Pm4AdtWriter, Pm4BinaryAdtPatcher, write-adt CLI — all deleted (corrupted output)
- Replaced with `pm4 match-report` (markdown output only)

## That's it