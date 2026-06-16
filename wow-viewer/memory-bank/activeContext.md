# Active Context — wow-viewer

**Last updated**: 2026-06-16 | **Focus**: Spec 064 — Blank map generation + relational ADT understanding

## Direction
WoW viewer. Libraries bridge to Unreal Engine.

## Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## Current: 064 — Blank Map Generation

### Why this matters
ADT/WDT/PM4 files are compressed relational databases. We must understand them as tables with indices and foreign keys before we can write correct ADTs. Step 1: generate valid blank ADT/WDT/WDL files that load in the viewer.

### What exists
- `LkAdtWriter` — writes full LK ADT from `LkAdtData` (working)
- `LkAdtReader` — reads LK ADT into `LkAdtData` (working)
- `LkWdtWriter`, `WdlWriter` — write WDT/WDL (working)
- `AlphaWdtWriter` — frozen (Rule 10)
- `AdtTerrainWriter` — patches MCVT+MCNR in existing ADT (working)
- `AdtPlacementWriter` — patches MDDF+MODF in existing ADT (working)

### Key gap
No blank factory — we can write `LkAdtData` but have no reusable way to construct one with valid blank defaults. Need `BlankAdtFactory`.

### Previous work reverted
- Pm4AdtWriter, Pm4BinaryAdtPatcher, write-adt CLI — all deleted (corrupted output)
- Replaced with `pm4 match-report` (markdown output only)
- LkAdtWriter untouched — not responsible for the broken patching

### Key data paths
- PM4 test data: `gillijimproject_refactor/test_data/NOT THE RIGHT FOLDER/World/Maps/development/`
- Staged clients: `output/tmp/wowarchive-clients/`

## That's it
Everything else (001, 029, 030/031/032, 038/040, 042, 045, 049, 053, 055, 056, 057) is not started or research only.