# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## Session Dirtied — Needs Fresh Start

This session accumulated too many wrong turns and hallucinated assumptions. Memory bank is updated with deliverables but the reasoning path is contaminated. Start fresh in next session. Key real deliverables:

- `pm4 dump-collision` CLI command in inspect tool (reads PM4 surfaces + WMO/M2 collision data)
- WMO collision comparison works and validated on 40 OIDs
- Python scorer validated (65/65 match C#)
- Vector3 serialization bug fixed in asset corpus export
- `TypedBounds` added to segment export JSON

Everything else from this session should be re-validated from scratch.

## 2026-06-15 — PM4 → ADT writing pipeline (wow-viewer)
- Built `Pm4AdtWriter` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`
- Added `pm4 write-adt` CLI command to `WowViewer.Tool.Inspect`
- Pipeline: PM4 → segment extraction → placement matching → LK ADT output (MDDF/MODF)
- Tested: 10 M2 + 15 WMO placements from development_00_00.pm4 written to valid ADT
- M2 CK24 types: 0x40/0x41 = m2, 0x42/0x43 = wmo, TypeFlags 0x03 = M2 top surfaces

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
No Python Zarr signal-store exists. The C# side can export segment signals and match against JSON corpora, but there's no Python tooling to:
- build Zarr stores from C# exports
- train/evaluate matchers
- validate proposal quality against known-tile ground truth

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)