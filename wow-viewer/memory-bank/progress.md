# Progress — wow-viewer

## 2026-06-16 — Blank map generation lands, MHDR offset fix, PM4 match patching planned
- Created `BlankAdtFactory` with `CreateBlank()`, `CreateBlankWdtOptions()`, `CreateBlankWdlTile()`, `CreateBlankAlphaTile()`
- `map generate-blank` CLI: `--format lk|alpha --texture <path> --tile-x --tile-y --map-name --output-dir`
- Default map name: `testing` (in Map.dbc across all versions)
- Default L0 texture: `tileset\ocean\westfallseafloor.blp` (configurable via `--texture`)
- Output directory: `World/Maps/<name>/` (viewer can discover from client root)
- **Critical fix**: MHDR ofsMCIN pointed to MCIN data instead of MCIN header in `LkAdtWriter` — viewer failed with `sig='?'`
- Alpha WDT generation works — `AlphaWdtWriter.Build()` with blank tile data produces valid inline MCNK WDT
- Viewer confirms LK blank ADT+WDT loads without errors (MCNK signatures valid, MCIN offsets correct)
- Committed: 713b78f2, b421ee9d, ffc97f62, 09244cbf, 94c77776, 83d15801 (revert), d393fb57 (AGENTS.md), fe3462dc (MHDR fix), 3fc9f357 (testing map name), d0de83b5 (World/Maps dir), 1db3620d (alpha WDT), 438f55a7 (texture param)
- AGENTS.md updated: WowViewer.App is the test viewer (not MdxViewer), validation target is WowViewer.App renders
- Next: PM4 match patching onto blank tiles — inject PM4 object placements into fresh blank ADTs

## 2026-06-15 — PM4 → ADT writing pipeline landed
- Built `Pm4AdtWriter` in `Core.PM4/Matching/` — converts PM4 match results to `LkAdtData`
- Added `pm4 write-adt` CLI command to inspect tool
- Pipeline: PM4 file → segment extraction → placement matching → LK ADT output
- Tested on development_00_00.pm4: 10 M2 (MDDF) + 15 WMO (MODF) placements written
- Output ADT verified valid with `map inspect` (version 18, 256 MCNK, correct chunk structure)
- M2 asset resolution from MPQ archives not yet tested (dev data uses loose file references)

## 2026-06-14 — Consolidation + weak signal tooling
- Replaced engine-program plan with viewer-first + UE bridge
- Archived 005, 020, 026, 033, 036, 059 (done/dead)
- Fixed stale status: 025/060→Complete, 043→stale noted
- Research specs 030/031/032/038/040 → consumed by 056
- Fixed 044 T006: removed dead MK Dataset from File menu + GUI