# Progress — wow-viewer

## 2026-06-16 — PM4 ADT writing removed, replaced with match-report
- Deleted `Pm4AdtWriter`, `Pm4BinaryAdtPatcher`, `Pm4AdtM2Placement`, `Pm4AdtWmoPlacement` — ADT patching was corrupting output ADTs
- Removed `pm4 write-adt` CLI command; replaced with `pm4 match-report` (markdown output)
- Added `FormatPm4MatchReport` and `WritePlacementSection` helpers in Program.cs
- Removed `FindBaseAdtPath` helper (only used by write-adt)
- Deleted `docs/PM4-ADT-RESTORATION.md`
- Updated spec 046: goal changed from ADT restoration to human-readable match reports
- Committed checkpoint before revert (5133bfe3)
- `LkAdtWriter` was never modified — it's clean

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
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ
