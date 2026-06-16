# Progress — wow-viewer

## 2026-06-16 — PM4 ADT patching reverted, spec 064 written for blank map generation
- Deleted Pm4AdtWriter, Pm4BinaryAdtPatcher — ADT patching was corrupting output
- Replaced `pm4 write-adt` with `pm4 match-report` (human-readable markdown)
- LkAdtWriter untouched — not part of PM4 matcher work
- Checkpoint commit: 5133bfe3, revert commit: 83d15801
- Wrote spec 064 (blank map generation + relational ADT understanding)
  - Phase 1: generate valid blank LK ADT/WDT/WDL that loads in viewer
  - Phase 2: document ADT as relational schema, prove lossless round-trip
  - Phase 3: Zarr ADT datastore (stretch)
- Key insight: ADT/WDT/PM4 are compressed relational databases. Must treat them as such.

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
