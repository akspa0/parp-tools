# Progress

## 2026-06-14 — Spec consolidation + tool fixes
- Replaced engine-program plan with viewer-first + UE bridge (509→35 lines)
- Archived 005, 020, 026, 036 (dead), 059 (Cata M2 done)
- Fixed stale status: 025→Complete, 060→Complete, 043→stale noted
- Marked research specs 030/031/032/038/040 consumed by 056
- Fixed 044 T006 false positive: removed dead MK Dataset GUI + menu item
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ reading
- Ran weak signal across 0.5.3/0.5.5/1.12.1/3.0.1 maps — tool proven on real data
- MkDatasetHarvester.cs restored (code important to VLM pipeline, GUI only removed)

## Prior sessions
- 046 PM4 matching: C# lib done (26/42), Python lane unstarted
- 058 PM4 scene graph: ~18/22 done
- Viewer shell 044: 10/13 (3 deferred P2)
- M2: 043 (chunked MDLX), 048 (1.12.1 MD20), 059 (Cata v109) all landed
- V18 terrain training: all tooling landed, full runs not yet launched
