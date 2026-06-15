# Progress — wow-viewer

## 2026-06-14 — Consolidation + weak signal tooling
- Replaced engine-program plan with viewer-first + UE bridge
- Archived 005, 020, 026, 036 (dead), 059 (Cata M2 done)
- Fixed stale status: 025/060→Complete, 043→stale noted
- Research specs 030/031/032/038/040 → consumed by 056
- Fixed 044 T006: removed dead MK Dataset from File menu + GUI
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ
- Proven on real data: 0.5.3 Azeroth (722), 0.5.5 Azeroth (586), 1.12.1 EmeraldDream (38)
- Memory bank compressed: activeContext + progress across both banks

## Prior work
- 046 PM4 matching C# lib complete (26/42), Python lane unstarted
- 058 PM4 scene graph ~18/22, 061 weak signal 15/21, 062 tile patcher 11/19
- Viewer shell 044: 10/13 (3 deferred P2)
- M2: 043/048/059 all landed
- V18 terrain training: all tooling landed, runs not yet launched
