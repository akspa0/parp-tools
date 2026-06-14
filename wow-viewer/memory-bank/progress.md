# Progress — wow-viewer

## 2026-06-14 Spec consolidation + direction change
- Replaced engine-program plan with viewer-first + UE bridge (509→35 lines)
- Archived dead specs: 005, 020, 026, 036
- Archived done spec: 059 (Cata M2)
- Fixed stale status: 025→Complete, 060→Complete, 043→stale tasks noted
- Marked research specs consumed by 056: 030, 031, 032, 038, 040
- Updated 28 references across docs/specs

## Spec 056 — ViewerApp + GPU + LOD Modernization (NEW 2026-06-10)
Full Spec Kit pack written: 7 US, 20 FR, 10 SC, 9 phases (0-8), 81 tasks. Convergence spec supersedes 036. Consumes research from 030/031/032/038.

## Spec 046 — PM4 Asset Matching (2026-06-08)
C# lib landed: segment builder, scorer, synthesizer, export service. 4 inspect CLI commands, 14 tests, 7 smoke proofs. Python/Zarr corpus lane unstarted.

## Completed specs (2026-05/06)
- 012: Real validation batch extraction (110/110)
- 014: MCAL rendering parity (7/7)
- 024: V18 canvas paste refinement (28/28)
- 025: Object roof mask library (22/22)
- 060: UI cleanup (20/20)

## M2/Format work (2026-06-05/11)
- 048: 1.12.1 era-aware MD20 reader (Ghidra traced, implemented, 7 tests)
- 059: Cata v0x109 M2 (done, archived)
- 057: Archive catalog scanner (10 tests)
- 058: PM4 Scene Graph panel (82%)
- 043: Chunked MDLX reader landed (0.5.3/0.7.0/0.8.0)

## V18 Terrain training (2026-06-04/06)
Focused on 0_5_3_3368 + 3_3_5_12340. Curation: 6763→4096. Tiny manifest (21 rows). All tooling landed. Full runs not yet launched.

## Previous work
- M2 3.0.1 no-draw fix, PM4 TypeFlags/segment builder/scorer rewrites
- Viewer shell (044) dockable host + menu declutter
- MSCN/MSPV = collision containment (not navmesh) finding
