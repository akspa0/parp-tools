# Tasks: ADT/v22 Terrain Reading and Rendering

**Input**: `specs/237-adt-v22-terrain/` (spec, plan, research, data-model, contracts/cli-contract.md, quickstart)
**Release**: v0.6 · **Branch**: `v0.5.4-dev`
**Tests**: included. The constitution requires real-data validation; synthetic tests guard behavior, and real-corpus tests are the sign-off.

**Corpus status (2026-09-16)**: **on hand**: 700 files (699 unique) in `wow-viewer/test_data/v22_adts/unknown/` (git-ignored; `$env:WOWVIEWER_AHDR_CORPUS` overrides), extensionless and named by FileDataID. **Revision 26, not the wiki's v22**; see evidence/phase0-first-look-2026-09-16.md. Tasks marked 🗂 use the real corpus.

Format: `- [ ] T### [P?] [US?] description (path)`. All paths are relative to `wow-viewer/`.

## Phase 1: Setup

- [ ] T001 Create evidence folder and README stating the receipt format (command, root, file count, SHA-256 of file list) in specs/237-adt-v22-terrain/evidence/README.md
- [ ] T002 [P] Add chunk ids `Alyr`, `Amap`, `Ashd`, `Acdo`, `Aloc`, `Aoch`, `Adst` to src/core/WowViewer.Core/Maps/MapChunkIds.cs
- [ ] T003 [P] Add a real-corpus helper resolving `GetWowViewerRoot()/test_data/v22_adts` (override: `WOWVIEWER_AHDR_CORPUS`) that skips when the folder is absent or empty, following the BlpPixelDecoderTests pattern, in tests/WowViewer.Core.Tests/AdtAhdrCorpus.cs

## Phase 2: Foundational (blocks all stories)

- [ ] T004 Add `AdtV22`, `AdtV22Error`, `AdtV26`, `AdtAhdrUnknownVersion` to src/core/WowViewer.Core/Files/WowFileKind.cs and src/core/WowViewer.Core/Maps/MapFileKind.cs, including the ADT-family predicate
- [ ] T005 Grep all `AdtV23` switch/predicate sites across src/ and tools/ and extend each one explicitly for the new kinds; list the sites touched in specs/237-adt-v22-terrain/evidence/t005-kind-sites.md
- [ ] T006 Detect AHDR-family files when `AHDR` is first **or** directly follows `MVER` (revision 26 corpus), with no filename/extension reliance. Select the kind from `AHDR.version` (22/23/26/other), keeping the `.error` split. First record what kind the current detector gives a revision-26 file (expected: a wrong MVER-branch kind), in src/core/WowViewer.Core.IO/Files/WowFileDetector.cs
- [ ] T007 Map the new kinds in src/core/WowViewer.Core.IO/Maps/MapFileSummaryReader.cs
- [ ] T008 [P] Replace the "every AHDR file is v23" assertions with version-parametrized cases (22, 23, 26, 99, plus .error, plus MVER-first extensionless layout) in tests/WowViewer.Core.Tests/WowFileDetectorTests.cs and tests/WowViewer.Core.Tests/MapFileSummaryReaderTests.cs
- [ ] T009 Widen the guard to all AHDR kinds in src/core/WowViewer.Core.IO/Maps/AdtV23SummaryReader.cs and print `ADT/v22 semantics:` vs `ADT/v23 semantics:` by kind in tools/inspect/WowViewer.Tool.Inspect/Program.cs (~line 9632)
- [ ] T010 [P] Create typed model records per data-model.md in src/core/WowViewer.Core/Maps/AdtAhdr/{AdtAhdrTile,AdtAhdrChunk,AdtAhdrLayer,AdtAhdrPlacement,AdtAhdrInventory,AdtAhdrDiagnostic}.cs

**Checkpoint**: detection is version-correct; the existing test suite is green.

## Phase F: User Story 0 — Fast-path wireframe (P1, do first) 🎯 FIRST SIGNAL

**Goal**: see revision-26 terrain in the viewer today, using only measured facts. **Independent test**: SC-009 (no cracks on ALOC-adjacent edges).
**Order**: T002, T004, T006, T008 from Setup/Foundational first; then these.

- [ ] T045 [US0] Minimal revision-26 reader (AHDR dims, ALOC X/Y, AVTX outer+inner; diagnostics, never throws) in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T046 [P] [US0] Slicer: outer row-major + inner block → per-chunk 145-entry 9-8-9 heights, plus height-derived normals, with an indexed-ramp known-answer test, in src/core/WowViewer.Core.IO/Maps/AdtAhdrTileSlicer.cs and tests/WowViewer.Core.Tests/AdtAhdrTileSlicerTests.cs
- [ ] T047 [US0] 🗂 Real-corpus test: every one of the 71 ALOC-adjacent tile pairs has identical sliced edge heights in tests/WowViewer.Core.Tests/AdtAhdrRealDataTests.cs
- [ ] T048 [US0] Minimal `AhdrTerrainAdapter` (content-sniff folder, place tiles by ALOC, empty layers/placements, provisional flag) in src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs
- [ ] T049 [US0] "Open AHDR terrain folder…" entry wired through TerrainManager into the existing terrain wireframe mode, with a provisional label, in src/viewer/WoWViewer/ViewerApp.cs and src/viewer/WoWViewer/Terrain/TerrainManager.cs
- [ ] T050 [US0] 🗂 Operator visual gate: screenshots, crack check, inner-grid sanity note in specs/237-adt-v22-terrain/evidence/fastpath-wireframe.md

**Note**: T034/T035/T036 (US3) later extend T046/T048 instead of re-creating them.

## Phase 3: User Story 1 — Know exactly what the files contain (P1) 🎯 MVP

**Goal**: a byte-accounting inventory of the real corpus. **Independent test**: the inventory runs on the corpus with 0 unexplained bytes and a full chunk table (SC-001, SC-002).

- [ ] T011 [US1] Implement a recursive chunk walk (top level → ACNK header + sub-chunks → ALYR fixed part + AMAP), recording id/offset/size/parent/gaps/overruns, in src/core/WowViewer.Core.IO/Maps/AdtAhdrInventoryReader.cs
- [ ] T012 [US1] Evaluate padded and unpadded sub-chunk walks per file and record which one accounts for all bytes (research R3) in src/core/WowViewer.Core.IO/Maps/AdtAhdrInventoryReader.cs
- [ ] T013 [US1] Add a documented-size table (AHDR 0x40, ALYR ≥0x20, ASHD 0x200, ACDO 0x38, AFBO 0x48, AVTX/ANRM from header) and disagreement reporting in src/core/WowViewer.Core.IO/Maps/AdtAhdrInventoryReader.cs
- [ ] T014 [US1] Add corpus aggregation (version histogram, chunk-occurrence table, unknown chunks, size histograms, distinct ATEX/ADOO names, file SHA-256) in src/core/WowViewer.Core.IO/Maps/AdtAhdrInventoryReader.cs
- [ ] T015 [P] [US1] Synthetic tests: unknown chunk surfaced, truncated file flagged, gap detected, wrong padding variant reports a gap (detector power) in tests/WowViewer.Core.Tests/AdtAhdrInventoryReaderTests.cs
- [ ] T016 [US1] CLI `adt-ahdr inventory --root [--recursive] [--json] [--limit]` with exit codes 0/1/2 per contract in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T017 [US1] 🗂 Run the inventory on the real corpus, confirm it reproduces every number in evidence/phase0-first-look-2026-09-16.md (699 revision-26 files, 0 unaccounted bytes, chunk table, ALOC tiles), and write the gate receipt, turning every unknown chunk into a named research item in specs/237-adt-v22-terrain/evidence/phase0-corpus-inventory.md and specs/237-adt-v22-terrain/research.md

**Checkpoint (Phase 0 gate)**: SC-001 and SC-002 hold on the real corpus.

## Phase 4: User Story 2 — Decode a tile (P1)

**Goal**: a full typed decode with measured layouts. **Independent test**: SC-002–SC-006 on the corpus.

- [ ] T018 [US2] Reader skeleton built on the inventory walk, with per-channel diagnostics and no throwing (FR-010), in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T019 [US2] 🗂 Measure ATEX/ADOO layout (one chunk of NUL-separated names vs chunk-per-name), then decode the name tables in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T020 [US2] Raw AVTX/ANRM decode into outer/inner arrays sized from the header in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T021 [US2] ACNK header decode with v22/v23/26 layout gated by revision, plus raw ALYR/ASHD/ACDO records and top-level ALOC (tile X/Y measured) / AOCH / ADST kept raw, in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T022 [P] [US2] Synthetic decode tests, including malformed/short channels producing diagnostics, in tests/WowViewer.Core.Tests/AdtAhdrReaderTests.cs
- [ ] T023 [US2] Layout probe engine (C# port of evidence/scripts/seam_probe.py, which must reproduce its 0.0000-vs-≥650 result): outer vertex order (transpose × flipI × flipJ) scored by cross-tile seam |Δh| using ALOC adjacency (fallback: inner-vs-outer-neighbour mean); ANRM permutation × sign scored by cosine vs height-derived normals; chunk-boundary step statistic for the height frame; each with a corrupted-candidate power control, in src/core/WowViewer.Core.IO/Maps/AdtAhdrLayoutProbe.cs
- [ ] T024 [US2] Placement frame probe: candidate position frames scored by placement Z vs decoded terrain height at XY; ACDO record size from payload-size modulo candidates; per-axis scale uniformity; uniqueId collisions, in src/core/WowViewer.Core.IO/Maps/AdtAhdrLayoutProbe.cs
- [ ] T025 [US2] CLI `adt-ahdr layout-probe --root [--json]`, printing INCONCLUSIVE when a margin or power check fails (exit 3), in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T026 [US2] 🗂 Run layout-probe on the corpus, record winners and margins, and fix the reader to the winning layouts in specs/237-adt-v22-terrain/evidence/phase1-layout-probe.md and src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T027 [US2] Alpha: infer the encoding per map from payload size (4096 / 2048 / other → compressed candidate, flagged), decode via the existing src/core/WowViewer.Core.IO/Maps/AdtMcalDecoder.cs **without modifying it**, and record the encoding and reason, in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T028 [US2] ASHD 64×64 bit expansion, with bit order confirmed against terrain slope in the probe, in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs
- [ ] T029 [P] [US2] v23-only AFBO and ACVT decode (synthetic tests; mark real validation "no v23 files" if none exist) in src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs and tests/WowViewer.Core.Tests/AdtAhdrReaderTests.cs
- [ ] T030 [US2] CLI `adt-ahdr dump --file [--chunk x,y] [--json]` per contract in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T031 [US2] 🗂 Real-corpus tests asserting SC-003 (≥99% clean decode), SC-005 (normal magnitude and agreement) and SC-006 (reference resolution) in tests/WowViewer.Core.Tests/AdtAhdrRealDataTests.cs
- [ ] T032 [US2] Write the measured format layout (the source of truth) in docs/architecture/adt-v22-format.md
- [ ] T033 [US2] 🗂 Phase 1 gate receipt with SC-002–SC-006 numbers in specs/237-adt-v22-terrain/evidence/phase1-decode.md

## Phase 5: User Story 3 — See it in the viewer (P2)

**Goal**: the corpus renders with seams, textures and placements. **Independent test**: operator visual check (SC-007).
**Depends on**: US2 gate only. Standalone: no external assets, so layers render as index colours and placements as labelled markers.

- [ ] T034 [US3] Slicer: whole-tile outer/inner grids → per-chunk 145-entry 9-8-9 heights/normals using the measured order, with an indexed-ramp known-answer test, in src/core/WowViewer.Core.IO/Maps/AdtAhdrTileSlicer.cs and tests/WowViewer.Core.Tests/AdtAhdrTileSlicerTests.cs
- [ ] T035 [US3] Adapter tile discovery: content-sniff every file regardless of name/extension, place each by ALOC[1]=X, ALOC[2]=Y, flag missing ALOC and duplicate tiles (no WDT exists; the tiles are the whole map) in src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs
- [ ] T036 [US3] `LoadTileWithPlacements`: fill TerrainChunkData (heights, normals, layers, 64×64 alpha, shadow, area id, holes, v23 MCCV), with WorldPosition from the shared tile/chunk math, in src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs
- [ ] T037 [US3] Map ACDO → MddfPlacement/ModfPlacement by model-name extension using the measured frame; dedupe by UniqueId, in src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs
- [ ] T038 [US3] Per-tile failure isolation, surfaced in the tile list, plus documented "unsupported" returns for phasing/placement-write/cartography members, in src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs
- [ ] T039 [US3] "Open ADT/v22 folder…" entry beside the Rosetta datastore open (~ViewerApp.cs:13017), wired through TerrainManager, in src/viewer/WoWViewer/ViewerApp.cs and src/viewer/WoWViewer/Terrain/TerrainManager.cs
- [ ] T040 [US3] 🗂 Operator visual gate with screenshots (seams, blending, objects on ground) in specs/237-adt-v22-terrain/evidence/phase2-render.md

## Phase 6: User Story 4 — Existing formats unaffected (P2)

- [ ] T044 [P] Investigate the unexplained revision-26 values (ALOC[0]=2869, ALOC fields 3/4, AOCH all-zero, ADST, AHDR+0x14 = 8396383) and record the results in specs/237-adt-v22-terrain/research.md R14
- [ ] T041 [US4] Run the full test suite and open one 0.5.3 Alpha map and one 3.3.5 LK map; record results in specs/237-adt-v22-terrain/evidence/phase2-regression.md

## Phase 7: Polish

- [ ] T042 [P] Document `adt-ahdr` commands, then diff docs/CLI-TOOLS.md and specs/237-adt-v22-terrain/quickstart.md against the real argument parser
- [ ] T043 [P] Update wow-viewer/memory-bank/activeContext.md and progress.md, and the 237 row in specs/STATUS.md

## Dependencies

- Setup → Foundational → US1 → US2 → US3. US4 runs after US3 (or after any phase touching shared code).
- 🗂 tasks block their story's checkpoint until the corpus arrives. T001–T016, T018, T020–T023, T027–T030 and T034 can be done on synthetic data first.
- No cross-spec dependencies: this is a new engine version with no external companions, and the tile files are the only input.

## Parallel examples

- Foundational: T008 and T010 alongside T006/T007.
- US2: T022 and T029 alongside the reader work (different files).
- Polish: T042 and T043 together.

## Implementation strategy

**First signal = US0 (fast-path wireframe)**: the files are public and hours old, and a wireframe needs only measured facts.
MVP for the full decoder = US1 (a truthful inventory plus the version fix). It is useful the day the corpus arrives, and
it decides every later layout. Then US2 → US3 incrementally, never starting a phase before the
previous gate passes (Constitution: one phase at a time).
