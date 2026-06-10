---
description: "Task list for spec 043 — 1.x Classic MDX (chunked, MDLX magic) model support in wow-viewer"
---

# Tasks: 043 — 1.x Classic MDX (Chunked) Model Support

**Input**: `specs/043-m2-chunked-mdx-classic-support/spec.md`
**Prerequisites**: `spec.md` (written). `plan.md` deliberately skipped — spec is detailed enough; the file paths below are explicit.

**Status (2026-06-03)**: Active implementation lane. The foundational `M2Chunked` reader/dispatcher, staged smoke tests, viewer standalone open-file runtime route, and `WowViewer.Tool.Inspect m2 inspect` chunked-MDX support are landed. Still open against the full draft: companion `.skin` / `.anim` ingestion inside `M2ChunkedModelReader`, richer multi-profile handling beyond the generated conversion skin, and any 2.x pre-`2.0.0` research.

**Tests**: Tests are explicitly required (SC-001, SC-002, SC-005, SC-006). Write them FIRST per the constitution. Mark with `[USn]`.

**Organization**: Tasks grouped by user story. US-1 (1.12.1 read) is the MVP. US-2 (0.5.3 + 1.x multi-version) extends US-1. US-3 (Ghidra) is deferred. US-4 (don't break 3.3.5) is a hard non-goal enforced per task. US-5 (0.5.3 stretch) is part of US-2.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies)
- **[Story]**: which user story
- Exact file paths included

---

## Phase 1: Foundational — Magic check + chunk walker + dispatch (blocks all readers)

**Goal**: Build the chunk walker, the magic check, and the dispatch seam. The dispatch is a single new file in `WowViewer.Core.IO/M2Chunked/` that routes `.mdx` to the new reader and `.m2` to the existing reader. No existing files are modified.

**Independent Test**: A test reads a 1.12.1 `.mdx` and confirms the dispatch returns a `M2ModelDocument` (even if the document is empty/stub at this point — the goal here is just the dispatch + chunk walk). The 3.3.5 path's existing tests pass unchanged.

- [ ] T001 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/MdxMagic.cs` — `public static class MdxMagic { public const uint MDLX = 0x584C444D; }`. Mirrors the gilli constant at `gillijimproject_refactor/MDX-L_Tool/Formats/Mdx/MdxHeaders.cs:9-10`. No new dependencies.
- [ ] T002 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ChunkedChunkHeader.cs` — `public readonly record struct M2ChunkedChunkHeader(string FourCC, uint Size, long Offset)`. Used by the chunk walker for dispatch.
- [ ] T003 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ChunkedChunkWalker.cs` — class. Constructor takes `BinaryReader`. Method `IEnumerable<M2ChunkedChunkHeader> Walk()` reads FourCC + uint32 size + payload offset, yields a header, and skips the payload. Handles unknown FourCCs (yield them with size 0; the reader decides what to do). Handles chunk size overruns (truncate to file end, yield header with truncated size + a warning flag). Handles files too small to contain the magic (returns empty enumeration; reader decides what to do).
- [ ] T004 [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` — class. Method `M2ModelDocument Read(string path)` checks the file's first 4 bytes:
  - If `MDLX` magic → calls `M2ChunkedModelReader.Read(path)` (created in Phase 2).
  - Otherwise → calls the existing `M2ModelReader.Read(path)` (no changes to `M2ModelReader`).
  The dispatch file is the ONLY entry point changed in the load chain. The existing `M2ModelReader.Load` / `M2ModelReader.Read` methods are not modified. The `WowViewer.Core.Runtime/M2/` load seam (wherever the viewer calls `M2ModelReader.Read`) is updated to call the dispatcher instead — but only at the runtime seam, not in the existing 3.3.5 reader.

**Checkpoint**: Dispatch seam in place. A 3.3.5 `.m2` file still loads through the existing path. A 1.12.1 `.mdx` file gets routed to the new reader (which throws `NotImplementedException` until Phase 2 lands).

---

## Phase 2: User Story 1 — 1.12.1 .mdx reader core (P1) 🎯 MVP

**Goal**: Read a 1.12.1 `.mdx` file's `VERS`, `MODL`, `MTLS`, `TEXS`, `GEOS` (with all sub-chunks), `BONE`, `HELP`, `PIVT`, `ATCH` chunks. Skip the rest (sequences, animations) for this phase. Emit an `M2ModelDocument`. The runtime does not need to play animations yet — the document's `Animations` list is empty for the MVP.

**Independent Test**: A test reads `Bear.mdx` (or another non-trivial 1.12.1 `.mdx`) from the staged 1.12.1 client and asserts `M2ModelDocument.Vertices.Length > 0` and `M2ModelDocument.Materials.Length > 0`. SC-001 satisfied.

### Tests for US-1 (write FIRST)

- [ ] T005 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/M2ChunkedChunkWalkerTests.cs` — 5+ tests:
  - Walks a synthetic 3-chunk file (`VERS\0\0\0\0x08` + `MODL\0\0\0\0x10` + `TEXS\0\0\0\0x20`).
  - Skips unknown FourCCs (`XYZW\0\0\0\0x04`) without error.
  - Truncates a chunk whose size overruns the file end with a warning.
  - Returns empty enumeration on a 2-byte file (smaller than magic).
  - Handles a `VERS` chunk with size 0 (no payload).
- [ ] T006 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/M2ChunkedModelReaderTests.cs` — 4+ tests using staged-client `*.mdx` files (paths to `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/World of Warcraft/Data/*.mdx`):
  - `M2ChunkedModelReader.Read(<Bear.mdx path>)` returns non-null `M2ModelDocument` with `Vertices.Length > 0`, `Materials.Length > 0`.
  - `M2ChunkedModelReader.Read(<Creature.mdx path>)` returns document with at least one bone (when the model has bones).
  - `M2ChunkedModelReader.Read(<CreatureNoBones.mdx path>)` returns document with `Bones.Length == 0` (i.e. missing-bones is a valid state, not an error).
  - `M2ChunkedModelReader.Read(<not-an-mdx.txt path>)` throws `InvalidDataException` with a clear "not a chunked MDX" message.
- [ ] T007 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/M2ModelReaderDispatcherTests.cs` — 3+ tests:
  - `Read(<*.m2 path>)` routes to existing 3.3.5 path (no `M2ChunkedModelReader` involvement; verify by checking the result matches a saved 3.3.5 golden).
  - `Read(<*.mdx path>)` routes to `M2ChunkedModelReader` (verify by checking the result has the chunked-format-specific fields populated).
  - `Read(<garbage file path>)` throws `InvalidDataException`.

### Implementation for US-1

- [ ] T008 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/VersChunkParser.cs` — parses `VERS` chunk. Reads a `uint Version`. Stores in a `M2ChunkedFileContext.Version` field. Emits a `M2ChunkedParseResult` with the parsed version. Default to `0x100` if chunk is missing (FR-021). Warn if version is unknown (FR-020).
- [ ] T009 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/ModlChunkParser.cs` — parses `MODL` chunk. Reads a fixed-length name string (typically 0x150 bytes) plus a count of skins and other `MODL` fields. Populates `M2ModelDocument.Name`, `M2ModelDocument.SkinCount`, and the render flags (translation table in the "Format Decisions" appendix; gilli source is the reference). FR-007.
- [ ] T010 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/MtlsChunkParser.cs` — parses `MTLS` chunk. Reads material count + per-material data. Populates `M2ModelDocument.Materials`. Each material has: flags, blending mode, texture-index-for-1st-2nd-3rd-layer, etc. Maps directly from gilli `MdxFile.cs` material section.
- [ ] T011 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/TexsChunkParser.cs` — parses `TEXS` chunk. Reads texture-path count + per-texture path (null-terminated string). Populates `M2ModelDocument.TexturePaths`. (Path resolution to BLP files is a runtime concern; this slice just records the paths.)
- [ ] T012 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/GeosChunkParser.cs` — parses `GEOS` chunk. Reads geoset count + per-geoset metadata. The actual vertex/normal/UV/index data comes from the `GEOS` sub-chunks (`VRTX`, `NRMS`, `PTYP`, `PCNT`, `PVTX`, `GNDX`, `MTGC`, `MATS`, `UVAS`, `UVBS`, `BIDX`, `BWGT`). This task wires the outer `GEOS` walker; sub-chunk parsers are separate tasks.
- [ ] T013 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/GeosSubChunkParsers/VrtxSubChunkParser.cs`, `NrmsSubChunkParser.cs`, `PtypSubChunkParser.cs`, `PcntSubChunkParser.cs`, `PvtxSubChunkParser.cs`, `GndxSubChunkParser.cs`, `MtgcSubChunkParser.cs`, `MatsSubChunkParser.cs`, `UvasSubChunkParser.cs`, `UvbsSubChunkParser.cs`, `BidxSubChunkParser.cs`, `BwgtSubChunkParser.cs` — 12 sub-chunk parsers (FR-005). All in `WowViewer.Core.IO.M2Chunked.ChunkParsers.GeosSubChunkParsers` namespace. Each parser reads its payload and appends to the in-progress `Geoset` state. All read from gilli `MdxFile.cs` as the primary reference.
- [ ] T014 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/BoneChunkParser.cs` — parses `BONE` chunk. Auto-detects stride (FR-OQ-4): 0x50 (0.5.3) or 0xAC (1.x). Populates `M2ModelDocument.Bones`.
- [ ] T015 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/HelpChunkParser.cs`, `PivtChunkParser.cs`, `AtchChunkParser.cs` — 3 more chunk parsers. Populate `M2ModelDocument.Helpers`, `M2ModelDocument.PivotPoints`, `M2ModelDocument.Attachments`.
- [ ] T016 [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ChunkedModelReader.cs` — the orchestrator. Constructor takes a `string path` (or `Stream` + companion directory). Method `M2ModelDocument Read()`:
  1. Opens the file, validates `MDLX` magic (FR-003). Throws `InvalidDataException` on mismatch.
  2. Walks the chunk list with `M2ChunkedChunkWalker`.
  3. For each chunk, dispatches to the matching chunk parser (FR-004).
  4. After walk, returns the populated `M2ModelDocument`.
  5. Logs warnings for unknown chunks (FR-018), truncated chunks (FR-019), and missing-but-defaulted chunks (FR-020, FR-021).
- [ ] T017 [US1] Wire `M2ModelReaderDispatcher` (from T004) to call `M2ChunkedModelReader.Read` for `MDLX` magic. (Replace the `throw new NotImplementedException()` placeholder with the real call.)

**Checkpoint**: US-1 done. A 1.12.1 `.mdx` opens via the new dispatcher and produces a non-null `M2ModelDocument` with vertices + materials + (when present) bones. SC-001 + SC-002 satisfied in tests. 3.3.5 path bit-for-bit unchanged (US-4). SC-003 verified by `git diff --stat`.

---

## Phase 3: User Story 1 — Skin + animation support (still US-1, but second slice)

**Goal**: Add `.skin` and `.anim` companion file resolution. Add `SEQS`, `GLBS` chunk parsers. Populate `M2ModelDocument.Skins` and `M2ModelDocument.Animations` (data only; playback is later).

**Independent Test**: A test reads a 1.12.1 `.mdx` with companion `Skin00.skin` and asserts `M2ModelDocument.Skins.Length > 0`. A separate test reads a 1.12.1 `.mdx` with companion `Anim00.anim` and asserts `M2ModelDocument.Animations.Length > 0`.

### Tests for US-1 (skin + anim)

- [ ] T018 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/SkinFileParserTests.cs` — 3+ tests:
  - Parses a synthetic `*.skin` file with `SKIN` magic, a vertex-index count, and a vertex-indices list. (The exact `*.skin` chunk format is documented in gilli and wowdev.wiki; for MVP the parser just reads indices into `M2ModelDocument.Skins[0].Indices`.)
  - Returns null/empty when the `.skin` file doesn't exist (the parent `.mdx` has inline skins instead).
  - Throws `InvalidDataException` on a malformed `.skin` file.
- [ ] T019 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/AnimFileParserTests.cs` — 3+ tests:
  - Parses a synthetic `*.anim` file with `ANIM` magic.
  - Returns empty list when the `.anim` file doesn't exist.
  - Throws on a malformed `.anim` file.

### Implementation for US-1 (skin + anim)

- [ ] T020 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/SkinFileParser.cs` — class. Method `M2ModelDocument.Skin Read(string skinPath)` reads a `*.skin` file. Companion path is resolved by `M2ChunkedModelReader` from the parent `.mdx`'s directory (FR-007). Maps to `M2ModelDocument.Skins[i]`.
- [ ] T021 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/AnimFileParser.cs` — class. Method `M2ModelDocument.Animation Read(string animPath)` reads a `*.anim` file. Maps to `M2ModelDocument.Animations[i]`. (Playback is out of scope; data only.)
- [ ] T022 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/SeqsChunkParser.cs` — parses `SEQS` chunk. Auto-detects record size (FR-010): 128 bytes = 0.5.3 legacy, 132+ bytes = 1.x. Populates a list of sequence records. Sequences are referenced by `M2ModelDocument.Animations` after companion file resolution.
- [ ] T023 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/ChunkParsers/GlbsChunkParser.cs` — parses `GLBS` chunk. Reads global sequence timestamps. Populates `M2ModelDocument.GlobalSequences`.
- [ ] T024 [US1] Modify `M2ChunkedModelReader.Read` (T016) to:
  1. After walking chunks, enumerate `.skin` files in the parent directory matching the model's name (e.g. `Bear00.skin`, `Bear01.skin`, ...). For each, call `SkinFileParser.Read`. Populate `M2ModelDocument.Skins`.
  2. Same for `.anim` files. Populate `M2ModelDocument.Animations`.
  3. If the `.mdx` has inline skins (0.5.3 era, no `.skin` companions), pull from the `GEOS` sub-chunks themselves (FR-009).

**Checkpoint**: US-1 fully done. SC-001 + SC-002 + SC-006 satisfied (visual recognition of a bear is the user's manual check; the test covers the data side). The viewer can now open a 1.12.1 `.mdx` end-to-end.

---

## Phase 4: User Story 2 — Multi-version support (0.5.3 + 1.x; 2.x deferred)

**Goal**: Verify the reader handles 0.5.3 era (smaller chunk set, inline skins) and 1.x era (full chunk set, external skins). The 2.x pre-2.0.0 era is deferred (OQ-1).

**Independent Test**: A 0.5.3 `.mdx` reads via the new reader and produces a non-null `M2ModelDocument` with `Vertices.Length > 0` (SC-005). The 0.5.3 inline-skin path is exercised.

- [ ] T025 [P] [US2] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/ZeroFiveThreeEraTests.cs` — 3+ tests using staged 0.5.3 `.mdx` files from `output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft/`:
  - `M2ChunkedModelReader.Read(<0.5.3.mdx path>)` returns non-null `M2ModelDocument` with `Vertices.Length > 0`.
  - Inline-skin path is exercised (no `.skin` companion exists; the document has `Skins[0]` populated from `GEOS` sub-chunks).
  - `BONE` stride is 0x50 (auto-detected; verified by reading the chunk size and checking the document's bone count matches `chunkSize / 0x50`).
- [ ] T026 [P] [US2] Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Chunked/OneXEraTests.cs` — 3+ tests using staged 1.12.1 `.mdx` files:
  - `BONE` stride is 0xAC (auto-detected).
  - External-skin path is exercised (companion `.skin` files are loaded; document's `Skins.Length > 0`).
  - `SEQS` record size is 132+ bytes (auto-detected; the parser uses the 1.x layout, not the 0.5.3 legacy layout).
- [ ] T027 [US2] Add 0.5.3 + 1.12.1 sample paths to a `MapTestPaths` constant in `wow-viewer/tests/WowViewer.Core.IO.Tests/`. The staged-client paths are `I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft\*.mdx` and `I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\Data\*.mdx`. Tests skip (not fail) if the path doesn't exist — the user is the only one with the 10 TB archive, and tests must not break on other machines.

**Checkpoint**: US-2 done. 0.5.3 + 1.12.1 both load. SC-005 satisfied. 2.x pre-2.0.0 deferred with a clear reactivation path (load 2.x `WoW.exe` in Ghidra; confirm chunked vs MD20).

---

## Phase 5: User Story 4 — Don't Break 3.3.5 (hard non-goal, verified per task)

**Goal**: Every commit in Phases 1-4 must satisfy the hard non-goal. This phase is a verification checklist, not a code change.

- [ ] T028 [US4] After every Phase 1-4 commit, run `git diff --stat HEAD~1` and verify ZERO files in `wow-viewer/src/core/WowViewer.Core/M2/`, `wow-viewer/src/core/WowViewer.Core.IO/M2/`, and `wow-viewer/src/core/WowViewer.Core.Runtime/M2/` are modified. The new code lives in `WowViewer.Core.IO/M2Chunked/` ONLY.
- [ ] T029 [US4] After every Phase 1-4 commit, run `dotnet test "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug --filter "FullyQualifiedName~M2|FullyQualifiedName~MdxToM2|FullyQualifiedName~M2ToMdx"` and verify the pre-043 test count is unchanged.
- [ ] T030 [US4] Save a "golden" 3.3.5 model load result (a `M2ModelDocument` JSON dump from a known 3.3.5 `.m2` file) before any 043 work. After each commit, re-load the same `.m2` and byte-compare the result against the golden.

**Checkpoint**: US-4 verified. 3.3.5 path bit-for-bit unchanged. SC-003 + SC-004 satisfied.

---

## Phase 6: User Story 3 — Ghidra Open Questions (P2, DEFERRED)

**Status (2026-06-02)**: DEFERRED. Ghidra work is expensive. The MVP slice relies on gilli `MdxFile.cs` as the primary reference (FR-002 in spec). Ghidra is the secondary check for the parts where gilli's choices are unclear. The "Format Decisions" appendix in spec 043 captures every decision with a gilli line; zero "TODO: Ghidra" rows are required for the MVP.

- [ ] T031 [US3-DEFERRED] (No implementation in this slice. Reactivation: load `I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\WoW.exe` (4,775,986 bytes) in Ghidra. Decompile `MDLFileRead`-equivalent. Cross-reference with the spec's "Format Decisions" appendix.)

---

## Phase 7: CLI command + integration (US-1, last slice)

**Goal**: Extend the existing `m2 inspect` command in `WowViewer.Tool.Inspect` for manual debugging of chunked MDX, and wire the chunked reader/dispatcher into the viewer's standalone file-open seam.

**Independent Test**: `WowViewer.Tool.Inspect m2 inspect --input <path-to-1.12.1-.mdx>` exits 0 and prints the chunk list, vertex count, material count, skin count, animation count. The viewer opens the same file via File > Open File and renders it (or at least loads it without error).

- [ ] T032 [P] [US1] Extend `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` so `m2 inspect --input <file.mdx|file.mdl>` routes `MDLX` containers through `M2ChunkedModelReader`, then prints a chunked-MDX summary: chunk list, `Vertices.Length`, `Materials.Length`, `Skins.Length`, `Animations.Length`, magic check, version. SC-007 satisfied.
- [ ] T033 [US1] Find the runtime load seam in `wow-viewer/src/viewer/WoWViewer/` (or `WowViewer.Core.Runtime/M2/`) where the existing 3.3.5 `M2ModelReader.Read` is called when the user opens a file. Replace the direct call with `M2ModelReaderDispatcher.Read`. Verify the existing 3.3.5 path still works (T028, T029, T030).

**Checkpoint**: Viewer can now open 1.12.1 `.mdx` files via File > Open File. SC-006 satisfied (user's manual visual check).

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Doc sync, memory bank update, cleanup.

- [ ] T034 [P] Update `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` (existing) — add a "1.x Classic MDX (chunked)" section to the M2 family overview, cross-linking spec 043 and the gilli `MDX-L_Tool` reference. Note that 0.5.3, 1.x, and 2.x pre-2.0.0 use the chunked format; 2.0.0+ uses MD20.
- [ ] T035 [P] Create `wow-viewer/docs/architecture/m2/m2-family-overview-2026-06-02.md` (new) — a one-page cheat-sheet mapping each WoW build era to its model format and the wow-viewer code that handles it:
  - 0.5.3: chunked MDX → `M2ChunkedModelReader` (spec 043)
  - 1.x (1.12.1): chunked MDX → `M2ChunkedModelReader` (spec 043)
  - 2.0.0.5610: MD20 v0x100 embedded → `M2BuildLegacy` (spec 037, future)
  - 3.0.1.8303: MD20 v0x104..0x108 embedded → `M2BuildLegacy` (spec 037, future)
  - 3.3.5.12340: MD20 v0x108..0x10A external-skin → existing 3.3.5 reader
- [ ] T036 [P] Update `wow-viewer/specs/037-m2-301-embedded-views-adapter/spec.md` — note that 1.x is NOT in 037's scope. 037 covers 2.0.0+ only. 043 covers 1.x and earlier. Cross-link the two specs.
- [ ] T037 Run `dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` and `dotnet test "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` — both must pass. Document the test counts in the spec's success criteria section.
- [ ] T038 Update the `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` to reflect that 043 is the active spec. 042 US-2/US-5 are deferred.

**Checkpoint**: All US-1, US-2, US-4 work integrated. Spec, docs, memory bank, build, and tests all in sync.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Foundational)**: No dependencies — new files only. Single commit.
- **Phase 2 (US-1 reader core)**: Depends on Phase 1. Can land in one commit (T005-T017) or split into tests+impl+integration.
- **Phase 3 (US-1 skin + anim)**: Depends on Phase 2. Companion file resolution requires the core reader to exist.
- **Phase 4 (US-2 multi-version)**: Depends on Phase 3. Tests 0.5.3 + 1.x with the same reader.
- **Phase 5 (US-4 don't break 3.3.5)**: Verification phase — runs after every commit in Phases 1-4.
- **Phase 6 (US-3 Ghidra DEFERRED)**: Reserved for follow-up slice. Do NOT implement in this round.
- **Phase 7 (CLI + integration)**: Depends on Phase 4. The CLI command needs the reader to exist; the viewer integration is the last step.
- **Phase 8 (Polish)**: Depends on Phases 1-7. (US-3 Ghidra not required for Phase 8.)

### Parallel Opportunities

- T001, T002, T003 (foundational types) — different files, parallel.
- T005, T006, T007 (Phase 2 tests) — different files, parallel.
- T008, T009, T010, T011, T012, T013, T014, T015 (Phase 2 chunk parsers) — all different files, parallel; T016 depends on all of them.
- T018, T019 (Phase 3 tests) — different files, parallel.
- T020, T021, T022, T023 (Phase 3 parsers) — different files, parallel; T024 depends on all of them.
- T025, T026, T027 (Phase 4 tests) — different files, parallel.
- T034, T035, T036 (Phase 8 docs) — different files, parallel.

### Within Each User Story

- Tests (T005, T006, T007, T018, T019, T025, T026) MUST be written and FAIL before implementation.
- Chunk parsers (T008-T015) are independent of each other and can land in any order.
- The orchestrator (T016) depends on all chunk parsers.

---

## Implementation Strategy

### MVP First (US-1 + US-4)

1. Phase 1: T001-T004 (foundational dispatch).
2. Phase 2: T005-T017 (1.12.1 reader core).
3. Phase 5: T028-T030 (verify 3.3.5 path unchanged).
4. Phase 3: T018-T024 (skin + anim support).
5. Phase 4: T025-T027 (0.5.3 + 1.x multi-version).
6. Phase 7: T032-T033 (CLI + viewer integration).
7. Phase 8: T034-T038 (polish).
8. **STOP and VALIDATE**: User manually opens a 1.12.1 `.mdx` in the viewer, confirms the model renders. This is SC-006.

### Suggested Commit Sequence

1. `feat(io): add M2Chunked namespace with magic, chunk header, and walker` (Phase 1, T001-T003)
2. `feat(io): add M2ModelReaderDispatcher for MDLX vs MD20 magic routing` (Phase 1, T004)
3. `feat(io): add M2ChunkedModelReader for 1.x classic MDX support` (Phase 2, T005-T017)
4. `feat(io): add SkinFileParser and AnimFileParser for companion files` (Phase 3, T018-T024)
5. `test(io): add 0.5.3 and 1.x era multi-version tests` (Phase 4, T025-T027)
6. `feat(tool): add m2-inspect-mdx CLI subcommand` (Phase 7, T032)
7. `feat(viewer): wire M2ModelReaderDispatcher into the file-open seam` (Phase 7, T033)
8. `docs(043): add 1.x MDX research notes and M2 family overview` (Phase 8, T034-T036)

### Reactivation Path for US-3 (Ghidra)

When the user wants to load 1.12.1 `WoW.exe` in Ghidra:
1. Load the binary in Ghidra.
2. Find the `MDLFileRead`-equivalent function (decompile around `0x0078b660` based on gilli's reference).
3. Cross-reference every "Format Decisions" row in spec 043's appendix with the decompiled code.
4. If a gilli choice contradicts the decompiled code, update the spec and (if needed) the reader.

### Reactivation Path for 2.x pre-2.0.0 chunked MDX (if it exists)

1. Stage a 2.x pre-2.0.0 client in `output/tmp/wowarchive-clients/`.
2. Pick any `.mdx` file. Check the first 4 bytes:
   - `MDLX` (0x584C444D) → it's chunked MDX. Resume 043 with a new test case.
   - `MD20` (0x3032444D) → it's MD20. Falls under spec 037 (future), not 043.
3. If chunked, add a 2.x era test in Phase 4. The reader should "just work" — the chunk walk is format-agnostic.

---

## Notes

- 38 tasks across 8 phases. 37 active tasks (Phases 1-5, 7, 8); 1 deferred (Phase 6 US-3 Ghidra).
- US-1 (1.12.1 read) is the MVP. US-2 (multi-version) extends it. US-3 (Ghidra) is deferred. US-4 (don't break 3.3.5) is a hard non-goal enforced per task. US-5 (0.5.3 stretch) is part of US-2.
- All new code lives in `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/` (new folder, new namespace `WowViewer.Core.IO.M2Chunked`). NO existing M2 files are modified. The hard non-goal from spec 037 is re-affirmed.
- gillijimproject_refactor's `MDX-L_Tool/Formats/Mdx/` is the primary reference. The "Format Decisions" appendix in spec 043 names every gilli line that drives a non-obvious choice. Ghidra is the secondary check, deferred.
- Animation playback and particle rendering are explicitly OUT OF SCOPE for this slice. The reader populates `M2ModelDocument.Animations` with the parsed data, but the runtime does not play them. This is a separate slice.
- Tests are interleaved with implementation per the spec's request (SC-001, SC-002, SC-005, SC-006, SC-007 explicitly require test coverage).
- The `MapTestPaths` constants in tests are guarded with `[Skip]` on missing-path so the tests don't break on machines that don't have the 10 TB archive. The user is the only one with full access; CI sees a subset.
- The 2.x pre-2.0.0 era chunked MDX (if it exists) is explicitly deferred — needs Ghidra first to confirm the format. The MVP ships 0.5.3 + 1.x.
