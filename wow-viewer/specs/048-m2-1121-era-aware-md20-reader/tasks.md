# Tasks: 1.12.1 Era-Aware MD20 Reader

**Input**: Design documents from `/specs/048-m2-1121-era-aware-md20-reader/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependency)
- **[Story]**: Which user story this belongs to

---

## Phase 1: Stage 1.12.1 Test Fixture

**Purpose**: ensure a real 1.12.1 .mdx is staged and reachable from the wow-viewer test runner.

- [x] T001 [P] Confirm `I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\` exists.
- [x] T002 [P] Confirm `creature\bear\bear.mdx` is reachable via the wow-viewer `ArchiveVirtualFileReader` + listfile pipeline.

**Checkpoint**: the test fixture can be loaded by a test without external setup.

---

## Phase 2: 1.12.1 Constants + Version + Era Tag

**Purpose**: capture the 1.12.1 layout as immutable code.

- [x] T010 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121Version.cs` with the `0x100` and `0x101` version enum.
- [x] T011 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121EraTag.cs` with the era tag enum + `ToDisplayString`.
- [x] T012 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121Constants.cs` with offsets, magic, and per-record strides from the Ghidra trace.

**Checkpoint**: `dotnet build` of `WowViewer.Core.IO` succeeds with 0 errors and 0 new warnings.

---

## Phase 3: Dispatcher Era-Aware Routing

**Purpose**: route 1.12.1 to the new reader without breaking 3.3.5 or MDLX.

- [x] T020 [US1] [US4] Add `M2DispatchResult` record to `M2ModelReaderDispatcher.cs`. Keep existing `Read` return type unchanged (FR-018).
- [x] T021 [US1] [US4] Add `ReadDetailed(...)` overloads that return the era tag alongside the document.
- [x] T022 [US4] Add `DetectEra(...)` helper that classifies a byte stream by magic + version.
- [x] T023 [US4] Reject `version == 0x104` (2.x TBC) with `NotSupportedException` mentioning spec 049.
- [x] T024 [US1] Route `Md20_1X_V100` / `Md20_1X_V101` to the new `M2Era1121ModelReader`.
- [x] T025 [US4] Confirm `Md20_3X_V108` still routes to the existing 3.3.5 reader.
- [x] T026 [US4] Confirm `Mdlx` magic still routes to the chunked reader.

**Checkpoint**: `dotnet build` clean. Pre-existing 3.3.5 tests untouched.

---

## Phase 4: Era-Aware Reader

**Purpose**: parse 1.12.1 .mdx into the existing `M2ModelDocument` shape.

- [x] T030 [US1] Implement `M2Era1121ModelReader.Read` with header parsing (magic + version + name + global loops).
- [x] T031 [US1] Implement `ReadTrackDefinition<T>` and `ValidateSpan` for bounds-checked table reads.
- [x] T032 [US1] Implement `ReadSequences` at `0x6c` stride.
- [x] T033 [US1] Implement `ReadColors` at `0x1c` stride.
- [x] T034 [US1] Implement `ReadTextureWeights` at `0x08` stride.
- [x] T035 [US1] Implement `ReadTextureTransforms` (placeholder for the 0x30 records).
- [x] T036 [US1] Implement `ReadLights` at `0x0c` stride (position + radius, per OQ-4 resolution).
- [x] T037 [US1] Implement `ReadCameras` at `0x2c` stride.
- [x] T038 [US1] Implement `ReadRibbons` at `0x7c` stride.
- [x] T039 [US1] Implement `ReadParticles` at `0xdc` stride.
- [x] T040 [US1] Walk the view count at `0x3c/0x40` and surface `ViewCount`.
- [x] T041 [US1] Bounds-check the 0x101-only 0x1f8/29-sub-table record (deferred decode, per A-006).

**Checkpoint**: build clean.

---

## Phase 5: View-Record Walker + Camera/Ribbon/Particle

**Purpose**: complete the view, camera, ribbon, particle reads.

- [x] T050 [US1] Wire `ReadViews` into `ParseM2` (count + sub-table bounds checks, per-record contents deferred to a future slice).
- [x] T051 [US1] Wire `ReadCameras` into `ParseM2`.
- [x] T052 [US1] Wire `ReadRibbons` into `ParseM2`.
- [x] T053 [US1] Wire `ReadParticles` into `ParseM2`.

**Checkpoint**: build clean.

---

## Phase 6: Tests

**Purpose**: 7 unit tests cover the reader end-to-end and the dispatcher's era routing.

- [x] T060 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.Tests/M2Era1121ModelReaderTests.cs`.
- [x] T061 [US1] Test US-1: read 1.12.1 fixture, assert `Version ∈ {0x100, 0x101}`, `SequenceCount > 0`, `ViewCount > 0`, `ModelName != null`.
- [x] T062 [US4] Test US-1 negative: dispatch 3.3.5 stream, assert `Era == Md20_3X_V108`.
- [x] T063 [US2] Test US-2: flags pass-through; reflection check that the 1.12.1 reader has no cvar accessor.
- [x] T064 [US4] Test US-3 regression: dispatch MDLX, assert `Era == Mdlx` and 1.12.1 reader not invoked.
- [x] T065 [US4] Test US-4: dispatch `version=0x104`, assert `NotSupportedException` mentions 049.
- [x] T066 [P] Test extra: synthetic 1.12.1 header with `0x101` magic byte; assert `Version == 0x101`.
- [x] T067 [P] Test extra: synthetic truncated stream; assert the reader does not throw `EndOfStreamException`.
- [x] T068 [P] Test extra: bounded `M2DispatchResult` + era tag display strings.
- [x] T069 [US4] Confirm pre-existing 3.3.5 tests still pass (FR-018): `ModelFootprintReaderTests.TryRead_SyntheticM2_ReturnsSingleXzHull` is a pre-existing failure on master, unrelated to 048.

**Checkpoint**: 7 new tests pass, 0 existing tests regressed.

---

## Phase 7: CLI Era Tag

**Purpose**: surface the era tag in the `m2 inspect` CLI.

- [x] T070 [US1] Add `using WowViewer.Core.IO.M2Era1121;` to `WowViewer.Tool.Inspect/Program.cs`.
- [x] T071 [US1] Update `m2 inspect` to call `M2ModelReaderDispatcher.ReadDetailed` and print `ERA: <era tag>` as the first line of output.
- [x] T072 [US1] For MDLX inputs, print `ERA: MDLX (chunked)`.
- [x] T073 [US1] For 3.3.5 inputs, print `ERA: 3.3.5 (MD20 v0x108)`.
- [x] T074 [US1] For 1.12.1 inputs, print `ERA: 1.12.1 (MD20 v0x100)` or `ERA: 1.12.1 (MD20 v0x101)`.
- [x] T075 [US1] For 2.x inputs, exit non-zero with the `NotSupportedException` message.

**Checkpoint**: `dotnet build` clean. CLI on 3.3.5 fixture shows the new era tag.

---

## Phase 8: Doc Sync + Memory Bank

**Purpose**: align the spec stack and memory bank with the implementation.

- [x] T080 [P] [US1] Edit `specs/043-m2-chunked-mdx-classic-support/spec.md` to defer 1.12.1 to spec 048; add a "Status (2026-06-05)" banner.
- [x] T081 [P] Edit `docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md` to add the spec 048 banner and update OQ-1..OQ-6 with their 048-resolution status.
- [x] T082 [P] Edit `docs/architecture/m2-native-client-research-2026-03-31.md` to add a 1.12.1 cross-reference section.
- [x] T083 [P] Update `gillijimproject_refactor/memory-bank/activeContext.md` (M2 / Runtime Continuity) with the spec 048 entry.
- [x] T084 [P] Update `gillijimproject_refactor/memory-bank/progress.md` (Completed / Landed) with the spec 048 entry.
- [x] T085 [P] Generate `wow-viewer/specs/048-m2-1121-era-aware-md20-reader/quickstart.md`, `tasks.md`, `data-model.md`, `research.md`.
- [x] T086 Final build: `dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` — 0 errors.
- [x] T087 Final test: `dotnet test "I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj" -c Debug` — 7 new tests pass, 0 regressions.
- [x] T088 Commit: per AGENTS.md commit style, single focused commit "Spec 048: 1.12.1 era-aware MD20 reader + 043 lane split".

**Checkpoint**: docs, memory bank, and commit aligned.
