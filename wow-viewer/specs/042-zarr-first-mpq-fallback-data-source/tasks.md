---
description: "Task list for spec 042 — Zarr-first / MPQ-fallback data source with build detection, DBD chain, and menu cleanup"
---

# Tasks: 042 — Zarr-First / MPQ-Fallback Data Source

**Input**: `specs/042-zarr-first-mpq-fallback-data-source/spec.md`
**Prerequisites**: `spec.md` (written). `plan.md` deliberately skipped — spec is detailed enough; the file paths below are explicit.

**Status (2026-06-02)**: Zarr-first work demoted to P3 per user redirect. Cross-build map comparison and editor tooling are explicitly OUT OF SCOPE. Build detection (US-1), DBD chain (US-3), and menu cleanup (US-4) remain P1. The user has prioritized MDX support from older clients (see spec 043) over Zarr work. Do not implement US-2 (Zarr-first terrain) or US-5 (perf bench) in this slice.

**Tests**: Tests are requested by the spec (SC-001, SC-004, SC-006). Build-detection tests (US-1) and DBD-chain tests (US-3) land as part of their respective phases. Mark them with `[USn]` and write FIRST per the constitution.

**Organization**: Tasks grouped by user story. US-4 (menu removal) lands first as a one-line cleanup, then US-1 (build detection) is foundational, then US-3 (DBD chain). US-2 (Zarr-first terrain) and US-5 (perf bench) are deferred to a follow-up slice.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies)
- **[Story]**: which user story
- Exact file paths included

---

## Phase 1: Cleanup (US-4, P2) — one-line, lands first

**Goal**: Drop the vestigial "Open MK Dataset..." File menu item and its `_wantOpenVlmProject` flag wiring. Legacy `VlmProjectLoader` class stays (FR-017) but the UI surface is gone.

**Independent Test**: `dotnet build` clean. Open File menu — "Open MK Dataset..." absent. "Open Zarr Dataset..." still present.

- [ ] T001 [US4] Delete the "Open MK Dataset..." `ImGui.MenuItem` block (lines 1697-1698) in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`. Leave "Open Zarr Dataset..." (line 1700) intact.
- [ ] T002 [US4] Delete the `_wantOpenVlmProject = true;` line in the menu code path of `ViewerApp.cs`. Leave the existing `_wantOpenZarrDataset` flag.
- [ ] T003 [US4] Delete the per-frame `_wantOpenVlmProject` reset block (the `if (_wantOpenVlmProject) _wantOpenVlmProject = false;` style lines) in `ViewerApp.cs`. Verify with `grep _wantOpenVlmProject` → 0 matches.
- [ ] T004 [US4] Run `dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` — must succeed with 0 errors. (Sanity: confirms no orphan references.)

**Checkpoint**: Menu cleanup lands. No behavior regression. (FR-015, FR-016, FR-017 satisfied.)

---

## Phase 2: Foundational — BuildKey + Channel + Datasets Root (blocks US-1, US-3)

**Goal**: Add the data types that US-1 and US-3 both need. Pure data + a `DatasetsRootResolver` static helper. No UI, no IO. **The Zarr-resolver helpers from US-2 are deferred — they belong in a follow-up slice and are NOT part of this phase.**

**Independent Test**: New types compile. `BuildKey.TryParse` unit tests pass (covered in T010-T013). `DatasetsRootResolver.Resolve` returns `<workspace>/output/datasets/`.

- [ ] T005 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/Channel.cs` — `public enum Channel { Retail, Ptr, PreRelease, Beta, Classic, ClassicPtr }` with a `Channel.TryParse(string, out Channel)` method mapping the canonical tokens.
- [ ] T006 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/BuildKey.cs` — readonly struct `(int Major, int Minor, int Patch, int Build, string Locale, Channel Channel)` with `ToCanonicalString()` returning `<X>.<Y>.<Z>.<B>_<locale>_<channel>` (lowercase channel), `IEquatable<BuildKey>`, `IComparable<BuildKey>` (orders by build number, then channel ordinal), and a static `BuildKey.None` sentinel.
- [ ] T007 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/DatasetsRootResolver.cs` — static `Resolve(string workspaceRoot) => Path.Combine(workspaceRoot, "output", "datasets")` plus a `Default()` overload that returns `<wow-viewer>/output/datasets/` (resolved relative to the executing assembly's repo root). The "default workspace" is the wow-viewer repo root (Principle I — repo independence). The `DatasetsRootResolver` is a simple constant-returning helper for now; the actual Zarr-resolver logic (per-build vs per-map layout) is deferred to the follow-up slice that implements US-2.
- [ ] T008 [US1] Verify the new namespace compiles standalone in `WowViewer.Core.csproj`. (Should be no-op since it's a new folder under the same project.)

**Checkpoint**: Foundational types compile in isolation.

---

## Phase 3: User Story 1 — Build Detection (P1) 🎯 MVP

**Goal**: Given a client folder, return a `BuildKey` from either the folder name or the `WoW.exe` / `Wow.exe` PE header. No UI prompts. No picker dialog. Cross-build map comparison is OUT OF SCOPE per user redirect — the build key is per-folder, not aggregated.

**Independent Test**: `BuildKeyDetector.Detect(folderPath)` returns a `BuildKey` for each of the 10 staged clients. SC-001 satisfied.

### Tests for US-1 (write FIRST)

- [ ] T009 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.Tests/Build/BuildKeyTests.cs` — 12+ tests covering: canonical-string round trip, `TryParse` accepts `3.3.5.12340_enUS_retail`, rejects `garbage`, equality, comparison, default `None` sentinel.
- [ ] T010 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.Tests/Build/BuildKeyDetectorTests.cs` — 8+ tests using synthetic folder names (no real client paths): `3.3.5.12340 enUS Retail` → expected, `1.12.1.5875 enUS Pre-release` → expected, `MyStuff/WoW-Clone` → returns `BuildKey.None`, etc. The PE-header tests use a synthetic in-memory PE blob (write minimal valid PE to `MemoryStream` with `FileVersion` and `ProductVersion` strings in the version resource).
- [ ] T011 [P] [US1] Create `wow-viewer/tests/WowViewer.Core.Tests/Build/DatasetsRootResolverTests.cs` — 3 tests: default returns the `output/datasets/` path under the repo, custom workspace root prepends correctly, missing directory is created (or surfaced as missing — pick one and document).

### Implementation for US-1

- [ ] T012 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/FolderNameBuildKeyParser.cs` — pure static class. `TryParse(string folderName, out BuildKey)` handles the canonical patterns:
  - `<X>.<Y>.<Z>.<B> <locale> <channel>` (most common in WoWArchive)
  - `<X>_<Y>_<Z>_<B>_<locale>_<channel>` (snake-case from staged-clients)
  - `<X>.<Y>.<Z>.<B>-<locale>-<channel>` (dash-separated)
  - `<X>.<Y>.<Z>.<B>.<locale>.<channel>` (dot-separated)
  - Locale tokens: `enUS`, `enGB`, `deDE`, `frFR`, `esES`, `esMX`, `ruRU`, `koKR`, `zhCN`, `zhTW`. Reject unknown locales.
  - Channel tokens: `retail`, `ptr`, `prerelease`, `beta`, `classic`, `classicptr`. Default to `retail` if absent.
- [ ] T013 [P] [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/PeHeaderBuildKeyParser.cs` — uses `System.Reflection.PortableExecutable.PEReader` to read `FileVersion` and `ProductVersion` from the Win32 VERSIONINFO resource. Handles the `WoW.exe` vs `Wow.exe` casing. Maps `ProductName` to channel: contains "PTR" → Ptr, contains "Beta" → Beta, contains "Classic" → Classic, else Retail. Returns `BuildKey.None` if version is unparseable.
- [ ] T014 [US1] Create `wow-viewer/src/core/WowViewer.Core/Build/BuildKeyDetector.cs` — static `BuildKey Detect(string folderPath)`:
  1. Try `FolderNameBuildKeyParser.TryParse(Path.GetFileName(folderPath), out var key)`. If parsed and not `None`, return.
  2. Find any `WoW*.exe` in the folder, pick the largest non-launcher one (FR-001 edge case).
  3. Try `PeHeaderBuildKeyParser.TryParse(exePath, out key)`. Return.
  4. Return `BuildKey.None`.
- [ ] T015 [US1] Add the new files to `WowViewer.Core.csproj` if not auto-included (verify via `dotnet build`).

**Checkpoint**: US-1 done. `BuildKeyDetector.Detect` works on all 10 staged clients. SC-001 satisfied in tests.

---

## Phase 4: User Story 3 — DBD Schema Fallback Chain (P1)

**Goal**: When a DBC/DB2 file is read, resolve the matching DBD schema. If the current build's DBD is missing, walk the build chain (current → previous → ... → oldest known) and pick the newest available. Log a clear status-bar note on fallback. **Cross-channel and cross-locale are hard boundaries** (OQ-2 resolved). **No wrong-build soft fallback across builds** (OQ-1 reverted: cross-build comparison is out of scope).

**Independent Test**: Synthesize a `DbdSchemaDirectory` with one DBD for build A. Request DBD for build B (newer than A). Resolver returns A's DBD with `IsFallback=true` and the status note names both builds.

### Tests for US-3

- [ ] T016 [P] [US3] Create `wow-viewer/tests/WowViewer.Core.Tests/Build/BuildChainWalkerTests.cs` — 6+ tests:
  - Single build in chain → returns same build
  - Multi-build chain → picks newest match
  - Cross-channel: request Ptr, only Retail available → null (no cross-channel fallback)
  - Cross-locale: request deDE, only enUS available → null (no cross-locale fallback)
  - Empty chain → null
  - 100-iteration determinism test (SC-006): same inputs → same output.
- [ ] T017 [P] [US3] Create `wow-viewer/tests/WowViewer.Core.Tests/Dbc/DbdSchemaResolverTests.cs` — 4+ tests using in-memory `DbdFile` mocks. Cover: exact match, fallback chain hit, fallback chain miss (returns clear error), schema-name matching by build key + db-file-name.

### Implementation for US-3

- [ ] T018 [P] [US3] Create `wow-viewer/src/core/WowViewer.Core/Build/BuildChainWalker.cs` — static class. `BuildKey? WalkBack(BuildKey from, IEnumerable<BuildKey> knownBuilds)` — filters `knownBuilds` to the same channel and locale, sorts descending by `BuildKey.CompareTo`, returns the first one with `Build < from.Build`. Returns null if no match. Channel/locale boundaries are hard.
- [ ] T019 [P] [US3] Create `wow-viewer/src/core/WowViewer.Core.IO/Dbc/DbdSchemaResolver.cs` — class. Constructor takes a `DbdSchemaDirectory` (path + listing). Method `ResolveDbd(BuildKey buildKey, string dbFileName) → DbdResolution?` with:
  - Tries exact match first.
  - On miss, calls `BuildChainWalker.WalkBack` repeatedly (newest → older) until a match is found.
  - Returns `DbdResolution` with `DbdFile`, `SourceBuildKey`, `IsFallback`, `FallbackChain` (list of build keys tried).
- [ ] T020 [US3] Wire `DbdSchemaResolver` into the existing DBC/DB2 readers in `wow-viewer/src/core/WowViewer.Core.IO/Dbc/`. Find the entry point where `DbcReader.Load` is called from a DBC file path and add the DBD resolution step. On miss (no DBD in chain), surface a clear exception naming the build key and the file (FR-014). On fallback, log to the viewer status bar (FR-012).
- [ ] T021 [US3] Add a viewer surface for the fallback note — when `DbdResolution.IsFallback == true`, the viewer shows "<file> parsed with DBD from <SourceBuildKey> (current <RequestedBuildKey> not present)" in the status bar for ~5 seconds. Implementation in `ViewerApp.cs` near the existing status-bar code.

**Checkpoint**: US-3 done. SC-004 + SC-006 satisfied in tests.

---

## Phase 5: DEFERRED — Zarr-First Terrain Streaming (US-2, originally P1, now deferred)

**Status (2026-06-02)**: DEFERRED. The user has explicitly redirected to MDX support (see spec 043). Cross-build map comparison is out of scope. The Zarr-first / MPQ-fallback data source has no current use case without cross-build comparison. This phase is preserved for the follow-up slice that resumes Zarr work, but **do not implement these tasks in the current slice**.

When this phase is resumed, the spec's US-2 will need to be revised:
- Remove the "per-map subdir" layout (OQ-5 reverted to per-build subdir).
- Remove the wrong-build soft fallback (OQ-1 reverted to hard "no cross-build fallback").
- Demote US-5 (perf bench) to P3 (the speedup case is no longer compelling without the per-map comparison use case).

**Goal (for the follow-up slice)**: Data-source layer resolves `<datasets>/<buildKey>/<mapName>.zarr/` first; falls back to MPQ per-tile transparently. Non-terrain assets (M2/MDX/WMO/DBC/listfile) always come from MPQ.

### Tests for US-2 (deferred)

- [ ] T022 [P] [US2-DEFERRED] Create `wow-viewer/tests/WowViewer.Core.Tests/Source/ZarrStoreResolverTests.cs` — 6+ tests for the per-build subdir layout (`<datasets>/<buildKey>/<mapName>.zarr/`) with no cross-build soft fallback. Cover: exact match, missing store, cross-locale null, empty zarr.json, path normalization.

### Implementation for US-2 (deferred)

- [ ] T023 [P] [US2-DEFERRED] Create `wow-viewer/src/core/WowViewer.Core/Source/ZarrStoreResolver.cs` — static class with `Resolve(DatasetsRoot root, string mapName, BuildKey buildKey) → ZarrResolution?` for exact match only. NO soft fallback across builds. NO cross-locale fallback.
- [ ] T024 [P] [US2-DEFERRED] Extend `wow-viewer/src/viewer/WoWViewer/Terrain/ZarrTileDatasetLoader.cs` — add `LoadTile(TerrainTileAddress addr) → TerrainTileTensorPack?` returning null on missing or corrupt chunk. (Spec 041 T-10 follow-up.)
- [ ] T025 [US2-DEFERRED] Create `wow-viewer/src/core/WowViewer.Core.IO/Source/CompositeDataSource.cs` — wraps a `ZarrTileDatasetLoader` and an `MpqDataSource`. Routes per-tile reads to Zarr when the resolver finds a match, otherwise MPQ.
- [ ] T026 [US2-DEFERRED] Modify `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs:LoadMpqDataSource` — after MPQ source is built, call `BuildKeyDetector.Detect` + `ZarrStoreResolver.Resolve`. If a Zarr store is found, wrap the data source in `CompositeDataSource` and update `_statusMessage`. Only on first map load.

**Checkpoint (deferred)**: US-2 done when resumed.

---

## Phase 6: DEFERRED — Performance Bench (US-5, originally P3)

**Status (2026-06-02)**: DEFERRED. The perf bench was a "nice to have" measurement slice. With Zarr work demoted, the bench has no current driver. Skip entirely in the current slice.

- [ ] T027 [P] [US5-DEFERRED] (No implementation in this slice.)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Doc sync, memory bank update, cleanup. Per the constitution, every spec change ships with an architecture doc update.

- [ ] T028 [P] Update `wow-viewer/docs/architecture/data-source-abstraction-2026-06-02.md` (new file) — capture the BuildKey/Channel design (US-1) and the DBD-fallback contract (US-3). The Zarr-resolver design is deferred to a follow-up slice and is NOT covered in this doc. The doc explicitly notes that the Zarr-first / MPQ-fallback composition is on hold pending the cross-build map comparison refactor.
- [ ] T029 [P] Update `wow-viewer/specs/041-mh2o-mclq-liquid-type-determination-fix/spec.md` — fix the stale T-09/T-10 references (now obsolete; replaced by 042 US-2 which is itself deferred). Replace the V14 architecture pivot section if any remains.
- [ ] T030 [P] Update `wow-viewer/.specify/memory/constitution.md` — add a clause under "Library-First" that the build-detection abstraction is owned by `WowViewer.Core/Build/` and the DBD resolver by `WowViewer.Core.IO/Dbc/DbdSchemaResolver.cs`. The Zarr source layer is a planned-but-not-yet-implemented extension.
- [ ] T031 Run `dotnet build "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` and `dotnet test "I:\parp\parp-tools\wow-viewer\WowViewer.slnx" -c Debug` — both must pass. Document the test counts in the spec's success criteria section.
- [ ] T032 Update the `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` to reflect that 042 (Phase 1-4) and 043 are the active specs. 042 US-2 and US-5 are explicitly deferred.

**Checkpoint**: All US-1, US-3, US-4 work integrated. Spec, docs, memory bank, build, and tests all in sync. (US-2 and US-5 deferred with a clear reactivation path.)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (US-4 cleanup)**: No dependencies — can land immediately. Single commit.
- **Phase 2 (Foundational)**: No dependencies — new files only. Single commit.
- **Phase 3 (US-1)**: Depends on Phase 2. Can land in one commit (T009-T015) or split into tests+impl.
- **Phase 4 (US-3)**: Depends on Phase 2. **Independent of US-4 and US-1** — can land in parallel with US-1 if there's another developer.
- **Phase 5 (US-2 DEFERRED)**: Reserved for follow-up slice. Do NOT implement in this round.
- **Phase 6 (US-5 DEFERRED)**: Reserved for follow-up slice. Do NOT implement in this round.
- **Phase 7 (Polish)**: Depends on Phases 1-4. (US-2 and US-5 not required for Phase 7.)

### Parallel Opportunities

- T005, T006, T007 (foundational types) — all different files, fully parallel.
- T009, T010, T011 (US-1 tests) — different files, parallel.
- T012, T013, T014 (US-1 impl) — `FolderNameBuildKeyParser`, `PeHeaderBuildKeyParser`, `BuildKeyDetector` are different files, but T014 depends on T012 and T013.
- T016, T017 (US-3 tests) — different files, parallel.
- T018, T019 (US-3 impl) — different files, parallel; T020 depends on both.
- T028, T029, T030 (Polish docs) — different files, parallel.

### Within Each User Story

- Tests (T009-T011, T016-T017) MUST be written and FAIL before implementation.
- Models before services (T005, T006 before T007).
- Helpers before orchestrator (T012, T013 before T014).
- Build detection before DBD resolution (Phase 3 before Phase 4).

---

## Implementation Strategy

### MVP First (US-4 + US-1 + US-3)

1. Phase 1: T001-T004 (one-line menu cleanup).
2. Phase 2: T005-T008 (foundational types).
3. Phase 3: T009-T015 (build detection).
4. Phase 4: T016-T021 (DBD chain).
5. Phase 7: T028-T032 (polish).
6. **STOP and VALIDATE**: Run all 10 staged clients through `BuildKeyDetector.Detect` and confirm 10/10 build keys match expectations. This is SC-001. Then run `dotnet test` and confirm all pre-043 test counts unchanged. This is US-4 SC.

### Suggested Commit Sequence

1. `chore(viewer): drop unused 'Open MK Dataset' menu item` (Phase 1)
2. `feat(core): add BuildKey + Channel + DatasetsRootResolver foundational types` (Phase 2)
3. `feat(core): add BuildKeyDetector with folder-name + PE-header parsing` (Phase 3)
4. `feat(core): add BuildChainWalker for deterministic build-chain resolution` (Phase 4 helpers)
5. `feat(io): add DbdSchemaResolver with channel/locale hard boundaries` (Phase 4)
6. `feat(io): wire DBD fallback into DBC/DB2 readers` (Phase 4 integration)
7. `docs(042): add data-source-abstraction architecture doc + memory bank sync` (Phase 7)

### Reactivation Path for US-2 / US-5

When the cross-build map comparison refactor lands (separate spec, separate slice):
1. Re-read spec 042 and tasks.md. The deferred phases (5, 6) are waiting.
2. Replace OQ-1 and OQ-5 resolutions with the new decisions (per-build subdir, no soft fallback).
3. Resume T022-T026 (US-2) and T027 (US-5) from the deferred phases.
4. Add the new architecture doc note explaining why US-2 was deferred and what unblocked it.

---

## Notes

- 32 tasks across 7 phases. 21 active tasks (Phases 1-4, 7); 11 deferred (Phases 5, 6).
- US-4 (menu removal) lands first as a one-line cleanup. US-1 (build detection) is foundational. US-3 (DBD chain) is the meat of this round. US-2 and US-5 are explicitly deferred.
- The user has redirected priorities to MDX support from older clients (see spec 043). The 042 work is the "data-source cleanup that makes the viewer less coupled to disk extraction" foundation; the heavy Zarr work waits for the cross-build map comparison refactor.
- Tests are interleaved with implementation per the spec's request (SC-001, SC-004, SC-006 explicitly require test coverage).
- The legacy `VlmProjectLoader` class is preserved (FR-017) for one release cycle; no task removes it.
- The 2017-2018 DB2 chaos is acknowledged in the spec's OQ-4 (resolved: simple chain, error on parse failure) — no task needs to handle it specially.
- Phase 5 and Phase 6 tasks are tagged with `-DEFERRED` suffix to make their status obvious in `tasks.md` review.
