# Tasks: CASC Data Source (Local Install + Remote CDN)

**Input**: `specs/238-casc-data-source/` (spec, plan, research, quickstart)
**Release**: v0.6 · **Branch**: `v0.5.4-dev`
**Tests**: included (constitution real-data validation). 🌐 = needs a real local install (`WOWVIEWER_CASC_LOCAL`) or network access to a CDN.

Format: `- [ ] T### [P?] [US?] description (path)`. All paths are relative to `wow-viewer/` unless noted.

## Phase 1: Setup (dependency + era survey)

- [ ] T001 Add TACTSharp as a git submodule at a pinned commit in wow-viewer/libs/wowdev/TACTSharp, and register it in the repo-root .gitmodules
- [ ] T002 Add a TACTSharp ProjectReference (net10.0, `GlobalPropertiesToRemove="ManagePackageVersionsCentrally"`, matching the DBCD pattern) in src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj, and confirm the solution builds
- [ ] T003 [P] Record the finding that existing `libs/*` gitlinks have no .gitmodules entries (not repaired here) in specs/238-casc-data-source/research.md
- [ ] T004 [P] Add a cache directory pattern to .gitignore and state in the CLI help that the cache is local-only (Data Policy)
- [ ] T005 🌐 Throwaway spike: open one local build, read 10 files by id, print identity, in tools/inspect/WowViewer.Tool.Inspect/Program.cs (`casc spike`; removed in T041)
- [ ] T006 🌐 Era survey: one real build per root era (6.x legacy root, 8.2+ root, newest retail), pass/fail per era, library decision (TACTSharp or fallback with license resolved), and questions for Marlamin, in specs/238-casc-data-source/evidence/phase0-era-survey.md

**Checkpoint (Phase 0 gate)**: era table and library decision recorded.

## Phase 2: Foundational

- [ ] T007 [P] Create `FileReadResult` (Ok / NotPresent / KeyUnavailable(keyId) / OfflineUnavailable / Failed(reason)) in src/core/WowViewer.Core.IO/Files/FileReadResult.cs
- [ ] T008 [P] Add `IFileDataIdReader` (ReadFileById, FileIdExists, TryResolveId, TryResolvePath) in src/core/WowViewer.Core.IO/Files/IArchiveReader.cs
- [ ] T009 [P] Create `CascStorageOptions` (mode Local/Remote/Hybrid, install path, product, region, locale, cache dir, ordered hosts, key-set source; no defaults pointing at machine paths) in src/core/WowViewer.Core.IO/Casc/CascStorageOptions.cs
- [ ] T010 [P] Create `CascBuildIdentity` (product, version, buildConfig, cdnConfig) in src/core/WowViewer.Core.IO/Casc/CascBuildIdentity.cs
- [ ] T011 [P] Create `CascListfile` bidirectional id↔path map over libs/wowdev/wow-listfile (ids without names allowed) in src/core/WowViewer.Core.IO/Casc/CascListfile.cs
- [ ] T012 [P] Unit tests for FileReadResult and CascListfile in tests/WowViewer.Core.Tests/FileReadResultTests.cs and tests/WowViewer.Core.Tests/CascListfileTests.cs

## Phase 3: User Story 1 — Open a local install (P1) 🎯 MVP

**Independent test**: read a known file by path and by id; hashes match an independent extraction (SC-001).

- [ ] T013 [US1] Local product/build discovery from the install in src/core/WowViewer.Core.IO/Casc/CascProductDiscovery.cs
- [ ] T014 [US1] `CascArchiveCatalog` local mode implementing IArchiveCatalog + IFileDataIdReader (ReadFile(path), ReadFileById, FileExists, GetAllKnownFiles, locale selection recorded) in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T015 [US1] Thread-safety stress test (parallel reads, stable hashes) in tests/WowViewer.Core.Tests/CascRealDataTests.cs
- [ ] T016 [US1] CLI `casc products`, `casc read --id|--path --out` in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T017 [US1] CLI `casc verify --sample N --reference <dir> [--json]` with a detector-power self-check (one altered reference byte must be reported) in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T018 [US1] 🌐 Gate: SC-001 on a real local build (≥1000 ids) and SC-003 (≥200 ids per surveyed era), naming the reference tool and version, in specs/238-casc-data-source/evidence/phase1-local-verify.md

## Phase 4: User Story 5 — Build traceability (P2)

- [ ] T019 [US5] Expose `CascBuildIdentity` from the catalog and add CLI `casc identity` in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs and tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T020 [US5] Include build identity in every `casc` CLI JSON output and log line in tools/inspect/WowViewer.Tool.Inspect/Program.cs

## Phase 5: User Story 2 — Stream from CDN (P1)

**Independent test**: open a remote build, load files, go offline, reload from cache (SC-002, SC-005).

- [ ] T021 [US2] Remote version list for product/region in src/core/WowViewer.Core.IO/Casc/CascProductDiscovery.cs
- [ ] T022 [US2] Remote mode in the catalog: on-demand reads into the configured cache, hash-verified before use, corrupt entries discarded and re-fetched, in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T023 [US2] Ordered host failover, per-file failure reporting, and coalescing of duplicate in-flight downloads in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T024 [US2] Offline behavior (cached → Ok, uncached → OfflineUnavailable) and cache stats in src/core/WowViewer.Core.IO/Casc/CascCacheStats.cs
- [ ] T025 [US2] Hybrid mode (local first, remote fill for the same build identity only; off by default) in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T026 [US2] CLI `casc builds --product --region`, `casc cache --stats|--verify`, and `--remote --cache` on read/verify, in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T027 [US2] 🌐 Gate: SC-002 (local vs remote byte identity on the same sample) and SC-005 (offline) in specs/238-casc-data-source/evidence/phase2-remote.md

## Phase 6: User Story 4 — Encryption (P2)

- [ ] T028 [US4] `CascKeyRing`: load a key set from configuration, refresh at runtime, and map unknown-key reads to KeyUnavailable(keyId) with no bytes, in src/core/WowViewer.Core.IO/Casc/CascKeyRing.cs
- [ ] T029 [US4] Wire the key ring into the catalog read path in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T030 [US4] 🌐 Gate: SC-006 (known key → valid header; unknown key → no bytes; refresh unlocks without restart) appended to specs/238-casc-data-source/evidence/phase2-remote.md

## Phase 7: User Story 3 — Historical builds (P2)

- [ ] T031 [US3] Open by explicit build identity through configured mirrors, reporting hosts tried and their responses on failure, in src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs
- [ ] T032 [US3] 🌐 Record which configured hosts serve which surveyed historical builds (R6) in specs/238-casc-data-source/evidence/phase2-historical.md

## Phase 8: User Story 6 — Viewer integration without regressions (P1)

- [ ] T033 [US6] Add default id-read members to IDataSource (MPQ/loose return NotPresent) in src/viewer/WoWViewer/DataSources/IDataSource.cs
- [ ] T034 [US6] `CascDataSource` thin wrapper with stats for the existing diagnostics UI in src/viewer/WoWViewer/DataSources/CascDataSource.cs
- [ ] T035 [US6] "Open CASC install…" and "Open remote build…" flows (product/region/build picker; identity in status bar) in src/viewer/WoWViewer/ViewerApp.cs
- [ ] T036 [US6] Route name-based lookups through the listfile so pre-FileDataID readers work unchanged in src/viewer/WoWViewer/DataSources/CascDataSource.cs
- [ ] T037 [US6] 🌐 Gate: SC-004 timings on a remote-build map, SC-007 (MPQ 0.5.3 + 3.3.5 load time ±5%, full suite green) in specs/238-casc-data-source/evidence/phase3-viewer.md

## Phase 9: Polish

- [ ] T038 [P] Write docs/architecture/casc-data-source.md (modes, cache, Principle VII interpretation, era table)
- [ ] T039 [P] Document the `casc` commands, then diff docs/CLI-TOOLS.md and specs/238-casc-data-source/quickstart.md against the real parser
- [ ] T040 [P] Update memory-bank/activeContext.md, progress.md and specs/STATUS.md row 13
- [ ] T041 Remove the T005 spike command from tools/inspect/WowViewer.Tool.Inspect/Program.cs

## Dependencies

- Setup (T001–T006 gate) → Foundational → US1 → {US5, US2} → {US4, US3} → US6 → Polish.
- Spec 239 may start after US1 (T018). Spec 237's viewer phase needs US6.
- **Open operator item**: acknowledge the Principle VII read-side cache interpretation (plan Constitution Check) before T022.

## Parallel examples

- Foundational: T007–T011 are all separate files.
- Polish: T038–T040.

## Implementation strategy

MVP = local install read by path and id with verified bytes (US1). Remote/CDN (US2) follows
immediately as the headline capability. Viewer integration comes last, once the catalog is proven.
