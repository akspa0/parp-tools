# Tasks: 057 Client Archive Version Selector

## Phase 1: Hot-Swap Freeze Fix (P1 — User Pain Point)

- [ ] T001 In `RestoreWorldAfterDataSourceReload()` (`ViewerApp.cs:9426`), add bounded WDT probe before full world reload:
  - Check if WDT exists in new data source.
  - If missing, set status: `Map "<name>" not present in <build-B>; remaining on <build-A>`.
  - Return early without hanging.

- [ ] T002 [P] Add `SwapOutcome` enum in `wow-viewer/src/viewer/WoWViewer/ClientSwapOrchestrator.cs` (or inline if simpler):
  - `Succeeded`, `MapMissingInTarget`, `FailedWithReason(string Reason)`, `Canceled`

- [ ] T003 Add generation counter for in-flight swap cancellation. If user clicks different build during swap, old handler no-ops on completion. No overlapping swaps. No partial world state.

- [ ] T004 Validate: Load 3.3.5 `development`, swap to 0.5.3 (no `development`), verify < 2s to clear status. No freeze.

## Phase 2: Archive Catalog Scanner (Library)

- [ ] T005 [P] Create `wow-viewer/src/core/WowViewer.Core.IO/Archive/WoWArchiveBuildStatus.cs`:
  - Enum: `MountLive`, `Staged`, `Available`

- [ ] T006 [P] Create `wow-viewer/src/core/WowViewer.Core.IO/Archive/WoWArchiveBuildEntry.cs`:
  - Record: `BuildVersion`, `Platform`, `Locale`, `Era`, `InnerPath`, `Status`

- [ ] T007 [P] Create `wow-viewer/src/core/WowViewer.Core.IO/Archive/WoWArchiveEra.cs`:
  - Static helper: `string Classify(string eraTag)` — maps `0.X` → Alpha, `1.X` → Vanilla, `2.X` → TBC, `3.X` → Wrath, `4.X` → Cata
  - `string ParsePlatform(string segment)` — extracts Windows/OSX
  - `string ParseLocale(string segment)` — extracts enUS/deDE/enGB/etc.

- [ ] T008 Create `wow-viewer/src/core/WowViewer.Core.IO/Archive/WoWArchiveCatalog.cs`:
  - `static WoWArchiveCatalog Scan(string archiveRootPath)`
  - Reads all `Clients_*.txt` files, deduplicates by build version (latest file wins)
  - Parses each line: split by `_`, extract era/platform/locale/build-version
  - Reads `Manifests/WoWArchive-16_*.json` as mirror: parse JSONL (header `JRMAN` + one JSON per file line), extract distinct build folders from `path` fields
  - Cross-references: a build appears in both `Clients_*.txt` AND has files in JSON manifest → confirmed
  - Status resolution:
    - Check `<archiveRoot>/Mount/<buildFolder>/World of Warcraft/` for MountLive
    - Check `output/tmp/wowarchive-clients/<build_underscored>/World of Warcraft/` for Staged
    - Staged takes precedence when both exist
    - Otherwise Available

- [ ] T009 Create `wow-viewer/tests/WowViewer.Core.Tests/Archive/WoWArchiveCatalogTests.cs`:
  - Parse real `Clients_2025-09-21.txt` (200+ entries)
  - Era classification for each prefix
  - Status resolution for known staged builds (3_3_5_12340, 0_5_3_3368, etc.)
  - Performance: scan completes in < 250ms
  - JSON manifest parsing (synthetic test with sample JRMAN data)

## Phase 3: Staging Service (Library)

- [ ] T010 Create `wow-viewer/src/core/WowViewer.Core.IO/Archive/StagingService.cs`:
  - `static StagingResult StageBuild(WoWArchiveBuildEntry entry, string archiveRoot, string stagingRoot)`
  - Copies build folder from `<archiveRoot>/Mount/<buildFolder>/` to `<stagingRoot>/<build_underscored>/`
  - Returns `StagingResult` with success/failure + path
  - Handles: mount path absent, disk full, partial copy rollback
  - No bulk operations. No background staging. No warmup.

- [ ] T011 Create `wow-viewer/tests/WowViewer.Core.Tests/Archive/StagingServiceTests.cs`:
  - Path resolution: `0_5_3_3368` → `0_5_3_3368`
  - Error handling: mount path does not exist → `StagingResult.Failed`
  - Staged path matches convention
  - No real mount required for tests (synthetic directory structure)

## Phase 4: Version Selector Panel (Viewer)

- [ ] T012 Add `ShellPanelId.VersionSelector` to enum in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs:62`.

- [ ] T013 Register panel: `new(ShellPanelId.VersionSelector, "Version Selector", ShellPanelLane.Left, 400f, ...)`. Add to `TopLeftQuadrantPanels` (alongside Navigator).

- [ ] T014 Create `wow-viewer/src/viewer/WoWViewer/ViewerApp_VersionSelector.cs` with full `DrawVersionSelectorPanel()`:
  - "Set WoWArchive base folder..." button + folder picker when no archive base set
  - Build list with status indicators: green = MountLive, blue = Staged, gray = Available, red = mount path broken
  - Per-build action buttons that adapt to status:
    - MountLive: "Load from mount" (no copy, reads direct)
    - Staged: "Load staged" (reads staged copy) + "Refresh from mount" (re-stage)
    - Available: "Stage + Load" (only if mount reachable) or "Mount not reachable; run MountAll.bat" (if not)
  - Per-build "Save as known-good" button
  - Filter widgets: platform dropdown, locale dropdown, era dropdown, "Clear filters" button
  - Highlight last-selected build and scroll to it on panel open
  - "this does not look like a WoWArchive root" fallback when single-client folder is set

- [ ] T015 [P] Persist archive base path in `ViewerSettings`. Add `WoWArchiveBasePath` string property.

- [ ] T016 [P] Persist filter state (`platform`, `locale`, `era`) per session in `ViewerSettings`. Add `VersionSelectorFilters` object or individual properties.

- [ ] T017 [P] Persist `LastSelectedBuildVersion` in `ViewerSettings` for scroll-to-highlight on reopen.

- [ ] T018 Validate: Panel opens, populates from real archive, filters work, filter state persists across sessions, action buttons dispatch correctly, non-archive folder shows fallback.

## Phase 5: Integration with Existing Flows

- [ ] T019 Create `wow-viewer/src/viewer/WoWViewer/ClientSwapOrchestrator.cs`:
  - `SwapOutcome RequestSwap(string gamePath, string buildVersion)` — dispatches to `LoadMpqDataSource`
  - Manages in-flight swap state via generation counter
  - Handles cancellation: new swap invalidates old swap's completion handler
  - Returns `SwapOutcome` for status bar/camera/world state derivation

- [ ] T020 Wire "Load from mount" / "Load staged" → `ClientSwapOrchestrator.RequestSwap(gamePath, ...)`. Status line shows SwapOutcome.

- [ ] T021 Wire "Stage + Load": call `StagingService.StageBuild()`, show progress/status during copy (this can be slow — 5+ GB builds), on success call `ClientSwapOrchestrator.RequestSwap(stagedPath, ...)`. On mount-not-reachable, show clear status without attempting copy.

- [ ] T022 [P] Wire "Save as known-good" on catalog entries → `AddOrUpdateKnownGoodClientPath()` with build-version from catalog, display name `<build> / <platform> / <locale>`. Persist via `SaveViewerSettings()`.

- [ ] T023 [P] Wire "Forget Known-Good" on catalog entries → `ForgetKnownGoodClientPath()`. Does NOT delete the staged path.

- [ ] T024 Ensure catalog result takes precedence over `FallbackClientBuildOptions` when both apply. Keep fallback as safety net for single-client folders.

- [ ] T025 Validate: Load via panel, save as known-good, restart, entry in File menu. Loose-map attach works against catalog-derived paths. Stage + Load handles mount-not-reachable gracefully.

## Phase 6: Polish + Cross-Cutting

- [ ] T026 [P] Update `wow-viewer/memory-bank/data-paths.md` to mention archive-root path and its env-var override.

- [ ] T027 [P] Add short "Mounting the archive" note in `wow-viewer/README.md` (informational only, not a step the viewer performs).

- [ ] T028 [P] Full build + test: `dotnet build` and `dotnet test` green.

- [ ] T029 [P] Memory bank update: `activeContext.md` and `progress.md`.

## Dependencies

- Phase 1: No dependencies. Start here — user's pain point, ships independently.
- Phase 2: No dependency on Phase 1. Can run in parallel.
- Phase 3: Depends on Phase 2 (needs `WoWArchiveBuildEntry` model).
- Phase 4: Depends on Phase 2 (needs catalog data). Can partially parallel with Phase 3.
- Phase 5: Depends on Phase 3 + 4 (needs staging service + panel).
- Phase 6: Depends on all.
