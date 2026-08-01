# Implementation Plan: Client Archive Version Selector

**Branch**: `057-client-archive-version-selector` | **Date**: 2026-06-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/057-client-archive-version-selector/spec.md`

## Summary

Add a catalog layer that reads the WoWArchive manifest files (`Clients_*.txt` and `Manifests/WoWArchive-16_*.json`) at the archive root **without mounting**, exposes every build in a dockable version-selector panel with filterable catalog (platform/locale/era), integrates with the existing known-good client surface, and includes a per-build staging service. Also fixes the hot-swap freeze when swapping to a build that does not contain the current map. All new catalog/swap code lives in a shared library per Library-First, so the data-harvester can also use the catalog scanner.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing viewer shell (`WoWViewer`), existing `LoadMpqDataSource` flow, existing `_knownGoodClientPaths` persistence via `ViewerSettings`

**Storage**: JSON settings file (`viewer_settings.json`) for archive base path, filter state, and last-selected build

**Testing**: `dotnet test` on new `WowViewer.Core.Tests` (catalog parser, staging service); real-data validation against `G:\WoW\WoWArchive-0.X-3.X\Clients_2025-09-21.txt`

**Target Platform**: Windows desktop

**Project Type**: desktop-app viewer + shared library

**Performance Goals**: Catalog parse < 250ms; hot-swap failure detection < 2s

**Constraints**: NEVER auto-mount. NEVER auto-stage. NEVER bulk-stage. Per-build explicit user click only. No `H:\CLIENTS` references.

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| I. Repo Independence | PASS | All new code in `wow-viewer/` |
| II. Library-First | PASS | Catalog scanner + staging service in shared library |
| III. Real-Data Validation | PASS | Validated against real `Clients_2025-09-21.txt` |
| IV–V. ML/Dataset | N/A | No ML/dataset work |
| VI. No Game Client Path Assumptions | PASS | Uses staged paths only |

## Archive Manifest Format (from real data)

### `Clients_*.txt` (one build per line)

```
0.X_Pre-Release_Windows_enUS_0.5.3.3368
0.X_Pre-Release_OSX_enUS_0.7.0.3694
1.X_PTR_Windows_deDE_0.12.0.5494
3.X_Retail_Windows_enUS_3.3.5.12340
4.X_Retail_Windows_enUS_4.0.0.11927
```

Parse: split by `_`, last segment is build version, first segment is era tag (`0.X` = Alpha, `1.X` = Vanilla, `2.X` = TBC, `3.X` = Wrath, `4.X` = Cata), middle segments are platform + locale.

### `Manifests/WoWArchive-16_*.json` (JSONL)

First line: `JRMAN` header. Subsequent lines: JSON objects with `path` field like `1.X_PTR_Windows_deDE_0.12.0.5496/World of Warcraft/Data/...`. Used as a mirror to verify builds have actual content and to cross-reference the `Clients_*.txt` entries.

### Path conventions

- Mount: `G:\WoW\WoWArchive-0.X-3.X\Mount\<build-folder>\World of Warcraft\`
- Staged: `output/tmp/wowarchive-clients/<build_underscored>/World of Warcraft/`

## Project Structure

```text
wow-viewer/src/core/WowViewer.Core.IO/Archive/
├── WoWArchiveCatalog.cs              # NEW: catalog scanner (parse Clients_*.txt + Manifests/*.json)
├── WoWArchiveBuildEntry.cs           # NEW: build entry model
├── WoWArchiveBuildStatus.cs          # NEW: enum (MountLive, Staged, Available)
├── WoWArchiveEra.cs                  # NEW: era enum + parser
└── StagingService.cs                 # NEW: per-build stage-from-mount helper

wow-viewer/src/viewer/WoWViewer/
├── ViewerApp.cs
│   ├── ShellPanelId enum             # Add VersionSelector
│   ├── FallbackClientBuildOptions    # Keep as safety net (line 160)
│   └── LoadMpqDataSource             # Existing (line 9106) — no change needed
├── ViewerApp.cs
│   └── RestoreWorldAfterDataSourceReload # HOTFIX: bounded WDT probe (line 9426)
├── ViewerApp_ClientDialogs.cs
│   └── (existing known-good surface) # Extend for catalog-derived entries
├── ViewerApp_VersionSelector.cs
│   └── DrawVersionSelectorPanel()    # NEW: panel render method
└── ClientSwapOrchestrator.cs         # NEW: SwapOutcome model + orchestrator

wow-viewer/tests/WowViewer.Core.Tests/Archive/
├── WoWArchiveCatalogTests.cs         # NEW: catalog parser tests
└── StagingServiceTests.cs            # NEW: staging service tests (synthetic)
```

## Implementation Phases

### Phase 1: Hot-Swap Freeze Fix (P1 — User Pain Point)

**Goal**: Eliminate the freeze when hot-swapping to a build that does not contain the current map. Ships independently — no catalog or panel needed.

**Approach**:
1. In `RestoreWorldAfterDataSourceReload()` (`ViewerApp.cs:9426`), add a bounded WDT probe:
   - Before attempting full world reload, check if the WDT file exists in the new data source.
   - If WDT is missing, set status line: `Map "<name>" not present in <build-B>; remaining on <build-A>`.
   - Return early without hanging.
2. Add a `SwapOutcome` enum/result type to track: `Succeeded`, `MapMissingInTarget`, `FailedWithReason`, `Canceled`.
3. Ensure in-flight swap cancellation: when user clicks a different build during a swap, cancel the in-progress swap and start the new one. No overlapping swaps. No partial world state.

**Validation**: Load 3.3.5 `development` map, swap to 0.5.3 (no `development`), verify < 2s to clear status message. No freeze.

### Phase 2: Archive Catalog Scanner (Library)

**Goal**: Parse the WoWArchive `Clients_*.txt` and `Manifests/WoWArchive-16_*.json` into a structured catalog without mounting.

**Approach**:
1. Create `WoWArchiveCatalog` in `WowViewer.Core.IO/Archive/`:
   - `static WoWArchiveCatalog Scan(string archiveRootPath)` — reads all `Clients_*.txt` files and JSON manifests, deduplicates by build version, returns sorted list.
   - Parse `Clients_*.txt` lines: split by `_`, extract era/platform/locale/build-version.
   - Parse `Manifests/WoWArchive-16_*.json` as a mirror: verify builds have content. The JSONL format (header `JRMAN` + one JSON object per file line) provides per-file paths that confirm a build folder exists in the archive.
   - Era derived from prefix: `0.X` = Alpha, `1.X` = Vanilla, `2.X` = TBC, `3.X` = Wrath, `4.X` = Cata.
2. Create `WoWArchiveBuildEntry` record: `BuildVersion`, `Platform`, `Locale`, `Era`, `InnerPath`, `Status` (MountLive/Staged/Available).
3. Create `WoWArchiveBuildStatus` enum.
4. Create `WoWArchiveEra` static helper for era classification.
5. Status resolution:
   - `MountLive`: check if `<archiveRoot>/Mount/<buildFolder>/World of Warcraft/` exists and is reachable.
   - `Staged`: check if `output/tmp/wowarchive-clients/<build_underscored>/World of Warcraft/` exists.
   - `Available`: catalog lists it but neither path reachable.
   - Staged takes precedence over MountLive when both exist.
6. Unit tests against real `Clients_2025-09-21.txt`.

**Validation**: `Scan()` on real archive root returns 200+ entries. Status indicators correct for known staged builds. Parse completes in < 250ms.

### Phase 3: Staging Service (Library)

**Goal**: Per-build stage-from-mount operation, called only on explicit user click. Exposed as a shared library service for both viewer and data-harvester use.

**Approach**:
1. Create `StagingService` in `WowViewer.Core.IO/Archive/`:
   - `static StagingResult StageBuild(WoWArchiveBuildEntry entry, string archiveRoot, string stagingRoot)`
   - Copies the build folder from `<archiveRoot>/Mount/<buildFolder>/` to `<stagingRoot>/<build_underscored>/`.
   - Returns `StagingResult` with success/failure + path.
2. No bulk operations. No background staging. No warmup. Per-build only.
3. Add synthetic unit tests (path resolution, error handling when mount path absent).

**Validation**: Unit tests pass. Manual test: stage a small build from mount to staging, verify output path matches convention.

### Phase 4: Version Selector Panel (Viewer)

**Goal**: Dockable panel that shows the full filtered catalog, persists filter state and preferences, and dispatches click-to-action.

**Approach**:
1. Add `ShellPanelId.VersionSelector` to enum in `ViewerApp.cs`.
2. Register panel: `new(ShellPanelId.VersionSelector, "Version Selector", ShellPanelLane.Left, 400f, ...)`. Add to `TopLeftQuadrantPanels` (alongside Navigator — this is a browsing/selection surface).
3. Create `ViewerApp_VersionSelector.cs` with `DrawVersionSelectorPanel()`:
   - "Set WoWArchive base folder..." button when no archive base is set.
   - Build list with status indicators (green = MountLive, blue = Staged, gray = Available).
   - Per-build action buttons: "Load from mount", "Load staged", "Stage + Load".
   - "Save as known-good" button on selected builds.
   - Filter widgets: platform dropdown, locale dropdown, era dropdown, clear-filters button.
   - Highlight last-selected build and scroll to it on open.
4. Persist archive base path in `ViewerSettings` (same surface as known-good clients).
5. Persist filter state (`platform`, `locale`, `era`) per session in `ViewerSettings`.
6. Persist `LastSelectedBuildVersion` in `ViewerSettings` for scroll-to-highlight on reopen.

**Validation**: Panel opens, populates from real archive, filters work, filter state persists across viewer restarts, action buttons dispatch correctly.

### Phase 5: Integration with Existing Flows

**Goal**: Wire the panel to `LoadMpqDataSource`, the known-good client surface, and the swap orchestrator.

**Approach**:
1. Create `ClientSwapOrchestrator` in `WoWViewer/`:
   - Handles "Load from mount" / "Load staged" / "Stage + Load" dispatch to `LoadMpqDataSource`.
   - Manages in-flight swap state and cancellation via generation counter.
   - Returns `SwapOutcome` that the status bar and camera/world state derive from.
2. "Load from mount" / "Load staged" → call `LoadMpqDataSource(gamePath, ...)` with the resolved path.
3. "Stage + Load" → call `StagingService.StageBuild()`, then `LoadMpqDataSource()` with the staged path. Show progress during staging.
4. "Save as known-good" on catalog-derived entries → add to `_knownGoodClientPaths` with build-version from catalog, display name `<build> / <platform> / <locale>`.
5. "Forget Known-Good" on catalog-derived entries → removes entry but does NOT delete the underlying staged path.
6. Catalog result takes precedence over `FallbackClientBuildOptions` when both apply.
7. Keep `FallbackClientBuildOptions` as safety net for single-client folders.

**Validation**: Load a build via panel, save as known-good, restart viewer, entry persists in File menu. Loose-map attach works against catalog-derived paths. Swap orchestrator handles cancellation cleanly.

### Phase 6: Polish + Cross-Cutting

**Goal**: Final integration, doc sync, and validation.

**Approach**:
1. Update `wow-viewer/memory-bank/data-paths.md` to mention the archive-root path and its env-var override.
2. Add short "Mounting the archive" note in `wow-viewer/README.md` (informational only, not a step the viewer performs).
3. Memory bank update.
4. Full build + test pass.

**Validation**: `dotnet build` green. `dotnet test` green. End-to-end: browse archive, stage a build, load it, save as known-good, restart, reload. Hot-swap to missing-map build returns clear status in < 2s.
