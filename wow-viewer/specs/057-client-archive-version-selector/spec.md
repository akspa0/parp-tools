# Feature Specification: Client Archive Version Selector

**Feature Branch**: `057-client-archive-version-selector`
**Created**: 2026-06-10
**Status**: Draft
**Owner**: wow-viewer (viewer + data-harvester coordination)
**Related**: closes the long-standing "hot-swap to a client that does not contain the current map" freeze; supersedes the hardcoded `FallbackClientBuildOptions` table in `ViewerApp.cs` for the cases it can cover; extends the existing **known-good client** surface in `File` menu.

**Input**: User description: "we should really just have a way to point the viewer to a WoWArchive base folder, and then show a version selector in the ui somewhere, so the user can load literally ANY build in that archive with the viewer. The perfect matching pair. Then, the user can save the clients they want to work with and load loose files on or 'favorite' in the file menu (which is what we do with 'save as known-good'). Small improvements"

## Hard Constraints (read first)

1. **The viewer MUST NOT auto-mount the WoWArchive.** The archive uses BTRFS block deduplication; the read path requires WinFSP + a RAM-backed disk cache that consumes gigabytes of system memory to de-duplicate on the fly. RAM is a vital resource on this machine. The viewer reads the manifest files at the archive root (`Clients_*.txt`, `Manifests/WoWArchive-16_*.json`, `Bundles/`) **offline without mounting anything**.
2. **The viewer MUST NOT stage a build the user has not explicitly requested.** "Stage + Load" only fires when the user clicks it for a specific build; never as a side effect of opening the version-selector, never in the background, never "to warm things up."
3. **The viewer MUST NOT keep the archive mounted across user actions.** If the user mounts the archive themselves via `MountAll.bat`, the viewer reads from that mount on demand; if the user later unmounts it, the viewer notices and degrades gracefully (catalog still readable, but "Load" needs the user to re-stage or re-mount).
4. **Per the workspace AGENTS.md RULE 9**: no `H:\CLIENTS` references anywhere in the new code, scripts, tests, or docs.

## Context

The viewer already has a "known-good client" feature in the `File` menu: paste a game folder path, optionally save it as a favorite, switch to it later, attach a loose-map folder on top of it. The fallback build list (`FallbackClientBuildOptions` in `ViewerApp.cs:160`) hardcodes 11 builds so the build-version dropdown has something to show even when the active MPQ has no detectable build.

What is missing is the **catalog layer**: the WoWArchive bundle at `G:\WoW\WoWArchive-0.X-3.X\` is a structured catalog of every deduplicated WoW client from 0.5.3 through 3.3.5, with platform, locale, and build-version metadata in `Clients_*.txt` and `Manifests/WoWArchive-16_*.json`. The bundle's `Bundles/` directory holds the deduplicated data; the user can opt-in to `MountAll.bat` to get a read-only FUSE/WinFSP mount at `G:\WoW\WoWArchive-0.X-3.X\Mount\` for actually reading client data. The viewer cannot currently read any of this, so:

- "Open Game Folder" requires the user to know the inner path of a single build (`.../0_5_3_3368/World of Warcraft`) and paste it.
- A user who wants to flip between two builds in the same archive has to remember (or document externally) the exact inner path of each.
- A user who hot-swaps to a build that does not contain the currently-loaded map gets a freeze, then a silent failure (no clear "map missing in this build" status).
- The 11-entry `FallbackClientBuildOptions` table is wrong for any new build added to the archive.

The data-harvester side already follows the **staging convention** at `output/tmp/wowarchive-clients/<build>/World of Warcraft` for the 6 builds it supports (`0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, `4_0_0_11927`). The viewer should consume the same convention. The catalog scan is a read-only operation against the archive root; the stage operation is per-build and only on user click.

## User Scenarios & Testing

### User Story 1 - Point at the WoWArchive and pick a build from a list (Priority: P1)

As a viewer user, I want to set my WoWArchive base folder once, then see every build it contains in a version-selector panel — populated by reading the archive's manifest files at the archive root, **without mounting anything** — so I can stage and load any build with a single explicit click instead of pasting per-build inner paths.

**Why this priority**: This is the foundation. Without the catalog, none of the other stories make sense. It is the smallest end-to-end slice that delivers value (the user can finally browse the archive they already have on disk) **without paying any RAM cost** until they explicitly choose to load a build.

**Independent Test**: Open the viewer, set the archive base to `G:\WoW\WoWArchive-0.X-3.X\` (the archive root, **not** the mount), observe the version-selector panel populates from `Clients_*.txt` and `Manifests/WoWArchive-16_*.json` without mounting, click `0_5_3_3368/Windows/enUS` and confirm "Stage + Load" is offered (not auto-fired), then click "Stage + Load" to verify the staging and load flow.

**Acceptance Scenarios**:

1. **Given** no archive base is set, **When** the user opens the version-selector panel, **Then** a single button "Set WoWArchive base folder..." is shown, and selecting a folder scans it for `Clients_*.txt` and `Manifests/WoWArchive-16_*.json` (read-only, no mount triggered).
2. **Given** a valid archive base is set, **When** the panel populates, **Then** every build the archive contains (across platforms and locales) appears as a list item with the format `<build-version> / <platform> / <locale>` and a status indicator (`mount-live` if the user has independently run `MountAll.bat` and the mount is reachable, `staged` if a copy exists in `output/tmp/wowarchive-clients/<build>/World of Warcraft`, `available` if the catalog lists it but neither the mount nor a staged copy is reachable).
3. **Given** a build is shown as `mount-live` (the user has independently run `MountAll.bat` and the mount is reachable), **When** the user clicks "Load from mount", **Then** the viewer reads the build from the mount path without copying anything to disk, and updates the active data source. **The viewer does not mount or unmount anything; it only reads.** The user is responsible for `MountAll.bat` lifecycle.
4. **Given** a build is shown as `available` (catalog lists it but the mount is not reachable and no staged copy exists), **When** the user clicks "Stage + Load", **Then** the viewer shows a clear status `Mount is not reachable; run MountAll.bat or load a loose client folder instead`, and does not attempt to copy from a missing source. The user can still load a loose folder or a previously-staged build.
5. **Given** a build is shown as `staged` (a copy exists in `output/tmp/wowarchive-clients/<build>/World of Warcraft`), **When** the user clicks "Load staged", **Then** the viewer loads the staged copy without any RAM cache or mount involvement.
6. **Given** the archive base is set to a non-archive folder (a single client, or any other directory), **When** the panel populates, **Then** it falls back to scanning the directory for a single `World of Warcraft/...` build and shows that build alone, with a clear "this does not look like a WoWArchive root" message.

---

### User Story 2 - Hot-swap to a build that does not contain the current map (Priority: P1)

As a viewer user, I want a clear, fast fallback when I hot-swap to a build that does not contain the map I had loaded, so the viewer tells me what happened instead of freezing and then going silent.

**Why this priority**: This is the user's actual pain point from the most recent session. Without it, US1 above is incomplete — US1 lets them pick any build, but a missing-map swap still freezes. The hot-swap-bug-fix is the first follow-up task in this spec.

**Independent Test**: Load `development_00_00` from the 3.3.5 build, then swap to a 0.5.3 build (which has no `development`). The viewer should detect the missing map within a small bounded window and return to a sane state with a clear status message.

**Acceptance Scenarios**:

1. **Given** a world is loaded from build A, **When** the user hot-swaps to build B which does not contain that map, **Then** the viewer shows status `Map "<name>" not present in <build-B>; remaining on <build-A>` (or a graceful "swapped but world cleared"), and does not freeze.
2. **Given** a world is loaded from build A, **When** the user hot-swaps to build B which has a different build of the same map (e.g. WDT file exists but the ADTs differ), **Then** the viewer loads the build B version and keeps the camera position.
3. **Given** the swap to build B fails (e.g. mount disappears, WDT read times out), **When** the failure is detected, **Then** the viewer restores the previous world (or a safe empty state) and reports the failure in the status line, with no freeze beyond a bounded timeout (target: under 2 seconds from click to status message — see FR-008).
4. **Given** a swap is in progress, **When** the user clicks a different build, **Then** the in-progress swap is canceled and the new swap starts (no overlapping swaps that corrupt state).

---

### User Story 3 - "Save as known-good" plus "favorite" plus loose-map attach (Priority: P2)

As a viewer user, I want the version-selector to integrate with the existing known-good client surface, so any build I stage or load from the catalog can be saved as a favorite and reloaded later, with the same loose-map attach flow I already use.

**Why this priority**: US1 + US2 give the basic catalog and safety. This story makes the catalog fit the user's existing mental model and File-menu flow. It is a smaller delta on top of US1 because the known-good surface already exists.

**Independent Test**: Load a build via the version-selector, click "Save as known-good" on it, restart the viewer, the build is in the `File > Open Saved Game Folder` menu with the right display name and build version.

**Acceptance Scenarios**:

1. **Given** a build is loaded from the catalog, **When** the user clicks "Save as known-good", **Then** the entry is added to `_knownGoodClientPaths` with the build-version from the catalog, a display name like `<build-version> / <platform> / <locale>`, and the inner path used.
2. **Given** a known-good client entry exists, **When** the viewer restarts, **Then** the entry persists in the File menu.
3. **Given** a known-good client entry is selected from the File menu, **When** the user chooses "Load Loose Map Folder Against Saved Base", **Then** the same loose-map attach flow that exists today runs against the catalog-derived path.
4. **Given** the user clicks "Forget Known-Good" on a catalog-derived entry, **When** the confirm dialog accepts, **Then** the entry is removed and the underlying staged path is **not** deleted (the user's archive copy is the source of truth).

---

### User Story 4 - Filter the catalog by platform, locale, and era (Priority: P3)

As a viewer user who works across multiple client versions, I want the version-selector panel to let me filter the catalog by platform (Windows / OSX), locale (enUS, enGB, deDE, etc.), and era (Alpha / Vanilla / TBC / Wrath / Cata), so I can find the build I need without scrolling through hundreds of entries.

**Why this priority**: The unfiltered catalog is usable. Filtering is a quality-of-life improvement that becomes important once the catalog has more than ~20 entries.

**Independent Test**: Set the archive base, observe the panel shows 200+ entries; filter to `platform=Windows, era=Wrath`; observe the panel narrows to the 5-ish Wrath Windows builds.

**Acceptance Scenarios**:

1. **Given** the panel is showing the full catalog, **When** the user toggles a platform filter, **Then** only entries with that platform remain visible.
2. **Given** the panel is showing a filtered list, **When** the user clicks "Clear filters", **Then** the full catalog reappears.
3. **Given** the user has a `LastSelectedBuildVersion` persisted in settings, **When** the panel populates, **Then** that build is highlighted and the panel scrolls to it.

---

### Edge Cases

- The archive base folder does not exist (user typed a bad path) → panel shows a "Folder not found" message and does not crash.
- The archive exists but contains no `Clients_*.txt` or `Manifests/WoWArchive-16_*.json` → panel shows "No WoWArchive catalog found in this folder" with a link to the WoWArchive docs.
- The mount path is reachable but a specific build's inner path is not (e.g. file deleted, mount glitch) → that build shows a red status, not a freeze; the others remain clickable.
- The user has both a `output/tmp/wowarchive-clients/<build>/` staged copy AND the mount path reachable → the staged copy takes precedence, with a small "using staged" indicator so the user can decide to refresh.
- A swap is requested for a build that the data-harvester's pre-existing 6-build set has — the data-harvester's hardcoded paths should still work, and the new catalog surface should not conflict with them.
- The user pastes a path that is the WoWArchive **mount** root, not the archive root → catalog scan may still succeed if `Clients_*.txt` etc. are present in the mount; if not, the panel shows the "not a WoWArchive root" fallback. Either way, no automatic re-mount.
- The user unmounts the archive mid-session (kills `MountAll.bat`, runs the unmount script) → the next "Load from mount" click surfaces a clear `mount is not reachable` status; the catalog itself stays populated because it was read from the unmounted files at the archive root.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST be able to persist a single user-selected "WoWArchive base folder" path, with the same persistence surface used for known-good clients.
- **FR-002**: The viewer MUST be able to parse the WoWArchive `Clients_<date>.txt` manifest and the `Manifests/WoWArchive-16_<date>.json` mirror to derive a list of `<build-version, platform, locale, inner-path>` entries. This parse is a **read-only file read**; it MUST NOT trigger a mount.
- **FR-003**: The viewer MUST be able to filter the catalog by platform, locale, and era, and the filters MUST be persisted per session.
- **FR-004**: When the user explicitly clicks "Stage + Load" for a single build, the viewer MAY copy the build folder from the mount path to the canonical staging path `output/tmp/wowarchive-clients/<build-version-underscored>/World of Warcraft`, reusing the existing staging convention from `data-paths.md`. This MUST only run on user click; it MUST NOT be triggered implicitly by opening the panel, by hover, by selection, or by anything other than the explicit user action.
- **FR-005**: The version-selector MUST integrate with the existing known-good client surface: any loaded build can be saved as known-good, and any known-good entry can be the source for a loose-map attach.
- **FR-006**: A new "Client" or "Version" panel MUST be reachable from the viewer UI (docked, like the existing Navigator and Inspector panels) and MUST NOT be modal.
- **FR-007**: The version-selector MUST call the existing `LoadMpqDataSource` flow when the user clicks a build; it MUST NOT bypass that flow.
- **FR-008**: A hot-swap to a build that does not contain the current map MUST complete within a bounded time window (target: under 2 seconds from click to status message) and MUST set a clear status line. The freeze on the previous "no fallback" path MUST be eliminated.
- **FR-009**: The hot-swap path MUST detect "WDT file not found in the new data source" and either (a) keep the previous world, or (b) clear the world to a safe empty state, with the status line clearly stating which choice was made.
- **FR-010**: An in-flight swap MUST be cancellable when the user clicks a different build; the cancel MUST leave no partial world state.
- **FR-011**: The catalog and version-selector code MUST live in a shared library (per the Library-First constitution), so the data-harvester can also use the catalog scanner for harvest MPQ planning.
- **FR-012**: The fallback build list (`FallbackClientBuildOptions` in `ViewerApp.cs`) MUST remain as a safety net for the case where the user has a single-client folder that predates the WoWArchive convention. The catalog result MUST take precedence when both apply.
- **FR-013**: No reference to `H:\CLIENTS` may appear in any new code, scripts, tests, or documentation. Staged client paths under `output/tmp/wowarchive-clients/` are the only trusted client access.
- **FR-014**: The viewer MUST NOT call `MountAll.bat`, `winfsp`, `rman-mount`, or any other mount tooling. Mount lifecycle is the user's responsibility.
- **FR-015**: The viewer MUST NOT use the WoWArchive mount path as a default read source at startup. Reading from the mount only happens when the user clicks "Load from mount" on a specific build in the panel.
- **FR-016**: The version-selector's "Stage + Load" action MUST be a per-build operation. It MUST NOT be presented as a bulk action that stages more than one build at a time.
- **FR-017**: The version-selector MUST clearly distinguish three states per build: `mount-live` (user has independently mounted the archive and the inner path is reachable), `staged` (a copy exists in `output/tmp/wowarchive-clients/<build>/World of Warcraft`), and `available` (catalog lists it, neither path reachable). Each state MUST have a distinct action button, and the user MUST explicitly choose which action to take.

### Key Entities

- **ArchiveCatalog** (shared lib): a parsed snapshot of the WoWArchive manifest, with a list of `ArchiveBuildEntry { BuildVersion, Platform, Locale, Era, InnerPath, MountStatus }` and a per-build `StagedPath` if present.
- **VersionSelectorPanel** (viewer): docked panel that shows the catalog, applies filters, and dispatches click-to-action. **Never** holds a persistent reference to the mount; reads the mount only inside an explicit action handler.
- **SwapOutcome** (shared lib): result of a hot-swap attempt — `Succeeded`, `MapMissingInTarget`, `FailedWithReason(reason)`, `Canceled`. The viewer status line and the camera/world state are derived from this.
- **StagingRequest** (shared lib): describes "stage this one build from this mount path to this staged path", with the same staging convention the data-harvester already uses. The shared lib exposes a function the viewer calls once per user click.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user with a mounted WoWArchive can browse every build it contains, filter by platform/locale/era, and choose to "Load from mount" or "Load staged" or "Stage + Load" with a single explicit click — without ever leaving the viewer or pasting a path. The viewer does not mount or unmount the archive; it only reads manifest files at the archive root and the mount path on demand.
- **SC-002**: A hot-swap to a build that does not contain the current map completes within 2 seconds and produces a clear status line. The previous "freeze + silent failure" behavior is gone.
- **SC-003**: The data-harvester's existing 6-build hardcoded paths still work after the catalog surface is added; no regression in the harvest pipeline.
- **SC-004**: Every known-good client the user has saved can be reloaded after a viewer restart, including catalog-derived entries.
- **SC-005**: The version-selector panel does not block the main thread for more than 250ms during a refresh (catalog parse is O(file size), not O(entries × disk IO)). Opening the panel MUST NOT trigger a mount or a stage.
- **SC-006**: New code follows the Library-First principle: the catalog scanner and swap orchestrator live in a shared library, not directly in the viewer.

## Assumptions

- The WoWArchive bundle is the user's primary source of clients. The hardcoded `output/tmp/wowarchive-clients/` six-build set is a stopgap, not a long-term catalog.
- The data-harvester's pre-existing 6-build hardcoded paths are not authoritative; the viewer should not depend on them for correctness, but should not break them either.
- The user is on Windows with WoWArchive present. Cross-platform mount paths are out of scope for this slice; the data-paths doc already documents the `WOWARCHIVE_MOUNT` env-var pattern for non-Windows hosts, and that pattern is what the catalog uses.
- The user is willing to wait a few seconds for a one-time catalog parse. The catalog is not designed for sub-second refresh.
- The fix to the "swap to missing map" freeze is more important than a polished UI for the version-selector. The version-selector is the path to making the fix discoverable.
- The user has or will have the WoWArchive bundle present on disk; the viewer treats it as a read-only catalog, not as a runtime dependency.
- "Stage + Load" is a per-user, per-build decision. There is no automation, no warmup, no pre-fetch, and no bulk operation.

## Out of Scope (explicit)

- A new download manager that fetches missing builds from the internet.
- Per-build thumbnail or preview rendering in the version-selector.
- The 1.12.1-era-aware MD20 reader work (spec 048) and the 2.x TBC lane (spec 049) — the catalog exposes those builds, but the viewer's existing per-build format support decides what actually loads.
- A Tauri/Electron-style detached "version-selector window" — the panel stays docked.
- Migrating the data-harvester's hardcoded 6-build dict to the catalog in this slice. The data-harvester continues to use its hardcoded paths until a follow-up slice lands.
- Auto-mounting the archive under any circumstance. The viewer never runs `MountAll.bat` or any equivalent.
- Bulk "stage everything" or "pre-warm the cache" actions. The user stages one build at a time, on demand.

## Follow-Up Tasks (each is a small, independently-shippable slice)

1. **Hot-swap-bug fix** (the user's actual pain point): when `LoadMpqDataSource` swaps a client and the new client's `RestoreWorldAfterDataSourceReload` cannot find the current map, the viewer currently freezes. Bounded fix in `ViewerApp.cs:RestoreWorldAfterDataSourceReload` and `LoadFileFromDataSource`: detect missing WDT, return a clear status, do not hang. Ship as `f9dbcbc4`-style perf/perf commit, no architecture change. This slice is small enough to land in one pass; it does not require US1 to ship first.
2. **Shared library: `WowViewer.Core.Archive`** — a `WowViewer.Core.Archive` project (or extension to an existing one) that exposes `ArchiveCatalog.Scan(archiveRootPath)`, `ArchiveBuildEntry`, and the manifest parsers. Library-First per the constitution. Pure C#, no UI, no OpenGL, no ImGui.
3. **Viewer panel**: dock the version-selector using the existing `ShellPanelId` infrastructure; reuse the existing right-sidebar lane; expose filters as ImGui widgets.
4. **Integration**: wire the panel to the existing `LoadMpqDataSource` flow via the action buttons; add per-build status indicators; preserve the "Open Saved Game Folder" menu as the legacy entry point.
5. **Docs**: update `wow-viewer/memory-bank/data-paths.md` to mention the archive-root path and its env-var override; add a short "Mounting the archive" note in `wow-viewer/README.md` that is *informational*, not a step the viewer performs.
