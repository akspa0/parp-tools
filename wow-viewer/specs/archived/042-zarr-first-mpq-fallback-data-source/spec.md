# Feature Specification: 042 — Zarr-First / MPQ-Fallback Data Source

**Feature Branch**: `042-zarr-first-mpq-fallback-data-source`
**Created**: 2026-06-02
**Updated**: 2026-06-02 — US-2 (Zarr-first terrain) and US-5 (perf bench) demoted to DEFERRED per user redirect. Cross-build map comparison and editor-like tooling are explicitly OUT OF SCOPE. The user has prioritized MDX support from older clients (see spec 043) over Zarr work. US-1 (build detection), US-3 (DBD chain), and US-4 (menu cleanup) remain P1. US-2 will be resumed when the cross-build map comparison refactor lands (separate spec, separate slice).
**Status**: Draft
**Input**: User description — "remove Open MK Dataset, on first client load check for `datasets/` with Zarr-archived clients, prefer Zarr if present, fall back to game-client MPQ otherwise; Zarr should provide a performance boost over MPQ streaming; build detection should come from the WoWArchive folder path or the WoW.exe PE header with no version picker; DBD files should chain to the older build's DBD if the current build's is missing (PTR/retail edge case)."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Auto-Build Detection From Folder Path (Priority: P1)

A user mounts the WoWArchive (~10 TB) and points the viewer at a client folder
(e.g. `G:\WoW\WoWArchive-0.X-3.X\3.3.5.12340 enUS Retail` or
`I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft`).
The viewer auto-detects the build key from either the folder name pattern or the
`WoW.exe` / `Wow.exe` PE header version resource. No version picker dialog is
shown. The build key is used to look up a pre-staged Zarr store and to look up
DBD schema definitions.

**Why this priority**: The 10 TB archive is the canonical source of truth for
clients. Folder-name and PE-header parsing are deterministic; a picker is
redundant and slows down the user's "open and go" flow. This unblocks the
Zarr-first / MPQ-fallback data-source composition.

**Independent Test**: Point the viewer at five different staged client folders
and confirm the build key, locale, and channel are extracted from the path or
PE header without any user prompt.

**Acceptance Scenarios**:
1. **Given** a folder named `3.3.5.12340 enUS Retail`, **When** the user
   opens it, **Then** the viewer resolves build key `3.3.5.12340_enUS_retail`
   from the folder name and falls back to the PE header only if the name is
   ambiguous.
2. **Given** a folder named `1.12.1.5875 enUS Pre-release`, **When** the
   user opens it, **Then** the build key encodes `1.12.1.5875_enUS_prerelease`
   and the channel tag `prerelease` is preserved for downstream
   DBD/DB2 schema selection.
3. **Given** a folder whose name doesn't match the canonical pattern
   (e.g. `MyStuff/WoW-Clone`), **When** the user opens it, **Then** the
   viewer parses the PE header `FileVersion` and `ProductVersion` resources
   of any `Wow*.exe` it finds, falling back to a clear "could not detect
   build" status if neither yields a parseable version.
4. **Given** a folder containing multiple `Wow*.exe` (e.g. a launcher copy
   and a retail copy), **When** the user opens it, **Then** the viewer
   prefers the largest non-launcher binary, or the one whose product name
   contains "WoW" rather than "WoW-Companion" / "WoW-Trial".

---

### User Story 2 - Zarr-First Terrain Streaming (Priority: **DEFERRED**)

**Status (2026-06-02)**: US-2 is DEFERRED. The user has explicitly redirected
priority to MDX support from older clients (see spec 043). Cross-build map
comparison and editor-like tooling are out of scope. The Zarr-first / MPQ-fallback
data source has no current use case without the cross-build comparison
refactor (which is a separate spec). This user story is preserved here so
the spec is complete, but **do not implement US-2 in the current slice**.

When this user story is resumed, the Zarr store layout will revert to
**per-build subdir** (`<datasets>/<buildKey>/<mapName>.zarr/`) — see
OQ-5 (REVERTED 2026-06-02). Cross-build soft fallback is also out (OQ-1
REVERTED). The two original questions are no longer relevant because
the cross-build comparison use case is gone.

When the user opens a client and then opens a map, the viewer first looks for
a pre-staged Zarr store at `wow-viewer/output/datasets/<buildKey>/<mapName>.zarr/`
(per-build subdir). If the store exists and the harvester's `zarr.json` declares
all the required arrays (height, alpha layers, blend, liquid basic type, etc.),
terrain data streams from the Zarr store. If the store is missing or incomplete,
the viewer falls back to reading terrain chunks from the MPQ archives
transparently — the user sees no error, only a status-bar note that says
"Zarr cache miss for <mapName>; using MPQ".

**Why this priority**: Zarr stores are 5-20× faster to stream than MPQ for
terrain-only reads because (a) chunks are pre-decoded into typed binary
arrays, (b) no MPQ hash-table lookup per file, (c) sequential read
patterns map cleanly to LZ4/Zstd decompression. This is the path the
tensor-pack and Zarr pipeline in spec 041 sets up; this spec wires it
into the data-source layer.

**Independent Test**: Stage a known client + a Zarr store for one map, open
it, confirm status bar says "Zarr" not "MPQ". Delete the Zarr store, open
the same map, confirm the viewer falls back to MPQ and the rendered result
is identical.

**Acceptance Scenarios**:
1. **Given** a client with `output/datasets/3.3.5.12340_enUS_retail/Azeroth.zarr/`
   containing all required arrays, **When** the user opens Azeroth,
   **Then** the status bar shows "Zarr: Azeroth (12,345 arrays)".
2. **Given** the same client but no Zarr store for `Kalimdor`, **When** the
   user opens Kalimdor, **Then** the status bar shows
   "MPQ fallback: Kalimdor (no Zarr cache for this map)".
3. **Given** a Zarr store that is missing the `liquid_basic_type_257`
   array (stale harvester output), **When** the user opens that map,
   **Then** the viewer logs a warning and falls back to MPQ for the
   affected tiles until the harvester is re-run.
4. **Given** both a Zarr store and the original MPQ client, **When** the
   user inspects a tile that is in Zarr, **Then** the MPQ is never read
   for that tile (verifiable via a debug counter).

---

### User Story 3 - DBD Schema Fallback Chain (Priority: P1)

When the viewer reads a DBC or DB2 file from the client, it needs a DBD
schema definition to interpret column widths and types. Some builds
(especially PTR builds sandwiched between retail releases) ship the
same column format as the previous build, so no fresh DBD was emitted.
The viewer should chain to the older build's DBD when the current build's
is missing, with a clear status-bar note about the fallback so the user
knows which schema they're getting.

**Why this priority**: PTR builds between retail releases (e.g. a
3.4.0 PTR that uses 3.3.5's column layout) are common in the 10 TB
archive. Forcing the user to manually point at a DBD directory on every
PTR load is friction. The chain is deterministic — newest-known
matching schema wins.

**Independent Test**: For a fictional PTR build `3.4.0.15000 enUS PTR`,
delete its DBD file from the schema directory. Open the client, read a
DB2 file, confirm the viewer logs a warning saying "Using DBD from
3.3.5.12340 enUS Retail (3.4.0.15000's DBD was not present)" and the
parse succeeds.

**Acceptance Scenarios**:
1. **Given** the user opens a PTR build with no DBD in the schema dir,
   **When** a DBC/DB2 is read, **Then** the viewer walks the build
   chain (current → previous → ... → oldest known) and uses the
   newest available matching DBD.
2. **Given** the user opens a build from 2017-2018 with a known
   "DB2-changes-every-other-build" instability window, **When** a
   DBC/DB2 read fails to parse against the chain-selected DBD,
   **Then** the viewer surfaces a clear error naming the build key,
   the DBD it tried, and the column offset where parsing failed,
   rather than silently falling back to a different schema.
3. **Given** the user opens a build with no matching DBD in the chain
   at all, **When** a DBC/DB2 is read, **Then** the viewer surfaces a
   "no DBD available" error and the user can supply one via the
   preferences dialog (future slice, just the error path here).

---

### User Story 4 - Remove "Open MK Dataset..." Menu (Priority: P2)

The "Open MK Dataset..." menu item pointed at a legacy JSON MK Dataset
harvester that was always named wrong and was never used. The new
Zarr-first flow (US-2) covers the same ground through auto-detection
plus a manual "Open Zarr Dataset..." override. The legacy menu item is
deleted.

**Why this priority**: Cleanup, but minor. Doesn't block any user
workflow because the legacy menu was vestigial.

**Independent Test**: Open the File menu, confirm "Open MK Dataset..."
is absent. "Open Zarr Dataset..." remains as a manual override for
users who want to point at a Zarr store outside the auto-detected path.

**Acceptance Scenarios**:
1. **Given** the viewer is running, **When** the user opens the File
   menu, **Then** "Open MK Dataset..." is not present.
2. **Given** the viewer is running, **When** the user opens the File
   menu, **Then** "Open Zarr Dataset..." is present and works as
   a manual override (loads a chosen `<mapName>.zarr/` store).
3. **Given** the codebase, **When** the change is merged, **Then**
   the only references to the legacy `VlmProjectLoader` JSON path
   are inside the `VlmProjectLoader` class itself; no UI surface
   references it.

---

### User Story 5 - Performance Win Is Measurable (Priority: **DEFERRED**)

**Status (2026-06-02)**: US-5 is DEFERRED. The perf bench was a measurement
slice gated on US-2. With US-2 deferred, US-5 has no driver. Skip entirely
in the current slice.

**Why this priority**: The win is the reason the user wants Zarr-first,
so it has to be visible. But this is a measurement slice, not a
feature.

**Independent Test**: Run a synthetic 16-tile load benchmark against
both paths (Zarr present vs Zarr missing → MPQ) and confirm
Zarr is faster. Publish the numbers in the spec's Success Criteria
section so future regressions are visible.

**Acceptance Scenarios**:
1. **Given** a 16-tile benchmark on a 3.3.5 client, **When** run
   against Zarr and against MPQ, **Then** the Zarr path completes
   in <70% of the MPQ wall-clock time for the same 16 tiles.
2. **Given** the viewer is running with a Zarr store loaded, **When**
   the user pans the camera, **Then** chunk-decode time per tile
   (Blosc+Zstd) is <50ms on the dev machine and never causes a
   frame stutter (>16ms).
3. **Given** the same viewer in MPQ fallback, **When** the user
   pans the camera, **Then** MPQ read time per tile is higher
   (baseline) and frame stutters are visible at low panning speeds.

---

### Edge Cases

- **Zarr store for a different build**: User has Zarr for 3.3.5.12340
  but opens 3.3.5.12341. The viewer must NOT use the wrong-build Zarr;
  it must fall back to MPQ and log "Zarr cache for 3.3.5.12341 missing;
  available: 3.3.5.12340". (Wrong-build fallback is a separate
  question — see Open Questions.)
- **Zarr store for a different locale**: User has Zarr for
  `3.3.5.12340_enUS_retail/Azeroth.zarr/` but opens the
  `3.3.5.12340_deDE_retail` client. The viewer must not silently
  reuse the enUS Zarr for deDE rendering; it must fall back to MPQ
  and warn.
- **Empty Zarr store**: User has
  `output/datasets/<buildKey>/Azeroth.zarr/zarr.json` but no arrays.
  Treated as a missing Zarr, falls back to MPQ.
- **Corrupt Zarr chunk**: A single chunk file in the Zarr store is
  truncated or fails Blosc/Zstd decode. The viewer falls back to MPQ
  for that specific tile and logs a warning naming the chunk path.
- **MPQ archive with no listfile**: Older clients (1.12) sometimes
  ship without a listfile or with an outdated one. The viewer
  already has MPQ heuristics for this (file-detection-based), so
  the Zarr-first layer doesn't need new behavior — but the
  fallback to MPQ must still work without a listfile.
- **Concurrent Zarr and MPQ load**: User opens a client (MPQ) and
  then triggers the Zarr fallback. The Zarr path becomes the
  primary tile source; the MPQ path remains active for non-terrain
  assets (M2/MDX/WMO/DBC). The data-source abstraction must keep
  both alive concurrently.
- **Build key in path uses non-canonical separator**: User has folder
  named `3.3.5.12340-enUS-Retail` (dashes instead of spaces) or
  `3.3.5.12340.enUS.Retail` (dots). The path parser must normalize
  these into a canonical build key.
- **Pre-release vs Retail vs PTR vs Beta**: User has multiple channels
  for the same build number (e.g. 3.4.0.15000 PTR and 3.4.0.15000
  Retail). The build key includes the channel, so two different
  Zarr stores are kept side-by-side.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Build Detection (US-1)

- **FR-001**: The viewer MUST detect the build key from the
  client folder name when the name matches a canonical pattern
  (`<X>.<Y>.<Z>.<B> <locale> <channel>` or
  `<X>_<Y>_<Z>_<B>`-style snake-case from the staged-clients
  directory).
- **FR-002**: The viewer MUST detect the build key from the
  `WoW.exe` / `Wow.exe` PE header `FileVersion` and
  `ProductVersion` resources when the folder name is ambiguous
  or absent.
- **FR-003**: The viewer MUST NOT show a version picker dialog.
  Build detection is fully automatic.
- **FR-004**: The build key MUST encode major, minor, patch, build,
  locale, and channel. Format: `<X>.<Y>.<Z>.<B>_<locale>_<channel>`.
  Channels: `retail`, `ptr`, `prerelease`, `beta`, `classic`, `classic_ptr`.
- **FR-005**: The viewer MUST preserve the channel tag across all
  downstream lookups (Zarr, DBD, DBC) so a PTR build never silently
  uses a retail schema.

#### Zarr-First Terrain Streaming (US-2)

- **FR-006**: The data-source layer MUST attempt to resolve
  `output/datasets/<buildKey>/<mapName>.zarr/` (per-build subdir)
  before falling back to MPQ. The build key comes from US-1, the
  map name from the loaded WDT.
- **FR-007**: The data-source layer MUST validate the Zarr store
  has all required terrain arrays (height, alpha layers, blend,
  liquid basic type, mcnk flags) before declaring it usable.
- **FR-008**: The data-source layer MUST transparently fall back
  to MPQ per-tile when the Zarr store is missing, partial, or
  corrupt. The user sees no error — only a status-bar note.
- **FR-009**: When a Zarr store is used, the MPQ source MUST NOT
  be read for any tile in that map. The non-terrain assets
  (M2/MDX/WMO/DBC/listfile) MUST still come from MPQ.
- **FR-010**: The "Open Zarr Dataset..." menu item MUST remain as
  a manual override that lets the user point at a
  `<datasets>/<buildKey>/<mapName>.zarr/` store (or any other Zarr
  store) outside the auto-detected path. The override is per-session.

#### DBD Schema Fallback (US-3)

- **FR-011**: The DBD schema lookup MUST walk the build chain from
  the current build backward (newest known matching) when the
  current build's DBD is missing.
- **FR-012**: When a DBD fallback is used, the viewer MUST log a
  status-bar note naming both the current build (whose DBD was
  missing) and the source build (whose DBD was used).
- **FR-013**: The DBD chain MUST be deterministic: same inputs
  always produce the same fallback build. No fuzzy matching.
- **FR-014**: If no DBD in the chain matches, the viewer MUST
  surface a clear error naming the build key and the DB file
  being read, rather than silently guessing.

#### Cleanup (US-4)

- **FR-015**: The "Open MK Dataset..." File menu item MUST be
  removed.
- **FR-016**: The "Open Zarr Dataset..." File menu item MUST
  remain and continue to function.
- **FR-017**: The legacy `VlmProjectLoader` class MAY be removed
  in a follow-up slice but MUST be retained in the codebase for
  one release cycle (no code references it from the UI; tests
  may continue to exercise it).

#### Performance (US-5)

- **FR-018**: The Zarr path MUST outperform the MPQ path for
  16-tile first-load benchmarks by ≥30% wall-clock time
  (acceptance threshold: Zarr < 70% of MPQ wall-clock).
- **FR-019**: Per-tile chunk decode time on the Zarr path MUST
  be <50ms on the dev machine for typical tiles (Blosc+Zstd).
- **FR-020**: Benchmarks MUST be reproducible via a `WowViewer.Tool.Bench`
  tool (or extension to the existing `WowViewer.Tool.Inspect`)
  and the results MUST be checked into the spec as Success
  Criteria evidence.

### Key Entities

- **BuildKey**: a structured tuple of `(int Major, int Minor, int Patch,
  int Build, string Locale, Channel Channel)`. Channel is an enum.
  Serialized to a canonical string `<X>.<Y>.<Z>.<B>_<locale>_<channel>`.
  Used as the directory name for Zarr stores and as the lookup key
  for DBD schemas.

- **Channel enum**: `Retail`, `Ptr`, `PreRelease`, `Beta`, `Classic`,
  `ClassicPtr`. Maps to the folder-name tokens above.

- **DataSource abstraction**: an `IDataSource` interface (existing,
  has `MpqDataSource` as the canonical impl). A new
  `CompositeDataSource` (US-2) wraps a primary Zarr source and a
  fallback MPQ source, routing per-tile reads to whichever is
  authoritative for that tile. The `IDataSource` interface itself
  is unchanged — the composite is an implementation detail.

- **DbdSchemaResolver**: a new resolver that walks the build chain
  in a deterministic order. Returns the matching `DbdFile` for a
  given `(BuildKey, fileName)`. Logs the fallback chain to the
  status bar.

- **ZarrStoreSummary** (existing from spec 041): extended with
  `BuildKey` and `Channel` so the viewer can confirm the store
  matches the loaded client.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Opening any of the 10 staged clients in
  `I:\parp\parp-tools\output\tmp\wowarchive-clients\` produces the
  correct build key (verified against the folder name OR the PE
  header version resource) with zero user prompts. Verified across
  all 10 clients in CI smoke test.
- **SC-002**: For a staged client with a pre-built
  `output/datasets/<buildKey>/<mapName>.zarr/`, opening the map
  shows "Zarr: <mapName>" in the status bar. Removing the Zarr
  store shows "MPQ fallback: <mapName>" without any other visible
  change. Both paths render identical pixels (regression-tested).
- **SC-003**: Zarr-first 16-tile first-load benchmark completes
  in <70% of MPQ-fallback wall-clock time. Numbers published in
  the spec's "Bench" appendix.
- **SC-004**: Opening a PTR build whose DBD is missing succeeds
  by walking the build chain to the prior retail build's DBD.
  A clear status-bar note names both builds.
- **SC-005**: The "Open MK Dataset..." menu item is absent from
  the File menu. The "Open Zarr Dataset..." menu item is present
  and functional.
- **SC-006**: The DBD chain is deterministic. Given the same set
  of DBD files on disk, the resolver always picks the same
  source build for the same input build. Verified via 100-iteration
  fuzz test.

---

## Assumptions

- **A-001**: The 10 TB WoWArchive is read-only. The viewer stages
  Zarr stores into `wow-viewer/output/datasets/` (writable). The
  Zarr output root is per-workspace, not per-client.
- **A-002**: Zarr stores are produced by the existing
  `WowViewer.Tool.Harvest harvest-stream` → `build_v16_dataset.py`
  pipeline (spec 041). This spec does NOT add new harvester
  commands.
- **A-003**: Build key normalization is lossy on purpose: a folder
  named `MyStuff/WoW-Clone` cannot be auto-detected and the viewer
  surfaces a clear "could not detect build" status. The user can
  re-point at a properly-named client.
- **A-004**: The DBD schema directory is `<workspace>/output/dbd/`
  (or another path exposed via a future preferences dialog). The
  build chain ordering is: same major.minor.patch, lower build
  number, descending. If the same build number exists in multiple
  channels, channel-specific DBDs win.
- **A-005**: The viewer is single-process; the data-source layer
  is not thread-safe across `IDataSource` instances. Internal
  locks are per-source.
- **A-006**: Performance numbers in SC-003 are measured on the
  dev machine (Windows 11, NVMe). They are not portable to other
  hardware; the spec only asserts the relative speedup, not the
  absolute numbers.

---

## Open Questions

- **OQ-1 (REVERTED 2026-06-02)**: Wrong-build Zarr is NOT used as a soft
  fallback. Cross-build map comparison is out of scope per the user.
  The Zarr resolver does an exact-match lookup only. If the build
  doesn't have a Zarr store, the viewer falls back to MPQ. The user
  can manually override via the "Open Zarr Dataset..." menu to point
  at a different build's Zarr if they want to compare. The default
  workflow is "open the client you have, get the best terrain path
  available for that build."

- **OQ-2**: Should the DBD chain walk across channels (e.g. PTR
  → retail of the same build number)? Default: NO, channel is a
  hard boundary. A PTR build's DBD chain only walks through
  prior PTRs.

- **OQ-3**: Should the build detection also handle the
  `WoW-Classic` / `WoW-Classic-Era` / `WoW-TBC-Classic` /
  `WoW-Wrath-Classic` product split, where the major.minor.patch
  numbers reset? Default: YES, channel enum has `Classic` /
  `ClassicPtr` slots, and the PE header `ProductName` resource is
  parsed to populate them. Tests for product-name detection
  land in a follow-up slice once a Classic client is staged.

- **OQ-4**: The user mentioned 2017-2018 "DB2 changes every other
  build" instability. Should the DBD chain be smarter for that
  window (e.g. snap to the nearest retail if the requested
  build is in the unstable window)? Default: NO, keep the
  chain simple. The 2017-2018 chaos is handled by the explicit
  fallback error in FR-014: if parse fails, the viewer surfaces
  the column-offset error and the user knows to investigate.

- **OQ-5 (REVERTED 2026-06-02)**: Zarr store layout reverts to **per-build
  subdir**: `<datasets>/<buildKey>/<mapName>.zarr/`. Cross-build map
  comparison is out of scope, so the per-map subdir layout (which
  was specifically chosen to make cross-build comparison easier)
  is no longer motivated. Per-build is simpler and matches the
  natural workspace model where each client build has its own
  output folder.
