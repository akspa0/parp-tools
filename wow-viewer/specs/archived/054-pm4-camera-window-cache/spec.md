# Spec 054: PM4 Per-File Camera Window Cache

**Status**: Draft | **Priority**: P1 | **Owner**: WoWViewer PM4 lane

## Problem

The PM4 overlay takes minutes to render data for the tile the camera is on after a long camera jump, even when the user has already visited that area and the data is on disk. The current user workaround is to disable the PM4 overlay, uncheck every option, re-enable the overlay, click "Reload PM4", and wait. This is a production UX blocker for PM4 work.

### Root cause (from code inspection of `WorldScene.cs:38-60` and `Pm4OverlayCacheService.cs:38-60`)

The PM4 disk cache key is the SHA-256 of `(splitByMscnRef | splitByConnectivity | per-file {path|length|writeTicks})` for **every PM4 file in the current camera window**. When the camera moves, the candidate file list changes (different files in the new window), so:

- The new window's file set produces a new SHA-256 — a brand new cache key, no reuse of the old window's disk cache.
- The in-memory state in `_pm4TileObjects` is kept across the load attempt (per the `replaceExisting` logic), but the disk cache is per-(map, candidate-signature) so the per-file decode cost is paid fresh every jump.
- Re-visiting the start area only helps if the new window's file set exactly matches the original window's file set (it usually does, since the window is computed deterministically from camera position + radius). But any partial overlap (window expanded, window shifted by one tile, different radius) misses the cache entirely.

Compounding the pain:

- The PM4 decode path (`BuildPm4TileObjects`) does real CPU work: line/triangle budget enforcement, Mslk ref-index walk, MCLY-style grouping, Ck24 split-by-MscnRef/connectivity. A single tile takes ~hundreds of milliseconds; a 5×5 window of 25 tiles is ~seconds; the 7×7 default is ~10s on a fast machine. Without a per-file cache hit, the second jump is just as expensive as the first.
- The cache file for the on-disk layer is a single big gzip blob containing every file in the window. There is no per-file decomposition. A user who has visited 10 different areas has 10 separate cache blobs; a jump into an area not in the current cache file forces a full re-decode of that area's files even if many of them were decoded and stored under a different window's cache key.
- The `candidateSignature` re-hash on every camera-move is itself unnecessary work: the signature is only sensitive to file content + the two split flags, not to the window.

## User Stories

### US1: Instant first-render on a re-visited tile (P1)

As a user, when I teleport the camera to a tile I've visited before, the PM4 overlay for that tile should appear in well under one second — using the in-memory or on-disk cache, not a fresh decode.

**Why this priority**: This is the user's literal pain point. Without it, the PM4 work is unusable for any workflow that involves camera jumps (which is most of them — object matching, WMO exploration, scene-graph inspection).

**Independent Test**: Load a map, jump the camera, wait for PM4 ready, jump back, time the second PM4 ready. Must be < 1s for a tile that was previously decoded and is still in the in-memory cache. Must be < 5s for a re-visit that requires loading the on-disk cache (the cost of gzip decompress + per-tile materialization, not the cost of a fresh decode).

**Acceptance Scenarios**:
1. **Given** the PM4 overlay is loaded for some tile A in the camera window, **When** the user teleports to tile A', then back to tile A, **Then** the PM4 overlay for tile A reappears without a fresh decode (in-memory hit, < 100ms).
2. **Given** the PM4 overlay was loaded for tile A in a prior session, then the viewer was closed and reopened, **When** the user teleports to tile A, **Then** the PM4 overlay for tile A loads from the on-disk per-file cache (< 5s for a single tile, no fresh decode of MSUR/MSLK/MSCN/MSPV).
3. **Given** the PM4 overlay was loaded for tile A in a prior session, **When** the user teleports to tile B (a new tile), then back to tile A, **Then** the PM4 overlay for tile A loads from the on-disk per-file cache (B does not invalidate A's cache; < 5s for tile A's re-render).

### US2: Visible status while loading new tiles (P1)

As a user, when the PM4 overlay is decoding new tiles for a camera jump, I want to see which tiles are still being decoded and which are already showing data, so I know the work is making progress and not stuck.

**Why this priority**: When the user already understands that some tiles need a fresh decode, the worst UX is "minutes with no visible feedback." The existing `_pm4Status` string already has the "PM4 loading: ... files, X/Y files" pattern; this story extends it with per-tile progress.

**Independent Test**: Trigger a jump that requires fresh decode of multiple tiles. The workbench status line must show the per-file progress (e.g., `PM4 loading tile (32,48) 4/9`) and the visible tile count must grow as each tile finishes.

**Acceptance Scenarios**:
1. **Given** the PM4 overlay is decoding N new tiles, **When** the user watches the workbench, **Then** the status line shows a count that increases from 0 to N as tiles complete.
2. **Given** the PM4 overlay is decoding new tiles, **When** the user looks at the viewport, **Then** each tile that has finished decoding shows its overlay immediately (progressive render), even before all tiles in the new window are done.

### US3: Cache stays warm across camera jumps (P1)

As a user, when I move the camera around a familiar area, the PM4 overlay should stay fast — re-visiting a tile I was just on should be effectively free, and visiting a new tile in the same area should use the on-disk cache if I've been there in a prior session.

**Why this priority**: This is the "stays warm" property. Without it, the cache helps only the exact re-visit; a 1-tile shift forces a fresh decode of the shifted tile.

**Independent Test**: Decode tile A, then shift camera to A+1, then back to A. Tile A's re-render is from in-memory cache (< 100ms). Decode tiles A through E, then shift to F+1. Tile F+1 is a fresh decode (correctly). Return to A; tile A re-renders from in-memory cache.

**Acceptance Scenarios**:
1. **Given** tiles A, B, C are in the in-memory PM4 cache, **When** the camera window expands to include A, B, C, D, E, **Then** A/B/C render from in-memory (free) and D/E decode fresh (correctly). Total time is the time to decode only D and E, not all five.
2. **Given** tiles A, B, C were decoded in a prior session, **When** the camera window includes A, B, C, **Then** all three render from the on-disk per-file cache (no fresh decode).
3. **Given** a tile X was decoded fresh in this session (no prior on-disk entry), **When** the session ends and a new session starts, **Then** the on-disk per-file cache for X is now available for the new session (i.e., a fresh decode also writes the per-file entry).

### US4: No "uncheck every option and re-check" workflow needed (P2)

As a user, the PM4 overlay's per-option toggles (MSCN Nodes, MSPV Nodes, Mesh Lines, Mesh Triangles, Solid Fill, X-Ray) should not affect the camera-window cache behavior. The current code already doesn't, but the user workaround implies a bug in this area — this story is the regression test.

**Why this priority**: Lower priority because the camera-window cache is the actual fix; this story is a regression test that ensures the per-option toggles don't accidentally invalidate the cache.

**Independent Test**: Decode a tile, toggle each option in turn, verify the camera-window cache hit/miss behavior is unaffected.

**Acceptance Scenarios**:
1. **Given** tile A is in the in-memory PM4 cache, **When** the user toggles each option in the workbench (MSCN Nodes, MSPV Nodes, Mesh Lines, Mesh Triangles, Solid Fill, X-Ray) in any order, **Then** tile A's data is not re-decoded (no cache miss caused by toggles).
2. **Given** tile A is in the on-disk per-file cache, **When** the user toggles each option, **Then** tile A's on-disk cache is still valid (signature unchanged).

## Functional Requirements

### FR-001: Per-file in-memory PM4 cache

- A new `Pm4PerFileCache` type stores, for each PM4 file path in the current map, the decoded `Pm4TileObjects` (list of `Pm4OverlayObject` per CK24) plus the position refs.
- Cache key: `string` (the normalized virtual path of the PM4 file in the data source).
- Cache value: a small record holding the per-tile objects list + position refs + the file's `LastWriteTicks` (so we can detect a content change).
- Lookup is O(1) by file path. Insertion is O(1). The cache grows as the camera visits new tiles; it does not shrink automatically.
- The cache is owned by `WorldScene` and is cleared on `ReloadPm4Overlay()` (full re-decode request from the user).

### FR-002: Per-file on-disk PM4 cache

- The on-disk cache layout changes from a single big `(map, candidate-signature)` blob to a directory of per-file entries: `output/cache/pm4-overlay/{dataSourceSegment}/{mapName}/files/{normalizedPath}.pm4cache`.
- Each per-file entry is a small gzip blob containing the per-tile objects + position refs for that one PM4 file.
- The on-disk cache version bumps from 7 to 8 (intentional format break; old per-window cache is invalidated; the new per-file cache is rebuilt on next use).
- Each per-file entry is content-keyed: the `LastWriteTicks` (or file length, or both) is recorded inside the entry and compared on read; a mismatch is treated as a miss.
- A fresh decode that succeeds writes the per-file entry to disk before returning. The in-memory cache is updated in the same call.

### FR-003: Camera-window load uses per-file cache first

- When the camera window changes, the load path iterates the new window's files one at a time (not as one batch).
- For each file: check in-memory cache → check on-disk per-file cache → fall through to fresh decode.
- A file that resolves in either cache is added to `_pm4TileObjects` immediately; the user sees progressive render (US2).
- A file that requires fresh decode runs through the existing `BuildPm4TileObjects` path; on success, the result is written to both caches.
- The load is still asynchronous (does not block the render thread) and is still cancellable on a new camera jump.

### FR-004: Camera-window status with per-file progress

- The existing `_pm4Status` string gains a per-file progress format: `PM4 loading: <current-file> <decoded>/<total> files, <memcache-hits>/<diskcache-hits> from cache`.
- The workbench status line in `ViewerApp_Pm4Utilities.cs:237-252` already reads `_pm4Status`; it picks up the new format automatically.
- A new optional per-tile counter in the status (`tiles = <visibleTileCount>`) gives the user a real-time view of how many tiles are showing data.

### FR-005: Per-file cache survives session restart

- On viewer close, the in-memory cache is lost (acceptable; on-disk cache is the persistence).
- On viewer open, the first PM4 load for any tile uses the on-disk per-file cache if available.
- The per-file on-disk cache lives at `output/cache/pm4-overlay/{dataSourceSegment}/{mapName}/files/{normalizedPath}.pm4cache` and survives across sessions.

### FR-006: Cache invalidation rules

- The per-file cache is invalidated (treated as miss) when:
  - The PM4 file's `LastWriteTicks` or `Length` on disk differs from what's recorded in the cache entry (content changed).
  - The user explicitly clicks "Reload PM4" (full invalidation of in-memory + on-disk for the current map).
  - The `Split CK24 by MscnRef` or `Split CK24 by Connectivity` toggle changes (re-decodes the affected files; old per-file entries with the old split flags are kept on disk for re-use if the user toggles back).
  - The data source identity changes (e.g., switching from a loose-file root to an MPQ root for the same map).
- The per-file cache is NOT invalidated by:
  - The `MSCN Nodes` / `MSPV Nodes` / `Mesh Lines` / `Mesh Triangles` / `Solid Fill` / `X-Ray` toggles (these are render-time only).
  - The camera window changing.
  - The "Reload PM4" button if the user only changed camera and the file content is the same (i.e., the next load should still hit the per-file cache; the "Reload" button only clears the in-memory state and forces a fresh read, but the on-disk per-file cache is the source of truth for content).

## Non-Functional Requirements

### NFR-001: No regression in non-cache scenarios

- The first-ever load of a map (no on-disk per-file cache, no in-memory cache) must take the same time as today. The new cache code adds at most one extra dictionary lookup per file (negligible).
- The cache version bump (7 → 8) invalidates the old per-window cache. The first load after this change rebuilds the per-file cache for the current camera window. Subsequent loads are faster.

### NFR-002: Bounded memory growth

- The in-memory per-file cache is bounded by the number of PM4 files the user has visited in the session. A small LRU cap (e.g., 256 files = ~256 × a few MB = low hundreds of MB) is acceptable; the user can raise/lower it via a viewer setting.
- Out-of-scope for this spec: a hard memory cap with eviction. We just need "doesn't grow unbounded forever" — a simple LRU is fine.

### NFR-003: Tests

- Bounded unit tests for `Pm4PerFileCache` (key/value insert/lookup/miss/evict).
- Bounded integration test: build a per-file cache from a real PM4 file, close the cache, reopen it, verify the loaded data matches.
- Bounded real-data smoke: load the development `00_00` tile, jump to `32_48` (no cache), then back to `00_00` (cache hit), and assert the second load is faster than the first.

## Success Criteria

1. A user who has visited N tiles in a session sees those N tiles re-render in < 100ms each on camera return (in-memory hit).
2. A user who has visited N tiles in a prior session sees those N tiles re-render from disk in < 5s each on next session start (on-disk per-file hit).
3. A camera jump that requires fresh decode of M new tiles takes only M × (single-tile decode cost), not the full window's decode cost.
4. The user no longer needs to "uncheck every option and re-check" the PM4 overlay to get it to load data for the current tile.
5. The PM4 workbench status line shows real per-file progress during a fresh decode.

## Out of Scope

- A hard memory cap with LRU eviction. The spec includes a simple bounded LRU but does not include eviction policy tuning.
- Networked/shared PM4 cache (e.g., a workspace-level cache that survives between users). Per-machine per-session only.
- Predictive pre-fetch (loading tiles the user is *likely* to jump to next). We do opportunistic loading inside the current window only.
- Compression changes (we keep gzip for now; the per-file entries are small and the bottleneck is the decode, not the I/O).
- Migration of the old `(map, candidate-signature)` cache blob to the new per-file format. The old blob is simply invalidated; the new entries are rebuilt on next use.
- Any change to the on-disk format of the per-tile object records (lines, triangles, etc.). The cache stores the same `Pm4OverlayObject` records; the storage granularity changes, not the payload.

## Cross-references

- Spec 051 (`wow-viewer/specs/051-pm4-mscn-mspv-visualization/`) — owns the visualization (MSCN/MSPV cubes) and the status counter work; spec 054 picks up the "fast camera window" lane that 051 did not own.
- Spec 050 (`wow-viewer/specs/050-pm4-wmo-group-matching/`) — the WMO group matcher consumes PM4 data; spec 054 makes that matcher faster on re-visits.
- `wow-viewer/src/core/WowViewer.Core.PM4/Signatures/` (planned, spec 051 Phase 3) — the per-object signature work; spec 054 reuses `Pm4OverlayObject` and does not change the signature API.
- `wow-viewer/docs/architecture/pm4-chunk-semantics.md` — the canonical doc for the data; spec 054 does not change semantics.
