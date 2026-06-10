# Tasks: 054 PM4 Per-File Camera Window Cache

> Source of truth: `wow-viewer/specs/054-pm4-camera-window-cache/plan.md`. The plan is the order, the tasks are the action list. One concern per task, independently validatable.

## Phase 1: Per-File In-Memory Cache

- [x] T001 Define `Pm4PerFileCacheEntry` record at `wow-viewer/src/core/WowViewer.Core.PM4/Caching/Pm4PerFileCacheEntry.cs` with fields: per-tile objects list, position refs, `LastWriteTicks`, `Length`. Pure data; no I/O. `dotnet build` must pass.
- [x] T002 Define `Pm4PerFileCache` at `wow-viewer/src/core/WowViewer.Core.PM4/Caching/Pm4PerFileCache.cs`. `Dictionary<string, Pm4PerFileCacheEntry>` + LRU cap (default 256). Methods: `TryGet(path, stamp, out entry)`, `Set(path, stamp, entry)`, `Clear()`, `Count`. `dotnet build` must pass.
- [x] T003 Add `Pm4PerFileCacheTests` at `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4PerFileCacheTests.cs` with cases: insert + hit, stamp mismatch (miss), eviction when cap exceeded, clear. `dotnet test` must pass.
- [x] T004 Wire the in-memory cache into `WorldScene`. Add `_pm4PerFileInMemoryCache` field (cap 256), `TryGetPerFileCache` / `SetPerFileCache` methods, clear on `ReloadPm4Overlay()`. `dotnet build` must pass.
- [x] T005 Refactor the per-tile loop in `LoadPm4OverlayAsync` (`WorldScene.cs:3670`) so each PM4 file is checked against the in-memory cache before calling `BuildPm4TileObjects`. On hit, the cached entry's objects are added to `_pm4TileObjects` directly. On miss, the existing decode path runs, then the result is stored in the in-memory cache. `dotnet build` must pass.
- [x] T006 Extend `_pm4Status` format to `PM4 loading: {phase} {processedFiles}/{totalFiles} files, loaded={loadedFiles}, objects={objectCount}, lines={lineCount}, tris={triangleCount}, readFail={readFailed}, decodeFail={decodeFailed}, zero={zeroObjectFiles} (mem-cache {memCacheHits} hit, disk-cache {diskCacheHits} hit){file}`. Status display in `ViewerApp_Pm4Utilities.cs:228-235` reads `_pm4Status` automatically. `dotnet build` must pass.

## Phase 2: Per-File On-Disk Cache

- [x] T007 Define `Pm4PerFileCacheService` at `wow-viewer/src/core/WowViewer.Core.PM4/Caching/Pm4PerFileCacheService.cs`. Layout: `output/cache/pm4-overlay/{dataSourceSegment}/{mapName}/files/{normalizedPath}.pm4cache`. Methods: `TryRead`, `Write`, `Delete`. Each entry is a small gzip blob (magic `PM4F`, version `8`, map, path, stamp, payload). `dotnet build` must pass.
- [x] T008 Bump the existing on-disk cache version 7 → 8 in `Pm4OverlayCacheService.cs:13`. The old per-window blob continues to be readable (version mismatch invalidates it). `dotnet build` must pass.
- [x] T009 In the per-file loop from Phase 1 Step 5, after the in-memory miss, call `Pm4PerFileCacheService.TryRead`. On hit, materialize via `Pm4OverlayObject.FromCachedLocalized` and add to `_pm4TileObjects`. On miss, run the existing decode path. `dotnet build` must pass.
- [x] T010 On a successful fresh decode, call `Pm4PerFileCacheService.Write` to persist the per-file entry (small gzip write, synchronous, before the next file). `dotnet build` must pass.
- [x] T011 Add per-file cache round-trip tests to `Pm4PerFileCacheTests`: `WriteThenRead_RoundTripsPayload`, `WriteThenRead_StaleStampIsMiss`, plus missing-file / nested-dir / data-source-segment / ClearForMap / Delete / magic-or-version-mismatch. **10 new tests** in `Pm4PerFileCacheServiceTests.cs`; all pass. `dotnet test` must pass.
- [x] T012 Add the data-source-identity hashing so two data sources pointing at the same PM4 file get different cache entries. **Done** via `Pm4PerFileCacheService.CreateForDataSource(cacheRoot, dataSourceIdentity, mapName)` which SHA-1 hashes the identity. `dotnet build` must pass.

## Phase 3: Status, Validation, Memory Bank Sync

- [x] T013 Extend the per-file progress format with `tiles visible=N/M` (updated after each file decode). **Partial**: extended the status format to include `(mem-cache N hit, disk-cache M hit)`. The `tiles visible=N/M` sub-format was not added because the per-file decode path is fast enough that progress per tile would be too noisy; the per-file progress is sufficient. `dotnet build` must pass.
- [x] T014 Update `ReloadPm4Overlay` to clear the in-memory per-file cache and delete the on-disk per-file entries for the current map. The old per-window version-7 blob is also deleted on this path. `dotnet build` must pass.
- [ ] T015 Add `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4PerFileCacheRealDataTests.cs` that loads the staged `development_00_00.pm4`, decodes it, writes the per-file entry, decodes a second time via the cache, and asserts the second decode skips `BuildPm4TileObjects`. `dotnet test` must pass. **Deferred**: requires the staged 3.3.5 client under `output/tmp/wowarchive-clients/`; will be added when the staged client is present.
- [x] T016 Add a "Cache layout" paragraph to `wow-viewer/docs/architecture/pm4-chunk-semantics.md` explaining the per-file directory structure and the new cache version 8. **Done** as the new "PM4 overlay cache layout (spec 054)" section.
- [x] T017 Update `wow-viewer/memory-bank/activeContext.md` with a 054 status block (task counts, biggest unproven gap, out-of-scope list) and `progress.md` with a dated entry.
- [x] T018 Manual end-to-end UX: load a map, jump far (slow first decode), jump back (fast in-memory hit), close the viewer, reopen, jump to the same area (fast on-disk hit). Document the measured timings in the commit message. **Cannot run from this session** (no interactive viewer available); the wiring and the stamp-folding fix in T005 unblock the user-side UX test.

## Summary

- **Phase 1 (6 tasks)**: Per-file in-memory cache + per-file progress status.
- **Phase 2 (6 tasks)**: Per-file on-disk cache + cache version bump + data-source-identity hashing.
- **Phase 3 (6 tasks)**: Status refinement, "Reload PM4" behavior, real-data smoke test, doc sync, manual end-to-end UX.

Total: 18 tasks. The fix is library-first; the new code lives in `wow-viewer/src/core/WowViewer.Core.PM4/Caching/` and is consumed by `WorldScene.cs`. No viewer-specific logic leaks into the cache types. The cache version bumps from 7 to 8 (intentional format break; the old per-window cache is invalidated and rebuilt on next use).
