# V16 Harvest Recovery Plan — 2026-05-17

## Goal

Fix the V16 corpus builder so archive-backed harvest stays in memory, incomplete
builds can resume at map granularity, and future rebuilds default to a faster
Zarr write profile.

This is a recovery plan for the current V16 lane. It does not replace the
existing dataset contract, and it does not invalidate already-finished `.zarr`
stores.

## Why This Exists

The current archive-backed path still stages ADT family files into
`%LOCALAPPDATA%\Temp\wowviewer_harvest_*` before feeding the tensor builders.
That breaks the intended `NativeMpqService` streaming contract and adds avoidable
disk churn. The current Python writer also uses relatively expensive Zarr
compression defaults and does not have a real resume contract for interrupted
multi-map builds.

## Phase 1 — In-Memory Archive Harvest

Goal: stop writing root/`_tex0`/`_obj0` ADT files to temp disk during
archive-backed harvest and discovery.

Steps:

1. Add an in-memory ADT tensor-pack builder path in `WowViewer.Core.IO`.
   - Input: logical source paths plus root/`_tex0`/`_obj0` byte arrays.
   - Output: unchanged `TerrainTileTensorPack`.

2. Move archive-backed harvest and `discover-maps` to the in-memory builder.
   - `BuildPackFromArchiveAdt(...)` should no longer create temp folders.
   - `SourceAdtPath` metadata should preserve the logical virtual ADT path, not
     a temp filesystem path.

3. Keep archive WL fallback and minimap BLP loading unchanged.
   - This phase is about removing ADT temp extraction only.

Validation intent:
- rejected-tile reports and metadata should show logical ADT paths under
  `World\Maps\...`, not `%TEMP%`
- split-ADT placements and object masks must remain available

## Phase 2 — Map-Level Resume

Goal: allow interrupted V16 builds to continue from completed maps without
starting the build from scratch.

Steps:

1. Add a resume manifest inside `<build>.zarr.partial/`.
   - Track completed maps, tile count, and build/write settings.

2. Persist index/placements sidecars after each completed map.
   - Resume should trust only completed-map state, not a partially streamed map.

3. Add `--resume` to `build_v16_dataset.py build`.
   - If a staged partial exists and settings match, skip completed maps and
     continue appending tiles.
   - If settings mismatch, fail loud instead of guessing.

Validation intent:
- restarting after interruption should reuse completed maps and continue with
  remaining maps only
- failed runs should still preserve diagnostics and rejected-tile reports

## Phase 3 — Faster Zarr Defaults

Goal: make future V16 rebuilds materially faster without breaking readers or
already-finished stores.

Steps:

1. Change future builder defaults to a faster codec profile.
   - Prefer `lz4` or low-level `zstd` over `zstd` level 5.

2. Surface codec settings in the builder CLI.
   - Existing stores remain readable because the schema does not change.

3. Keep writes single-writer for now.
   - Do not introduce concurrent writers in this phase.
   - Race-safe parallelism can be revisited later once the write path is sane.

Validation intent:
- new builds can use a faster codec without changing dataset schema
- existing `V16Dataset` reader still opens old and new stores together

## Out Of Scope For This Slice

- content-addressed deduplication across builds
- collapsing all builds into one physical mega-store
- multi-process Zarr writing
- training-model architecture changes

## Implementation Order

1. Land this plan and sync continuity.
2. Remove temp ADT extraction from archive-backed harvest/discovery.
3. Add map-level resume support to the Python builder.
4. Switch future Zarr defaults to a faster profile and document it.
