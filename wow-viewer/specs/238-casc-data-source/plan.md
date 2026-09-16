# Implementation Plan: CASC Data Source (Local Install + Remote CDN)

**Branch**: `v0.5.4-dev` (v0.6 release line) | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md) | **Release**: v0.6

> Authored from the template by hand: `setup-plan.ps1` rejects non-numbered branches. Use
> `$env:SPECIFY_FEATURE_DIRECTORY = 'specs/238-casc-data-source'` for speckit scripts.

## Summary

Vendor **TACTSharp** as a properly registered submodule. Wrap it in a Core.IO `CascArchiveCatalog`
that implements the existing `IArchiveCatalog` plus a new id-addressed read surface. Expose it to the
viewer through a thin `CascDataSource : IDataSource`. Local, remote and hybrid storage modes share one
catalog. The remote cache holds verbatim hash-verified objects. Every supported root-manifest era is
proven on a real build against an independent extraction.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: TACTSharp (wowdev, MIT). Existing `IArchiveCatalog`/`IDataSource` seams. `libs/wowdev/wow-listfile`. `libs/wowdev/DBCD` (unchanged here).

**Storage**: Local CASC (read-only). A remote object cache on local disk, with its location set by operator configuration.

**Testing**: xUnit. Unit tests for the id/path resolution and result types. Real-build tests skip unless `WOWVIEWER_CASC_LOCAL` / `WOWVIEWER_CASC_REMOTE_BUILD` are set.

**Target Platform**: Windows desktop viewer, cross-platform core and CLI

**Project Type**: Library + CLI + desktop viewer

**Performance Goals**: SC-004 (a remote map opens in under 2 minutes cold, under 20 seconds warm). A local read performs comparably to MPQ reads of similar size.

**Constraints**: Constitution I (vendored), VI (no hardcoded roots/hosts), VII (read-only containers). Thread-safe under streaming.

**Scale/Scope**: One new catalog, one data source, one CLI command group, and `IDataSource`/`IArchiveReader` id-read extensions

**Environment (2026-09-16)**: a local CASC install is in progress at `I:\wow12\World of Warcraft` (operator configuration; never hardcoded). **Local mode is the first working path once it completes**; remote CDN follows. Its first consumer is Spec 237's asset resolution.

**NEEDS CLARIFICATION → research**: TACTSharp's era coverage (R2); hybrid-mode behavior (R5); historical build mirrors (R6); the independent reference tool for SC-001 (R8)

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Repo independence | PASS (action) | TACTSharp is vendored as a submodule under `wow-viewer/libs/wowdev/TACTSharp`, registered in `.gitmodules`. The existing unregistered `libs/*` gitlinks are recorded as a separate finding and not fixed silently. |
| II. Library-first | PASS | The catalog lives in `WowViewer.Core.IO/Casc`. The viewer data source and CLI are thin. |
| III. Real-data validation | PASS (gated) | SC-001/002/003 use real builds with hashes and receipts. |
| VI. No client path assumptions | PASS | Install path, cache dir, hosts, region, locale and keys are all configuration. |
| VII. Containers are inputs | PASS | CASC is read only. The remote cache stores verbatim downloaded input objects for reuse, and no command exports it. **Operator confirmed 2026-09-16: caches are fine** (the project already uses caches). |
| Data Policy (BYOD) | PASS | The cache is local and excluded from git and releases. |
| Format reader ownership | PASS | No CASC reader exists in wow-viewer. |
| One phase at a time / bite-sized | PASS | 4 phases, at most 10 steps each. |

## Project Structure

```text
libs/wowdev/TACTSharp/                        # NEW submodule (pinned commit)
.gitmodules                                   # register TACTSharp

src/core/WowViewer.Core.IO/
├── Files/IArchiveReader.cs                   # + IFileDataIdReader (ReadFileById, FileIdExists, TryResolveId)
├── Files/FileReadResult.cs                   # NEW: Ok / NotPresent / KeyUnavailable / OfflineUnavailable / Failed
└── Casc/
    ├── CascStorageOptions.cs                 # mode (Local/Remote/Hybrid), paths, hosts, region, locale, keys
    ├── CascBuildIdentity.cs                  # product, version, buildConfig, cdnConfig
    ├── CascProductDiscovery.cs               # local .build.info + remote version service
    ├── CascArchiveCatalog.cs                 # IArchiveCatalog + IFileDataIdReader over TACTSharp
    ├── CascListfile.cs                       # path<->id map (reuses the vendored listfile)
    ├── CascKeyRing.cs                        # key set load/refresh
    └── CascCacheStats.cs

src/viewer/WoWViewer/DataSources/
├── IDataSource.cs                            # + default id-read members (non-breaking)
└── CascDataSource.cs                         # NEW thin wrapper
src/viewer/WoWViewer/ (open-client UI)        # "Open CASC install…" / "Open remote build…"

tools/inspect/WowViewer.Tool.Inspect/Program.cs   # casc products|builds|identity|read|verify|cache
docs/CLI-TOOLS.md, docs/architecture/casc-data-source.md

tests/WowViewer.Core.Tests/
├── CascListfileTests.cs, FileReadResultTests.cs
└── CascRealDataTests.cs                      # env-gated
```

## Phases

### Phase 0: Dependency + era survey (research gate)

1. Add TACTSharp as a submodule at a pinned commit and register it in `.gitmodules`. Build it against net10.0 inside the solution.
2. Record the unregistered existing `libs/*` gitlinks in `research.md` as a finding. Do not repair them in this spec unless the operator directs.
3. Spike (throwaway CLI command): open one local build, read 10 files by id, print identity.
4. Survey TACTSharp's root/encoding era support against at least one real build per era (6.x pre-TVFS root, 8.2+ root, the newest retail) and record the results.
5. Decide remote host/mirror configuration for historical builds (R6), and confirm with Marlamin where useful.
6. **Gate**: `evidence/phase0-era-survey.md` holds the per-era pass/fail table and the library decision (TACTSharp, or the fallback with its license resolved).

### Phase 1: Core catalog, local mode (US1, US5; FR-001, 003, 004, 008, 010)

1. `FileReadResult` + `IFileDataIdReader` in Core.IO, with unit tests.
2. `CascStorageOptions` + `CascBuildIdentity`.
3. `CascProductDiscovery` (local): enumerate products and builds from the install.
4. `CascListfile`: load the vendored listfile into bidirectional maps. Unknown-name ids are allowed.
5. `CascArchiveCatalog` (local): `ReadFile(path)`, `ReadFileById`, `FileExists`, `GetAllKnownFiles`, and locale selection.
6. Concurrency: a thread-safety stress test (many parallel reads, results stable).
7. CLI `casc products`, `casc identity`, `casc read --id|--path --out`.
8. CLI `casc verify --sample N --reference <dir>`: hash comparison against an independent extraction (R8).
9. **Gate**: `evidence/phase1-local-verify.md` shows SC-001 on a real local build, plus SC-003 for each surveyed era.

### Phase 2: Remote + cache + keys (US2, US3, US4; FR-002, 005, 006, 007, 012)

1. Remote discovery: the version list for a product/region, and opening by explicit build identity.
2. On-demand remote reads through TACTSharp with a configured cache directory. Verify the hash before use; discard and re-fetch on mismatch.
3. Host failover and per-file failure reporting. Coalesce duplicate in-flight downloads.
4. Offline behavior: cached objects serve; uncached objects return `OfflineUnavailable`.
5. `CascKeyRing`: load a key set from configuration, refresh at runtime, and return `KeyUnavailable` (with key id) for unknown keys.
6. Hybrid mode (local first, remote fill) behind an explicit option (R5).
7. CLI `casc builds --product --region`, `casc cache --stats|--verify`.
8. **Gate**: `evidence/phase2-remote.md` shows SC-002 (local vs remote byte identity), SC-005 (offline) and SC-006 (keys).

### Phase 3: Viewer integration (FR-009, 011; US6)

1. `IDataSource` default id-read members; MPQ and loose sources keep the defaults (returning not-present).
2. `CascDataSource` wrapper, including stats for the existing data-source diagnostics UI.
3. Viewer "Open CASC install…" and "Open remote build…" flows (product, region, build picker), with the build identity shown in the status bar.
4. Asset-path fallbacks: existing name-based lookups go through the listfile, so pre-FileDataID readers work unchanged where names exist.
5. Regression: open 0.5.3 and 3.3.5 MPQ clients; time the load (SC-007); run the full test suite.
6. **Gate**: `evidence/phase3-viewer.md` shows the SC-004 timings on a remote build map and the regression results. Map rendering of modern formats is **not** claimed here (Spec 239).

## Complexity Tracking

| Item | Why | Simpler alternative rejected |
|---|---|---|
| New submodule (TACTSharp) | CASC/TACT/BLTE/encryption/remote protocol is a large, moving target maintained by the domain experts | Writing a CASC reader in-house duplicates maintained MIT code and would lag every client format change |
