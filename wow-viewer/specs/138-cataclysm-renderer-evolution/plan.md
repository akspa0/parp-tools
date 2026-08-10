# Implementation Plan: Cross-Era Terrain Source and Renderer Foundation

**Branch**: `138-cataclysm-renderer-evolution` | **Date**: 2026-08-09 | **Spec**: [spec.md](./spec.md)

## Summary

Build one profile-gated terrain path that can read and render basic terrain from the legacy MPQ
line through 11.x CASC clients. Preserve `NativeMpqService` for 0.x–5.x, add a narrow CascLib
adapter for early CASC builds, evaluate TACTSharp for later CASC builds, and use wow.export and
WoWTools.Minimaps as comparative authorities. Archive choice, format choice, and optional terrain
signals must be explicit in build-scoped provenance rather than inferred or silently substituted.

## Technical Context

**Languages/Versions**: C#/.NET 10 for source adapters, profile contracts, terrain readers, and
tools; C++ only behind the CascLib native boundary; Python/uv remains downstream and does not read
client archives directly.

**Primary Dependencies**: Existing `IArchiveReader`/`IArchiveCatalog`, `NativeMpqService`,
`WowViewer.Core.IO` format readers, the already-integrated DBCD/WoWDBDefs/listfile authorities,
pinned MIT CascLib reference, and a capability-scoped TACTSharp adapter candidate. TACT.Net is
reference-only until GPL-3.0 review is complete.

**Storage**: Configured loose client roots, MPQ archives, local CASC installations, optional CDN-
structured caches, and read-only listfile/DB definition inputs. No client root is stored in source.

**Testing**: Focused C# unit/contract tests, archive probe fixtures, real-data reads from approved
configured client roots, `dotnet build`, and `dotnet test`. User-run extraction, broad harvest,
GPU work, and long benchmarks are prepared but not launched by Codex.

**Target Platform**: Windows x64 first, with the managed source contract kept portable. Native
CascLib loading must report architecture and dependency failures clearly.

**Project Type**: Shared .NET library plus thin inspect/harvest CLI surfaces.

**Performance Goals**: Correctness first. Record open time, first-file latency, repeated-file
latency, and peak memory for each adapter during validation; do not set a performance signoff
threshold until equivalent 6.x, 7.x, and later fixtures exist.

**Constraints**: Repo-independent; one owner per archive/format surface; configured client roots;
profile-gated optional chunks; fail-closed provenance; no legacy parser rewrite; no accidental GPL
runtime dependency; one phase at a time.

**Scale/Scope**: Basic terrain source and rendering across representative 0.5.3, 1.x, 3.3.5,
4.0.0, 6.x, 7.x, and 11.x builds. Full parity for objects, liquids, modern materials, WMO/M2
features, and all later chunks remains separate capability work.

## Constitution Check

| Principle / constraint | Status | Notes |
|---|---|---|
| Repo independence | PASS | New ownership remains under `wow-viewer`; external projects are references or pinned dependencies with an explicit boundary. |
| Library first | PASS | Adapters implement the existing shared archive seam; terrain tools remain thin. |
| Real-data validation | PASS | Each adapter requires a configured, fingerprinted client fixture and a content probe. |
| No hardcoded client paths | PASS | Roots, products, locale, tags, and caches are runtime configuration. |
| Existing readers remain canonical | PASS | `NativeMpqService` and existing ADT/WDT/BLP readers are reused. |
| License boundary | PASS WITH GATE | CascLib/TACTSharp references are MIT; TACT.Net is GPL-3.0 and remains isolated pending review. |
| User runs heavy work | PASS | Broad extraction, harvest, and benchmarks are operator-run commands only. |
| One phase at a time | PASS | Each phase has a probe or real-data gate before the next adapter or terrain capability. |

## Project Structure

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/
│   ├── Files/                         # existing IArchiveReader/IArchiveCatalog seam
│   ├── Archive/                       # source profile and provenance models
│   └── Casc/                          # CascLib interop and TACTSharp adapter boundary
├── tools/inspect/WowViewer.Tool.Inspect/
│   └── archive/                       # source/profile/probe reporting
├── tools/harvest/WowViewer.Tool.Harvest/
│   └── existing harvest path           # consumes the shared archive contract
├── tests/WowViewer.Core.Tests/
│   ├── Archive/                        # profile, provenance, and adapter contract tests
│   └── Maps/                           # cross-profile terrain fixtures
└── specs/138-cataclysm-renderer-evolution/
    ├── spec.md
    ├── research.md
    ├── data-model.md
    ├── quickstart.md
    ├── contracts/source-profile.schema.json
    └── tasks.md                         # generated later by speckit-tasks
```

**Structure Decision**: The archive source boundary belongs in `WowViewer.Core.IO`, the existing
canonical owner for client-file access. The renderer and harvester receive virtual bytes plus
profile capabilities and do not reference CascLib, TACTSharp, wow.export, or MPQ internals.

## Phase 0 — Source inventory and provenance contract (P1)

1. Pin the exact GitHub repository, commit/package version, license, and local checkout state for
   CascLib, TACTSharp, WoWTools.Minimaps, wow.export, and any new archive dependency; record the
   already-integrated DBCD/WoWDBDefs/listfile versions without creating a second metadata path.
2. Define the source-profile JSON contract and common provenance record without changing archive
   behavior.
3. Add a read-only probe that reports source kind, build identity, listfile availability, virtual
   path lookup, FileDataID lookup where supported, and capability failures.
4. Assemble representative configured fixtures for 0.5.3, 1.x, 3.3.5, 4.0.0, 6.x, 7.x, and
   11.x; record hashes and paths without copying proprietary data into the repository.
5. Compare the first terrain file reads against existing `NativeMpqService` output and the chosen
   external reference tool for each applicable build.

**Gate**: Every fixture has a reproducible profile and probe report. No adapter is selected by
directory heuristic alone, and a missing capability is reported explicitly.

## Phase 1 — CascLib compatibility adapter (P1)

1. Add a minimal native loading boundary for the pinned CascLib build and fail clearly on missing
   DLL, wrong architecture, unsupported product, or invalid storage.
2. Implement virtual-path and FileDataID reads through the shared archive contract.
3. Implement listfile integration without changing the semantics of the existing listfile cache.
4. Add disposal, concurrent-read, and repeated-read tests using a small user-provided fixture.
5. Prove the adapter on a 6.x build before connecting it to terrain discovery.

**Gate**: 6.x source probe and one terrain tile read succeed with provenance; no MPQ fallback is
   silently used.

## Phase 2 — Later CASC adapter and source selection (P1)

1. Evaluate TACTSharp behind the same shared contract on 7.x and one later build.
2. Record whether local, CDN-structured, online, encrypted, locale, and install-tag modes pass;
   unsupported modes remain explicit capability gaps.
3. Add deterministic adapter selection from the build/profile record, with CascLib fallback only
   when the profile explicitly allows it and the fallback is separately recorded.
4. Keep TACT.Net and `Warcraft.NET` as isolated reference/acquisition inputs until license and
   runtime ownership review is complete.
5. Compare selected reads and map/minimap discovery with WoWTools.Minimaps and wow.export.

**Gate**: 7.x and later source reads produce equivalent bytes for the selected file set or a
   documented, build-scoped difference. Adapter selection and fallback are visible in provenance.

## Phase 3 — Common terrain profile and basic render path (P1)

1. Define the minimum terrain capability set: height, normals, layer textures/alpha, map tile
   ownership, liquid presence, and available baked color/light/shadow signals.
2. Route monolithic and split ADT ownership through the profile without duplicating terrain readers.
3. Keep `MCCV`, `MCLV`, and `MCSH` optional and build-scoped; do not assume later `MCTV`/`MCMT`
   fields exist in the 4.0.0 profile without evidence.
4. Integrate the profile with the existing terrain/minimap compositor and renderer using explicit
   capability checks.
5. Validate one basic tile per era checkpoint and report missing optional signals separately from
   base terrain failure.

### 0.5.3 transfer-corpus sub-gate

Before accepting any real 0.5.3 tile into the v60 transfer sample, complete these independently
checkable slices:

1. Honor every Alpha `MCLY.offsAlpha` and add regression coverage for non-contiguous alpha blocks.
2. Add a height-written mask and remove zero-as-missing gap filling from absolute MCVT extraction.
3. Split raw packed MCSH provenance from any synthetic `terrain_shadow_256` signal and make missing
   required signals fail closed.
4. Label Alpha MDDF/MODF masks as auxiliary placement labels until MCRF and screen-space
   visibility semantics are implemented.
5. Read external WDL separately and record the 0.5.3 LIT/solar evidence without claiming exact
   time interpolation before it is recovered.

**Gate**: Basic terrain loads through the same consumer path for 0.5.3, 1.x, 3.3.5, 4.0.0, 6.x,
7.x, and 11.x representative fixtures, with per-build provenance and capability reports.

## Phase 4 — Later terrain capabilities (P2)

1. Add one later chunk or signal at a time, beginning with the highest-value terrain signal shown
   by the profile matrix.
2. Add real-data fixtures and a focused visual/semantic test for each signal before enabling it in
   another profile.
3. Use wow.export and WoWTools.Minimaps for byte/visual comparison where they expose equivalent
   outputs, while preserving viewer-owned contracts.
4. Add modern objects, liquids, materials, and WMO/M2 integration only after base terrain remains
   green across all earlier checkpoints.

**Gate**: Each capability has independent evidence, fallback behavior, and regression coverage;
   aggregate “the map loads” status is not sufficient.

## Deferred / explicitly out of scope

- Full 0.5.3 shadow parity until the cross-era source/profile foundation is working; the initial
  transfer corpus must first pass the 0.5.3 reader/provenance sub-gate above.
- Shipping a GPL-3.0 TACT.Net dependency without a completed license and distribution review.
- Copying wow.export or WoWTools.Minimaps into the runtime architecture.
- Broad client harvesting, GPU training, or long performance benchmarks during source-adapter work.
- Promoting unverified dossier claims such as `MCTV` or `MCMT` into universal format behavior.

## Complexity Tracking

No constitution violations are proposed. The multiple adapter projects are intentionally isolated
behind one existing archive contract because no single referenced library currently proves all
required source modes and eras.
