# Implementation Plan: Modern Client Assets (Post-5.0.1, FileDataID Era)

**Branch**: `v0.5.4-dev` (v0.6 release line) | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md) | **Release**: v0.6

> Authored from the template by hand. Use `$env:SPECIFY_FEATURE_DIRECTORY = 'specs/239-modern-client-assets'` for speckit scripts.

## Summary

The work is survey-first. Phase 0 builds a coverage survey, so the gap between "what a modern build
contains" and "what the viewer reads" is measured per format and per tier before any reader changes.
Phase 1 adds one FileDataID reference resolver used by every reader. Phases 2–4 extend the canonical
Core.IO readers tier by tier (WDT/ADT, then M2/WMO, then DB2-driven context), each proven on a real
build from the tier. Renderer changes are limited to what the decoded data needs (height blending,
modern liquid/material paths). Shader-combination fidelity stays with Spec 198.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Spec 238 `IFileDataIdReader`/`CascDataSource`; vendored DBCD (WDB2–WDC5) + WoWDBDefs; existing Core.IO WDT/ADT/M2/WMO readers; `StandardTerrainAdapter`; `WarcraftNetM2Adapter` (existing viewer path)

**Storage**: N/A (reads through the data source)

**Testing**: xUnit synthetic chunk tests. Real-build tests gated on `WOWVIEWER_CASC_LOCAL` + tier build config.

**Target Platform**: Windows desktop viewer, cross-platform core and CLI

**Project Type**: Library + CLI + desktop viewer

**Performance Goals**: Tile and model load time for a modern build within 1.5x of an equivalent 5.0.1 map on the same machine, excluding cold remote downloads.

**Constraints**: FR-011 (pre-6.0 unchanged); FR-012 (Core.IO owns readers); Terrain Alpha Risk Area (no MCAL changes without Alpha + LK checks)

**Scale/Scope**: 5 format families × 3 era tiers. Scope is bounded by the coverage survey backlog, not by the full format universe.

**NEEDS CLARIFICATION → research**: tier representative builds (R1); reuse of Warcraft.NET types vs Core.IO extension (R3); the DB2 definition selection mechanism (R6); the height-blend overlap with Spec 197 (R7)

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Repo independence | PASS | No new external references. wow.export is consulted online as a reference and not vendored. |
| II. Library-first / reader ownership | PASS | Readers are extended in Core.IO; the CLI survey and viewer consume them. Warcraft.NET stays a cross-check (R3). |
| III. Real-data validation | PASS (gated) | Every tier is proven on a real build (FR-010). |
| VI. No client path assumptions | PASS | Builds come from Spec 238 configuration. |
| VII. Containers are inputs | PASS | Read only. |
| Terrain alpha risk area | PASS (guarded) | Any alpha/blend change is checked against 0.5.3 and 3.3.5 baselines (Phase 2 regression step). |
| One phase at a time / bite-sized | PASS | 5 phases, at most 10 steps each. |

## Project Structure

```text
src/core/WowViewer.Core.IO/
├── Files/FileReference.cs                    # NEW: path | fileDataId, origin, required
├── Files/FileReferenceResolver.cs            # NEW: resolve through IFileDataIdReader / path reader
├── Survey/AssetCoverageSurvey.cs             # NEW: per-format chunk known/unknown/failed counts
├── Maps/ (WDT, ADT readers)                  # + MAID, MDID/MHID, MTXP, placement FDID flags
├── M2Chunked/ (existing chunked reader)      # + SFID/TXID/SKID/BFID/AFID resolution
├── Wmo/ (WMO readers)                        # + GFID/MODI/material texture ids, era chunks
└── Dbc/Db2TableProvider.cs                   # NEW: table by fileDataId + build-matched definition via DBCD

src/viewer/WoWViewer/
├── Terrain/StandardTerrainAdapter.cs         # id-based tile resolution path
├── Rendering/ (M2, WMO, terrain)             # height blend, modern liquid/material hookups
└── DataSources/ (DB providers)               # DB2-by-id provider for Map/AreaTable/Light/Liquid

tools/inspect/WowViewer.Tool.Inspect/Program.cs   # asset-survey
docs/architecture/modern-client-assets.md         # tier matrix; measured chunk layouts
```

## Phases

### Phase 0: Tier builds + coverage survey (US4; FR-008)

1. The operator picks one representative build per tier (A: 6.x–8.0, B: 8.1–8.3, C: 9.x+), all openable through Spec 238. Record them in `research.md` R1.
2. `AssetCoverageSurvey` over the existing chunk walker: per format, count known/unknown chunk ids and read failures.
3. CLI `asset-survey --build <cfg> --map <name> [--tiles N]`: walks the WDT → ADTs → placed M2/WMO, with JSON output.
4. Run on each tier build → `evidence/phase0-coverage-tier{A,B,C}.json` plus a summary table.
5. Cross-check the unknown chunk list against Warcraft.NET's era chunk folders and wow.export's loaders, and annotate each unknown chunk with where it is documented.
6. **Gate**: ranked backlog in `research.md` (chunk → formats → renderer impact) and SC-004 met.

### Phase 1: File reference resolver (FR-001, FR-009)

1. `FileReference` + `FileReferenceResolver` (path or id through the Spec 238 reader, returning `FileReadResult`).
2. Required vs optional semantics with diagnostics; unit tests.
3. Adapter from existing name-based call sites: no behavior change for pre-6.0 (FR-011 tests).
4. **Gate**: unit tests pass, and the 0.5.3/3.3.5/5.0.1 regression renders are unchanged.

### Phase 2: WDT + ADT (US1; FR-002, FR-003, FR-007 terrain)

1. WDT `MAID` per-tile ids, plus the name-based fallback by `MPHD` flag.
2. ADT placements: file-id flag handling for doodads and WMOs.
3. `MDID`/`MHID` texture ids and `MTXP` params. Reconcile with Spec 197's height-texturing work (R7).
4. `StandardTerrainAdapter` id-based tile resolution through the resolver.
5. Height-based blending in the terrain renderer (reuse 197 if present).
6. Liquids: modern `MH2O` → liquid tables by id (depends on Phase 4 step 1 for tables; fall back to the default material until then).
7. Regression: Alpha 0.5.3 and LK 3.3.5 terrain blending unchanged (Terrain Alpha Risk Area).
8. **Gate**: `evidence/phase2-terrain.md` shows SC-001 terrain for each tier build, with screenshots.

### Phase 3: M2 + WMO (US2; FR-004, FR-005)

1. Chunked M2: `SFID` skins and `TXID` textures through the resolver (static display).
2. `SKID`/`BFID`/`AFID`: resolve for bind/idle pose, degrading to bind pose when missing.
3. Decide Core.IO ownership vs the existing `WarcraftNetM2Adapter` path (R3), and document it.
4. WMO root: `GFID` group ids, `MODI` doodad ids, material texture ids.
5. WMO era chunks surfaced by the Phase 0 backlog (tier-ordered).
6. Encrypted or absent model → placeholder with reason (US2 scenario 4).
7. **Gate**: `evidence/phase3-objects.md` shows SC-002 per tier and a city screenshot.

### Phase 4: DB2-driven world context (US3; FR-006)

1. `Db2TableProvider`: table by file id + build-matched definition via DBCD/WoWDBDefs.
2. Map list from `Map.db2`; area names from `AreaTable.db2`.
3. Light/sky tables for the tier builds (the existing light service consumes them).
4. Liquid type/material/object tables (feeds Phase 2 step 6).
5. Definition-mismatch reporting and degradation.
6. **Gate**: `evidence/phase4-db2.md` shows SC-003.

### Phase 5: Tier sign-off (SC-005, SC-006)

1. The operator visually compares each tier zone to in-game references or minimaps.
2. Full regression suite, and pre-6.0 render comparisons.
3. Update `docs/architecture/modern-client-assets.md` with the tier matrix and proven builds.
4. **Gate**: `evidence/phase5-signoff.md`.

## Complexity Tracking

No constitution violations.
