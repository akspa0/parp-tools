# Tasks: Modern Client Assets (Post-5.0.1, FileDataID Era)

**Input**: `specs/239-modern-client-assets/` (spec, plan, research)
**Release**: v0.6 · **Branch**: `v0.5.4-dev` · **Depends on**: Spec 238 remote gate T027 (or local T018, whichever lands first) for all real-build work
**Tests**: included. 🌐 = needs a real tier build opened through Spec 238.

Format: `- [ ] T### [P?] [US?] description (path)`. All paths are relative to `wow-viewer/`.

## Phase 1: Setup

- [ ] T001 Record tier builds in specs/239-modern-client-assets/research.md (R1): tier C defaults to the current live retail build via CDN; tiers A/B are chosen from the builds Spec 238 T006 finds servable (operator has no candidates yet)
- [ ] T002 [P] Create evidence folder README (receipt format including build identity from Spec 238) in specs/239-modern-client-assets/evidence/README.md

## Phase 2: Foundational — file reference resolver (blocks all stories)

- [ ] T003 [P] Create `FileReference` (path | fileDataId, origin chunk/field, required flag) in src/core/WowViewer.Core.IO/Files/FileReference.cs
- [ ] T004 Create `FileReferenceResolver` over IFileDataIdReader + path reader, returning FileReadResult with required/optional diagnostics, in src/core/WowViewer.Core.IO/Files/FileReferenceResolver.cs
- [ ] T005 [P] Resolver unit tests (id, path, missing optional silent, missing required reported) in tests/WowViewer.Core.Tests/FileReferenceResolverTests.cs
- [ ] T006 Pre-6.0 regression baseline: capture 0.5.3 / 3.3.5 / 5.0.1 reference renders and stats before any reader change in specs/239-modern-client-assets/evidence/baseline-pre60.md

## Phase 3: User Story 4 — Coverage survey (P2, done first: it ranks the backlog)

**Independent test**: survey a tier build map → per-format known/unknown/failed counts (SC-004).

- [ ] T007 [US4] `AssetCoverageSurvey` over the existing chunk walker (per format: files read, chunk ids known/unknown with counts, failures by reason including KeyUnavailable) in src/core/WowViewer.Core.IO/Survey/AssetCoverageSurvey.cs
- [ ] T008 [US4] CLI `asset-survey --build <cfg> --map <name> [--tiles N] [--json]` walking WDT → ADTs → placed M2/WMO in tools/inspect/WowViewer.Tool.Inspect/Program.cs
- [ ] T009 [US4] 🌐 Run the survey on tier A, B and C builds and save specs/239-modern-client-assets/evidence/phase0-coverage-tier{A,B,C}.json
- [ ] T010 [US4] Annotate each unknown chunk with where it is documented (Warcraft.NET era chunk folders, wow.export loaders, wowdev.wiki) and write a ranked backlog in specs/239-modern-client-assets/research.md

**Checkpoint (Phase 0 gate)**: ranked backlog recorded; SC-004 met.

## Phase 4: User Story 1 — Modern terrain (P1) 🎯 MVP

**Independent test**: a tier zone renders all tiles with ≥99% tile references resolved (SC-001).

- [ ] T011 [US1] WDT `MAID` per-tile file ids with the MPHD-flag name-based fallback in src/core/WowViewer.Core.IO/Maps/WdtTileIndexReader.cs
- [ ] T012 [P] [US1] Synthetic MAID tests (id slots, optional-slot absence) in tests/WowViewer.Core.Tests/WdtMaidTests.cs
- [ ] T013 [US1] 🌐 Placement file-id flags (MDDF 0x40, MODF 0x8), with power check: flagged values resolve as ids and fail as name indices (R5), in src/core/WowViewer.Core.IO/Maps/AdtPlacementReader.cs
- [ ] T014 [US1] `MDID`/`MHID` texture ids and `MTXP` params; reconcile with Spec 197 height-texturing work so only one implementation exists (R7), in src/core/WowViewer.Core.IO/Maps/AdtTextureReader.cs
- [ ] T015 [US1] Id-based tile resolution through FileReferenceResolver in src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs
- [ ] T016 [US1] Height-based blending in the terrain renderer (reuse 197's if landed) in src/viewer/WoWViewer/Terrain/TerrainRenderer.cs
- [ ] T017 [US1] Modern MH2O liquid resolution through liquid tables (default material until US3 T028 lands) in src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs
- [ ] T018 [US1] Terrain Alpha Risk Area regression: 0.5.3 + 3.3.5 blending compared to the T006 baseline in specs/239-modern-client-assets/evidence/phase2-alpha-regression.md
- [ ] T019 [US1] 🌐 Gate: SC-001 per tier with screenshots in specs/239-modern-client-assets/evidence/phase2-terrain.md

## Phase 5: User Story 2 — Modern doodads and WMOs (P1)

**Independent test**: ≥98% placements load per tier zone; a city renders (SC-002).

- [ ] T020 [US2] Decide Core.IO ownership vs the existing WarcraftNetM2Adapter path per format (R3) and record a field-by-field agreement check in specs/239-modern-client-assets/research.md
- [ ] T021 [US2] Chunked M2 `SFID` skins and `TXID` textures through the resolver extending the existing chunk walker in src/core/WowViewer.Core.IO/M2Chunked/M2ChunkedModelReader.cs (and src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs per T020)
- [ ] T022 [US2] `SKID`/`BFID`/`AFID` resolution for bind/idle pose with bind-pose degradation in src/core/WowViewer.Core.IO/M2Chunked/
- [ ] T023 [US2] WMO root `GFID` group ids, `MODI` doodad ids, and material texture ids in src/core/WowViewer.Core.IO/Wmo/
- [ ] T024 [US2] WMO/M2 era chunks from the T010 backlog, tier-ordered, one chunk per commit in src/core/WowViewer.Core.IO/Wmo/ and src/core/WowViewer.Core.IO/M2Chunked/
- [ ] T025 [US2] Encrypted or absent model → placeholder with recorded reason in src/viewer/WoWViewer/Rendering/ModelRenderer.cs
- [ ] T026 [US2] 🌐 Gate: SC-002 per tier plus a city screenshot in specs/239-modern-client-assets/evidence/phase3-objects.md

## Phase 6: User Story 3 — DB2-driven world context (P2)

**Independent test**: map list, area names and lights from the build's own tables (SC-003).

- [ ] T027 [US3] `Db2TableProvider`: table by file id plus build-matched definition via DBCD/WoWDBDefs (R6) in src/core/WowViewer.Core.IO/Dbc/Db2TableProvider.cs
- [ ] T028 [US3] Map list (`Map.db2`), area names (`AreaTable.db2`) and liquid type/material/object tables in src/viewer/WoWViewer/DataSources/ (DB2-by-id provider)
- [ ] T029 [US3] Light/sky tables consumed by the existing light service in src/viewer/WoWViewer/Terrain/LightService.cs
- [ ] T030 [US3] Definition-mismatch reporting and graceful feature degradation in src/core/WowViewer.Core.IO/Dbc/Db2TableProvider.cs
- [ ] T031 [US3] 🌐 Gate: SC-003 per tier in specs/239-modern-client-assets/evidence/phase4-db2.md

## Phase 7: User Story 5 — Older clients unaffected (P1)

- [ ] T032 [US5] Full test suite plus 0.5.3 / 3.3.5 / 5.0.1 render comparison against the T006 baseline (SC-006) in specs/239-modern-client-assets/evidence/phase5-regression.md

## Phase 8: Polish and tier sign-off

- [ ] T033 Operator visual comparison per tier against in-game references or build minimaps (SC-005) in specs/239-modern-client-assets/evidence/phase5-signoff.md
- [ ] T034 [P] Write docs/architecture/modern-client-assets.md (tier matrix, proven builds, measured chunk layouts)
- [ ] T035 [P] Document `asset-survey` and diff docs/CLI-TOOLS.md against the real parser
- [ ] T036 [P] Update memory-bank/activeContext.md, progress.md and specs/STATUS.md row 14

## Dependencies

- Spec 238 T027 (remote) or T018 (local) → Setup → Foundational → US4 (survey) → US1 → US2 → US3 → US5 → Polish.
- T017 (liquids) completes fully after T028.
- T014/T016 coordinate with Spec 197. Shader-combination fidelity is delegated to Spec 198.

## Parallel examples

- Foundational: T003 and T005 alongside T004.
- US1: T012 alongside T011.
- Polish: T034–T036.

## Implementation strategy

Survey first (US4), so every reader change is backed by measured occurrence. Then the MVP: terrain for one
zone on the **first tier build available**, which is tier C (live retail via the CDN) as of 2026-09-16. Tier B remains the
most thorough resolver test once a B build is identified; tier A follows as regression coverage.
